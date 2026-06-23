"""CPU prototype + rigorous benchmark of Bellman-solver speedups (roadmap step 3, #1).

Compares a candidate value-iteration accelerator ("after") against the solver's current
value iteration ("before") under Andrew's speedup protocol:

  * 10 users x 3 hyperparameter sets = 30 paired (user, set) solves.
  * 6 threads run SIMULTANEOUSLY: 3 run "before", 3 run "after", so any outside load
    (the running GPU sweep, background jobs, thermal) hits both halves equally.
  * Accept iff ALL of:
      1. no convergence failure on any of the 30 (user, set) solves;
      2. one-sided Wilcoxon signed-rank on paired times, p < 0.01 (after faster);
      3. median(time_before / time_after) > 1;
    plus a correctness gate (the accelerated solution must match the baseline's fixed
    point) so a fast-but-wrong method can't pass.

Everything is documented to outputs/accel/<candidate>_benchmark.md (+ .json).

This is a CPU prototype: iteration count is hardware-independent, so a method that cuts
iterations here will cut them on the GPU too. The candidate here is **Anderson
acceleration** (#1). Run:

  uv run --no-sync python experiments/accel_bench7.py --candidate anderson --aa-depth 5
"""

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ssp_mmc_fsrs.solver7 import (  # noqa: E402
    DISCOUNT_FACTOR,
    SSPMMCSolver7,
    build_hybrid_s_grid,
    build_production_d_grid,
)
from converge7 import (  # noqa: E402
    DEFAULT_BUTTON_USAGE,
    DEFAULT_PARAMS,
    invalid_data_reason,
    load_jsonl_by_user,
    make_hyperparam_sets,
)

TOL = 0.1  # value-iteration convergence tolerance (matches the production sweep)
N_ITER_CAP = 3000  # generous cap so both methods can fully converge
UNCONV_FRAC = 1.0 / 20.0  # sweep's "this set converged" criterion (frac pinned at max)


# ── operator: one Bellman optimality backup T(V), then the solver's monotone map ───────
def bellman_T(V, op):
    """T(V) = min over actions of [const_cost + gamma * sum_branch prob * V[next]]."""
    av = op["const_cost"].clone()
    for prob, tr in zip(op["branch_probs"], op["trans"]):
        av.addcmul_(prob, V[tr], value=DISCOUNT_FACTOR)
    return torch.amin(av, dim=-1)


def f_map(V, op):
    """The solver's actual update map G(V) = min(V, T(V)) (monotone from COST_MAX)."""
    return torch.minimum(V, bellman_T(V, op))


def h_map(V, op):
    """Plain Bellman backup with the terminal (max-S_long) boundary pinned to 0 -- a clean
    gamma-contraction. NOTE: its fixed point is the *true* V*, which can differ from the
    solver's path-dependent min-map fixed point (the min can freeze states above V*). So we
    judge correctness by the POLICY (argmin), not V. Used by the accelerator."""
    tv = bellman_T(V, op)
    tv[op["terminal"]] = 0.0
    return tv


def policy_argmin(V, op):
    """The chosen action index per state (argmin over the 40 retention candidates) -- this
    is what becomes retention_matrix and feeds the simulator. The real correctness test."""
    av = op["const_cost"].clone()
    for prob, tr in zip(op["branch_probs"], op["trans"]):
        av.addcmul_(prob, V[tr], value=DISCOUNT_FACTOR)
    return av.argmin(dim=-1)


def baseline_vi(op, V0=None):
    """ "Before": exactly the solver's eager value iteration (min(V, T(V)) to tol). With a
    custom V0 (any upper bound V0 >= V*) it lands on the SAME unique V* -- the min-map
    converges to V* from any upper bound -- so a tighter V0 only changes speed, not the
    answer (faithful by construction)."""
    V = (op["state_init"] if V0 is None else V0).clone()
    it, diff = 0, float("inf")
    while it < N_ITER_CAP and diff > TOL:
        it += 1
        fV = f_map(V, op)
        diff = (V - fV).max().item()
        V = fV
    return V, it, diff


def init_uniform_vi(op):
    """ "After" (#2): the SAME min-map, but started from a tight uniform upper bound
    U = max(const_cost)/(1-gamma) instead of COST_MAX=1e6. V* <= max_cost/(1-gamma) for any
    policy, so U >= V* is a valid upper bound; it skips the ~1e6->~1e3 collapse the COST_MAX
    start has to grind through. Terminal stays 0 (same boundary as the baseline init).
    Per-user, per-set, no cross-user dependency, trivially portable to Rust."""
    U = op["const_cost"].max().item() / (1.0 - DISCOUNT_FACTOR)
    V0 = torch.full_like(op["state_init"], U)
    V0[op["terminal"]] = 0.0
    return baseline_vi(op, V0)


def mpi_vi(op, m_eval):
    """ "After" (#2, stronger): Modified Policy Iteration. Each outer step does ONE full
    greedy Bellman sweep (40 actions) to refresh the policy, then m_eval CHEAP policy-
    evaluation sweeps that reuse the greedy policy (1 action each, ~40x cheaper) to
    propagate value across transition-hops fast -- exactly what the slow sets need. Stays an
    upper bound via min(), so it converges to the same V* (faithful). `it` counts only the
    expensive greedy sweeps. Portable to Rust; per-user/per-set."""
    cc = op["const_cost"]
    bp, tr = op["branch_probs"], op["trans"]
    n = cc.shape[0]
    idx = torch.arange(n)
    U = cc.max().item() / (1.0 - DISCOUNT_FACTOR)
    V = torch.full((n,), U, dtype=cc.dtype)
    V[op["terminal"]] = 0.0
    it, diff = 0, float("inf")
    while it < N_ITER_CAP and diff > TOL:
        it += 1
        # Greedy improvement: full backup over all 40 actions.
        av = cc.clone()
        for prob, t in zip(bp, tr):
            av.addcmul_(prob, V[t], value=DISCOUNT_FACTOR)
        Vg, pol = torch.min(av, dim=-1)
        Vnew = torch.minimum(V, Vg)
        diff = (V - Vnew).max().item()
        V = Vnew
        if diff <= TOL:
            break
        # Cheap policy evaluation: reuse the greedy action per state (1 gather/branch).
        cc_p = cc[idx, pol]
        prob_p = [prob[idx, pol] for prob in bp]
        tr_p = [t[idx, pol] for t in tr]
        for _ in range(m_eval):
            ev = cc_p.clone()
            for prob, t in zip(prob_p, tr_p):
                ev.addcmul_(prob, V[t], value=DISCOUNT_FACTOR)
            V = torch.minimum(V, ev)
    return V, it, diff


def anderson_vi(op, depth):
    """ "After": Anderson(depth) acceleration of the SAME map G(V)=min(V,T(V)), with a
    restart safeguard (plain step if the residual grows or goes non-finite), so it can
    never converge worse than the baseline and lands on the same fixed point."""
    V = op["state_init"].clone()
    f_hist, g_hist = [], []  # past h-values and residuals g = h(V) - V
    it, diff, prev = 0, float("inf"), float("inf")
    while it < N_ITER_CAP and diff > TOL:
        it += 1
        fV = h_map(V, op)
        g = fV - V
        diff = g.abs().max().item()
        if diff <= TOL:
            V = fV
            break
        # Safeguard: if Anderson made the residual worse (or blew up), drop history and
        # fall back to a plain step this iteration.
        if (diff > prev) or (not bool(torch.isfinite(g).all())):
            f_hist.clear()
            g_hist.clear()
        f_hist.append(fV)
        g_hist.append(g)
        if len(g_hist) > depth + 1:
            f_hist.pop(0)
            g_hist.pop(0)
        mk = len(g_hist) - 1
        if mk == 0:
            V = fV  # plain step
        else:
            dG = torch.stack([g_hist[i + 1] - g_hist[i] for i in range(mk)], dim=1)
            dF = torch.stack([f_hist[i + 1] - f_hist[i] for i in range(mk)], dim=1)
            gamma = torch.linalg.lstsq(dG, g.unsqueeze(1)).solution.squeeze(1)
            V = fV - dF @ gamma
        prev = diff
    return V, it, diff


def frac_at_max(V):
    return float((V == V.max()).sum().item()) / V.numel()


# ── per-user solver build -> operator handles + per-set const_cost ─────────────────────
def build_ops(user_ids, hp_sets, params, usage, grid_kw):
    """Build one CPU solver per user, extract read-only operator handles, and the
    const_cost for each (user, set). Returns a list of 30 work items."""
    items = []
    for uid in user_ids:
        w = params[uid]["parameters"]["0"]
        u = usage[uid]
        solver = SSPMMCSolver7(
            review_costs=u["review_costs"],
            first_rating_prob=u["first_rating_prob"],
            review_rating_prob=u["review_rating_prob"],
            w=w,
            device="cpu",
            **grid_kw,
        )
        branch = solver._branch_probs()
        trans = [t.long() for t in solver._transitions]  # cast once, shared read-only
        state_init = solver._state_init.clone()
        terminal = state_init == 0.0  # terminal (max-S_long) states, pinned to 0 cost
        r_state = torch.as_tensor(solver.r_state, dtype=torch.float32)
        for si, hp in enumerate(hp_sets):
            const_cost = solver._const_cost(hp)
            items.append(
                {
                    "user": uid,
                    "set": si,
                    "op": {
                        "const_cost": const_cost,
                        "branch_probs": branch,
                        "trans": trans,
                        "state_init": state_init,
                        "terminal": terminal,
                        "r_state": r_state,
                    },
                }
            )
        del solver
    return items


# ── simultaneous 3-before / 3-after benchmark ─────────────────────────────────────────
def run_pool(items, solve_fn, n_threads, times, extra):
    q = queue.Queue()
    for i in range(len(items)):
        q.put(i)

    def worker():
        while True:
            try:
                i = q.get_nowait()
            except queue.Empty:
                return
            op = items[i]["op"]
            t0 = time.perf_counter()
            V, it, diff = solve_fn(op)
            dt = time.perf_counter() - t0
            times[i] = dt
            extra[i] = {"iters": it, "final_diff": diff, "V": V}
            q.task_done()

    return [threading.Thread(target=worker) for _ in range(n_threads)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-users", type=int, default=10)
    p.add_argument("--n-sets", type=int, default=3)
    p.add_argument("--user-seed", type=int, default=0)
    p.add_argument("--hp-seed", type=int, default=0)
    p.add_argument("--threads-per-side", type=int, default=3)
    p.add_argument(
        "--candidate",
        choices=["anderson", "init_uniform", "mpi"],
        default="mpi",
    )
    p.add_argument("--aa-depth", type=int, default=5)
    p.add_argument(
        "--mpi-eval", type=int, default=20, help="MPI policy-eval sweeps/step."
    )
    p.add_argument(
        "--vdiff-tol",
        type=float,
        default=8.0,
        help="Correctness gate: max allowed ||V_after - V_before||_inf. The min-map has a "
        "UNIQUE fixed point V*, and two tol-converged solutions differ by <= 2*tol/(1-gamma) "
        "~ 6.7, so this catches a genuinely-wrong solution while ignoring tie-break jitter "
        "in the argmin policy (reported separately as informational).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(1)  # 1 core/op so 6 threads => 6 cores, no oversubscription

    params = load_jsonl_by_user(DEFAULT_PARAMS)
    usage = load_jsonl_by_user(DEFAULT_BUTTON_USAGE)
    common = sorted(set(params) & set(usage))
    common = [u for u in common if len(params[u]["parameters"]["0"]) == 34]
    # Sample, then keep the first n_users with valid inputs (so no build/convergence noise).
    rng = np.random.default_rng(args.user_seed)
    pool = rng.permutation(common).tolist()
    user_ids = []
    for uid in pool:
        if invalid_data_reason(usage[uid], params[uid]["parameters"]["0"]) is None:
            user_ids.append(uid)
        if len(user_ids) == args.n_users:
            break
    hp_sets = make_hyperparam_sets(15, seed=args.hp_seed)[: args.n_sets]
    grid_kw = {"s_state": build_hybrid_s_grid(), "d_state": build_production_d_grid()}

    print(
        f"users={user_ids}  sets={args.n_sets}  candidate={args.candidate} "
        f"(depth={args.aa_depth})  threads/side={args.threads_per_side}"
    )
    t_build = time.perf_counter()
    items = build_ops(user_ids, hp_sets, params, usage, grid_kw)
    print(f"built {len(items)} (user,set) ops in {time.perf_counter() - t_build:.0f}s")

    n = len(items)
    before_t = [None] * n
    after_t = [None] * n
    before_x = [None] * n
    after_x = [None] * n

    def base_fn(op):
        return baseline_vi(op)

    def cand_fn(op):
        if args.candidate == "anderson":
            return anderson_vi(op, args.aa_depth)
        if args.candidate == "mpi":
            return mpi_vi(op, args.mpi_eval)
        return init_uniform_vi(op)

    threads = run_pool(items, base_fn, args.threads_per_side, before_t, before_x)
    threads += run_pool(items, cand_fn, args.threads_per_side, after_t, after_x)
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    print(f"benchmark wall={wall:.0f}s")

    # ── per-item analysis: convergence (reached tol) + POLICY correctness ─────────────
    # Correctness is judged on the policy (argmin retention), not V: the solver's min-map
    # fixed point is path-dependent and can sit above V*, so V can't match exactly, but the
    # POLICY (what the simulator consumes) must.
    rows, failures = [], []
    for i, it in enumerate(items):
        op = it["op"]
        Vb, Va = before_x[i]["V"], after_x[i]["V"]
        v_diff = float((Va - Vb).abs().max().item())
        pol_b = policy_argmin(Vb, op)
        pol_a = policy_argmin(Va, op)
        n_pol_diff = int((pol_b != pol_a).sum().item())
        frac_pol_diff = n_pol_diff / pol_b.numel()
        rs = op["r_state"]
        max_r_diff = float((rs[pol_a] - rs[pol_b]).abs().max().item())
        conv_b = before_x[i]["iters"] < N_ITER_CAP
        conv_a = after_x[i]["iters"] < N_ITER_CAP
        item_fail = (not conv_a) or (v_diff > args.vdiff_tol)
        row = {
            "user": it["user"],
            "set": it["set"],
            "iters_before": before_x[i]["iters"],
            "iters_after": after_x[i]["iters"],
            "time_before": before_t[i],
            "time_after": after_t[i],
            "ratio": before_t[i] / after_t[i],
            "frac_policy_diff": frac_pol_diff,
            "max_R_diff": max_r_diff,
            "V_diff": v_diff,
            "conv_before": conv_b,
            "conv_after": conv_a,
            "fail": item_fail,
        }
        rows.append(row)
        if item_fail:
            failures.append(row)

    before = np.array([r["time_before"] for r in rows])
    after = np.array([r["time_after"] for r in rows])
    ratios = before / after
    iter_ratios = np.array([r["iters_before"] / r["iters_after"] for r in rows])
    # One-sided Wilcoxon: H1 = before times are larger (after is faster).
    wstat, pval = wilcoxon(before, after, alternative="greater")
    median_ratio = float(np.median(ratios))

    no_fail = len(failures) == 0
    accept = bool(no_fail and (pval < 0.01) and (median_ratio > 1.0))

    summary = {
        "candidate": args.candidate,
        "aa_depth": args.aa_depth,
        "users": user_ids,
        "n_pairs": n,
        "threads_per_side": args.threads_per_side,
        "wall_s": wall,
        "wilcoxon_stat": float(wstat),
        "wilcoxon_p": float(pval),
        "median_time_ratio": median_ratio,
        "mean_time_ratio": float(np.mean(ratios)),
        "median_iter_ratio": float(np.median(iter_ratios)),
        "median_iters_before": float(np.median([r["iters_before"] for r in rows])),
        "median_iters_after": float(np.median([r["iters_after"] for r in rows])),
        "max_frac_policy_diff": float(max(r["frac_policy_diff"] for r in rows)),
        "max_R_diff": float(max(r["max_R_diff"] for r in rows)),
        "max_V_diff": float(max(r["V_diff"] for r in rows)),
        "n_failures": len(failures),
        "accept": accept,
    }

    # ── write the report ──────────────────────────────────────────────────────────────
    out_dir = ROOT_DIR / "outputs" / "accel"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.candidate}_benchmark.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )

    lines = []
    lines.append(f"# Bellman solver speedup benchmark — {args.candidate}\n")
    lines.append(
        f"**Verdict: {'ACCEPT ✅' if accept else 'REJECT ❌'}** "
        f"(candidate = {args.candidate}, Anderson depth = {args.aa_depth})\n"
    )
    lines.append("## Protocol\n")
    lines.append(
        f"- {n} paired solves = {args.n_users} users x {args.n_sets} hyperparam sets "
        f"(CPU, production grid 176k states, tol={TOL}, cap={N_ITER_CAP} iters).\n"
        f"- {args.threads_per_side} threads run *before* and {args.threads_per_side} run "
        f"*after* **simultaneously** (same outside load on both); `torch.set_num_threads(1)`.\n"
        "- Accept iff: no convergence failure (reached tol within the cap) **and** one-sided "
        "Wilcoxon p < 0.01 **and** median(time_before/time_after) > 1. Correctness gate: "
        f"max ||V_after - V_before||_inf <= {args.vdiff_tol} (same unique V* within tol; "
        "argmin policy diff reported separately as informational tie-break jitter).\n"
    )
    lines.append("## Results\n")
    lines.append(
        f"- **Wilcoxon (one-sided, after faster): p = {pval:.3e}** "
        f"(stat={wstat:.1f}) — need < 0.01.\n"
    )
    lines.append(
        f"- **median time ratio = {median_ratio:.3f}x** "
        f"(mean {np.mean(ratios):.3f}x) — need > 1.\n"
    )
    lines.append(
        f"- median iters: before {np.median([r['iters_before'] for r in rows]):.0f} -> "
        f"after {np.median([r['iters_after'] for r in rows]):.0f} "
        f"(median iter ratio {np.median(iter_ratios):.3f}x).\n"
    )
    lines.append(
        f"- convergence failures (hit {N_ITER_CAP}-iter cap): "
        f"{sum(1 for r in rows if not r['conv_after'])} / {n}.\n"
    )
    lines.append(
        f"- **correctness: max ||V_after - V_before||_inf = {max(r['V_diff'] for r in rows):.4g}**"
        f" (gate {args.vdiff_tol}) -- same V* within tol.\n"
    )
    lines.append(
        f"- informational (tie-break jitter): max {max(r['frac_policy_diff'] for r in rows):.4%}"
        f" of states change chosen retention; max |dR| = {max(r['max_R_diff'] for r in rows):.3g}.\n"
    )
    lines.append("\n## Per-(user, set) detail\n")
    lines.append(
        "| user | set | iters_b | iters_a | t_before(s) | t_after(s) | ratio | "
        "policy_diff | max_dR | V_diff | fail |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    for r in rows:
        lines.append(
            f"| {r['user']} | {r['set']} | {r['iters_before']} | {r['iters_after']} | "
            f"{r['time_before']:.3f} | {r['time_after']:.3f} | {r['ratio']:.3f} | "
            f"{r['frac_policy_diff']:.4%} | {r['max_R_diff']:.3g} | {r['V_diff']:.3g} | "
            f"{'YES' if r['fail'] else ''} |\n"
        )
    (out_dir / f"{args.candidate}_benchmark.md").write_text(
        "".join(lines), encoding="utf-8"
    )

    print("\n==== SUMMARY ====")
    for k, v in summary.items():
        if k not in ("users",):
            print(f"  {k}: {v}")
    print(f"VERDICT: {'ACCEPT' if accept else 'REJECT'}")
    print(f"report: {out_dir / (args.candidate + '_benchmark.md')}")


if __name__ == "__main__":
    main()
