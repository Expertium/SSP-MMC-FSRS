"""Grid-fineness accuracy check for the FSRS-7 SSP-MMC solver (step 3, grid-tradeoff point 2).

Measures how the *simulated* outcomes -- mean memorized cards and mean study time per day --
change when the Bellman solver's S-grid fineness changes. For the same users and the same
random SSP-MMC cost-hyperparameter sets, we build a policy at a FINE reference grid
(``--ref-n-s``, default 225) and a COARSE grid (``--n-s``, default 135), simulate both with
*identical* simulator seeds (so the only difference is the grid), and report per-(user, set)
and aggregate AVERAGE and MAX relative differences for each metric.

Only the S axis changes; the difficulty grid (0.1 steps -> 91) and the action grid
(0.01 retention steps -> 40) are already fine and held fixed. ``discount_factor`` is the
solver default 0.97 (off-limits as a knob, per CLAUDE.md).

VRAM: the 225-pt build peaks ~4.4 GB (calibrated against tests/bench_solver7_batched.py),
chosen to fit alongside the running 10k convergence sweep under a <5 GB budget.

Per-user inputs (CLAUDE.md): FSRS-7 params from FSRS-7-short-secs-recency.jsonl (34 each),
per-user costs + Hard/Good/Easy probabilities from button_usage.jsonl -- used for BOTH the
Bellman solve and the simulation.

Run:
  uv run --no-sync python experiments/grid_accuracy7.py
  uv run --no-sync python experiments/grid_accuracy7.py --n-users 10 --n-sets 3 --learn-span 365
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ssp_mmc_fsrs import fsrs7  # noqa: E402
from ssp_mmc_fsrs.solver7 import SSPMMCSolver7, COST_MAX  # noqa: E402
from ssp_mmc_fsrs.simulation7 import simulate  # noqa: E402
from converge7 import (  # noqa: E402
    make_hyperparam_sets,
    load_jsonl_by_user,
    invalid_data_reason,
    DEFAULT_PARAMS,
    DEFAULT_BUTTON_USAGE,
)


def _normalize(p):
    """Renormalize a probability vector to sum to 1 (button_usage stores 4-decimal probs
    that sum to e.g. 1.0001, which trips np.random.choice's strict tolerance)."""
    a = np.asarray(p, dtype=float)
    return (a / a.sum()).tolist()


def make_ssp_mmc_policy(solver, retention_matrix, w, device):
    """FSRS-7 SSP-MMC simulator policy: look up the solved target retention for each card's
    ``(d, s_long, s_short)`` state and invert the dual forgetting curve to an interval.

    Mirrors the FSRS-6 ``ssp_mmc_policy`` closure but for the 3-component state and the
    6-arg FSRS-7 policy signature. Cards that have reached the top S_long grid point are
    "done" (the solver pins cost 0 there) -- they get an effectively-infinite interval, so
    the simulator never reviews them again (matching the FSRS-6 terminal convention).

    IMPORTANT: capture only the small grid info (s_state, d-range, sizes) -- NOT the solver
    object, whose transition tables are several GB on the GPU. Capturing the solver would
    keep one user's tables alive (via this closure) into the next user's build and blow up
    VRAM (~2x -> OOM).
    """
    rm = torch.as_tensor(retention_matrix, device=device, dtype=torch.float64)
    w_t = torch.as_tensor(w, device=device, dtype=torch.float64)
    s_state = solver._s_state_t.clone()  # (s_size,) float32 grid -- tiny
    d_state = solver._d_state_t.clone()  # (d_size,) float32 grid -- tiny
    d_uniform = solver._d_uniform
    d_min, d_max, d_size = solver.d_min, solver.d_max, solver.d_size
    s_size = solver.s_size
    grid_dtype = solver.dtype  # solver grid is float32; sim state is float64

    def s2i(s):  # first grid point >= s (searchsorted dtypes must match -> cast)
        idx = torch.searchsorted(s_state, s.to(grid_dtype).contiguous())
        return idx.clamp_(0, s_size - 1)

    def d2i(d):  # mirrors solver.d2i_torch (floor for uniform, nearest for custom)
        if d_uniform:
            idx = torch.floor((d - d_min) / (d_max - d_min) * d_size).to(torch.long)
            return idx.clamp_(0, d_size - 1)
        dv = d.to(grid_dtype).contiguous()
        hi = torch.searchsorted(d_state, dv).clamp(0, d_size - 1)
        lo = (hi - 1).clamp(0, d_size - 1)
        pick_hi = (d_state[hi] - dv).abs() <= (dv - d_state[lo]).abs()
        return torch.where(pick_hi, hi, lo)

    def policy(s_long, s_short, d, prev_interval, grade, ease):
        sl_idx = s2i(s_long)
        ss_idx = s2i(s_short)
        d_idx = d2i(d)
        target_r = rm[d_idx, sl_idx, ss_idx]
        interval = fsrs7.forgetting_curve_inverse(target_r, s_long, s_short, d, w_t)
        terminal = sl_idx >= (s_size - 1)
        interval = torch.where(
            terminal, torch.full_like(interval, float("inf")), interval
        )
        return interval, ease

    return policy


def parse_args():
    p = argparse.ArgumentParser(
        description="FSRS-7 SSP-MMC grid-fineness accuracy check."
    )
    p.add_argument("--parameters", type=Path, default=DEFAULT_PARAMS)
    p.add_argument("--button-usage", type=Path, default=DEFAULT_BUTTON_USAGE)
    p.add_argument("--n-users", type=int, default=10, help="First N valid users.")
    p.add_argument("--n-sets", type=int, default=3, help="Random hyperparameter sets.")
    p.add_argument(
        "--hp-seed",
        type=int,
        default=0,
        help="Seed for hyperparam sets. Default 0 reuses the SAME sets the convergence "
        "sweep validated as converging on these users, so grid effects aren't confounded "
        "by a non-converged solve.",
    )
    p.add_argument("--ref-n-s", type=int, default=225, help="Fine reference S grid.")
    p.add_argument("--n-s", type=int, default=135, help="Coarse S grid to compare.")
    p.add_argument("--parallel", type=int, default=8, help="Monte-Carlo replica decks.")
    p.add_argument("--deck-size", type=int, default=10_000)
    p.add_argument("--learn-span", type=int, default=365, help="Simulated days.")
    p.add_argument(
        "--solve-iter",
        type=int,
        default=2000,
        help="Max value-iteration steps per solve (cap so a non-converging random "
        "hyperparameter set can't run to the 100k default at 225 pts). Converged sets "
        "stop at ~530 iters regardless; both grids use the same cap, so it stays paired.",
    )
    p.add_argument(
        "--sim-seed", type=int, default=42, help="Shared across grids (paired)."
    )
    p.add_argument("--device", type=str, default=None, help="cuda/cpu (default: auto).")
    p.add_argument(
        "--results",
        type=Path,
        default=ROOT_DIR / "outputs" / "checkpoints" / "grid_accuracy7.json",
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(1)  # be polite to the concurrently running convergence sweep
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    params = load_jsonl_by_user(args.parameters)
    usage = load_jsonl_by_user(args.button_usage)
    common = sorted(set(params) & set(usage))
    common = [u for u in common if len(params[u]["parameters"]["0"]) == 34]

    # First N users with usable data (skip the data-quality casualties).
    user_ids = []
    for u in common:
        if invalid_data_reason(usage[u], params[u]["parameters"]["0"]) is None:
            user_ids.append(u)
        if len(user_ids) == args.n_users:
            break

    hp_sets = make_hyperparam_sets(args.n_sets, seed=args.hp_seed)
    grids = [args.ref_n_s, args.n_s]  # reference first

    print(f"device: {device}  |  users: {user_ids}")
    print(f"grids (S points): ref={args.ref_n_s}, coarse={args.n_s}")
    print(
        f"sim: parallel={args.parallel}, deck={args.deck_size}, "
        f"span={args.learn_span}d, seed={args.sim_seed} (paired across grids)"
    )
    print(f"hyperparameter sets ({args.n_sets}, seed={args.hp_seed}):")
    for i, hp in enumerate(hp_sets):
        print(f"  set {i}: {hp}")

    # results[(n_s, user, set_idx)] = {"memorized","time","frac_at_max"}
    results = {}
    ref, coarse = args.ref_n_s, args.n_s
    total_cells = len(grids) * len(user_ids) * args.n_sets

    def write_results(summary=None):
        """Atomic incremental save (tmp + replace). ``complete`` flags a finished run so
        an external retry loop knows to stop. The 225-pt run needs ~10 GB and shares a
        12 GB card, so a transient OOM/TDR crash is possible -- this makes it resumable."""
        args.results.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "users": user_ids,
            "ref_n_s": ref,
            "coarse_n_s": coarse,
            "parallel": args.parallel,
            "deck_size": args.deck_size,
            "learn_span": args.learn_span,
            "solve_iter": args.solve_iter,
            "sim_seed": args.sim_seed,
            "hp_seed": args.hp_seed,
            "max_same_day": 8,
            "hyperparam_sets": hp_sets,
            "results": {f"{ns}|{uid}|{si}": v for (ns, uid, si), v in results.items()},
            "complete": summary is not None,
            "summary": summary,
        }
        tmp = args.results.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        tmp.replace(args.results)

    # Resume: load any cells already computed in a prior (possibly crashed) attempt.
    if args.results.exists():
        try:
            prev = json.loads(args.results.read_text(encoding="utf-8"))
            for k, v in prev.get("results", {}).items():
                ns, uid, si = (int(x) for x in k.split("|"))
                results[(ns, uid, si)] = v
            print(
                f"resuming: {len(results)}/{total_cells} cells already done", flush=True
            )
        except (json.JSONDecodeError, OSError) as e:
            print(f"could not resume ({e}); starting fresh", flush=True)

    t_start = time.perf_counter()

    for n_s in grids:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for user_id in user_ids:
            pending = [
                (si, hp)
                for si, hp in enumerate(hp_sets)
                if (n_s, user_id, si) not in results
            ]
            if not pending:
                print(f"n_s={n_s:3d} user {user_id}: already done, skip", flush=True)
                continue
            w = params[user_id]["parameters"]["0"]
            u = usage[user_id]
            first_rating_prob = _normalize(u["first_rating_prob"])
            review_rating_prob = _normalize(u["review_rating_prob"])
            tb = time.perf_counter()
            solver = SSPMMCSolver7(
                review_costs=u["review_costs"],
                first_rating_prob=first_rating_prob,
                review_rating_prob=review_rating_prob,
                w=w,
                device=device,
                n_s=n_s,
            )
            peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            build_dt = time.perf_counter() - tb
            for set_idx, hp in pending:
                cost_matrix, retention_matrix = solver.solve(
                    hp, n_iter=args.solve_iter, verbose=False
                )
                # Self-check: fraction of states still pinned at COST_MAX = unconverged
                # (same <5% criterion as converge7). Recorded so convergence is visible,
                # not assumed.
                frac_at_max = float((cost_matrix >= COST_MAX * 0.99).mean())
                policy = make_ssp_mmc_policy(solver, retention_matrix, w, device)
                _, _, memorized_cnt, cost = simulate(
                    parallel=args.parallel,
                    w=w,
                    policy=policy,
                    device=device,
                    deck_size=args.deck_size,
                    learn_span=args.learn_span,
                    learn_costs=u["learn_costs"],
                    review_costs=u["review_costs"],
                    first_rating_prob=first_rating_prob,
                    review_rating_prob=review_rating_prob,
                    seed=args.sim_seed,
                )
                mem = float(memorized_cnt.mean())
                tsec = float(cost.mean())  # mean study time per deck-day (seconds)
                results[(n_s, user_id, set_idx)] = {
                    "memorized": mem,
                    "time": tsec,
                    "frac_at_max": frac_at_max,
                }
                print(
                    f"  n_s={n_s:3d} user {user_id} set {set_idx}: "
                    f"memorized={mem:8.1f}  time/day={tsec / 60:6.2f} min  "
                    f"frac@max={frac_at_max:.2%}",
                    flush=True,
                )
                write_results()  # incremental save after every cell (crash recovery)
                del policy, retention_matrix, cost_matrix, memorized_cnt, cost
            del solver
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            print(
                f"n_s={n_s:3d} user {user_id}: build {build_dt:.1f}s, "
                f"peak VRAM {peak:.2f} GB",
                flush=True,
            )

    # ---- Relative differences: coarse (n_s) vs fine reference (ref_n_s). ----
    rel_mem, rel_time = [], []
    rows = []
    for user_id in user_ids:
        for set_idx in range(args.n_sets):
            r = results[(ref, user_id, set_idx)]
            c = results[(coarse, user_id, set_idx)]
            dm = abs(c["memorized"] - r["memorized"]) / r["memorized"]
            dt = abs(c["time"] - r["time"]) / r["time"]
            rel_mem.append(dm)
            rel_time.append(dt)
            rows.append((user_id, set_idx, r, c, dm, dt))

    print(f"\n==== {coarse}-pt vs {ref}-pt reference (relative differences) ====")
    print(
        f"{'user':>6} {'set':>3} | {'mem ref':>9} {'mem coarse':>10} {'d_mem%':>7} | "
        f"{'t ref(m)':>9} {'t coarse(m)':>11} {'d_t%':>7}"
    )
    for user_id, set_idx, r, c, dm, dt in rows:
        print(
            f"{user_id:>6} {set_idx:>3} | {r['memorized']:>9.1f} {c['memorized']:>10.1f} "
            f"{100 * dm:>6.2f}% | {r['time'] / 60:>9.2f} {c['time'] / 60:>11.2f} "
            f"{100 * dt:>6.2f}%"
        )

    summary = {
        "memorized_rel_diff_mean": float(np.mean(rel_mem)),
        "memorized_rel_diff_max": float(np.max(rel_mem)),
        "time_rel_diff_mean": float(np.mean(rel_time)),
        "time_rel_diff_max": float(np.max(rel_time)),
    }
    print("\n---- summary ----")
    print(
        f"memorized: avg rel diff = {100 * summary['memorized_rel_diff_mean']:.2f}%, "
        f"max = {100 * summary['memorized_rel_diff_max']:.2f}%"
    )
    print(
        f"time/day:  avg rel diff = {100 * summary['time_rel_diff_mean']:.2f}%, "
        f"max = {100 * summary['time_rel_diff_max']:.2f}%"
    )
    print(f"total wall time: {time.perf_counter() - t_start:.0f}s")

    write_results(summary)
    print(f"Saved results to {args.results}")
    print("GRID ACCURACY COMPLETE")


if __name__ == "__main__":
    main()
