"""FSRS-7 SSP-MMC hyperparameter optimizer (roadmap step 5).

Goal: find SSP-MMC cost-hyperparameter sets that **Pareto-beat fixed desired retention (DR)**
on the **knowledge-vs-workload** tradeoff, tuned over many users with the GRU as the
pseudo-ground-truth p(recall). This is the FSRS-7 successor to ``hyperparameter_optimizer.py``
(FSRS-6, kept untouched); it drops that file's per-user-input aggregation, dead knee-point
code, label/policy_configs machinery, and the broken candidate-proposal RNG.

Two objectives (aggregated as the **mean across users**):
  * ``average_knowledge`` (MAXIMIZE) -- per user, the mean over days of ``sum_cards p_recall``
    (i.e. ``memorized.mean()``). Across-days mean, NOT last-day: last-day is gameable by a
    policy that crams all reviews just before the horizon.
  * ``time_per_day_min`` (MINIMIZE) -- per user, mean daily study time in minutes
    (``cost.mean() / 60``).

Pipeline per (hp, user): build+solve the Bellman (Python/GPU ``SSPMMCSolver7``) with the hp ->
SSP-MMC retention table -> Rust ``simulate_fsrs7`` (policy="ssp_mmc") with the per-user GRU.
The **simulator batches all users in one call** (its ``parallel`` axis = users, run with
rayon); only the per-user Bellman solves loop. The DR baseline needs no solve (DR inverts the
curve inside the sim), so it is one batched sim per retention level.

Deliverable (research output): the aggregate Pareto front + the subset of hp sets that
dominate the aggregate fixed-DR front. Written to ``<out>/pareto7.json``.

Run:
    uv run --no-sync python -m experiments.hyperparameter_optimizer7 \
        --n-users 20 --total-trials 200 --seed 42
    (add --dr-baseline-only to just (re)build the DR front; --regen-dr-baseline to refresh it)

NOTE (performance): per trial = N x (build + solve) + one batched sim, STREAMED -- each user's
transitions (~113 MB on the production grid) are rebuilt, solved, reduced to a ~1.4 MB retention
table, then freed, so peak VRAM stays ~3 GB for ANY N (caching all N is impossible: 1000 users
-> ~113 GB, fits neither VRAM nor RAM). The per-user build+solve is ~0.34 s, GPU-saturated (the
step-3 MPI+Triton work already made it cheap), so cross-user solve batching can't help; the sim
is ~0.22 s/user on CPU (rayon). => ~9.4 min/trial at 1000 users (~16-24 h for a 100-150 trial
tune). See the step5-optimizer memory.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.path.insert(0, str(ROOT / "experiments"))

import lib  # type: ignore  # noqa: E402

import ssp_mmc_rust  # noqa: E402
from ssp_mmc_fsrs import fsrs7  # noqa: E402
from ssp_mmc_fsrs.gru import BatchedGRU  # noqa: E402
from ssp_mmc_fsrs.simulation7 import S_MIN_SECS  # noqa: E402
from ssp_mmc_fsrs.solver7 import (  # noqa: E402
    SSPMMCSolver7,
    build_hybrid_s_grid,
    build_production_d_grid,
)

warnings.filterwarnings("ignore")

# ── fixed config (mirrors experiments/profile_pipeline.py; NOT the bench harness) ──────────
FSRS7_PARAMS = (
    ROOT.parent / "srs-benchmark" / "result" / "FSRS-7-short-secs-recency.jsonl"
)
BUTTON_USAGE = ROOT.parent / "Anki-button-usage" / "button_usage.jsonl"
GRU_DIR = ROOT / "outputs" / "gru_weights" / "GRU-short-secs"

S_GRID = build_hybrid_s_grid()
D_GRID = build_production_d_grid()
S_MIN = float(S_MIN_SECS)
S_MAX = float(fsrs7.S_MAX)
N_ITER = fsrs7.NEWTON_N_ITER  # sim's Newton scheduling-inverse count
MAX_COST = 86400 / 2
LEARN_LIMIT = 10
REVIEW_LIMIT = 9999
MAX_SAME_DAY = 8
DECK = 3000
SPAN = 365 * 3
SIM_SEED = 42
CUDA = torch.cuda.is_available()

# DR baseline retention grid (the fixed-DR policies we compare against).
DR_MIN, DR_MAX, DR_STEP = 0.60, 0.99, 0.01

# The 13-D SSP-MMC cost-hyperparameter search space. Names == SSPMMCSolver7.solve() keys;
# bounds mirror converge7.make_hyperparam_sets (base_fail is PINNED 1.0 inside the solver, so
# it is not searched).
PARAMETERS = [
    {"name": "transform_s_long", "type": "choice", "values": ["no_log", "log"]},
    {"name": "transform_s_short", "type": "choice", "values": ["no_log", "log"]},
    {
        "name": "exp_s_long",
        "type": "range",
        "bounds": [0.1, 10.0],
        "log_scale": True,
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "exp_s_short",
        "type": "range",
        "bounds": [0.1, 10.0],
        "log_scale": True,
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "exp_d",
        "type": "range",
        "bounds": [0.1, 10.0],
        "log_scale": True,
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "base_succ",
        "type": "range",
        "bounds": [-5.0, 5.0],
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "w_fail_s_long",
        "type": "range",
        "bounds": [-5.0, 5.0],
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "w_fail_s_short",
        "type": "range",
        "bounds": [-5.0, 5.0],
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "w_fail_d",
        "type": "range",
        "bounds": [-5.0, 5.0],
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "w_succ_s_long",
        "type": "range",
        "bounds": [-5.0, 5.0],
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "w_succ_s_short",
        "type": "range",
        "bounds": [-5.0, 5.0],
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "w_succ_d",
        "type": "range",
        "bounds": [-5.0, 5.0],
        "value_type": "float",
        "digits": 2,
    },
    {
        "name": "w_retention",
        "type": "range",
        "bounds": [0.0, 3.0],
        "value_type": "float",
        "digits": 2,
    },
]
PARAM_NAMES = [p["name"] for p in PARAMETERS]
CHOICE_NAMES = [p["name"] for p in PARAMETERS if p["type"] == "choice"]

OBJECTIVES = {
    "average_knowledge": ObjectiveProperties(minimize=False),
    "time_per_day_min": ObjectiveProperties(minimize=True),
}

# Manual-candidate exploration cadence (push candidates toward regions where SSP-MMC is still
# DR-dominated). Defaults; overridable via CLI.
MANUAL_CANDIDATE_START_TRIAL = 40
MANUAL_CANDIDATE_INTERVAL = 10

# Hypervolume early-stop (kept from the FSRS-6 optimizer). Because time is MINIMIZED, the
# hypervolume is computed on (knowledge, -time) so "bigger is better" on both axes; the
# reference point's y is below any achievable -time (time in [0, MAX_COST/60]).
HYPERVOLUME_TOLERANCE = 1e-3
HYPERVOLUME_PATIENCE = 3
HYPERVOLUME_CHECK_INTERVAL = 5
HYPERVOLUME_EPS = 1e-12
HYPERVOLUME_REF_POINT = (0.0, -(MAX_COST / 60.0 + 1.0))


# ── per-user data ─────────────────────────────────────────────────────────────────────────
class Users:
    """Per-user FSRS-7 params + costs/probs + GRU weights, stacked for the batched simulator.

    The simulator's ``parallel`` axis = users, so every per-user input is one row of a
    ``(N, ...)`` array aligned by ``uids`` order.
    """

    def __init__(self, uids):
        self.uids = list(uids)
        w, lc, rc, frp, rrp = [], [], [], [], []
        for uid in self.uids:
            wi, _, _ = lib.load_fsrs_weights(str(FSRS7_PARAMS), uid)
            cfg = lib.normalize_button_usage(
                lib.load_button_usage_config(str(BUTTON_USAGE), uid)
            )
            w.append(np.asarray(wi, np.float64))
            lc.append(np.asarray(cfg["learn_costs"], np.float64))
            rc.append(np.asarray(cfg["review_costs"], np.float64))
            frp.append(np.asarray(cfg["first_rating_prob"], np.float64))
            rrp.append(np.asarray(cfg["review_rating_prob"], np.float64))
        self.w = np.ascontiguousarray(np.stack(w))  # (N, 34)
        self.lc = np.ascontiguousarray(np.stack(lc))  # (N, 4)
        self.rc = np.ascontiguousarray(np.stack(rc))  # (N, 4)
        self.frp = np.ascontiguousarray(np.stack(frp))  # (N, 4)
        self.rrp = np.ascontiguousarray(np.stack(rrp))  # (N, 3)
        gru = BatchedGRU.from_pth_paths(
            [str(GRU_DIR / f"user_{uid}.pth") for uid in self.uids],
            device="cpu",
            dtype=torch.float64,
        )
        self.gflat = np.ascontiguousarray(
            gru.flat_weights().numpy(), np.float64
        )  # (N, 505)

    def __len__(self):
        return len(self.uids)


def parse_user_ids(spec):
    """Parse '1,2,5-10' style id specs into a sorted unique list."""
    out = []
    for tok in str(spec).replace(" ", "").split(","):
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            a, b = int(a), int(b)
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


# ── evaluation ────────────────────────────────────────────────────────────────────────────
def _solve_all(hp, users, progress_every=250):
    """Stream per-user build -> solve(hp) -> extract retention -> free (bounded VRAM at any N).

    Per-user transitions are ~113 MB on the production grid, so caching all N (1000 -> ~113 GB)
    fits neither VRAM nor RAM. Instead each trial REBUILDS transitions per user and frees them
    immediately, keeping only the small (~1.4 MB) retention tables -> peak VRAM ~constant (~3 GB)
    regardless of N. The build is ~3x the solve, but it's the price of not OOMing. (Measured
    ~0.34 s/user, GPU-saturated, so this is the throughput floor; see step5-optimizer memory.)
    Returns the stacked (N, d*s*s) retention table for the batched sim.
    """
    n = len(users)
    ret_rows = []
    for i in range(n):
        solver = SSPMMCSolver7(
            review_costs=users.rc[i],
            first_rating_prob=users.frp[i],
            review_rating_prob=users.rrp[i],
            w=users.w[i],
            s_state=S_GRID,
            d_state=D_GRID,
        )
        _, rm = solver.solve(hp, verbose=False)
        ret_rows.append(np.asarray(rm, np.float64).reshape(-1))
        del solver
        if CUDA:
            torch.cuda.empty_cache()
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    solved {i + 1}/{n} users", flush=True)
    return np.ascontiguousarray(np.stack(ret_rows))


def _simulate(users, policy, policy_param, retention_table):
    """One batched simulate_fsrs7 call over all users; returns (mem, cost), each (N, SPAN)."""
    s_grid = np.ascontiguousarray(S_GRID, np.float64) if policy == "ssp_mmc" else None
    d_grid = np.ascontiguousarray(D_GRID, np.float64) if policy == "ssp_mmc" else None
    out = ssp_mmc_rust.simulate_fsrs7(
        len(users),
        DECK,
        SPAN,
        SIM_SEED,
        users.w,
        users.lc,
        users.rc,
        users.frp,
        users.rrp,
        MAX_COST,
        LEARN_LIMIT,
        REVIEW_LIMIT,
        S_MIN,
        S_MAX,
        MAX_SAME_DAY,
        N_ITER,
        policy,
        float(policy_param),
        users.gflat,
        retention_table,
        s_grid,
        d_grid,
    )
    return np.asarray(out[2]), np.asarray(out[3])


def _objectives_from(mem, cost):
    """(N,SPAN) memorized + cost -> aggregate (mean knowledge, mean time/day in minutes)."""
    knowledge = mem.mean(axis=1)  # per user: across-days mean of sum_cards p_recall
    time_min = cost.mean(axis=1) / 60.0  # per user: mean minutes/day
    return float(knowledge.mean()), float(time_min.mean())


def evaluate_hp(hp, users):
    """Stream-solve the Bellman per user with `hp`, simulate all users batched -> objectives."""
    t0 = time.perf_counter()
    ret_table = _solve_all(hp, users)  # (N, d*s*s)
    t_solve = time.perf_counter() - t0
    t1 = time.perf_counter()
    mem, cost = _simulate(users, "ssp_mmc", 0.0, ret_table)
    t_sim = time.perf_counter() - t1
    print(f"    [solve {t_solve:.0f}s + sim {t_sim:.0f}s]", flush=True)
    return _objectives_from(mem, cost)


def evaluate_dr(r, users):
    """Fixed-DR policy at retention `r` (no Bellman solve needed)."""
    mem, cost = _simulate(users, "dr", r, None)
    return _objectives_from(mem, cost)


def multi_objective_function(param_dict, users):
    print(f"\nEvaluating {param_dict}", flush=True)
    knowledge, time_min = evaluate_hp(param_dict, users)
    print(
        f"  knowledge={knowledge:.1f} cards  time={time_min:.2f} min/day\n", flush=True
    )
    # SEM left as None (noiseless): with 1 deck/user the per-user estimate is noisy, but the
    # mean over users is the objective; passing across-user SEM is a possible future refinement.
    return {
        "average_knowledge": (knowledge, None),
        "time_per_day_min": (time_min, None),
    }


# ── DR baseline ───────────────────────────────────────────────────────────────────────────
def generate_dr_baseline(users, path):
    dr = []
    for r in np.arange(DR_MIN, DR_MAX, DR_STEP):
        knowledge, time_min = evaluate_dr(float(r), users)
        dr.append(
            {
                "dr": float(r),
                "average_knowledge": knowledge,
                "time_per_day_min": time_min,
            }
        )
        print(f"  DR={r:.2f}: knowledge={knowledge:.1f}  time={time_min:.2f} min/day")
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dr, open(path, "w"), indent=2)
    print(f"Saved DR baseline to {path}")
    return dr


def load_or_make_dr_baseline(users, path, force=False):
    if path.exists() and not force:
        return json.load(open(path))
    return generate_dr_baseline(users, path)


# ── Pareto extraction, DR comparison ──────────────────────────────────────────────────────
def pareto_frontier(ax):
    """ax Pareto frontier of the OBSERVED (raw simulated) objectives, not GP-predicted means.

    `use_model_predictions=False` makes ax return the actual measured (knowledge, time) at each
    Pareto-optimal trial -- the honest values to compare against the measured DR front (the GP
    posterior mean can differ from what the simulator actually produced).
    """
    return ax.get_pareto_optimal_parameters(use_model_predictions=False)


def _frontier_points(frontier):
    """ax frontier -> list of (param_dict, knowledge, time_min), sorted by knowledge."""
    pts = []
    for _, payload in frontier.items():
        params, metrics = payload[0], payload[1][0]
        pts.append(
            (
                {k: params[k] for k in PARAM_NAMES},
                float(metrics["average_knowledge"]),
                float(metrics["time_per_day_min"]),
            )
        )
    return sorted(pts, key=lambda p: p[1])


def _dr_points(dr_baseline):
    return [
        (e["dr"], float(e["average_knowledge"]), float(e["time_per_day_min"]))
        for e in dr_baseline
    ]


def _hypervolume_2d(points, ref_point):
    """Dominated area of a maximize-both 2D point set vs a reference point worse than all."""
    ref_x, ref_y = ref_point
    filtered = [(x, y) for x, y in points if x > ref_x and y > ref_y]
    if not filtered:
        return 0.0
    filtered.sort(key=lambda p: p[0], reverse=True)
    max_y = ref_y
    area = 0.0
    for idx, (x, y) in enumerate(filtered):
        if y > max_y:
            max_y = y
        x_next = filtered[idx + 1][0] if idx + 1 < len(filtered) else ref_x
        width = x - x_next
        if width > 0:
            area += width * (max_y - ref_y)
    return area


def _hypervolume(frontier):
    # maximize-both transform: (knowledge, -time).
    pts = [(k, -t) for _, k, t in _frontier_points(frontier)]
    return _hypervolume_2d(pts, HYPERVOLUME_REF_POINT)


def _dr_dominated_indices(ssp_points, dr_points):
    """Indices of SSP points dominated by some DR point (>= knowledge AND <= time, one strict).

    These are the regions where fixed DR still wins -> the candidate proposer pushes there.
    """
    bad = []
    for i, (_, k, t) in enumerate(ssp_points):
        for _, kd, td in dr_points:
            if kd >= k and td <= t and (kd > k or td < t):
                bad.append(i)
                break
    return bad


def _dr_efficient(dr_points):
    """DR points on the (max knowledge, min time) Pareto frontier, sorted by knowledge.

    The raw DR curve FOLDS BACK at high retention (past DR~0.94 knowledge drops while daily time
    explodes toward the cost cap), so the high-DR tail is dominated. Keeping only the efficient
    lower-left envelope makes knowledge->time monotonic, so np.interp below is valid (otherwise
    the fold-back gives two wildly different times at the same knowledge and corrupts the compare).
    """
    pts = [(k, t) for _, k, t in dr_points]
    eff = [
        (k, t)
        for k, t in pts
        if not any((k2 >= k and t2 <= t and (k2 > k or t2 < t)) for k2, t2 in pts)
    ]
    return sorted(set(eff))


def _beats_dr(ssp_points, dr_points):
    """SSP points that beat the DR front: less daily time than DR needs for the same knowledge."""
    eff = _dr_efficient(dr_points)
    ks = [k for k, _ in eff]
    ts = [t for _, t in eff]
    beats = []
    for params, k, t in ssp_points:
        dr_t = float(
            np.interp(k, ks, ts)
        )  # np.interp clamps outside the DR knowledge range
        if t < dr_t - 1e-9:
            beats.append(
                {
                    "params": params,
                    "knowledge": k,
                    "time_per_day_min": t,
                    "dr_time_at_knowledge": dr_t,
                }
            )
    return beats


def report_and_save(frontier, dr_baseline, out_dir):
    ssp_points = _frontier_points(frontier)
    dr_points = _dr_points(dr_baseline)
    beats = _beats_dr(ssp_points, dr_points)

    print("\nPareto-optimal SSP-MMC sets (knowledge, time/day min):")
    for params, k, t in ssp_points:
        print(f"    ({k:.1f}, {t:.2f})  {params}")
    print(f"\n{len(beats)}/{len(ssp_points)} Pareto sets beat the DR front:")
    for b in beats:
        print(
            f"    knowledge={b['knowledge']:.1f}  time={b['time_per_day_min']:.2f}  "
            f"(DR needs {b['dr_time_at_knowledge']:.2f} min/day)  {b['params']}"
        )

    out = {
        "front": [
            {"params": p, "knowledge": k, "time_per_day_min": t}
            for p, k, t in ssp_points
        ],
        "beats_dr": beats,
        "dr_front": [
            {"dr": d, "knowledge": k, "time_per_day_min": t} for d, k, t in dr_points
        ],
    }
    path = out_dir / "pareto7.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nSaved Pareto results to {path}")
    return out


# ── candidate proposer (kept, fixed) ──────────────────────────────────────────────────────
def propose_new_candidate(ssp_points, dr_points, rng):
    """Blend/mutate toward regions where SSP-MMC is still DR-dominated, to help ax explore.

    The `rng` is seeded ONCE PER AX ITERATION by the caller (``default_rng(seed + trial)``), so
    different iterations propose different candidates -- that per-iteration variation was the
    seed's intent. Within a call we draw sequentially, so each parameter advances the stream and
    gets its own value. (The FSRS-6 bug: it *reseeded with that same per-iteration seed inside
    the per-parameter loop*, collapsing every parameter to one identical draw.) Bounds/param set
    come from PARAMETERS (generalized past the hardcoded a0..a9 / FSRS-6 clamp ranges); the
    categoricals (transform_*) are carried from the better candidate.
    """
    bad = _dr_dominated_indices(ssp_points, dr_points)
    if not bad:
        print("No DR-dominated SSP-MMC points -> no manual candidate proposed.")
        return None
    j = min(bad)
    worse = ssp_points[j][0]
    better = ssp_points[max(j - 1, 0)][0]
    # If the categorical transforms agree, average the two; otherwise mutate the better one.
    strategy = (
        "average" if all(better[c] == worse[c] for c in CHOICE_NAMES) else "mutate"
    )

    new = {}
    for spec in PARAMETERS:
        name, ptype = spec["name"], spec["type"]
        if ptype == "choice":
            new[name] = better[name]
            continue
        lo, hi = spec["bounds"]
        if strategy == "average":
            wb, ww = rng.uniform(1.5, 4.0), rng.uniform(0.7, 1.0)
            val = (wb * better[name] + ww * worse[name]) / (wb + ww)
        else:  # mutate the better candidate per-parameter
            val = better[name] * (1.0 + rng.normal(0.0, 0.1))
        new[name] = float(np.clip(round(val, 2), lo, hi))
    print(f"Manually proposed candidate ({strategy}): {new}")
    return new


# ── optimizer driver ──────────────────────────────────────────────────────────────────────
def run_optimizer(
    ax,
    users,
    dr_baseline,
    out_dir,
    checkpoint,
    total_trials,
    seed,
    manual_start,
    manual_interval,
):
    completed = len(ax.experiment.trials)
    stable_checks = 0
    best_hv = None

    for i in range(completed, total_trials):
        if i > 0 and i % HYPERVOLUME_CHECK_INTERVAL == 0:
            hv = _hypervolume(pareto_frontier(ax))
            improvement = float("inf") if best_hv is None else hv - best_hv
            if best_hv is None or hv > best_hv + HYPERVOLUME_EPS:
                best_hv = hv
            print(
                f"[trial {i}] hypervolume={hv:.1f} best={best_hv:.1f} (improvement {improvement:.1f})"
            )
            # RELATIVE tolerance: the absolute 1e-3 from the FSRS-6 optimizer assumed a tiny
            # cards/min 2nd objective; with time in minutes the hypervolume area is ~1e6, so an
            # absolute 1e-3 never fires. "<0.1% of the current hypervolume" is scale-invariant.
            rel_improvement = improvement / max(abs(best_hv), HYPERVOLUME_EPS)
            stable_checks = (
                stable_checks + 1 if rel_improvement < HYPERVOLUME_TOLERANCE else 0
            )
            if stable_checks >= HYPERVOLUME_PATIENCE:
                print("Hypervolume plateaued -> early stop.")
                break

        manual = i >= manual_start and i % manual_interval == 0
        if manual:
            ssp_points = _frontier_points(pareto_frontier(ax))
            cand = propose_new_candidate(
                ssp_points, _dr_points(dr_baseline), np.random.default_rng(seed + i)
            )
            if cand is not None:
                params, trial_index = cand, ax.attach_trial(parameters=cand)[1]
            else:
                params, trial_index = ax.get_next_trial()
        else:
            params, trial_index = ax.get_next_trial()

        print(f"Starting trial {i + 1}/{total_trials}")
        ax.complete_trial(
            trial_index=trial_index,
            raw_data=multi_objective_function(params, users),
        )
        ax.save_to_json_file(str(checkpoint))

    frontier = pareto_frontier(ax)
    return report_and_save(frontier, dr_baseline, out_dir)


def _label(uids):
    if uids == list(range(uids[0], uids[-1] + 1)):
        return f"users_{uids[0]}-{uids[-1]}"
    return f"users_n{len(uids)}_{uids[0]}-{uids[-1]}"


def main():
    ap = argparse.ArgumentParser(
        description="FSRS-7 SSP-MMC hyperparameter optimizer (step 5)."
    )
    ap.add_argument(
        "--user-ids",
        default=None,
        help="ids/ranges, e.g. '1,2,5-10'. Overrides --n-users.",
    )
    ap.add_argument(
        "--n-users", type=int, default=20, help="Use users 1..N (default 20)."
    )
    ap.add_argument("--total-trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manual-start", type=int, default=MANUAL_CANDIDATE_START_TRIAL)
    ap.add_argument("--manual-interval", type=int, default=MANUAL_CANDIDATE_INTERVAL)
    ap.add_argument("--dr-baseline-only", action="store_true")
    ap.add_argument("--regen-dr-baseline", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default outputs/checkpoints/<label>).",
    )
    args = ap.parse_args()

    uids = (
        parse_user_ids(args.user_ids)
        if args.user_ids
        else list(range(1, args.n_users + 1))
    )
    out_dir = args.out or (lib.CHECKPOINTS_DIR / _label(uids))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Users: {len(uids)} ({uids[0]}..{uids[-1]}) | grid S={len(S_GRID)} D={len(D_GRID)} | "
        f"deck={DECK} span={SPAN} | out={out_dir}"
    )

    users = Users(uids)
    dr_path = out_dir / "dr_baseline7.json"
    dr_baseline = load_or_make_dr_baseline(users, dr_path, force=args.regen_dr_baseline)
    if args.dr_baseline_only:
        return

    checkpoint = out_dir / f"ssp_mmc7_seed{args.seed}.json"
    if checkpoint.exists():
        print(f"Resuming from {checkpoint}")
        ax = AxClient.load_from_json_file(str(checkpoint))
        ax._random_seed = args.seed
    else:
        ax = AxClient(random_seed=args.seed, verbose_logging=False)
        ax.create_experiment(
            name="SSP-MMC-FSRS7", parameters=PARAMETERS, objectives=OBJECTIVES
        )
        ax.save_to_json_file(str(checkpoint))

    run_optimizer(
        ax,
        users,
        dr_baseline,
        out_dir,
        checkpoint,
        args.total_trials,
        args.seed,
        args.manual_start,
        args.manual_interval,
    )


if __name__ == "__main__":
    main()
