"""FSRS-7 Bellman convergence sweep (roadmap step 3).

For each user: build the FSRS-7 SSP-MMC solver (per-user FSRS-7 params + per-user button
costs + per-user Hard/Good/Easy probabilities) and run **15 random cost-hyperparameter
sets**. A set "converges" if fewer than 5% of states stay pinned at COST_MAX after value
iteration (same criterion as the FSRS-6 `converge.py`). A **user is flagged unconverged if
ANY of its 15 sets fails** (per the user's instruction).

The 15 hyperparameter sets are generated once (seeded) and SHARED across all users, so the
per-user convergence comparison is apples-to-apples. Ranges mirror the FSRS-6 hyperparameter
optimizer (exponents log-uniform [0.1,10]; weights uniform [-5,5]; w_retention [0,3]; the
two transforms are independent log/no_log choices). ``base_fail`` is pinned to 1.0.

Per-user data sources (see CLAUDE.md / memory per-user-data-sources):
  * FSRS-7 params: ../srs-benchmark/result/FSRS-7-short-secs-recency.jsonl (34 params)
  * costs/probs:   ../Anki-button-usage/button_usage.jsonl

Run, e.g.:
  uv run --no-sync python experiments/converge7.py --n-users 100
  uv run --no-sync python experiments/converge7.py --n-users 20 --n-iter 2000 --calibrate
"""

import argparse
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

from ssp_mmc_fsrs.solver7 import SSPMMCSolver7  # noqa: E402

DEFAULT_PARAMS = (
    ROOT_DIR.parent / "srs-benchmark" / "result" / "FSRS-7-short-secs-recency.jsonl"
)
DEFAULT_BUTTON_USAGE = ROOT_DIR.parent / "Anki-button-usage" / "button_usage.jsonl"

N_HYPERPARAM_SETS = 15


def make_hyperparam_sets(n=N_HYPERPARAM_SETS, seed=0):
    """Generate ``n`` random cost-hyperparameter sets (shared across all users).

    Ranges mirror the FSRS-6 optimizer search space, extended to FSRS-7's 13 knobs.
    """
    rng = np.random.default_rng(seed)
    sets = []
    for _ in range(n):
        sets.append(
            {
                "transform_s_long": rng.choice(["no_log", "log"]).item(),
                "transform_s_short": rng.choice(["no_log", "log"]).item(),
                # exponents: log-uniform in [0.1, 10]
                "exp_s_long": float(10 ** rng.uniform(-1, 1)),
                "exp_s_short": float(10 ** rng.uniform(-1, 1)),
                "exp_d": float(10 ** rng.uniform(-1, 1)),
                # base_fail pinned to 1.0 inside the solver
                "base_succ": float(rng.uniform(-5, 5)),
                "w_fail_s_long": float(rng.uniform(-5, 5)),
                "w_fail_s_short": float(rng.uniform(-5, 5)),
                "w_fail_d": float(rng.uniform(-5, 5)),
                "w_succ_s_long": float(rng.uniform(-5, 5)),
                "w_succ_s_short": float(rng.uniform(-5, 5)),
                "w_succ_d": float(rng.uniform(-5, 5)),
                "w_retention": float(rng.uniform(0, 3)),
            }
        )
    return sets


def load_jsonl_by_user(path):
    out = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            out[int(entry["user"])] = entry
    return out


def frac_at_max(cost_matrix):
    actual_max = cost_matrix.max()
    return float((cost_matrix == actual_max).sum()) / cost_matrix.size


def parse_args():
    p = argparse.ArgumentParser(
        description="FSRS-7 Bellman convergence sweep (step 3)."
    )
    p.add_argument("--parameters", type=Path, default=DEFAULT_PARAMS)
    p.add_argument("--button-usage", type=Path, default=DEFAULT_BUTTON_USAGE)
    p.add_argument(
        "--n-users", type=int, default=100, help="Number of users to sample."
    )
    p.add_argument("--user-seed", type=int, default=0, help="Seed for user sampling.")
    p.add_argument(
        "--hp-seed", type=int, default=0, help="Seed for hyperparameter sets."
    )
    p.add_argument(
        "--n-iter", type=int, default=3000, help="Max value-iteration steps."
    )
    p.add_argument("--n-s", type=int, default=None, help="Override S grid size (N_S).")
    p.add_argument("--unconverged-frac", type=float, default=1.0 / 20.0)
    p.add_argument("--device", type=str, default=None, help="cuda/cpu (default: auto).")
    p.add_argument(
        "--calibrate",
        action="store_true",
        help="Verbose per-set iteration counts (timing calibration on a small sample).",
    )
    p.add_argument(
        "--results",
        type=Path,
        default=ROOT_DIR / "outputs" / "checkpoints" / "convergence7_results.json",
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(1)

    params = load_jsonl_by_user(args.parameters)
    usage = load_jsonl_by_user(args.button_usage)
    common = sorted(set(params) & set(usage))
    # Keep only users with valid 34-param FSRS-7 vectors.
    common = [u for u in common if len(params[u]["parameters"]["0"]) == 34]

    rng = np.random.default_rng(args.user_seed)
    n = min(args.n_users, len(common))
    user_ids = sorted(rng.choice(common, size=n, replace=False).tolist())

    hp_sets = make_hyperparam_sets(N_HYPERPARAM_SETS, seed=args.hp_seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_s_kw = {"n_s": args.n_s} if args.n_s else {}

    print(
        f"Users: {n} (of {len(common)} available)  |  hyperparam sets: {len(hp_sets)}  |  "
        f"device: {device}  |  n_iter cap: {args.n_iter}  |  N_S: {args.n_s or 'default'}"
    )

    results = {}
    unconverged = []
    build_times, solve_times = [], []
    t_start = time.perf_counter()

    for ui, user_id in enumerate(user_ids, 1):
        w = params[user_id]["parameters"]["0"]
        u = usage[user_id]
        tb = time.perf_counter()
        solver = SSPMMCSolver7(
            review_costs=u["review_costs"],
            first_rating_prob=u["first_rating_prob"],
            review_rating_prob=u["review_rating_prob"],
            w=w,
            device=device,
            **n_s_kw,
        )
        build_times.append(time.perf_counter() - tb)

        user_converged = True
        worst_frac = 0.0
        for si, hp in enumerate(hp_sets):
            ts = time.perf_counter()
            cost_matrix, _ = solver.solve(
                hp, n_iter=args.n_iter, verbose=args.calibrate
            )
            solve_times.append(time.perf_counter() - ts)
            fm = frac_at_max(cost_matrix)
            worst_frac = max(worst_frac, fm)
            set_converged = fm < args.unconverged_frac
            if not set_converged:
                user_converged = False
            if args.calibrate:
                print(
                    f"  user {user_id} set {si:2d}: frac_at_max={fm:.4%} "
                    f"{'OK' if set_converged else 'NOT CONVERGED'}"
                )

        results[user_id] = {
            "converged": user_converged,
            "worst_frac_at_max": worst_frac,
        }
        if not user_converged:
            unconverged.append(user_id)

        # Free GPU memory between users.
        del solver
        if device == "cuda":
            torch.cuda.empty_cache()

        conv_count = sum(1 for r in results.values() if r["converged"])
        print(
            f"[{ui}/{n}] user {user_id}: "
            f"{'CONVERGED' if user_converged else 'UNCONVERGED'} "
            f"(worst frac_at_max={worst_frac:.3%})  |  running rate: "
            f"{conv_count}/{ui} = {conv_count / ui:.1%}"
        )

    elapsed = time.perf_counter() - t_start
    n_unconv = len(unconverged)
    print("\n==== FSRS-7 convergence sweep complete ====")
    print(f"Users tested:       {n}")
    print(f"Converged:          {n - n_unconv} ({(n - n_unconv) / n:.1%})")
    print(f"Unconverged:        {n_unconv} ({n_unconv / n:.1%})")
    print(f"Unconverged IDs:    {unconverged}")
    print(
        f"Timing: build mean={np.mean(build_times):.2f}s, "
        f"solve mean={np.mean(solve_times):.2f}s/set, total={elapsed:.0f}s"
    )

    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "n_users": n,
                "n_iter": args.n_iter,
                "n_s": args.n_s,
                "unconverged_users": unconverged,
                "results": {str(k): v for k, v in results.items()},
                "hyperparam_sets": hp_sets,
            },
            fh,
            indent=2,
        )
    print(f"Saved results to {args.results}")


if __name__ == "__main__":
    main()
