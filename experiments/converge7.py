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

from ssp_mmc_fsrs.solver7 import (  # noqa: E402
    SSPMMCSolver7,
    build_hybrid_s_grid,
    build_production_d_grid,
)

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


def invalid_data_reason(usage_entry, w):
    """Return a reason string if this user's inputs are unusable (NaN/inf probs or costs,
    all-zero review costs, rating probs that don't sum to ~1, NaN params), else None. Such
    users are a DATA-quality issue (too few reviews to estimate stats), not a solver-
    convergence question, so the sweep skips them rather than flagging them unconverged."""
    arr = np.asarray
    for key in (
        "review_costs",
        "learn_costs",
        "first_rating_prob",
        "review_rating_prob",
    ):
        v = arr(usage_entry[key], dtype=float)
        if not np.all(np.isfinite(v)):
            return f"{key}_nonfinite"
    if float(np.sum(usage_entry["review_costs"])) <= 0:
        return "review_costs_zero"
    if abs(float(np.sum(usage_entry["review_rating_prob"])) - 1.0) > 0.05:
        return "review_rating_prob_sum"
    if not np.all(np.isfinite(arr(w, dtype=float))):
        return "w_nonfinite"
    return None


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
    p.add_argument(
        "--grid",
        choices=["production", "loguniform"],
        default="production",
        help="production = hybrid-S(71) + coarser-D(35) ~= 176k states (~9x fewer than the "
        "old uniform 135/91 grid, much faster); loguniform = old uniform N_S grid (--n-s).",
    )
    p.add_argument(
        "--shard",
        type=str,
        default=None,
        help="i/N: process the i-th of N strided shards of the user list, for running "
        "several processes in parallel on disjoint users (fills an under-used GPU).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Hyperparameter sets per batched solve (4 is the measured sweet spot).",
    )
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

    # Strided shard for parallel runs (each process takes every N-th user, kept balanced).
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        user_ids = user_ids[si::sn]
    n = len(user_ids)

    hp_sets = make_hyperparam_sets(N_HYPERPARAM_SETS, seed=args.hp_seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.grid == "production":
        grid_kw = {
            "s_state": build_hybrid_s_grid(),
            "d_state": build_production_d_grid(),
        }
        grid_desc = "production hybrid-S(71)+coarser-D(35) ~176k states"
    else:
        grid_kw = {"n_s": args.n_s} if args.n_s else {}
        grid_desc = f"loguniform N_S={args.n_s or 'default'}"

    print(
        f"Users: {n} (of {len(common)} available)  |  hyperparam sets: {len(hp_sets)}  |  "
        f"device: {device}  |  n_iter cap: {args.n_iter}  |  grid: {grid_desc}"
        + (f"  |  shard {args.shard}" if args.shard else "")
    )

    results = {}
    unconverged = []
    errored = []
    invalid = []
    # Resume: if the results file exists, skip users already done (so a crash/restart picks
    # up where it left off). A user that crashed hard mid-solve was never recorded, so it is
    # NOT in `results` and gets retried.
    if args.results.exists():
        try:
            prev = json.loads(args.results.read_text(encoding="utf-8"))
            results = {int(k): v for k, v in prev.get("results", {}).items()}
            unconverged = list(prev.get("unconverged_users", []))
            errored = list(prev.get("errored_users", []))
            invalid = list(prev.get("invalid_data_users", []))
            done = set(results)
            before = len(user_ids)
            user_ids = [u for u in user_ids if u not in done]
            print(
                f"Resuming from {args.results}: {len(done)} done, "
                f"{len(user_ids)} of {before} remaining"
            )
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Could not resume from {args.results} ({exc}); starting fresh.")

    build_times, solve_times = [], []
    t_start = time.perf_counter()

    def write_results():
        # Incremental save after every user so a crash never loses the whole run.
        args.results.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.results.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "n_users_target": n,
                    "n_users_done": len(results),
                    "n_iter": args.n_iter,
                    "n_s": args.n_s,
                    "unconverged_users": sorted(unconverged),
                    "errored_users": sorted(errored),
                    "invalid_data_users": sorted(invalid),
                    "results": {str(k): v for k, v in results.items()},
                    "hyperparam_sets": hp_sets,
                },
                fh,
                indent=2,
            )
        tmp.replace(args.results)

    for ui, user_id in enumerate(user_ids, 1):
        try:
            w = params[user_id]["parameters"]["0"]
            u = usage[user_id]
            reason = invalid_data_reason(u, w)
            if reason is not None:
                invalid.append(user_id)
                results[user_id] = {"converged": None, "invalid_data": reason}
                print(
                    f"[{len(results)}/{n}] user {user_id}: SKIP invalid data ({reason})"
                )
                write_results()
                continue
            tb = time.perf_counter()
            solver = SSPMMCSolver7(
                review_costs=u["review_costs"],
                first_rating_prob=u["first_rating_prob"],
                review_rating_prob=u["review_rating_prob"],
                w=w,
                device=device,
                **grid_kw,
            )
            build_times.append(time.perf_counter() - tb)

            user_converged = True
            worst_frac = 0.0
            failed_sets = []
            ts = time.perf_counter()
            # Batched over hyperparameter sets (they share this user's transitions/recall);
            # batch_size=4 is the measured sweet spot (~2.1x, lowest VRAM).
            verdicts = solver.measure_convergence_batched(
                hp_sets, n_iter=args.n_iter, batch_size=args.batch_size
            )
            solve_times.append(time.perf_counter() - ts)
            for si, (set_converged, fm, iters) in enumerate(verdicts):
                worst_frac = max(worst_frac, fm)
                if not set_converged:
                    user_converged = False
                    failed_sets.append({"set": si, "frac_at_max": fm, "iters": iters})
                if args.calibrate:
                    print(
                        f"  user {user_id} set {si:2d}: frac_at_max={fm:.4%} "
                        f"iters={iters} {'OK' if set_converged else 'NOT CONVERGED'}"
                    )

            results[user_id] = {
                "converged": user_converged,
                "worst_frac_at_max": worst_frac,
                "failed_sets": failed_sets,
            }
            if not user_converged:
                unconverged.append(user_id)

            del solver
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:  # one bad user must not kill a multi-hour run
            errored.append(user_id)
            results[user_id] = {"converged": None, "error": repr(exc)[:200]}
            print(f"[{len(results)}/{n}] user {user_id}: ERROR {repr(exc)[:160]}")
            if device == "cuda":
                torch.cuda.empty_cache()
            write_results()
            continue

        write_results()
        conv_count = sum(1 for r in results.values() if r.get("converged") is True)
        print(
            f"[{len(results)}/{n}] user {user_id}: "
            f"{'CONVERGED' if user_converged else 'UNCONVERGED'} "
            f"(worst frac_at_max={worst_frac:.3%})  |  running rate: "
            f"{conv_count}/{len(results)} = {conv_count / max(len(results), 1):.1%}"
        )

    elapsed = time.perf_counter() - t_start
    n_unconv = len(unconverged)
    n_err = len(errored)
    n_inv = len(invalid)
    # Convergence rate is over VALID, non-errored users (invalid-data users excluded).
    n_valid = n - n_inv - n_err
    print("\n==== FSRS-7 convergence sweep complete ====")
    print(f"Users tested:       {n}")
    print(f"Invalid-data (skipped): {n_inv}")
    print(f"Errored:            {n_err}")
    print(f"Valid users:        {n_valid}")
    rate = (n_valid - n_unconv) / n_valid if n_valid else 0.0
    print(f"Converged:          {n_valid - n_unconv} ({rate:.2%} of valid)")
    print(
        f"Unconverged:        {n_unconv} ({n_unconv / n_valid if n_valid else 0:.2%} of valid)"
    )
    print(f"Unconverged IDs:    {sorted(unconverged)}")
    if errored:
        print(f"Errored IDs:        {sorted(errored)}")
    if invalid:
        print(f"Invalid-data IDs:   {sorted(invalid)}")
    print(
        f"Timing: build mean={np.mean(build_times):.2f}s, "
        f"solve mean={np.mean(solve_times):.2f}s/user (15 sets, bs={args.batch_size}), "
        f"total={elapsed:.0f}s"
    )
    write_results()
    print(f"Saved results to {args.results}")


if __name__ == "__main__":
    main()
