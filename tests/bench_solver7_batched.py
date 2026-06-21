"""Benchmark + parity for the batched-hyperparameter-set value iteration.

The 15 cost-hyperparameter sets of one user share the same transition tables and recall, so
processing several at once reads those (~1 GB) tables once per group instead of once per set.
This compares the batched path against the verified per-set path on real users: verdicts
must match exactly, and we measure the wall-clock speedup + VRAM at several batch sizes.

Run:  uv run --no-sync python tests/bench_solver7_batched.py [n_users] [n_iter]
"""

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "experiments", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ssp_mmc_fsrs.solver7 import SSPMMCSolver7, HAS_TRITON  # noqa: E402
from converge7 import (  # noqa: E402
    make_hyperparam_sets,
    load_jsonl_by_user,
    DEFAULT_PARAMS,
    DEFAULT_BUTTON_USAGE,
)

torch.set_num_threads(1)
BATCH_SIZES = [4, 8, 15]


def main():
    if not HAS_TRITON:
        print("Triton unavailable; batched path needs CUDA. SKIP.")
        return 0
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 1500

    params = load_jsonl_by_user(DEFAULT_PARAMS)
    usage = load_jsonl_by_user(DEFAULT_BUTTON_USAGE)
    common = sorted(set(params) & set(usage))
    common = [u for u in common if len(params[u]["parameters"]["0"]) == 34]
    user_ids = common[:n_users]
    hp_sets = make_hyperparam_sets(15, seed=0)

    seq_times = []
    batch_times = {bs: [] for bs in BATCH_SIZES}
    peak_vram = {bs: 0.0 for bs in BATCH_SIZES}
    mismatches = 0

    for user_id in user_ids:
        w = params[user_id]["parameters"]["0"]
        u = usage[user_id]
        solver = SSPMMCSolver7(
            review_costs=u["review_costs"],
            first_rating_prob=u["first_rating_prob"],
            review_rating_prob=u["review_rating_prob"],
            w=w,
        )
        # Sequential reference (verified path).
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        seq = [solver.measure_convergence(hp, n_iter=n_iter) for hp in hp_sets]
        torch.cuda.synchronize()
        seq_times.append(time.perf_counter() - t0)
        seq_verdicts = [v[0] for v in seq]

        for bs in BATCH_SIZES:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            bat = solver.measure_convergence_batched(
                hp_sets, n_iter=n_iter, batch_size=bs
            )
            torch.cuda.synchronize()
            batch_times[bs].append(time.perf_counter() - t0)
            peak_vram[bs] = max(peak_vram[bs], torch.cuda.max_memory_allocated() / 1e9)
            bat_verdicts = [v[0] for v in bat]
            if bat_verdicts != seq_verdicts:
                mismatches += 1
                for i, (a, c) in enumerate(zip(seq_verdicts, bat_verdicts)):
                    if a != c:
                        print(
                            f"  MISMATCH user {user_id} set {i}: "
                            f"seq={a} batch(bs={bs})={c} "
                            f"(seq frac {seq[i][1]:.2e}, batch frac {bat[i][1]:.2e})"
                        )
        del solver
        torch.cuda.empty_cache()

    seq_mean = sum(seq_times) / len(seq_times)
    print(f"\nusers: {len(user_ids)}  n_iter cap: {n_iter}")
    print(f"verdict mismatches (batched vs sequential): {mismatches}")
    print(f"\nsequential 15 sets: {seq_mean:.2f}s/user")
    for bs in BATCH_SIZES:
        bm = sum(batch_times[bs]) / len(batch_times[bs])
        print(
            f"batched bs={bs:2d}: {bm:.2f}s/user  speedup={seq_mean / bm:.2f}x  "
            f"VRAM peak={peak_vram[bs]:.2f} GB"
        )
    print("\nRESULT:", "PASS" if mismatches == 0 else "FAIL")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
