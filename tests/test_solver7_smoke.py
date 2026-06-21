"""Smoke test for the FSRS-7 Bellman solver (solver7.SSPMMCSolver7).

Checks: (1) it runs end-to-end and returns (d, s, s)-shaped matrices; (2) the f32 interval
inverse round-trip max|R_pred - R| is small (so f32 meshes are fine); (3) a benign uniform
cost set converges (well under 5% of states stuck at COST_MAX, the converge.py criterion);
(4) the optimal retention lies inside the action range. Runs a small CPU grid first, then
the full grid on GPU if available.

Run with:  uv run --no-sync python tests/test_solver7_smoke.py
"""

import sys
import time

import numpy as np
import torch

from ssp_mmc_fsrs import solver7
from ssp_mmc_fsrs.solver7 import SSPMMCSolver7, COST_MAX

torch.set_num_threads(1)

W = [
    0.1104,
    2.2395,
    3.9221,
    11.7841,
    6.1686,
    0.6457,
    3.6807,
    1.9795,
    0.0,
    1.3826,
    0.7024,
    0.5999,
    0.8146,
    0.6398,
    1.0,
    1.3207,
    0.6707,
    3.8668,
    0.4416,
    0.0934,
    1.8631,
    0.6162,
    1.0869,
    0.1567,
    0.0801,
    0.2421,
    0.9464,
    0.1433,
    0.7145,
    0.0,
    0.5667,
    0.3734,
    0.5333,
    0.3048,
]
REVIEW_COSTS = [23.0, 11.68, 7.33, 5.6]
FIRST_RATING_PROB = [0.24, 0.094, 0.495, 0.171]
REVIEW_RATING_PROB = [0.224, 0.631, 0.145]

# Benign hyperparameters: uniform per-rating cost, no modifiers, no retention term.
BENIGN = dict(
    transform_s_long="log",
    transform_s_short="log",
    exp_s_long=1.0,
    exp_s_short=1.0,
    exp_d=1.0,
    base_succ=1.0,
    w_fail_s_long=0.0,
    w_fail_s_short=0.0,
    w_fail_d=0.0,
    w_succ_s_long=0.0,
    w_succ_s_short=0.0,
    w_succ_d=0.0,
    w_retention=1.0,
)


def frac_at_max(cost_matrix):
    actual_max = cost_matrix.max()
    return float((cost_matrix == actual_max).sum()) / cost_matrix.size


def run(device, n_s, label):
    print(f"\n=== {label}: device={device}, N_S={n_s} ===")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    solver = SSPMMCSolver7(
        review_costs=REVIEW_COSTS,
        first_rating_prob=FIRST_RATING_PROB,
        review_rating_prob=REVIEW_RATING_PROB,
        w=W,
        device=device,
        n_s=n_s,
    )
    print(
        f"grid: s_size={solver.s_size}, d_size={solver.d_size}, r_size={solver.r_size}, "
        f"n_states={solver.n_states:,}"
    )
    print(
        f"s_state[:5]={np.round(solver.s_state[:5], 5)} ... "
        f"s_state[-3:]={np.round(solver.s_state[-3:], 1)}"
    )
    t0 = time.perf_counter()
    cost_matrix, retention_matrix = solver.solve(BENIGN)
    dt = time.perf_counter() - t0
    check = solver.inverse_check

    fm = frac_at_max(cost_matrix)
    converged = fm < 1.0 / 20.0
    print(
        f"interior inverse round-trip max|R_pred-R| = {check:.2e}  (f32, reachable only)"
    )
    print(f"solve wall time = {dt:.1f}s")
    print(
        f"cost: min={cost_matrix.min():.2f} max={cost_matrix.max():.2f} (COST_MAX={COST_MAX:.0f})"
    )
    print(
        f"fraction of states at max = {fm:.4%}  -> {'CONVERGED' if converged else 'NOT converged'}"
    )
    print(
        f"target retention: min={retention_matrix.min():.3f} max={retention_matrix.max():.3f} "
        f"(action range [{solver.r_min}, {solver.r_max}])"
    )
    if device == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_resv = torch.cuda.max_memory_reserved() / 1e9
        print(f"VRAM peak: allocated={peak_alloc:.2f} GB, reserved={peak_resv:.2f} GB")

    ok = True
    ok &= cost_matrix.shape == (solver.d_size, solver.s_size, solver.s_size)
    ok &= check < 5e-3  # interior round-trip must be tight even in f32
    ok &= converged
    ok &= retention_matrix.min() >= solver.r_min - 1e-6
    ok &= retention_matrix.max() <= solver.r_max + 1e-6
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def main():
    all_ok = True
    all_ok &= run("cpu", n_s=30, label="small CPU correctness")
    if torch.cuda.is_available():
        all_ok &= run("cuda", n_s=solver7.N_S, label="full GPU")
    else:
        print("\n(CUDA not available; skipping full GPU run)")
    print("\nOVERALL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
