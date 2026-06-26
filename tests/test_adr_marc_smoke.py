"""Smoke test for the Cost ADR policy (ssp_mmc_fsrs.adr) and the MARC solver (ssp_mmc_fsrs.marc).

Run with:  uv run --no-sync python tests/test_adr_marc_smoke.py

ADR checks: DR stays in [retention_min, retention_max]; raising cost_weight lowers average DR
(the price-of-time monotonicity); the all-zero-coefficient policy is a CONSTANT DR (fixed DR is
ADR's special case); retention_table has the SSP-MMC grid shape and is constant along s_short.

MARC checks (tiny CPU grid): solve() returns (d,s,s) matrices; target retention stays inside the
action range [R_MIN, R_MAX]; raising lambda (price of time) lowers the mean target retention
(more time-pressure -> longer intervals -> lower R). These are sanity checks, not a benchmark.
"""

import sys

import numpy as np
import torch

from ssp_mmc_fsrs import adr
from ssp_mmc_fsrs.marc import MARCSolver7
from ssp_mmc_fsrs.solver7 import R_MIN, R_MAX

torch.set_num_threads(1)

# Default FSRS-7 params + per-user inputs (from tests/test_solver7_smoke.py).
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


def test_adr():
    print("=== ADR ===")
    ok = True
    s_grid = np.geomspace(1e-3, 36500.0, 40)
    d_grid = np.linspace(1.0, 10.0, 10)

    # Range + price-of-time monotonicity across the cost_weight sweep.
    avg = []
    for cw in [0.0, 64.0, 256.0, 1024.0]:
        tbl = adr.retention_table(adr.DEFAULT_ADR_COEF, cw, s_grid, d_grid)
        ok &= tbl.shape == (len(d_grid), len(s_grid), len(s_grid))
        ok &= float(tbl.min()) >= adr.RETENTION_MIN - 1e-9
        ok &= float(tbl.max()) <= adr.RETENTION_MAX + 1e-9
        # constant along the s_short axis (axis=2)
        ok &= bool(np.allclose(tbl[:, :, 0:1], tbl, atol=0))
        avg.append(float(tbl.mean()))
    monotone = all(avg[i] >= avg[i + 1] - 1e-9 for i in range(len(avg) - 1))
    ok &= monotone
    print(f"    avg DR by cost_weight [0,64,256,1024] = {[round(a, 3) for a in avg]}")
    print(
        f"    range/shape/s_short-constant ok={ok and True}, monotone-down={monotone}"
    )

    # Fixed DR is ADR's special case: all-zero coefficients -> constant DR everywhere.
    zero = adr.retention_table(np.zeros(15), 0.0, s_grid, d_grid)
    const = bool(np.allclose(zero, zero.flat[0], atol=1e-12))
    expected = adr.RETENTION_MIN + (adr.RETENTION_MAX - adr.RETENTION_MIN) * 0.5
    ok &= const and abs(float(zero.flat[0]) - expected) < 1e-9
    print(
        f"    zero-coef DR constant={const} value={float(zero.flat[0]):.4f} (exp {expected:.4f})"
    )
    print("    ADR:", "PASS" if ok else "FAIL")
    return ok


def test_marc():
    print("=== MARC (tiny CPU grid) ===")
    ok = True
    # Tiny grids so eager value iteration is fast on CPU.
    s_grid = np.geomspace(1e-4, 365 * 10, 24)
    d_grid = np.linspace(1.0, 10.0, 8)
    means = []
    for lam in [1e-4, 1e-3, 1e-2]:
        solver = MARCSolver7(
            review_costs=REVIEW_COSTS,
            first_rating_prob=FIRST_RATING_PROB,
            review_rating_prob=REVIEW_RATING_PROB,
            w=W,
            device="cpu",
            s_state=s_grid,
            d_state=d_grid,
        )
        value_matrix, retention_matrix = solver.solve(lam, convergence_tol=1e-5)
        ok &= value_matrix.shape == (len(d_grid), len(s_grid), len(s_grid))
        ok &= retention_matrix.min() >= R_MIN - 1e-6
        ok &= retention_matrix.max() <= R_MAX + 1e-6
        means.append(float(retention_matrix.mean()))
    monotone = all(means[i] >= means[i + 1] - 1e-6 for i in range(len(means) - 1))
    ok &= monotone
    print(
        f"    mean target R by lambda [1e-4,1e-3,1e-2] = {[round(m, 3) for m in means]}"
    )
    print(f"    range/shape ok, monotone-down with lambda={monotone}")
    print("    MARC:", "PASS" if ok else "FAIL")
    return ok


def main():
    all_ok = True
    all_ok &= test_adr()
    all_ok &= test_marc()
    print("\nOVERALL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
