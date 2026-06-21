"""Parity test: Rust simulate_fsrs7() must match Python simulation7.simulate(rng_kind="shared").

Run with:  uv run --no-sync python tests/test_simulate7_parity.py

Definition of done for the FSRS-7 Rust port (roadmap step 2d). Both sides are f64 and share
the same counter-based RNG (now with a per-round dimension for same-day reviews), so we
expect very tight agreement -- only cross-language libm differences in exp/pow/log remain,
which can occasionally flip a near-threshold recall or nudge an interval across a day
boundary. Both get the SAME per-deck params (one user broadcast across all decks), as both
scheduler (DR/fixed/Memrise/SM-2) and predictor (the dual forgetting curve).
"""

import sys

import numpy as np
import torch

import ssp_mmc_rust
from ssp_mmc_fsrs import fsrs7
from ssp_mmc_fsrs.simulation7 import simulate, S_MIN_SECS
from ssp_mmc_fsrs.policies7 import (
    create_fixed_interval_policy,
    create_dr_policy,
    make_memrise_policy,
    make_anki_sm2_policy,
)

torch.set_num_threads(1)

# FSRS-7 default parameters (srs-benchmark FSRS7.init_w, 34 params).
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
LEARN_COSTS = [33.79, 24.3, 13.68, 6.5]
REVIEW_COSTS = [23.0, 11.68, 7.33, 5.6]
FIRST_RATING_PROB = [0.24, 0.094, 0.495, 0.171]
REVIEW_RATING_PROB = [0.224, 0.631, 0.145]

MAX_COST = 86400 / 2
LEARN_LIMIT = 10
REVIEW_LIMIT = 9999
MAX_SAME_DAY = 10
S_MAX = fsrs7.S_MAX
N_ITER = fsrs7.INVERSE_N_ITER


def tile(x, parallel):
    return np.ascontiguousarray(np.tile(np.asarray(x, dtype=np.float64), (parallel, 1)))


def run(policy_name, py_policy, policy_param, parallel, deck_size, learn_span, seed=42):
    py = simulate(
        parallel=parallel,
        w=W,
        policy=py_policy,
        device="cpu",
        deck_size=deck_size,
        learn_span=learn_span,
        max_cost_perday=MAX_COST,
        learn_limit_perday=LEARN_LIMIT,
        review_limit_perday=REVIEW_LIMIT,
        learn_costs=LEARN_COSTS,
        review_costs=REVIEW_COSTS,
        first_rating_prob=FIRST_RATING_PROB,
        review_rating_prob=REVIEW_RATING_PROB,
        seed=seed,
        s_min=S_MIN_SECS,
        s_max=S_MAX,
        max_same_day=MAX_SAME_DAY,
        rng_kind="shared",
    )
    rs = ssp_mmc_rust.simulate_fsrs7(
        parallel,
        deck_size,
        learn_span,
        int(seed),
        tile(W, parallel),
        tile(LEARN_COSTS, parallel),
        tile(REVIEW_COSTS, parallel),
        tile(FIRST_RATING_PROB, parallel),
        tile(REVIEW_RATING_PROB, parallel),
        MAX_COST,
        LEARN_LIMIT,
        REVIEW_LIMIT,
        float(S_MIN_SECS),
        float(S_MAX),
        MAX_SAME_DAY,
        N_ITER,
        policy_name,
        float(policy_param),
    )
    return py, rs


def compare(py, rs):
    names = ["review", "learn", "memorized", "cost"]
    ok = True
    lines = []
    for n, a, b in zip(names, py, rs):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        sa = a.sum()
        sum_rel = abs(sa - b.sum()) / max(abs(sa), 1e-9)
        if n in ("review", "learn"):
            exact = np.array_equal(a, b)
            max_cell = float(np.max(np.abs(a - b)))
            ncells = a.size
            ndiff = int((a != b).sum())
            good = exact or (sum_rel < 1e-3 and ndiff <= max(1, ncells // 1000))
            tag = (
                "exact"
                if exact
                else f"{ndiff}/{ncells} cells differ, max={max_cell:.0f}, sum_rel={sum_rel:.1e}"
            )
        else:
            per_cell = np.allclose(a, b, rtol=1e-4, atol=1e-4)
            good = per_cell or sum_rel < 1e-4
            max_rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-9)))
            tag = (
                f"OK max_rel={max_rel:.1e}"
                if per_cell
                else f"aggregate sum_rel={sum_rel:.1e}"
            )
        ok = ok and good
        lines.append(f"    {n:10s} {'OK ' if good else 'DIFF'} {tag}")
    return ok, lines


def main():
    policies = [
        ("fixed", create_fixed_interval_policy(10.0), 10.0),
        ("dr", create_dr_policy(0.9, W), 0.9),
        ("memrise", make_memrise_policy(), 0.0),
        ("sm2", make_anki_sm2_policy(), 0.0),
    ]
    configs = [
        dict(parallel=4, deck_size=500, learn_span=120),
        dict(parallel=6, deck_size=1200, learn_span=200),
    ]
    all_ok = True
    for policy_name, py_policy, param in policies:
        for cfg in configs:
            py, rs = run(policy_name, py_policy, param, **cfg)
            ok, lines = compare(py, rs)
            all_ok = all_ok and ok
            print(f"[{policy_name:8s}] {cfg}  ->  {'PASS' if ok else 'FAIL'}")
            for ln in lines:
                print(ln)

    print()
    if all_ok:
        print("PASS: Rust simulate_fsrs7 matches Python for all policies.")
        return 0
    print("FAIL: see diffs above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
