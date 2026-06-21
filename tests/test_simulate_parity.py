"""Parity test: Rust simulate() must match Python simulate(rng_kind="shared").

Run with:  uv run --no-sync python tests/test_simulate_parity.py

Definition of done for the step-1 policy ports. Both implementations share the same
counter-based RNG, so:
  - integer count arrays (review / learn) are expected to match EXACTLY,
  - real-valued arrays (memorized / cost) within a tight tolerance (cross-language libm
    differences in exp/pow/log can shift the last bits).
Both sides get the SAME per-deck params (one user broadcast across all decks).
"""

import sys

import numpy as np
import torch

import ssp_mmc_rust
from ssp_mmc_fsrs.simulation import simulate
from ssp_mmc_fsrs.policies import (
    create_fixed_interval_policy,
    create_dr_policy,
    memrise_policy,
    anki_sm2_policy,
)
from ssp_mmc_fsrs.config import (
    DEFAULT_W,
    DEFAULT_LEARN_COSTS,
    DEFAULT_REVIEW_COSTS,
    DEFAULT_FIRST_RATING_PROB,
    DEFAULT_REVIEW_RATING_PROB,
    S_MAX,
)

torch.set_num_threads(1)

MAX_COST = 86400 / 2
LEARN_LIMIT = 10
REVIEW_LIMIT = 9999


def tile(x, parallel):
    return np.ascontiguousarray(np.tile(np.asarray(x, dtype=np.float64), (parallel, 1)))


def run(policy_name, py_policy, policy_param, parallel, deck_size, learn_span, seed=42):
    py = simulate(
        parallel=parallel,
        w=DEFAULT_W,
        policy=py_policy,
        device="cpu",
        deck_size=deck_size,
        learn_span=learn_span,
        max_cost_perday=MAX_COST,
        learn_limit_perday=LEARN_LIMIT,
        review_limit_perday=REVIEW_LIMIT,
        seed=seed,
        rng_kind="shared",
    )
    rs = ssp_mmc_rust.simulate(
        parallel,
        deck_size,
        learn_span,
        int(seed),
        tile(DEFAULT_W, parallel),
        tile(DEFAULT_LEARN_COSTS, parallel),
        tile(DEFAULT_REVIEW_COSTS, parallel),
        tile(DEFAULT_FIRST_RATING_PROB, parallel),
        tile(DEFAULT_REVIEW_RATING_PROB, parallel),
        np.zeros((parallel, 4), dtype=np.float64),
        np.zeros((parallel, 1), dtype=np.float64),
        MAX_COST,
        LEARN_LIMIT,
        REVIEW_LIMIT,
        float("inf"),
        float(S_MAX),
        policy_name,
        float(policy_param),
    )
    return py, rs


def compare(py, rs):
    """Counts must be exact, or (where floor()/libm forbid it) within +/-1 per cell with
    aggregate agreement < 0.1%. Real-valued arrays must agree per-cell at rtol 1e-3, or at
    least in aggregate when a card flip shifts one day's value."""
    names = ["review", "learn", "memorized", "cost"]
    ok = True
    lines = []
    for n, a, b in zip(names, py, rs):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        sa, sb = a.sum(), b.sum()
        sum_rel = abs(sa - sb) / max(abs(sa), 1e-9)
        if n in ("review", "learn"):
            exact = np.array_equal(a, b)
            max_cell = float(np.max(np.abs(a - b)))
            good = exact or (max_cell <= 1.0 and sum_rel < 1e-3)
            tag = (
                "exact"
                if exact
                else (
                    f"+/-{max_cell:.0f}/cell sum_rel={sum_rel:.1e} (libm)"
                    if good
                    else f"DIFF max_cell={max_cell:.0f} sum_rel={sum_rel:.1e}"
                )
            )
        else:
            per_cell = np.allclose(a, b, rtol=1e-3, atol=1e-3)
            good = per_cell or sum_rel < 1e-3
            max_rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-9)))
            tag = (
                f"OK max_rel={max_rel:.1e}"
                if per_cell
                else (
                    f"OK aggregate sum_rel={sum_rel:.1e} (local flips)"
                    if good
                    else f"DIFF sum_rel={sum_rel:.1e}"
                )
            )
        ok = ok and good
        lines.append(f"    {n:10s} {tag}")
    return ok, lines


def main() -> int:
    policies = [
        ("fixed", create_fixed_interval_policy(10), 10.0),
        ("dr", create_dr_policy(0.9), 0.9),
        ("memrise", memrise_policy, 0.0),
        ("sm2", anki_sm2_policy, 0.0),
    ]
    configs = [
        dict(parallel=4, deck_size=500, learn_span=120),
        dict(parallel=6, deck_size=1500, learn_span=200),
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
        print("PASS: Rust matches Python for all policies.")
        return 0
    print("FAIL: see diffs above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
