"""Sanity + correctness checks for the FSRS-7 simulator (simulation7.simulate).

Run with:  uv run --no-sync python tests/test_simulate7_sanity.py

The FSRS-7 memory recurrence is already verified bit-exact vs the reference
(test_fsrs7_parity) and the interval inverter vs scipy.brentq (newton_steps_study). What's
NEW here is the same-day hybrid loop and the daily budgets, so this checks:
  - all four policies run and produce finite outputs in sane ranges,
  - daily budgets are respected (cost <= max_cost_perday, learns <= learn_limit_perday),
  - the same-day cap is enforced EXACTLY (a fixed tiny-interval policy with a huge budget
    must give learn + (cap-1) reviews per new card on day 0),
  - higher DR => more reviews (shorter intervals, more same-day repeats).
"""

import sys

import numpy as np
import torch

from ssp_mmc_fsrs.simulation7 import simulate
from ssp_mmc_fsrs import fsrs7
from ssp_mmc_fsrs.policies7 import (
    create_dr_policy,
    create_fixed_interval_policy,
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

MAX_COST = 86400 / 2
LEARN_LIMIT = 10
DEVICE = "cpu"


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def run_policy(policy, **kw):
    defaults = dict(
        parallel=4,
        w=W,
        policy=policy,
        device=DEVICE,
        deck_size=400,
        learn_span=120,
        max_cost_perday=MAX_COST,
        learn_limit_perday=LEARN_LIMIT,
        seed=42,
    )
    defaults.update(kw)
    return simulate(**defaults)


def test_all_policies_sane():
    print("test_all_policies_sane")
    ok = True
    policies = {
        "dr0.70": create_dr_policy(0.70, W),
        "dr0.90": create_dr_policy(0.90, W),
        "dr0.99": create_dr_policy(0.99, W),
        "fixed10": create_fixed_interval_policy(10.0),
        "memrise": make_memrise_policy(),
        "sm2": make_anki_sm2_policy(),
    }
    for name, pol in policies.items():
        review, learn, mem, cost = run_policy(pol, deck_size=400, learn_span=120)
        finite = all(np.isfinite(a).all() for a in (review, learn, mem, cost))
        budget = cost.max() <= MAX_COST + 1e-6
        learns_ok = learn.max() <= LEARN_LIMIT + 1e-9
        mem_ok = (mem >= -1e-9).all() and (mem <= 400 + 1e-6).all()
        nonneg = (review >= 0).all() and (learn >= 0).all() and (cost >= 0).all()
        ok &= check(
            f"{name}: finite={finite} cost<=max={budget} learn<=lim={learns_ok} "
            f"0<=mem<=deck={mem_ok} nonneg={nonneg}",
            finite and budget and learns_ok and mem_ok and nonneg,
        )
    return ok


def test_same_day_cap_exact():
    print("test_same_day_cap_exact")
    cap = 10
    # Fixed ~14 min interval + effectively unlimited budget: each new card should be
    # learned once and then reviewed (cap - 1) times the same day, hitting the cap.
    review, learn, mem, cost = run_policy(
        create_fixed_interval_policy(0.01),
        parallel=2,
        deck_size=100,
        learn_span=5,
        max_cost_perday=1e12,
        review_limit_perday=10**9,
        learn_limit_perday=5,
        max_same_day=cap,
    )
    learn0_ok = np.all(learn[:, 0] == 5)
    review0_ok = np.all(review[:, 0] == (cap - 1) * 5)
    # No day may exceed cap reviews per card: total reviews/day <= cap * (cards seen).
    return check(
        f"day0 learns==5 ({learn[:, 0].tolist()}) and "
        f"day0 reviews==(cap-1)*5={9 * 5} ({review[:, 0].tolist()})",
        learn0_ok and review0_ok,
    )


def test_higher_dr_more_reviews():
    print("test_higher_dr_more_reviews")
    totals = {}
    for dr in (0.70, 0.90, 0.99):
        review, learn, mem, cost = run_policy(
            create_dr_policy(dr, W), deck_size=400, learn_span=180
        )
        totals[dr] = review.sum()
    mono = totals[0.70] < totals[0.90] < totals[0.99]
    return check(
        f"reviews: dr0.70={totals[0.70]:.0f} < dr0.90={totals[0.90]:.0f} "
        f"< dr0.99={totals[0.99]:.0f}",
        mono,
    )


def test_dr_roundtrip():
    """White-box: scheduling at DR then evaluating the curve at that interval returns DR
    (the inverter is consistent with the curve), across a batch of random states."""
    print("test_dr_roundtrip")
    rng = np.random.default_rng(0)
    w_t = torch.tensor(W, dtype=torch.float64)
    n = 5000
    s_long = torch.tensor(np.exp(rng.uniform(np.log(1e-3), np.log(3000), n)))
    s_short_raw = np.exp(rng.uniform(np.log(1e-4), np.log(300), n))
    s_short = torch.tensor(np.minimum(s_short_raw, s_long.numpy()))
    d = torch.tensor(rng.uniform(1, 10, n))
    ok = True
    for dr in (0.60, 0.80, 0.99):
        t = fsrs7.forgetting_curve_inverse(dr, s_long, s_short, d, w_t)
        p = fsrs7.forgetting_curve(t, s_long, s_short, d, w_t)
        # Only where the interval isn't clamped to a bound (unreachable dr).
        interior = (t > fsrs7.MIN_INTERVAL_DAYS * 1.001) & (t < fsrs7.S_MAX * 0.999)
        err = (p[interior] - dr).abs().max().item() if interior.any() else 0.0
        ok &= check(f"dr={dr}: max|p(t)-dr| on interior = {err:.2e}", err < 2e-3)
    return ok


def main():
    results = [
        test_all_policies_sane(),
        test_same_day_cap_exact(),
        test_higher_dr_more_reviews(),
        test_dr_roundtrip(),
    ]
    print()
    if all(results):
        print("PASS: FSRS-7 simulator sanity checks all green.")
        return 0
    print("FAIL: see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
