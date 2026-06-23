"""Per-user (batched) parameters in the FSRS-7 simulator.

Run with:  uv run --no-sync python tests/test_simulate7_peruser.py

The step-5 design runs ``parallel`` = different USERS in one call, each with their own FSRS-7
``w``, costs, and rating probabilities. This checks:
  1. the per-user ``w`` (carried as (34, P, 1)) makes fsrs7 produce row-for-row exactly what
     a per-user loop of the shared (34,) path produces (forgetting_curve / update_state /
     init_state / inverse),
  2. per-user ``review_costs`` actually route per user (a user with huge costs does fewer
     reviews under a binding budget),
  3. per-user ``first_rating_prob`` actually route per user (an always-Easy user ends up
     with more knowledge than an always-Again user).
"""

import sys

import numpy as np
import torch

from ssp_mmc_fsrs import fsrs7
from ssp_mmc_fsrs.simulation7 import simulate
from ssp_mmc_fsrs.policies7 import create_dr_policy, create_fixed_interval_policy

torch.set_num_threads(2)

W = np.array(
    [
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
)


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return cond


def test_fsrs_peruser_matches_rowloop():
    """Batched (34, P, 1) w == per-row loop of the shared (34,) path, bit for bit."""
    print("test_fsrs_peruser_matches_rowloop")
    rng = np.random.default_rng(0)
    P, deck = 5, 200
    ws = np.stack([W * (1 + 0.05 * rng.standard_normal(34)) for _ in range(P)])
    ws[:, 8] = 0.0  # keep the structurally-zero params zero
    s_long = torch.tensor(np.exp(rng.uniform(np.log(1e-2), np.log(3000), (P, deck))))
    s_short = torch.minimum(
        torch.tensor(np.exp(rng.uniform(np.log(1e-3), np.log(300), (P, deck)))), s_long
    )
    d = torch.tensor(rng.uniform(1, 10, (P, deck)))
    dt = torch.tensor(np.exp(rng.uniform(np.log(1e-3), np.log(400), (P, deck))))
    rating = torch.tensor(rng.integers(1, 5, (P, deck)))
    wb = torch.tensor(ws).transpose(0, 1).unsqueeze(-1)  # (34, P, 1)

    def maxdiff(batched, rows):
        return float((batched - torch.stack(rows)).abs().max())

    ok = True
    fc = maxdiff(
        fsrs7.forgetting_curve(dt, s_long, s_short, d, wb),
        [
            fsrs7.forgetting_curve(
                dt[p], s_long[p], s_short[p], d[p], torch.tensor(ws[p])
            )
            for p in range(P)
        ],
    )
    ok &= check("forgetting_curve", fc == 0.0, f"max|d|={fc:.1e}")
    ub = fsrs7.update_state(dt, rating, s_long, s_short, d, wb, 1e-4)
    ur = [
        fsrs7.update_state(
            dt[p], rating[p], s_long[p], s_short[p], d[p], torch.tensor(ws[p]), 1e-4
        )
        for p in range(P)
    ]
    for i, nm in enumerate(("s_long", "s_short", "d")):
        diff = maxdiff(ub[i], [ur[p][i] for p in range(P)])
        ok &= check(f"update_state.{nm}", diff == 0.0, f"max|d|={diff:.1e}")
    ib = fsrs7.init_state(rating, wb, 1e-4)
    ir = [fsrs7.init_state(rating[p], torch.tensor(ws[p]), 1e-4) for p in range(P)]
    for i, nm in enumerate(("s_long", "s_short", "d")):
        diff = maxdiff(ib[i], [ir[p][i] for p in range(P)])
        ok &= check(f"init_state.{nm}", diff == 0.0, f"max|d|={diff:.1e}")
    inv = maxdiff(
        fsrs7.forgetting_curve_inverse(0.9, s_long, s_short, d, wb),
        [
            fsrs7.forgetting_curve_inverse(
                0.9, s_long[p], s_short[p], d[p], torch.tensor(ws[p])
            )
            for p in range(P)
        ],
    )
    ok &= check("forgetting_curve_inverse", inv == 0.0, f"max|d|={inv:.1e}")
    return ok


def test_peruser_costs_route():
    """Under a binding budget, the user with huge per-review costs does far fewer reviews."""
    print("test_peruser_costs_route")
    # row0 cheap reviews, row1 expensive; shared w/probs; small daily budget so it binds.
    review_costs = np.array(
        [[10.0, 10.0, 10.0, 10.0], [2000.0, 2000.0, 2000.0, 2000.0]]
    )
    learn_costs = np.array([[10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0]])
    w = np.stack([W, W])
    rev, learn, mem, cost = simulate(
        parallel=2,
        w=w,
        policy=create_dr_policy(0.9, w),
        device="cpu",
        deck_size=400,
        learn_span=120,
        max_cost_perday=5000.0,
        learn_costs=learn_costs,
        review_costs=review_costs,
        first_rating_prob=[0.2, 0.1, 0.6, 0.1],
        review_rating_prob=[0.1, 0.8, 0.1],
        seed=1,
    )
    cheap, pricey = rev[0].sum(), rev[1].sum()
    return check(
        "cheap user reviews >> expensive user",
        cheap > 1.5 * pricey,
        f"reviews cheap={cheap:.0f} expensive={pricey:.0f}",
    )


def test_peruser_first_rating_routes():
    """An always-Easy user ends with more knowledge than an always-Again user."""
    print("test_peruser_first_rating_routes")
    frp = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # row0: always Again
            [0.0, 0.0, 0.0, 1.0],
        ]
    )  # row1: always Easy
    w = np.stack([W, W])
    rev, learn, mem, cost = simulate(
        parallel=2,
        w=w,
        policy=create_fixed_interval_policy(1.0),
        device="cpu",
        deck_size=400,
        learn_span=60,
        first_rating_prob=frp,
        review_rating_prob=[0.1, 0.8, 0.1],
        seed=2,
    )
    again_know, easy_know = mem[0, -1], mem[1, -1]
    return check(
        "Easy-first knowledge > Again-first",
        easy_know > again_know,
        f"final knowledge again={again_know:.1f} easy={easy_know:.1f}",
    )


def main():
    results = [
        test_fsrs_peruser_matches_rowloop(),
        test_peruser_costs_route(),
        test_peruser_first_rating_routes(),
    ]
    print()
    if all(results):
        print("PASS: per-user batched parameters route correctly.")
        return 0
    print("FAIL: see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
