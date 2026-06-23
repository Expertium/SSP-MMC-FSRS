"""Smoke + sanity checks for the GRU recall-predictor path of simulation7.simulate.

Run with:  uv run --no-sync python tests/test_simulate7_gru.py

Requires a few per-user GRU weights to exist (train via experiments/train_gru_per_user.py;
users 1-3 are produced first). Checks:
  - the GRU path runs and produces finite outputs in sane ranges, budgets respected,
  - GRU p(recall) is a sane, monotonically-decreasing function of the interval (incl. sub-day),
  - knowledge DIFFERS from the FSRS-only (gru=None) path -> the GRU is actually arbitrating.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from ssp_mmc_fsrs.gru import BatchedGRU
from ssp_mmc_fsrs.simulation7 import simulate
from ssp_mmc_fsrs.policies7 import create_dr_policy

torch.set_num_threads(1)

REPO = Path(__file__).resolve().parents[1]
WDIR = REPO / "outputs" / "gru_weights" / "GRU-short-secs"

# FSRS-7 default parameters (34) -- scheduler params; GRU handles recall.
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


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return cond


def _weight_paths():
    paths = [WDIR / f"user_{u}.pth" for u in (1, 2, 3)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(
            f"SKIP: missing GRU weights {[p.name for p in missing]}; "
            "run experiments/train_gru_per_user.py first."
        )
        return None
    return paths


def test_gru_curve_sane(gru):
    """p(recall) in (0,1) and decreasing with the interval, including sub-day intervals."""
    print("test_gru_curve_sane")
    deck = gru.parallel  # reuse parallel as a tiny deck just to get a (P, deck) shape
    h = gru.init_hidden(deck)
    # advance one learning review (dt=0, rating=3) so h is non-trivial
    rating = torch.full((gru.parallel, deck), 3, dtype=torch.int64)
    h = gru.step(h, torch.zeros((gru.parallel, deck), dtype=torch.float64), rating)
    ts = [0.01, 0.5, 1.0, 7.0, 30.0, 365.0]
    ps = [
        gru.p_recall(h, torch.full((gru.parallel, deck), t, dtype=torch.float64))
        for t in ts
    ]
    in_range = all((0.0 < p).all() and (p < 1.0).all() for p in ps)
    means = [float(p.mean()) for p in ps]
    decreasing = all(means[i] >= means[i + 1] - 1e-9 for i in range(len(means) - 1))
    return check(
        "p in (0,1) and non-increasing in t",
        in_range and decreasing,
        f"means over t={ts}: {[round(m, 3) for m in means]}",
    )


def test_gru_path_runs_and_differs(paths):
    print("test_gru_path_runs_and_differs")
    parallel = len(paths)
    gru = BatchedGRU.from_pth_paths(paths, device=DEVICE, dtype=torch.float64)
    pol = create_dr_policy(0.90, W)
    common = dict(
        parallel=parallel,
        w=W,
        policy=pol,
        device=DEVICE,
        deck_size=300,
        learn_span=120,
        max_cost_perday=MAX_COST,
        learn_limit_perday=LEARN_LIMIT,
        seed=42,
    )
    rev_g, learn_g, mem_g, cost_g = simulate(**common, gru=gru)
    rev_f, learn_f, mem_f, cost_f = simulate(**common, gru=None)

    finite = all(np.isfinite(a).all() for a in (rev_g, learn_g, mem_g, cost_g))
    budget = cost_g.max() <= MAX_COST + 1e-6
    learns_ok = learn_g.max() <= LEARN_LIMIT + 1e-9
    mem_ok = (mem_g >= -1e-9).all() and (mem_g <= 300 + 1e-6).all()
    # The GRU arbitrates recall + knowledge -> trajectories must differ from FSRS-only.
    differs = abs(float(mem_g.sum()) - float(mem_f.sum())) > 1e-6

    ok = True
    ok &= check(
        "finite + budgets + mem range",
        finite and budget and learns_ok and mem_ok,
        f"cost<=max={budget} learn<=lim={learns_ok} 0<=mem<=deck={mem_ok}",
    )
    ok &= check(
        "knowledge differs from FSRS-only path",
        differs,
        f"sum(mem) gru={mem_g.sum():.1f} vs fsrs={mem_f.sum():.1f}",
    )
    return ok, gru


def main():
    paths = _weight_paths()
    if paths is None:
        return 0  # not a failure: weights not trained yet
    ok_run, gru = test_gru_path_runs_and_differs(paths)
    ok_curve = test_gru_curve_sane(gru)
    print()
    if ok_run and ok_curve:
        print("PASS: GRU simulator path checks all green.")
        return 0
    print("FAIL: see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
