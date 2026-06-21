"""Benchmark: Rust simulate() vs the current Python (torch) simulator.

Run with:  uv run --no-sync python tests/bench_simulate.py

Times the fixed-interval policy on CPU at 1 thread (matching the current "CPU is busy"
constraint). Python uses its normal torch path (rng_kind="torch") -- the realistic
baseline we want Rust to beat. Reports wall-clock and speedup. (Different RNG between the
two, but this measures speed, not parity; see test_simulate_parity.py for correctness.)
"""

import time

import numpy as np
import torch

import ssp_mmc_rust
from ssp_mmc_fsrs.simulation import simulate
from ssp_mmc_fsrs.policies import create_fixed_interval_policy
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
FIXED_IVL = 10.0


def bench_python(parallel, deck_size, learn_span, reps=2):
    kw = dict(
        parallel=parallel,
        w=DEFAULT_W,
        policy=create_fixed_interval_policy(FIXED_IVL),
        device="cpu",
        deck_size=deck_size,
        learn_span=learn_span,
        max_cost_perday=MAX_COST,
        learn_limit_perday=LEARN_LIMIT,
        review_limit_perday=REVIEW_LIMIT,
        seed=42,
        rng_kind="torch",
    )
    simulate(**kw)  # warmup
    return min(_time(lambda: simulate(**kw)) for _ in range(reps))


def bench_rust(parallel, deck_size, learn_span, reps=2):
    def tile(x):
        return np.ascontiguousarray(np.tile(np.asarray(x, np.float64), (parallel, 1)))

    args = (
        parallel,
        deck_size,
        learn_span,
        42,
        tile(DEFAULT_W),
        tile(DEFAULT_LEARN_COSTS),
        tile(DEFAULT_REVIEW_COSTS),
        tile(DEFAULT_FIRST_RATING_PROB),
        tile(DEFAULT_REVIEW_RATING_PROB),
        np.zeros((parallel, 4)),
        np.zeros((parallel, 1)),
        MAX_COST,
        LEARN_LIMIT,
        REVIEW_LIMIT,
        float("inf"),
        float(S_MAX),
        "fixed",
        FIXED_IVL,
    )
    ssp_mmc_rust.simulate(*args)  # warmup
    return min(_time(lambda: ssp_mmc_rust.simulate(*args)) for _ in range(reps))


def _time(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def main():
    print(f"{'config':28s} {'python(torch)':>14s} {'rust':>10s} {'speedup':>9s}")
    for parallel, deck_size, learn_span in [
        (10, 10_000, 365),
        (100, 10_000, 365),
    ]:
        tp = bench_python(parallel, deck_size, learn_span)
        tr = bench_rust(parallel, deck_size, learn_span)
        cfg = f"p={parallel} deck={deck_size} span={learn_span}"
        print(f"{cfg:28s} {tp:12.3f}s {tr:8.3f}s {tp / tr:7.1f}x")


if __name__ == "__main__":
    main()
