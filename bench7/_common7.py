"""Shared config + data loading for the FSRS-7 whole-sim speedup harness (bench7).

Both benchmarked variants ("before"/"after") import this so they run the *identical*
workload in the *identical* order. Config is fixed across iterations so the drift checks
vs iteration 0 stay valid.

Workload (per the sim7 speedup protocol): 20 users x 3 SSP-MMC hyperparameter sets =
**60 datapoints**. Each datapoint = one full pipeline run for a (hp, user) pair: build +
solve the Bellman (Python/GPU) with that hp -> SSP-MMC policy -> Rust simulate_fsrs7 with
the per-user GRU recall predictor. Per datapoint we record wall time (whole pipeline AND
sim-only), total knowledge (final memorized), and time_spent (total study cost = the
"time/day" workload metric).

SMOKE mode (env BENCH7_SMOKE=1) shrinks everything for a fast pipeline check.

This module stays torch-free so the *driver* can import it cheaply; load_users() imports
the heavy deps (experiments/lib) lazily, only in the worker.
"""

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SMOKE = os.environ.get("BENCH7_SMOKE") == "1"

N_USERS = 3 if SMOKE else 20
DECK = 500 if SMOKE else 3000
SPAN = 120 if SMOKE else 365 * 3
SEED = 42
REPS = 1 if SMOKE else 2  # per datapoint: min of REPS timings

WDIR = ROOT / "outputs" / "gru_weights" / "GRU-short-secs"
FSRS7 = "../srs-benchmark/result/FSRS-7-short-secs-recency.jsonl"
BU = "../Anki-button-usage/button_usage.jsonl"

# Rust simulate_fsrs7 limits (match experiments/profile_pipeline.py).
MAX_COST = 86400 / 2
LEARN_LIMIT = 10
REVIEW_LIMIT = 9999
MAX_SAME_DAY = 8
S_MIN = 1e-4  # FSRS-7 --secs reference floor (S_MIN_SECS)
S_MAX = 36500.0
N_ITER = 7  # fsrs7.NEWTON_N_ITER (the sim's Newton scheduling-inverse count)
DISCOUNT_FACTOR = 0.97

# Base SSP-MMC hyperparameters (match profile_pipeline.HP), then 3 sets that vary
# w_retention -- the knowledge<->workload tradeoff knob -- so the 3 policies (and thus the
# 3 sim outputs) per user genuinely differ.
BASE_HP = {
    "transform_s_long": "log",
    "transform_s_short": "log",
    "exp_s_long": 1.0,
    "exp_s_short": 1.0,
    "exp_d": 1.0,
    "base_succ": 1.0,
    "w_fail_s_long": 0.5,
    "w_fail_s_short": 0.5,
    "w_fail_d": 0.5,
    "w_succ_s_long": 0.5,
    "w_succ_s_short": 0.5,
    "w_succ_d": 0.5,
    "w_retention": 1.0,
}


def _hp(**over):
    h = dict(BASE_HP)
    h.update(over)
    return h


HP_SETS = [
    _hp(w_retention=0.5),
    _hp(w_retention=1.0),
    _hp(w_retention=2.0),
]


def datapoint_order():
    """Canonical (hp_idx, uid) order of the 60 datapoints -- hp outer, user inner."""
    return [(hi, uid) for hi in range(len(HP_SETS)) for uid in range(1, N_USERS + 1)]


def load_users(n=N_USERS):
    """First `n` users (ids 1..n): FSRS-7 params + per-user costs/rating-probs. Imports the
    heavy experiments/lib lazily so the driver needn't pull torch."""
    import sys

    sys.path.insert(0, str(ROOT / "experiments"))
    import lib  # type: ignore

    users = []
    for uid in range(1, n + 1):
        w, _, _ = lib.load_fsrs_weights(FSRS7, uid)
        cfg = lib.normalize_button_usage(lib.load_button_usage_config(BU, uid))
        users.append(
            {
                "uid": uid,
                "w": np.asarray(w, np.float64),
                "lc": np.asarray(cfg["learn_costs"], np.float64),
                "rc": np.asarray(cfg["review_costs"], np.float64),
                "frp": np.asarray(cfg["first_rating_prob"], np.float64),
                "rrp": np.asarray(cfg["review_rating_prob"], np.float64),
            }
        )
    return users
