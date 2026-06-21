"""Shared config + data loading for the speedup benchmark harness.

Both benchmarked variants ("before"/"after") import this so they run the *identical*
workload in the *identical* order, which is what makes the 150 before/after time pairs
comparable. Config is fixed across iterations so drift checks vs iteration 0 are valid.

Workload (per the speedup protocol): first 50 users (real per-user FSRS-6 params +
per-user costs/rating-probs), policy = fixed desired retention at DR in {0.70, 0.90,
0.99}, parallel=1 per user so each (user, DR) yields its own wall-clock time. That's
3 * 50 = 150 (user, DR) pairs.

SMOKE mode (env BENCH_SMOKE=1) shrinks everything for a fast pipeline check.
"""

import json
import os
from pathlib import Path

import numpy as np

FSRS6_PATH = Path(r"C:\Users\Andrew\srs-benchmark\result\FSRS-6-short-recency.jsonl")
BUTTON_PATH = Path(r"C:\Users\Andrew\Anki-button-usage\button_usage.jsonl")

SMOKE = os.environ.get("BENCH_SMOKE") == "1"

N_USERS = 3 if SMOKE else 50
DECK_SIZE = 500 if SMOKE else 10_000
LEARN_SPAN = 120 if SMOKE else 365 * 5
DRS = (0.70, 0.90, 0.99)
SEED = 42
REPS = 3

# Simulator limits (the validated "unlim_time_lim_reviews" defaults).
MAX_COST_PERDAY = 86400 / 2
LEARN_LIMIT_PERDAY = 10
REVIEW_LIMIT_PERDAY = 9999
SIM_S_MAX = float("inf")  # simulate() stability clamp (matches parity tests)
POLICY_S_MAX = float(365 * 25)  # S_MAX used by the DR policy


def _read_jsonl_by_user(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["user"]] = rec
    return out


def load_users(n=N_USERS):
    """First `n` users (ids 1..n) with FSRS-6 params + per-user costs/rating-probs."""
    f6 = _read_jsonl_by_user(FSRS6_PATH)
    bu = _read_jsonl_by_user(BUTTON_PATH)
    users = []
    for uid in range(1, n + 1):
        w = np.asarray(f6[uid]["parameters"]["0"], dtype=np.float64)
        b = bu[uid]
        rec = {
            "uid": uid,
            "w": w,
            "learn_costs": np.asarray(b["learn_costs"], dtype=np.float64),
            "review_costs": np.asarray(b["review_costs"], dtype=np.float64),
            "first_rating_prob": np.asarray(b["first_rating_prob"], dtype=np.float64),
            "review_rating_prob": np.asarray(b["review_rating_prob"], dtype=np.float64),
        }
        for k, v in rec.items():
            if k != "uid" and not np.all(np.isfinite(v)):
                raise ValueError(f"user {uid}: non-finite {k}: {v}")
        users.append(rec)
    return users


def _row(x, k):
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(1, k))


def make_args(user, dr):
    """Positional args for ssp_mmc_rust.simulate(...) — one user (parallel=1), DR policy."""
    return (
        1,
        DECK_SIZE,
        LEARN_SPAN,
        SEED,
        _row(user["w"], user["w"].size),
        _row(user["learn_costs"], 4),
        _row(user["review_costs"], 4),
        _row(user["first_rating_prob"], 4),
        _row(user["review_rating_prob"], 3),
        np.zeros((1, 4), dtype=np.float64),
        np.zeros((1, 1), dtype=np.float64),
        MAX_COST_PERDAY,
        LEARN_LIMIT_PERDAY,
        REVIEW_LIMIT_PERDAY,
        SIM_S_MAX,
        POLICY_S_MAX,
        "dr",
        float(dr),
    )


def pair_order(users):
    """The canonical (dr, uid) order of the 150 pairs — DR outer, user inner."""
    return [(dr, u["uid"]) for dr in DRS for u in users]
