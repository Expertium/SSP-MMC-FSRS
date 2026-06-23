"""Speed up the FSRS-7 interval inversion (``fsrs7.forgetting_curve_inverse``).

Run:  uv run --no-sync python tests/inverse_speedup_study.py

The dual-stability forgetting curve has no closed-form inverse, so scheduling a card to a
target retention DR needs a root-find. Today that's a 12-iteration vectorized Brent. This
study asks: can we go faster (fewer transcendentals / iterations) while keeping the
interval within 0.1% of the gold standard?

WHAT IT DOES
------------
1. 100 test cases = real FSRS-7 params for users 1..100. For each user we generate a
   *realistic* memory state (S_long, S_short, D) by running that user's own FSRS-7
   recurrence for a random number of reviews (so the triple honours the model's own joint
   dynamics, not an arbitrary box), then draw DR ~ U[0.60, 0.99].

2. GOLD STANDARD = Brent with 20 steps (a scalar pure-Python mirror of the shipped
   vectorized Brent). We also compute an INDEPENDENT golden with scipy.brentq (run to
   ~machine precision) purely to confirm both gold and candidate are genuinely accurate.

3. CANDIDATE ("after") = weighted-geomean initial guess + a few Newton steps in log(t).
   The two single-component inverses t1* (short curve alone = DR) and t2* (long curve
   alone = DR) bracket the root; we start at their weighted geometric mean (weight =
   w1/(w1+w2), the mixture weight) and refine with Newton. KEY TRICK: Newton's derivative
   reuses the SAME two pow() values as the function eval ((1+a*t)^(d-1) = (1+a*t)^d /
   (1+a*t)), so each Newton step costs ~2 pow -- the same as one Brent f-eval -- but
   converges quadratically, so far fewer steps are needed. u is clamped to the bracket
   each step as a cheap (transcendental-free) safeguard.

4. SPEED PROTOCOL (the project's standard): "before" (Brent-20) and "after" (candidate)
   are timed in TWO SIMULTANEOUS PROCESSES so external load hits both equally. We then run
   a one-sided Wilcoxon signed-rank test (before > after) over the 100 paired per-case
   times (require p < 0.01) and report median(time_before / time_after) (require > 1).

This is a SCALAR pure-Python microbenchmark on purpose: it isolates iteration count and
per-iteration transcendental cost -- exactly the quantities that carry over to the Rust
Bellman build (~millions of scalar inverses) and the per-review simulator cost. Numbers
here are a proxy for the eventual Rust port, not absolute wall-clock for the torch path.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

SRS_BENCH = r"C:\Users\Andrew\srs-benchmark"
FSRS7_PARAMS = os.path.join(SRS_BENCH, "result", "FSRS-7-short-secs-recency.jsonl")

MIN_T = 1.0 / 1440.0  # 1 minute, in days
MAX_T = 36500.0
S_MIN = 1e-4
S_MAX = 36500.0
SCALE = 1.0 - 2e-5
LOG_MIN = math.log(MIN_T)
LOG_MAX = math.log(MAX_T)

N_USERS = 100
GOLD_ITERS = 20  # Brent steps for the gold standard

exp = math.exp
log = math.log


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


# ── scalar FSRS-7 recurrence (mirror of src/ssp_mmc_fsrs/fsrs7.py) ──────────────────
# Only used to GENERATE realistic states; not on the timed path. Validated against the
# torch fsrs7 module in the driver (see _validate_recurrence).


def short_component_recall(t, s_short, w):
    t = max(t, 0.0)
    mag = clamp(w[23] * s_short ** (w[33] - 0.3), 0.01, 0.95)
    decay1 = -mag
    factor1 = exp(min(log(w[25]) / decay1, 60.0)) - 1.0
    return (t / s_short * factor1 + 1.0) ** decay1


def forgetting_curve(t, s, s_short, d, w):
    t = max(t, 0.0)
    r1 = short_component_recall(t, s_short, w)
    decay2 = -clamp(w[24], 0.01, 0.95)
    factor2 = w[26] ** (1.0 / decay2) - 1.0
    d_ts = exp((d - 5.0) * (w[32] - 0.3))
    r2 = (t / s * factor2 * d_ts + 1.0) ** decay2
    weight1 = w[27] * s_short ** (-w[29])
    weight2 = w[28] * s ** w[30] * exp((d - 5.0) * (w[31] - 0.5))
    retention = (weight1 * r1 + weight2 * r2) / (weight1 + weight2)
    return retention * SCALE + 1e-5


def _init_d(rating, w):
    return w[4] - exp(w[5] * (rating - 1.0)) + 1.0


def _next_stability(last_s, last_d, r, rating, start, w):
    hard = w[start + 6] if rating == 2 else 1.0
    easy = w[start + 7] if rating == 4 else 1.0
    new_s_fail = (
        w[start + 3]
        * ((last_s + 1.0) ** w[start + 4] - 1.0)
        * exp((1.0 - r) * w[start + 5])
    )
    pls = min(last_s, new_s_fail)
    sinc = (
        exp(w[start] - 1.5)
        * (11.0 - last_d)
        * last_s ** (-w[start + 1])
        * (exp((1.0 - r) * w[start + 2]) - 1.0)
        * hard
        * easy
        + 1.0
    )
    return max(pls, last_s * sinc) if rating > 1 else pls


def _next_difficulty(last_d, rating, retention, w):
    delta0 = -w[6] * (rating - 3.0)
    delta = delta0 * (retention + 0.1) if rating == 1 else delta0
    new_d = last_d + delta * (10.0 - last_d) / 9.0
    reverted = 0.01 * _init_d(4, w) + 0.99 * new_d
    return clamp(reverted, 1.0, 10.0)


def init_state(rating, w):
    s0 = w[rating - 1]
    return (
        clamp(s0, S_MIN, S_MAX),
        clamp(0.8 * s0, S_MIN, S_MAX),
        clamp(_init_d(rating, w), 1.0, 10.0),
    )


def update_state(dt, rating, s_long, s_short, d, w):
    last_s = clamp(s_long, S_MIN, S_MAX)
    last_ss = clamp(s_short, S_MIN, S_MAX)
    last_d = clamp(d, 1.0, 10.0)
    r = forgetting_curve(dt, last_s, last_ss, last_d, w)
    upd_sl = _next_stability(last_s, last_d, r, rating, 7, w)
    r1 = short_component_recall(dt, last_ss, w)
    upd_ss = _next_stability(last_ss, last_d, r1, rating, 15, w)
    if rating == 1:
        upd_ss = min(upd_ss, 0.8 * upd_sl)
    upd_d = _next_difficulty(last_d, rating, r, w)
    return (
        clamp(upd_sl, S_MIN, S_MAX),
        clamp(upd_ss, S_MIN, S_MAX),
        clamp(upd_d, 1.0, 10.0),
    )


# ── coefficient setup shared by every inverter (identical work for before & after) ──


def setup_coeffs(s_long, s_short, d, w):
    decay1 = -clamp(w[23] * s_short ** (w[33] - 0.3), 0.01, 0.95)
    factor1 = exp(min(log(w[25]) / decay1, 60.0)) - 1.0
    a1 = factor1 / s_short
    decay2 = -clamp(w[24], 0.01, 0.95)
    factor2 = w[26] ** (1.0 / decay2) - 1.0
    d_ts = exp((d - 5.0) * (w[32] - 0.3))
    a2 = factor2 * d_ts / s_long
    weight1 = w[27] * s_short ** (-w[29])
    weight2 = w[28] * s_long ** w[30] * exp((d - 5.0) * (w[31] - 0.5))
    wt_sum = max(weight1 + weight2, 1e-9)
    return decay1, a1, decay2, a2, weight1, weight2, wt_sum


def _bracket(dr, decay1, a1, decay2, a2):
    """Single-component inverses t1*, t2* in log space, clamped to [LOG_MIN, LOG_MAX]."""
    t1 = max((dr ** (1.0 / decay1) - 1.0) / a1, 1e-12)
    t2 = max((dr ** (1.0 / decay2) - 1.0) / a2, 1e-12)
    u1 = clamp(log(t1), LOG_MIN, LOG_MAX)
    u2 = clamp(log(t2), LOG_MIN, LOG_MAX)
    return (u1, u2) if u1 <= u2 else (u2, u1)


# ── BEFORE: scalar Brent (gold at n_iter = 20) ──────────────────────────────────────


def brent_inverse(dr, s_long, s_short, d, w, n_iter):
    decay1, a1, decay2, a2, weight1, weight2, wt_sum = setup_coeffs(
        s_long, s_short, d, w
    )
    tol = 1e-12

    def f(u):
        t = exp(u)
        p = (
            weight1 * (a1 * t + 1.0) ** decay1 + weight2 * (a2 * t + 1.0) ** decay2
        ) / wt_sum * SCALE + 1e-5
        return p - dr

    a, b = _bracket(dr, decay1, a1, decay2, a2)
    fa, fb = f(a), f(b)
    if abs(fa) < abs(fb):
        a, b, fa, fb = b, a, fb, fa
    c, fc, dd, mflag = a, fa, a, True
    for _ in range(n_iter):
        if fa != fc and fb != fc:
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            dsec = fb - fa if fb - fa != 0.0 else 1e-30
            s = b - fb * (b - a) / dsec
        lo_b = (3.0 * a + b) / 4.0
        bis = (
            (s - lo_b) * (s - b) >= 0.0
            or (mflag and abs(s - b) >= abs(b - c) / 2.0)
            or ((not mflag) and abs(s - b) >= abs(c - dd) / 2.0)
            or (mflag and abs(b - c) < tol)
            or ((not mflag) and abs(c - dd) < tol)
        )
        if bis:
            s = 0.5 * (a + b)
        mflag = bis
        fs = f(s)
        dd, c, fc = c, b, fb
        if fa * fs < 0.0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        if abs(fa) < abs(fb):
            a, b, fa, fb = b, a, fb, fa
    return exp(clamp(b, LOG_MIN, LOG_MAX))


# ── AFTER: weighted-geomean init + Newton steps in log(t) ───────────────────────────


def newton_inverse(dr, s_long, s_short, d, w, n_iter):
    decay1, a1, decay2, a2, weight1, weight2, wt_sum = setup_coeffs(
        s_long, s_short, d, w
    )
    u_lo, u_hi = _bracket(dr, decay1, a1, decay2, a2)
    # Weighted geomean init in log(t): weight = mixture weight of the short component.
    alpha = weight1 / wt_sum
    u = (
        alpha * u_lo + (1.0 - alpha) * u_hi
    )  # u_lo, u_hi are already log t1*, log t2* (sorted)
    if u < u_lo:
        u = u_lo
    elif u > u_hi:
        u = u_hi
    c1 = weight1 / wt_sum * SCALE
    c2 = weight2 / wt_sum * SCALE
    for _ in range(n_iter):
        t = exp(u)
        i1 = a1 * t + 1.0
        i2 = a2 * t + 1.0
        p1 = i1**decay1
        p2 = i2**decay2
        f = c1 * p1 + c2 * p2 + 1e-5 - dr
        # dp/du = dp/dt * t; (1+a*t)^(decay-1) = p / (1+a*t), so no extra pow().
        dfdu = (c1 * decay1 * a1 * p1 / i1 + c2 * decay2 * a2 * p2 / i2) * t
        if -1e-300 < dfdu < 1e-300:
            break
        u = u - f / dfdu
        if u < u_lo:
            u = u_lo
        elif u > u_hi:
            u = u_hi
    return exp(clamp(u, LOG_MIN, LOG_MAX))


# ── independent golden (scipy.brentq on a pure re-derivation) ────────────────────────


def golden_inverse(dr, s_long, s_short, d, w):
    from scipy.optimize import brentq

    c = setup_coeffs(s_long, s_short, d, w)
    decay1, a1, decay2, a2, weight1, weight2, wt_sum = c

    def f(t):
        p = (
            weight1 * (a1 * t + 1.0) ** decay1 + weight2 * (a2 * t + 1.0) ** decay2
        ) / wt_sum * SCALE + 1e-5
        return p - dr

    if f(MIN_T) <= 0.0:
        return MIN_T
    if f(MAX_T) >= 0.0:
        return MAX_T
    return brentq(f, MIN_T, MAX_T, xtol=1e-12, rtol=8.9e-16, maxiter=300)


# ── case generation ─────────────────────────────────────────────────────────────────


def load_params(n):
    out = []
    with open(FSRS7_PARAMS) as fh:
        for line in fh:
            rec = json.loads(line)
            p = rec.get("parameters", {}).get("0")
            if p and len(p) == 34:
                out.append((rec.get("user"), [float(x) for x in p]))
            if len(out) >= n:
                break
    return out


def _realistic_state(rng, w):
    """One realistic (s_long, s_short, d) from this user's own FSRS-7 recurrence: a random
    number of pass-heavy reviews at intervals near each step's stability (jittered)."""
    first_p = (0.15, 0.10, 0.65, 0.10)
    rev_p = (0.10, 0.10, 0.70, 0.10)
    rating = int(rng.choice([1, 2, 3, 4], p=first_p))
    sl, ss, d = init_state(rating, w)
    for _ in range(int(rng.integers(0, 25))):
        dt = sl * float(rng.uniform(0.5, 2.0))
        r = int(rng.choice([1, 2, 3, 4], p=rev_p))
        sl, ss, d = update_state(dt, r, sl, ss, d, w)
    return sl, ss, d


def make_cases(n, seed=20260623):
    """One realistic test case per user (n users), each with DR ~ U[0.60, 0.99]."""
    rng = np.random.default_rng(seed)
    cases = []
    for user, w in load_params(n):
        sl, ss, d = _realistic_state(rng, w)
        cases.append(
            {
                "user": user,
                "w": w,
                "s_long": sl,
                "s_short": ss,
                "d": d,
                "dr": float(rng.uniform(0.60, 0.99)),
            }
        )
    return cases


def production_iter_scan(n_users=10000, traj=4, seed=99):
    """The 100-case sample is too small to surface the ~1% of users whose realistic states
    floor decay1 (flat, ill-conditioned curve). This scans MANY realistic states across ALL
    users to pick the Newton iteration count that keeps the SIMULATOR worst-case < 0.1%.

    Returns (chosen_n, rows) where rows = [(n, max_rel, n_over_0.1%)] vs Brent-20."""
    rng = np.random.default_rng(seed)
    params = load_params(n_users)
    cases = []
    n_floor = 0
    for _, w in params:
        for _ in range(traj):
            sl, ss, d = _realistic_state(rng, w)
            cases.append((w, sl, ss, d, float(rng.uniform(0.60, 0.99))))
            if clamp(w[23] * ss ** (w[33] - 0.3), 0.01, 0.95) <= 0.0101:
                n_floor += 1
    gold = [brent_inverse(dr, sl, ss, d, w, GOLD_ITERS) for w, sl, ss, d, dr in cases]
    rows = []
    for n in range(4, 9):
        rel = np.array(
            [
                abs(newton_inverse(dr, sl, ss, d, w, n) - g) / max(g, MIN_T)
                for (w, sl, ss, d, dr), g in zip(cases, gold)
            ]
        )
        rows.append((n, float(rel.max()), int((rel > 1e-3).sum())))
    chosen = next((n for n, mx, nov in rows if nov == 0 and mx < 1e-4), None)
    if chosen is not None:  # one extra step of safety margin if available
        chosen = min(chosen + 1, 8)
    return chosen or 7, rows, len(cases), n_floor


# ── timing worker (one method, one process) ─────────────────────────────────────────

METHODS = {
    "before": (brent_inverse, GOLD_ITERS)
}  # "after" filled in at runtime via --niter


def time_method(fn, n_iter, cases, reps, inner):
    times, intervals = [], []
    # warmup (JIT-free Python, but warms the loop / branch predictor / caches)
    for c in cases:
        fn(c["dr"], c["s_long"], c["s_short"], c["d"], c["w"], n_iter)
    for c in cases:
        dr, sl, ss, d, w = c["dr"], c["s_long"], c["s_short"], c["d"], c["w"]
        best = float("inf")
        val = 0.0
        for _ in range(reps):
            t0 = time.perf_counter()
            for _ in range(inner):
                val = fn(dr, sl, ss, d, w, n_iter)
            dt = time.perf_counter() - t0
            if dt < best:
                best = dt
        times.append(best / inner)
        intervals.append(val)
    return times, intervals


def run_worker(args):
    with open(args.cases) as fh:
        cases = json.load(fh)
    if args.method == "before":
        fn, n_iter = brent_inverse, GOLD_ITERS
    else:
        fn, n_iter = newton_inverse, args.niter
    times, intervals = time_method(fn, n_iter, cases, args.reps, args.inner)
    with open(args.out, "w") as fh:
        json.dump({"times": times, "intervals": intervals}, fh)


# ── driver ──────────────────────────────────────────────────────────────────────────


def _validate_recurrence(cases):
    """Confirm the scalar curve here matches the shipped torch fsrs7 forgetting_curve."""
    import torch

    from ssp_mmc_fsrs import fsrs7

    worst = 0.0
    for c in cases[:25]:
        w = c["w"]
        wt = torch.tensor(w, dtype=torch.float64)
        for t in (0.5, c["s_long"], 3.0 * c["s_long"] + 0.1):
            a = forgetting_curve(t, c["s_long"], c["s_short"], c["d"], w)
            b = float(
                fsrs7.forgetting_curve(
                    torch.tensor(t, dtype=torch.float64),
                    torch.tensor(c["s_long"], dtype=torch.float64),
                    torch.tensor(c["s_short"], dtype=torch.float64),
                    torch.tensor(c["d"], dtype=torch.float64),
                    wt,
                )
            )
            worst = max(worst, abs(a - b))
    return worst


def accuracy_table(cases, gold):
    """Worst-case relative interval error vs gold for each candidate config."""
    rows = []
    for n in range(0, 7):
        errs = [
            abs(
                newton_inverse(c["dr"], c["s_long"], c["s_short"], c["d"], c["w"], n)
                - g
            )
            / max(g, MIN_T)
            for c, g in zip(cases, gold)
        ]
        rows.append(
            (
                f"newton-{n}",
                n,
                max(errs),
                float(np.percentile(errs, 99)),
                float(np.median(errs)),
            )
        )
    for n in range(1, 13):
        errs = [
            abs(
                brent_inverse(c["dr"], c["s_long"], c["s_short"], c["d"], c["w"], n) - g
            )
            / max(g, MIN_T)
            for c, g in zip(cases, gold)
        ]
        rows.append(
            (
                f"brent-{n}",
                n,
                max(errs),
                float(np.percentile(errs, 99)),
                float(np.median(errs)),
            )
        )
    return rows


def robustness_check():
    """Document WHY Newton is simulator-only: over the full Bellman Cartesian grid (a box
    product of the SHARED S grid for S_long AND S_short, each spanning [S_MIN, S_MAX]),
    many cells are dynamically-unreachable corners where decay1 clamps to its 0.01 floor
    and the curve is nearly flat (df -> 0). There only bisection's guaranteed bracket-
    halving converges, so Brent's ~12 iters are near-optimal and Newton blows up."""
    params = load_params(40)
    s_longs = np.geomspace(1e-3, 3650.0, 12)
    ds = np.linspace(1.0, 10.0, 5)
    drs = np.round(np.append(np.arange(0.60, 0.99, 0.03), 0.99), 4)
    states = [
        (sl, ss, d) for sl in s_longs for ss in np.geomspace(1e-4, sl, 6) for d in ds
    ]

    n_flat = n_clamp = tot = 0
    worst = {"newton-4": 0.0, "newton-5": 0.0, "brent-12": 0.0, "brent-20": 0.0}
    for _, w in params:
        for sl, ss, d in states:
            mag = clamp(w[23] * ss ** (w[33] - 0.3), 0.01, 0.95)
            flat = mag <= 0.0101
            for dr in drs:
                g = golden_inverse(dr, sl, ss, d, w)
                if g <= MIN_T * 1.0001 or g >= MAX_T * 0.9999:
                    n_clamp += 1
                if flat:
                    n_flat += 1
                for nm, fn, n in (
                    ("newton-4", newton_inverse, 4),
                    ("newton-5", newton_inverse, 5),
                    ("brent-12", brent_inverse, 12),
                    ("brent-20", brent_inverse, 20),
                ):
                    e = abs(fn(dr, sl, ss, d, w, n) - g) / max(g, MIN_T)
                    if e > worst[nm]:
                        worst[nm] = e
                tot += 1
    print(
        "\n--- robustness over the FULL Bellman grid (40 users x box of S_long x S_short x D x DR) ---"
    )
    print(
        f"cases: {tot};  clamped-to-bound (unreachable): {n_clamp} ({100 * n_clamp / tot:.0f}%);  "
        f"flat/ill-conditioned (decay1 floored): {n_flat} ({100 * n_flat / tot:.0f}%)"
    )
    print(f"{'method':>10} {'max_rel_err_vs_scipy':>22}")
    for nm in ("newton-4", "newton-5", "brent-12", "brent-20"):
        print(f"{nm:>10} {worst[nm]:>22.3e}")
    print(
        "=> Newton (any fixed step count) is UNSAFE on the full grid's flat corners; Brent's"
    )
    print(
        "   bisection fallback is near-optimal there. Newton is the SIMULATOR inverse (realistic"
    )
    print(
        "   states only); the BELLMAN build keeps Brent (or Brent + per-cell early-exit in Rust)."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", dest="method", choices=["before", "after"])
    ap.add_argument("--cases")
    ap.add_argument("--out")
    ap.add_argument("--niter", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--inner", type=int, default=2000)
    args = ap.parse_args()

    if args.method:  # worker mode
        run_worker(args)
        return 0

    # ── driver mode ──
    from scipy.stats import wilcoxon

    print(f"Loading FSRS-7 params + generating {N_USERS} realistic cases ...")
    cases = make_cases(N_USERS)
    rec_err = _validate_recurrence(cases)
    print(f"scalar forgetting_curve vs torch fsrs7: max abs diff = {rec_err:.2e}")

    sl = np.array([c["s_long"] for c in cases])
    ss = np.array([c["s_short"] for c in cases])
    dd = np.array([c["d"] for c in cases])
    drs = np.array([c["dr"] for c in cases])
    print(
        f"cases: S_long [{sl.min():.3g}, {sl.max():.3g}], S_short [{ss.min():.3g}, {ss.max():.3g}], "
        f"D [{dd.min():.2f}, {dd.max():.2f}], DR [{drs.min():.3f}, {drs.max():.3f}]"
    )

    gold = [
        brent_inverse(c["dr"], c["s_long"], c["s_short"], c["d"], c["w"], GOLD_ITERS)
        for c in cases
    ]
    indep = [
        golden_inverse(c["dr"], c["s_long"], c["s_short"], c["d"], c["w"])
        for c in cases
    ]
    gold_vs_indep = max(abs(g - i) / max(i, MIN_T) for g, i in zip(gold, indep))
    print(
        f"GOLD (Brent-{GOLD_ITERS}) vs independent scipy.brentq golden: max rel diff = {gold_vs_indep:.2e}"
    )

    print(
        "\n--- accuracy sweep (worst-case rel interval error vs GOLD over 100 cases) ---"
    )
    print(
        f"{'method':>10} {'fevals~':>8} {'max_rel':>10} {'p99_rel':>10} {'median_rel':>11}"
    )
    rows = accuracy_table(cases, gold)
    for name, n, mx, p99, med in rows:
        fevals = n + 2 if name.startswith("newton") else n + 2
        print(f"{name:>10} {fevals:>8} {mx:>10.2e} {p99:>10.2e} {med:>11.2e}")

    bare = next(
        (n for name, n, mx, _, _ in rows if name.startswith("newton") and mx < 1e-3),
        None,
    )
    print(
        f"\n100-case bare minimum under 0.1%: newton-{bare} (but 100 cases UNDER-SAMPLE)"
    )

    # The 100 cases miss the ~1% of users whose realistic states floor decay1, so the
    # production iteration count is set from a large all-user realistic scan instead.
    print(
        "\n--- production iteration count (many realistic states across ALL 10k users) ---"
    )
    chosen, scan_rows, n_scan, n_floor = production_iter_scan()
    print(
        f"realistic cases scanned: {n_scan} (decay1 floored: {n_floor}, {100 * n_floor / n_scan:.2f}%)"
    )
    print(f"{'newton-n':>10} {'max_rel_vs_Brent20':>20} {'cases_over_0.1%':>16}")
    for n, mx, nov in scan_rows:
        print(f"{'newton-' + str(n):>10} {mx:>20.3e} {nov:>16}")
    print(
        f"=> chosen production candidate: newton-{chosen} (smallest with 0 cases over 0.1%, + 1 margin step)"
    )

    # ── parallel timed protocol ──
    here = os.path.dirname(os.path.abspath(__file__))
    cases_path = os.path.join(here, "_inv_cases.json")
    before_out = os.path.join(here, "_inv_before.json")
    after_out = os.path.join(here, "_inv_after.json")
    with open(cases_path, "w") as fh:
        json.dump(cases, fh)

    py = sys.executable
    script = os.path.abspath(__file__)
    reps, inner = args.reps, args.inner
    cmd_before = [
        py,
        script,
        "--worker",
        "before",
        "--cases",
        cases_path,
        "--out",
        before_out,
        "--reps",
        str(reps),
        "--inner",
        str(inner),
    ]
    cmd_after = [
        py,
        script,
        "--worker",
        "after",
        "--cases",
        cases_path,
        "--out",
        after_out,
        "--niter",
        str(chosen),
        "--reps",
        str(reps),
        "--inner",
        str(inner),
    ]
    print(f"\nrunning before & after in parallel (reps={reps}, inner={inner}) ...")
    p1 = subprocess.Popen(cmd_before)
    p2 = subprocess.Popen(cmd_after)
    p1.wait()
    p2.wait()

    with open(before_out) as fh:
        bef = json.load(fh)
    with open(after_out) as fh:
        aft = json.load(fh)
    tb = np.array(bef["times"])
    ta = np.array(aft["times"])
    iv_b = np.array(bef["intervals"])
    iv_a = np.array(aft["intervals"])

    # accuracy of the timed candidate vs gold + vs independent golden
    rel_vs_gold = np.abs(iv_a - np.array(gold)) / np.maximum(np.array(gold), MIN_T)
    rel_vs_indep = np.abs(iv_a - np.array(indep)) / np.maximum(np.array(indep), MIN_T)
    # confirm the before-worker reproduced the in-driver gold
    gold_repro = np.max(np.abs(iv_b - np.array(gold)))

    ratios = tb / ta
    stat, p = wilcoxon(tb, ta, alternative="greater")
    med_ratio = float(np.median(ratios))

    print("\n================  RESULTS  ================")
    print(f"before = Brent-{GOLD_ITERS} (gold) ; after = newton-{chosen}")
    print(f"before-worker reproduced gold: max abs interval diff = {gold_repro:.2e}")
    print("\nACCURACY (candidate intervals):")
    print(f"  max  rel diff vs GOLD          = {rel_vs_gold.max():.3e}   (need < 1e-3)")
    print(f"  p99  rel diff vs GOLD          = {np.percentile(rel_vs_gold, 99):.3e}")
    print(f"  median rel diff vs GOLD        = {np.median(rel_vs_gold):.3e}")
    print(f"  max  rel diff vs scipy golden  = {rel_vs_indep.max():.3e}")
    print("\nSPEED (100 paired per-case times):")
    print(f"  median(time_before/time_after) = {med_ratio:.3f}   (need > 1)")
    print(
        f"  mean before = {tb.mean() * 1e6:.3f} us/call ; mean after = {ta.mean() * 1e6:.3f} us/call"
    )
    print(
        f"  one-sided Wilcoxon (before>after): stat = {stat:.1f}, p = {p:.3e}   (need < 0.01)"
    )

    acc_ok = rel_vs_gold.max() < 1e-3
    speed_ok = med_ratio > 1.0
    sig_ok = p < 0.01
    print("\nVERDICT:")
    print(f"  accuracy  <0.1% : {'PASS' if acc_ok else 'FAIL'}")
    print(f"  speedup   >1    : {'PASS' if speed_ok else 'FAIL'}")
    print(f"  Wilcoxon  <0.01 : {'PASS' if sig_ok else 'FAIL'}")
    print(f"  => {'ALL PASS' if (acc_ok and speed_ok and sig_ok) else 'NOT ALL PASS'}")

    for pth in (cases_path, before_out, after_out):
        try:
            os.remove(pth)
        except OSError:
            pass

    robustness_check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
