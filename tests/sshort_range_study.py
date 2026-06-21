"""Measure the realistic range of S_short (and S_long) under DR-scheduled trajectories.

Why: the step-3 Bellman solver puts S_long AND S_short on the SAME log S grid
(0.0001 -> S_MAX=36500) and the cost function normalizes each by S_MAX. We want to confirm
(a) what range S_short actually occupies, so the grid covers it and isn't mostly wasted,
and (b) that log-normalizing S_short by S_MAX gives a usable spread (not ~0 everywhere).
(This measurement is what drove extending S_MAX from FSRS-6's 9125 to the 36500 clamp.)

Method: Monte-Carlo many cards through repeated reviews. Each review the interval is chosen
to hit a target retention `dr` via fsrs7.forgetting_curve_inverse (the DR policy), recall is
sampled from the dual forgetting curve, ratings come from review_rating_prob on success and
are 1 on failure. We chain fsrs7.update_state and record S_short / S_long percentiles. We do
this for the default params and a sample of real FSRS-7 users, across several DR values.

Run with:  uv run --no-sync python tests/sshort_range_study.py
"""

import json
import sys

import numpy as np
import torch

from ssp_mmc_fsrs import fsrs7

torch.set_num_threads(1)

S_MIN = 1e-4  # FSRS-7 --secs floor (solver grid floor)

W_DEFAULT = [
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
FIRST_RATING_PROB = [0.24, 0.094, 0.495, 0.171]
REVIEW_RATING_PROB = [0.224, 0.631, 0.145]  # hard/good/easy | success
FSRS7_PARAMS = "C:/Users/Andrew/srs-benchmark/result/FSRS-7-short-secs-recency.jsonl"


def load_user_params(n_users, seed=0):
    """Load up to n_users real FSRS-7 parameter vectors (34 each)."""
    rows = []
    with open(FSRS7_PARAMS, "r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=min(n_users, len(rows)), replace=False)
    out = []
    for i in idx:
        p = rows[int(i)]["parameters"]["0"]
        if len(p) == 34:
            out.append(p)
    return out


def simulate_trajectories(w_list, dr, n_cards, n_reviews, seed=0):
    """Chain reviews for n_cards cards over n_reviews steps at target retention `dr`.

    w_list: list of 34-param vectors; cards are split evenly across them.
    Returns flattened arrays of all recorded (s_long, s_short, d) over all cards/steps
    (post-first-review states only).
    """
    gen = torch.Generator().manual_seed(seed)
    w_stack = torch.tensor(np.array(w_list), dtype=torch.float64)  # (U, 34)
    U = w_stack.shape[0]
    per = n_cards // U
    # Assign each card a user row.
    user_of_card = torch.arange(U).repeat_interleave(per)
    n = user_of_card.numel()
    w = w_stack[user_of_card]  # (n, 34) per-card params

    first_prob = torch.tensor(FIRST_RATING_PROB, dtype=torch.float64)
    succ_prob = torch.tensor(REVIEW_RATING_PROB, dtype=torch.float64)

    # First review: sample initial rating, init state per-card.
    first_rating = (
        torch.multinomial(first_prob, n, replacement=True, generator=gen) + 1
    ).double()

    # init_state is written for a single shared w; do it per-card manually.
    ridx = first_rating.long().clamp(1, 4) - 1
    s_long = (
        torch.gather(w[:, 0:4], 1, ridx[:, None]).squeeze(1).clamp(S_MIN, fsrs7.S_MAX)
    )
    s_short = (0.8 * torch.gather(w[:, 0:4], 1, ridx[:, None]).squeeze(1)).clamp(
        S_MIN, fsrs7.S_MAX
    )
    d = (w[:, 4] - torch.exp(w[:, 5] * (first_rating - 1)) + 1.0).clamp(
        fsrs7.D_MIN, fsrs7.D_MAX
    )

    rec_long, rec_short, rec_d = [], [], []
    for _ in range(n_reviews):
        # DR policy: interval to reach `dr` given current state (per-card w).
        t = _inv_per_card(dr, s_long, s_short, d, w)
        # Recall probability at that interval, then sample recall.
        r = _curve_per_card(t, s_long, s_short, d, w)
        recalled = torch.rand(n, generator=gen) <= r
        # Rating: success -> {2,3,4} via succ_prob; fail -> 1.
        succ_rating = (
            torch.multinomial(succ_prob, n, replacement=True, generator=gen) + 2
        ).double()
        rating = torch.where(recalled, succ_rating, torch.ones(n, dtype=torch.float64))
        s_long, s_short, d = _update_per_card(t, rating, s_long, s_short, d, w)
        rec_long.append(s_long.clone())
        rec_short.append(s_short.clone())
        rec_d.append(d.clone())

    return (
        torch.stack(rec_long).flatten().numpy(),
        torch.stack(rec_short).flatten().numpy(),
        torch.stack(rec_d).flatten().numpy(),
    )


# Per-card variants of the fsrs7 functions (w is (n,34) instead of a shared vector). We
# index columns directly; the math is identical to fsrs7.py.
def _wc(w, i):
    return w[:, i]


def _short_recall(t, s_short, w):
    t = t.clamp(min=0.0)
    decay1_mag = (_wc(w, 23) * s_short.pow(_wc(w, 33) - 0.3)).clamp(0.01, 0.95)
    decay1 = -decay1_mag
    factor1 = (_wc(w, 25).log() * decay1.pow(-1.0)).clamp(max=60.0).exp() - 1.0
    return ((t / s_short) * factor1 + 1.0).pow(decay1)


def _curve_per_card(t, s, s_short, d, w):
    t = t.clamp(min=0.0)
    r1 = _short_recall(t, s_short, w)
    decay2 = -_wc(w, 24).clamp(0.01, 0.95)
    factor2 = _wc(w, 26).pow(decay2.pow(-1.0)) - 1.0
    d_ts = ((d - 5.0) * (_wc(w, 32) - 0.3)).exp()
    r2 = ((t / s) * factor2 * d_ts + 1.0).pow(decay2)
    weight1 = _wc(w, 27) * s_short.pow(-_wc(w, 29))
    weight2 = _wc(w, 28) * s.pow(_wc(w, 30)) * ((d - 5.0) * (_wc(w, 31) - 0.5)).exp()
    retention = (weight1 * r1 + weight2 * r2) / (weight1 + weight2)
    return retention * (1.0 - 2e-5) + 1e-5


def _next_stability(last_s, last_d, r, rating, start, w):
    ones = torch.ones_like(last_s)
    hard = torch.where(rating == 2, _wc(w, start + 6), ones)
    easy = torch.where(rating == 4, _wc(w, start + 7), ones)
    new_s_fail = (
        _wc(w, start + 3)
        * ((last_s + 1.0).pow(_wc(w, start + 4)) - 1.0)
        * ((1.0 - r) * _wc(w, start + 5)).exp()
    )
    pls = torch.minimum(last_s, new_s_fail)
    sinc = (_wc(w, start) - 1.5).exp() * (11.0 - last_d) * last_s.pow(
        -_wc(w, start + 1)
    ) * (((1.0 - r) * _wc(w, start + 2)).exp() - 1.0) * hard * easy + 1.0
    new_s_success = torch.maximum(pls, last_s * sinc)
    return torch.where(rating > 1, new_s_success, pls)


def _update_per_card(delta_t, rating, s_long, s_short, d, w):
    last_s = s_long.clamp(S_MIN, fsrs7.S_MAX)
    last_ss = s_short.clamp(S_MIN, fsrs7.S_MAX)
    last_d = d.clamp(fsrs7.D_MIN, fsrs7.D_MAX)
    r = _curve_per_card(delta_t, last_s, last_ss, last_d, w)
    upd_s_long = _next_stability(last_s, last_d, r, rating, 7, w)
    r1 = _short_recall(delta_t, last_ss, w)
    upd_s_short = _next_stability(last_ss, last_d, r1, rating, 15, w)
    relearn = torch.minimum(upd_s_short, 0.8 * upd_s_long)
    upd_s_short = torch.where(rating == 1, relearn, upd_s_short)
    # difficulty
    delta_d = -_wc(w, 6) * (rating - 3)
    delta_d = torch.where(rating == 1, delta_d * (r + 0.1), delta_d)
    new_d = last_d + delta_d * (10.0 - last_d) / 9.0
    init_d4 = _wc(w, 4) - torch.exp(_wc(w, 5) * (4 - 1)) + 1.0
    new_d = (0.01 * init_d4 + 0.99 * new_d).clamp(fsrs7.D_MIN, fsrs7.D_MAX)
    return (
        upd_s_long.clamp(S_MIN, fsrs7.S_MAX),
        upd_s_short.clamp(S_MIN, fsrs7.S_MAX),
        new_d.clamp(fsrs7.D_MIN, fsrs7.D_MAX),
    )


def _inv_per_card(dr, s_long, s_short, d, w):
    """Per-card vectorized Brent inverse (mirrors fsrs7.forgetting_curve_inverse)."""
    import math

    decay1 = -(_wc(w, 23) * s_short.pow(_wc(w, 33) - 0.3)).clamp(0.01, 0.95)
    factor1 = (_wc(w, 25).log() * decay1.pow(-1.0)).clamp(max=60.0).exp() - 1.0
    a1 = factor1 / s_short
    decay2 = -_wc(w, 24).clamp(0.01, 0.95)
    factor2 = _wc(w, 26).pow(decay2.pow(-1.0)) - 1.0
    d_ts = ((d - 5.0) * (_wc(w, 32) - 0.3)).exp()
    a2 = factor2 * d_ts / s_long
    weight1 = _wc(w, 27) * s_short.pow(-_wc(w, 29))
    weight2 = (
        _wc(w, 28) * s_long.pow(_wc(w, 30)) * ((d - 5.0) * (_wc(w, 31) - 0.5)).exp()
    )
    wt_sum = (weight1 + weight2).clamp(min=1e-9)
    min_t, max_t = fsrs7.MIN_INTERVAL_DAYS, fsrs7.S_MAX
    log_min, log_max = math.log(min_t), math.log(max_t)
    scale = 1.0 - 2e-5
    dr_t = torch.as_tensor(dr, dtype=s_long.dtype)
    tol = 1e-12

    def f_of_u(u):
        t = u.exp()
        p = (
            weight1 * (a1 * t + 1.0).pow(decay1) + weight2 * (a2 * t + 1.0).pow(decay2)
        ) / wt_sum
        return p * scale + 1e-5 - dr_t

    t1 = ((dr_t.pow(1.0 / decay1) - 1.0) / a1).clamp(min=1e-12)
    t2 = ((dr_t.pow(1.0 / decay2) - 1.0) / a2).clamp(min=1e-12)
    a = torch.minimum(t1, t2).log().clamp(log_min, log_max)
    b = torch.maximum(t1, t2).log().clamp(log_min, log_max)
    fa, fb = f_of_u(a), f_of_u(b)
    sw = fa.abs() < fb.abs()
    a, b = torch.where(sw, b, a), torch.where(sw, a, b)
    fa, fb = torch.where(sw, fb, fa), torch.where(sw, fa, fb)
    c, fc = a.clone(), fa.clone()
    dd = a.clone()
    mflag = torch.ones_like(b, dtype=torch.bool)
    for _ in range(fsrs7.INVERSE_N_ITER):
        use_iq = (fa != fc) & (fb != fc)
        s_iq = (
            a * fb * fc / ((fa - fb) * (fa - fc))
            + b * fa * fc / ((fb - fa) * (fb - fc))
            + c * fa * fb / ((fc - fa) * (fc - fb))
        )
        dsec = torch.where(fb - fa == 0, torch.full_like(fa, 1e-30), fb - fa)
        s_sec = b - fb * (b - a) / dsec
        s = torch.where(use_iq, s_iq, s_sec)
        lo_b = (3.0 * a + b) / 4.0
        not_between = (s - lo_b) * (s - b) >= 0.0
        cond2 = mflag & ((s - b).abs() >= (b - c).abs() / 2.0)
        cond3 = (~mflag) & ((s - b).abs() >= (c - dd).abs() / 2.0)
        cond4 = mflag & ((b - c).abs() < tol)
        cond5 = (~mflag) & ((c - dd).abs() < tol)
        bis = not_between | cond2 | cond3 | cond4 | cond5
        s = torch.where(bis, 0.5 * (a + b), s)
        mflag = bis
        fs = f_of_u(s)
        dd = c
        c, fc = b, fb
        side = fa * fs < 0.0
        a = torch.where(side, a, s)
        fa = torch.where(side, fa, fs)
        b = torch.where(side, s, b)
        fb = torch.where(side, fs, fb)
        sw = fa.abs() < fb.abs()
        a, b = torch.where(sw, b, a), torch.where(sw, a, b)
        fa, fb = torch.where(sw, fb, fa), torch.where(sw, fa, fb)
    return b.clamp(log_min, log_max).exp()


def pct(x):
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]
    vals = np.percentile(x, qs)
    return {q: v for q, v in zip(qs, vals)}


def report(name, s_long, s_short, d):
    print(f"\n=== {name} ===   (n={s_short.size:,})")
    for label, arr in [("S_long", s_long), ("S_short", s_short), ("D", d)]:
        p = pct(arr)
        print(f"  {label:8s} " + "  ".join(f"p{q}={p[q]:.4g}" for q in p))
    # How much of the 100-pt log grid does S_short use? (fraction of S_short <= a few cuts)
    for cut in [1.0, 10.0, 100.0, 1000.0]:
        frac = float((s_short <= cut).mean())
        print(f"           S_short <= {cut:>7.0f} d : {frac * 100:5.1f}%")
    # Normalized-feature spread for the cost function.
    s_max = float(fsrs7.S_MAX)  # 36500 (solver grid S_MAX, extended from FSRS-6's 9125)
    log_norm = np.log1p(s_short) / np.log1p(s_max)
    lin_norm = s_short / s_max
    print(
        f"  log1p(S_short)/log1p(S_MAX): p1={np.percentile(log_norm, 1):.3f} "
        f"p50={np.percentile(log_norm, 50):.3f} p99={np.percentile(log_norm, 99):.3f}"
    )
    print(
        f"  S_short/S_MAX (linear)     : p1={np.percentile(lin_norm, 1):.4f} "
        f"p50={np.percentile(lin_norm, 50):.4f} p99={np.percentile(lin_norm, 99):.4f}"
    )


def main():
    n_cards = 4000
    n_reviews = 40
    # Default params.
    for dr in [0.7, 0.9, 0.97]:
        sl, ss, dd = simulate_trajectories([W_DEFAULT], dr, n_cards, n_reviews, seed=1)
        report(f"default w, DR={dr}", sl, ss, dd)

    # Real users (pooled).
    users = load_user_params(200, seed=0)
    print(f"\nLoaded {len(users)} real FSRS-7 users.")
    for dr in [0.7, 0.9, 0.97]:
        sl, ss, dd = simulate_trajectories(
            users, dr, len(users) * 20, n_reviews, seed=2
        )
        report(f"200 real users, DR={dr}", sl, ss, dd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
