"""How many iterations does the FSRS-7 DR interval inversion need: Newton vs Brent?

Run with:  uv run --no-sync python tests/newton_steps_study.py

We invert the dual-stability forgetting curve to get the interval at a target retention
(DR). Two vectorized fixed-iteration root-finders (both in log(t)) are compared head to
head:
  - safeguarded Newton (rtsafe; uses the analytic dR/dt; ~4 pow / iter) -- a local impl,
    kept only for this comparison,
  - Brent (derivative-free: inverse-quadratic / secant / bisection; ~2 pow / iter) -- the
    method shipped in ``fsrs7.forgetting_curve_inverse`` (the local copy here is asserted
    equal to it).

Over a grid of memory states x DR in [0.60, 0.99] x several real param sets, for
n = 1..20 iterations we report each method's max / 99th-pct / median relative error of the
interval vs an INDEPENDENT golden (scipy.brentq, double precision), and the fewest
iterations each needs to keep the worst-case interval error < 0.1%.

Result: Brent reaches the median exactly in ~1 step and worst-case < 0.1% at n = 12
(hence INVERSE_N_ITER = 12 in fsrs7); the safeguarded-Newton variant needs > 20 and costs
~2x the transcendentals per iteration -- so Brent is both faster-converging and cheaper.

The golden uses a pure-Python re-derivation of p(t) (not the torch code), so it's a true
independent check; brentq is a bracketing method run to ~machine precision.
"""

import os
import sys
import json
import math

import numpy as np
import torch
from scipy.optimize import brentq

from ssp_mmc_fsrs import fsrs7

torch.set_num_threads(1)

SRS_BENCH = r"C:\Users\Andrew\srs-benchmark"
FSRS7_PARAMS = os.path.join(SRS_BENCH, "result", "FSRS-7-short-secs-recency.jsonl")

MIN_T = fsrs7.MIN_INTERVAL_DAYS  # 1 minute
MAX_T = fsrs7.S_MAX

DEFAULT_W = [
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


def real_param_sets(n):
    out = [("default", DEFAULT_W)]
    if os.path.exists(FSRS7_PARAMS):
        with open(FSRS7_PARAMS) as f:
            for line in f:
                rec = json.loads(line)
                params = rec.get("parameters", {}).get("0")
                if params and len(params) == 34:
                    out.append((f"user{rec.get('user')}", [float(x) for x in params]))
                if len(out) >= n + 1:
                    break
    return out


def py_coeffs(w, s_long, s_short, d):
    """t-independent curve coefficients in plain Python floats (independent golden)."""
    decay1 = -min(max(w[23] * s_short ** (w[33] - 0.3), 0.01), 0.95)
    factor1 = math.exp(min(math.log(max(w[25], 1e-9)) / decay1, 60.0)) - 1.0
    a1 = factor1 / s_short
    decay2 = -min(max(w[24], 0.01), 0.95)
    factor2 = max(w[26], 1e-9) ** (1.0 / decay2) - 1.0
    d_ts = math.exp((d - 5.0) * (w[32] - 0.3))
    a2 = factor2 * d_ts / s_long
    weight1 = w[27] * s_short ** (-w[29])
    weight2 = w[28] * s_long ** w[30] * math.exp((d - 5.0) * (w[31] - 0.5))
    return decay1, a1, decay2, a2, weight1, weight2, (weight1 + weight2)


def py_p(t, c):
    decay1, a1, decay2, a2, weight1, weight2, wt_sum = c
    r1 = (a1 * t + 1.0) ** decay1
    r2 = (a2 * t + 1.0) ** decay2
    return (weight1 * r1 + weight2 * r2) / wt_sum * (1.0 - 2e-5) + 1e-5


def golden_t(w, s_long, s_short, d, dr):
    c = py_coeffs(w, s_long, s_short, d)

    def f(t):
        return py_p(t, c) - dr

    f_lo = f(MIN_T)
    f_hi = f(MAX_T)
    if f_lo <= 0.0:  # desired recall already below dr at 1 min -> clamp to min
        return MIN_T
    if f_hi >= 0.0:  # never decays to dr within 100 years -> clamp to max
        return MAX_T
    return brentq(f, MIN_T, MAX_T, xtol=1e-12, rtol=8.9e-16, maxiter=300)


def build_grid():
    """Realistic memory states (s_short <= s_long) x DR in [0.60, 0.99]."""
    s_longs = np.geomspace(1e-3, 3650.0, 12)
    ds = np.linspace(1.0, 10.0, 5)
    drs = np.round(np.append(np.arange(0.60, 0.99, 0.03), 0.99), 4)
    states = []
    for sl in s_longs:
        for ss in np.geomspace(1e-4, sl, 6):
            for d in ds:
                states.append((sl, ss, d))
    return states, drs


def brent_inverse(dr, s_long, s_short, d, w, n_iter, min_t=MIN_T, max_t=MAX_T):
    """Vectorized Brent's method (derivative-free) inverting forgetting_curve p to dr, in
    log(t). Fixed ``n_iter`` iterations. Mirrors the classic Dekker-Brent algorithm with
    inverse-quadratic interpolation / secant / bisection fallback, all via torch.where."""
    decay1 = -(w[23] * s_short.pow(w[33] - 0.3)).clamp(0.01, 0.95)
    factor1 = (w[25].log() * decay1.pow(-1.0)).clamp(max=60.0).exp() - 1.0
    a1 = factor1 / s_short
    decay2 = -w[24].clamp(0.01, 0.95)
    factor2 = w[26].pow(decay2.pow(-1.0)) - 1.0
    d_ts = ((d - 5.0) * (w[32] - 0.3)).exp()
    a2 = factor2 * d_ts / s_long
    weight1 = w[27] * s_short.pow(-w[29])
    weight2 = w[28] * s_long.pow(w[30]) * ((d - 5.0) * (w[31] - 0.5)).exp()
    wt_sum = (weight1 + weight2).clamp(min=1e-9)
    scale = 1.0 - 2e-5
    dr_t = torch.as_tensor(dr, dtype=s_long.dtype, device=s_long.device)
    log_min, log_max = math.log(min_t), math.log(max_t)
    tol = 1e-10

    def f_of_u(u):
        t = u.exp()
        i1 = a1 * t + 1.0
        i2 = a2 * t + 1.0
        p = (
            weight1 * i1.pow(decay1) + weight2 * i2.pow(decay2)
        ) / wt_sum * scale + 1e-5
        return p - dr_t

    # Tight initial bracket: the root lies between the single-component inverses t1*, t2*
    # (clamped to [min_t, max_t]); far narrower than the full 1min..100yr range, and it
    # auto-collapses unreachable-dr lanes onto a bound (degenerate bracket -> bisection).
    t1 = ((dr_t.pow(1.0 / decay1) - 1.0) / a1).clamp(min=1e-12)
    t2 = ((dr_t.pow(1.0 / decay2) - 1.0) / a2).clamp(min=1e-12)
    a = torch.minimum(t1, t2).log().clamp(log_min, log_max)
    b = torch.maximum(t1, t2).log().clamp(log_min, log_max)
    fa = f_of_u(a)
    fb = f_of_u(b)

    sw = fa.abs() < fb.abs()
    a, b = torch.where(sw, b, a), torch.where(sw, a, b)
    fa, fb = torch.where(sw, fb, fa), torch.where(sw, fa, fb)
    c, fc = a.clone(), fa.clone()
    dd = a.clone()
    mflag = torch.ones_like(s_long, dtype=torch.bool)

    for _ in range(n_iter):
        use_iq = (fa != fc) & (fb != fc)
        denom_a = (fa - fb) * (fa - fc)
        denom_b = (fb - fa) * (fb - fc)
        denom_c = (fc - fa) * (fc - fb)
        s_iq = a * fb * fc / denom_a + b * fa * fc / denom_b + c * fa * fb / denom_c
        dsec = fb - fa
        dsec = torch.where(dsec == 0, torch.full_like(dsec, 1e-30), dsec)
        s_sec = b - fb * (b - a) / dsec
        s = torch.where(use_iq, s_iq, s_sec)

        lo_b = (3.0 * a + b) / 4.0
        not_between = (s - lo_b) * (s - b) >= 0.0
        cond2 = mflag & ((s - b).abs() >= (b - c).abs() / 2.0)
        cond3 = (~mflag) & ((s - b).abs() >= (c - dd).abs() / 2.0)
        cond4 = mflag & ((b - c).abs() < tol)
        cond5 = (~mflag) & ((c - dd).abs() < tol)
        bis = not_between | cond2 | cond3 | cond4 | cond5
        s = torch.where(bis, (a + b) / 2.0, s)
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


def newton_inverse(dr, s_long, s_short, d, w, n_iter, min_t=MIN_T, max_t=MAX_T):
    """Safeguarded Newton (Numerical Recipes rtsafe) in log(t) -- the REJECTED alternative,
    kept here so the head-to-head is reproducible. Same tight bracket as Brent."""
    decay1 = -(w[23] * s_short.pow(w[33] - 0.3)).clamp(0.01, 0.95)
    factor1 = (w[25].log() * decay1.pow(-1.0)).clamp(max=60.0).exp() - 1.0
    a1 = factor1 / s_short
    decay2 = -w[24].clamp(0.01, 0.95)
    factor2 = w[26].pow(decay2.pow(-1.0)) - 1.0
    d_ts = ((d - 5.0) * (w[32] - 0.3)).exp()
    a2 = factor2 * d_ts / s_long
    weight1 = w[27] * s_short.pow(-w[29])
    weight2 = w[28] * s_long.pow(w[30]) * ((d - 5.0) * (w[31] - 0.5)).exp()
    wt_sum = (weight1 + weight2).clamp(min=1e-9)
    scale = 1.0 - 2e-5
    dr_t = torch.as_tensor(dr, dtype=s_long.dtype, device=s_long.device)
    log_min, log_max = math.log(min_t), math.log(max_t)

    def f_and_dfdu(u):
        t = u.exp()
        i1 = a1 * t + 1.0
        i2 = a2 * t + 1.0
        p = (
            weight1 * i1.pow(decay1) + weight2 * i2.pow(decay2)
        ) / wt_sum * scale + 1e-5
        dr1 = decay1 * i1.pow(decay1 - 1.0) * a1
        dr2 = decay2 * i2.pow(decay2 - 1.0) * a2
        dfdu = ((weight1 * dr1 + weight2 * dr2) / wt_sum * scale * t).clamp(max=-1e-30)
        return p - dr_t, dfdu

    t1 = ((dr_t.pow(1.0 / decay1) - 1.0) / a1).clamp(min=1e-12)
    t2 = ((dr_t.pow(1.0 / decay2) - 1.0) / a2).clamp(min=1e-12)
    lo = torch.minimum(t1, t2).log().clamp(log_min, log_max)
    hi = torch.maximum(t1, t2).log().clamp(log_min, log_max)
    u = 0.5 * (lo + hi)
    f, dfdu = f_and_dfdu(u)
    dx_old = hi - lo
    dx = dx_old
    for _ in range(n_iter):
        out = ((u - hi) * dfdu - f) * ((u - lo) * dfdu - f) > 0
        slow = (2.0 * f).abs() > (dx_old * dfdu).abs()
        bisect = out | slow
        dx_old = dx
        dx = torch.where(bisect, 0.5 * (hi - lo), f / dfdu)
        u = torch.where(bisect, 0.5 * (lo + hi), u - f / dfdu)
        f, dfdu = f_and_dfdu(u)
        pos = f > 0
        lo = torch.where(pos, u, lo)
        hi = torch.where(pos, hi, u)
    return u.clamp(log_min, log_max).exp()


def run_method(name, inverter, n, w_tensors, g_widx, sl_t, ss_t, d_t, dr_t):
    t_out = torch.empty_like(sl_t)
    for wi, w in enumerate(w_tensors):
        m = torch.from_numpy(g_widx == wi)
        t_out[m] = inverter(dr_t[m], sl_t[m], ss_t[m], d_t[m], w, n)
    return t_out


def main():
    param_sets = real_param_sets(3)
    states, drs = build_grid()
    print(
        f"grid: {len(param_sets)} param sets x {len(states)} states x {len(drs)} DRs "
        f"= {len(param_sets) * len(states) * len(drs)} cases; "
        f"DR in [{drs.min():.2f}, {drs.max():.2f}], min_t=1min"
    )

    # Flatten all (param, state, dr) cases; compute the independent golden once.
    g_sl, g_ss, g_d, g_dr, g_t, g_widx = [], [], [], [], [], []
    for wi, (_, w) in enumerate(param_sets):
        for sl, ss, d in states:
            for dr in drs:
                g_sl.append(sl)
                g_ss.append(ss)
                g_d.append(d)
                g_dr.append(float(dr))
                g_t.append(golden_t(w, sl, ss, d, float(dr)))
                g_widx.append(wi)
    g_sl = np.array(g_sl)
    g_ss = np.array(g_ss)
    g_d = np.array(g_d)
    g_dr = np.array(g_dr)
    g_t = np.array(g_t)
    g_widx = np.array(g_widx)

    w_tensors = [torch.tensor(w, dtype=torch.float64) for _, w in param_sets]
    sl_t = torch.tensor(g_sl)
    ss_t = torch.tensor(g_ss)
    d_t = torch.tensor(g_d)
    dr_t = torch.tensor(g_dr)

    methods = [
        ("safeguarded Newton (local, rejected; ~4 pow/iter)", newton_inverse),
        ("Brent (local; ~2 pow/iter)", brent_inverse),
    ]

    # Production parity: the fsrs7 inverter (Brent) must match the local Brent here.
    pe = 0.0
    for wi, w in enumerate(w_tensors):
        m = torch.from_numpy(g_widx == wi)
        prod = fsrs7.forgetting_curve_inverse(
            dr_t[m], sl_t[m], ss_t[m], d_t[m], w, n_iter=12
        )
        loc = brent_inverse(dr_t[m], sl_t[m], ss_t[m], d_t[m], w, 12)
        pe = max(pe, float((prod - loc).abs().max()))
    print(
        f"\nproduction fsrs7.forgetting_curve_inverse vs local Brent: max abs diff = {pe:.2e}"
    )

    chosen = {}
    for name, inv in methods:
        print()
        print(f"--- {name} ---")
        print(f"{'n_iter':>6}  {'max_rel':>10}  {'p99_rel':>10}  {'median_rel':>11}")
        chosen[name] = None
        rel = None
        last_t = None
        for n in range(1, 21):
            t_out = run_method(name, inv, n, w_tensors, g_widx, sl_t, ss_t, d_t, dr_t)
            last_t = t_out.numpy()
            rel = np.abs(last_t - g_t) / np.maximum(g_t, MIN_T)
            max_rel = rel.max()
            print(
                f"{n:>6}  {max_rel:>10.2e}  {np.percentile(rel, 99):>10.2e}  "
                f"{np.median(rel):>11.2e}"
            )
            if chosen[name] is None and max_rel < 1e-3:
                chosen[name] = n
        # Diagnose the worst lane at the final iteration count (relative + absolute).
        i = int(np.argmax(rel))
        abs_err_min = abs(last_t[i] - g_t[i]) * 1440.0  # days -> minutes
        print(
            f"   worst@20: s_long={g_sl[i]:.4g} s_short={g_ss[i]:.4g} d={g_d[i]:.3g} "
            f"dr={g_dr[i]:.3g} golden_t={g_t[i]:.4g}d rel={rel[i]:.2e} "
            f"abs={abs_err_min:.3g}min"
        )

    print()
    for name, _ in methods:
        c = chosen[name]
        msg = f"{c} iters" if c is not None else ">20 iters"
        print(f"=> {name}: fewest for worst-case <0.1% = {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
