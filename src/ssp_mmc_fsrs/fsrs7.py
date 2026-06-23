"""FSRS-7 memory model (dual-stability, 34 parameters).

A faithful, vectorized port of the reference recurrence in
``C:\\Users\\Andrew\\srs-benchmark\\models\\fsrs_v7.py`` (``class FSRS7(FSRS6)``,
itself ported from the Rust fsrs-rs at
https://github.com/Expertium/fsrs-rs-speed-autoresearch). Roadmap step 2.

The memory state has **three** components instead of FSRS-6's two:

* ``s_long``  — long-term stability (days)
* ``s_short`` — short-term stability (days), which carries within-/few-day recall
* ``d``       — difficulty (1..10)

The forgetting curve mixes a short-term recall component ``r1`` (driven by
``s_short``) and a long-term recall component ``r2`` (driven by ``s_long``, with the
difficulty effect applied to the time-scale). Unlike FSRS-6's single power curve this
mixture is **not** analytically invertible — policies that need "interval for a target
retention" must root-find (roadmap step 2c).

These are pure functions of ``w`` (a 34-element tensor, indexable by ``int`` to get a
0-dim scalar) so they can be shared by the simulator, the Bellman solver, and the
policies. They mirror ``fsrs_v7.py`` op-for-op; ``tests/test_fsrs7_parity.py`` checks
numerical agreement against the live reference. Param-block layout (matching the
reference's index comments):

* ``0..3``   initial stability by rating
* ``4..6``   difficulty
* ``7..14``  long-term stability update  (``next_stability`` with ``start=7``)
* ``15..22`` short-term stability update (``next_stability`` with ``start=15``)
* ``23..33`` forgetting curve: 23 decay1, 24 decay2, 25 base1, 26 base2,
  27 base_weight1, 28 base_weight2, 29 s_weight_power1, 30 s_weight_power2,
  31 d_weight, 32 d_decay, 33 s_decay1
"""

import math

import torch

# Memory-state clamp bounds (model.rs S_MIN/S_MAX, D_MIN/D_MAX in fsrs-rs). S_MAX and the
# difficulty bounds are fixed; the stability floor (s_min) is config-dependent and passed
# in (0.0001 for the FSRS-7 `--secs` reference).
S_MAX = 36500.0
D_MIN = 1.0
D_MAX = 10.0


def short_component_recall(t, s_short, w):
    """Short-term recall component ``r1``, driven by the short-term stability (its decay
    is S-modulated via ``s_decay1`` = w[33]). Shared by the forgetting curve (the
    mixture) and the short-term stability update (which reads ``r1``, not the mixed R)."""
    t = t.clamp(min=0.0)
    t_over_s_short = t / s_short
    decay1_mag = (w[23] * s_short.pow(w[33] - 0.3)).clamp(0.01, 0.95)
    decay1 = -decay1_mag
    # factor1 built in log-space with the exponent clamped at 60 so value+gradient stay
    # finite (mirrors the reference's stabilized factor1).
    factor1 = (w[25].log() * decay1.pow(-1.0)).clamp(max=60.0).exp() - 1.0
    return (t_over_s_short * factor1 + 1.0).pow(decay1)


def forgetting_curve(t, s, s_short, d, w):
    """Dual-stability forgetting curve. Mixes the short component ``r1`` (from
    ``s_short``) and the long component ``r2`` (from ``s``, difficulty on the time-scale),
    weighted by ``weight1``/``weight2`` (``weight2`` is difficulty-modulated). Final
    rescale ``p = 1e-5 + (1 - 2e-5) * retention``."""
    t = t.clamp(min=0.0)
    t_over_s_long = t / s

    # Short-term component r1 reads the short-term S (shared with the short-term update).
    r1 = short_component_recall(t, s_short, w)

    # Long-term component r2 reads the long-term S; difficulty acts on the horizontal
    # TIME-SCALE (decay2 itself is not d-modulated).
    decay2 = -w[24].clamp(0.01, 0.95)
    factor2 = w[26].pow(decay2.pow(-1.0)) - 1.0
    d_timescale = ((d - 5.0) * (w[32] - 0.3)).exp()
    r2 = (t_over_s_long * factor2 * d_timescale + 1.0).pow(decay2)

    # Mixture weights keyed to each S; weight2 is D-modulated (d_weight).
    weight1 = w[27] * s_short.pow(-w[29])
    weight2 = w[28] * s.pow(w[30]) * ((d - 5.0) * (w[31] - 0.5)).exp()

    retention = (weight1 * r1 + weight2 * r2) / (weight1 + weight2)
    return retention * (1.0 - 2e-5) + 1e-5


def next_stability(last_s, last_d, r, rating, start, w):
    """Stability after a review. ``start`` selects the parameter block: 7 for the
    long-term S, 15 for the short-term S. On success returns ``max(pls, last_s*sinc)``;
    on a lapse returns the post-lapse stability ``pls`` (which is D-INDEPENDENT in the
    finished model). Not clamped here — the caller clamps to ``s_min..S_MAX``."""
    ones = torch.ones_like(last_s)
    hard_penalty = torch.where(rating == 2, w[start + 6], ones)
    easy_bonus = torch.where(rating == 4, w[start + 7], ones)

    new_s_fail = (
        w[start + 3]
        * ((last_s + 1.0).pow(w[start + 4]) - 1.0)
        * ((1.0 - r) * w[start + 5]).exp()
    )
    pls = torch.minimum(last_s, new_s_fail)

    sinc = (w[start] - 1.5).exp() * (11.0 - last_d) * last_s.pow(-w[start + 1]) * (
        ((1.0 - r) * w[start + 2]).exp() - 1.0
    ) * hard_penalty * easy_bonus + 1.0
    new_s_success = torch.maximum(pls, last_s * sinc)

    success = rating > 1
    return torch.where(success, new_s_success, pls)


def init_d(rating, w):
    """Initial difficulty for a first review (inherited FSRS-5/6 form). Not clamped."""
    return w[4] - torch.exp(w[5] * (rating - 1)) + 1.0


def linear_damping(delta_d, old_d):
    return delta_d * (10.0 - old_d) / 9.0


def mean_reversion(init, current):
    """Fixed 1% / 99% reversion (FSRS-7 overrides the FSRS-4/5 ``w[7]`` reversion)."""
    return 0.01 * init + 0.99 * current


def next_difficulty(last_d, rating, retention, w):
    """Difficulty update with surprise-weighted lapse: on a lapse (rating==1) scale
    ``delta_d`` by ``retention + 0.1``. Clamped to ``D_MIN..D_MAX``."""
    delta_d = -w[6] * (rating - 3)
    delta_d_lapse = delta_d * (retention + 0.1)
    delta_d = torch.where(rating == 1, delta_d_lapse, delta_d)
    new_d = last_d + linear_damping(delta_d, last_d)
    new_d = mean_reversion(init_d(4, w), new_d)
    return new_d.clamp(D_MIN, D_MAX)


def init_state(rating, w, s_min):
    """First-review memory state ``(s_long, s_short, d)``.

    ``s_long`` = initial stability by rating (w[0..3]); ``s_short`` = 0.8 * s_long;
    ``d`` = clamped init_d. Stabilities clamped to ``s_min..S_MAX``.

    ``w`` may be a shared 1-D ``(34,)`` tensor or a per-user ``(34, P, 1)`` tensor (the
    simulator's batched layout). In the per-user case the initial stability is gathered
    per user (each deck row reads its own ``w[0..3]``)."""
    rating_idx = rating.long().clamp(1, 4) - 1
    if w.dim() == 1:
        s0 = w[rating_idx]
    else:
        # w is (34, P, 1); rating_idx is (P, deck). Gather each user's own w[0..3].
        w2 = w.squeeze(-1).transpose(0, 1)  # (P, 34)
        s0 = torch.gather(w2, 1, rating_idx)  # (P, deck)
    s_long = s0.clamp(s_min, S_MAX)
    s_short = (0.8 * s0).clamp(s_min, S_MAX)
    d = init_d(rating, w).clamp(D_MIN, D_MAX)
    return s_long, s_short, d


def update_state(delta_t, rating, s_long, s_short, d, w, s_min, s_max=S_MAX):
    """Memory state ``(s_long, s_short, d)`` after a (non-first) review at elapsed
    ``delta_t``. The mixed retrievability drives the long-term S and difficulty updates;
    the short-term S uses its own recall ``r1``; on a lapse the short-term S is capped at
    ``0.8 * post-lapse long-term S``."""
    last_s = s_long.clamp(s_min, s_max)
    last_s_short = s_short.clamp(s_min, s_max)
    last_d = d.clamp(D_MIN, D_MAX)

    retrievability = forgetting_curve(delta_t, last_s, last_s_short, last_d, w)
    upd_s_long = next_stability(last_s, last_d, retrievability, rating, 7, w)

    r1 = short_component_recall(delta_t, last_s_short, w)
    upd_s_short = next_stability(last_s_short, last_d, r1, rating, 15, w)
    relearn = torch.minimum(upd_s_short, 0.8 * upd_s_long)
    upd_s_short = torch.where(rating == 1, relearn, upd_s_short)

    upd_d = next_difficulty(last_d, rating, retrievability, w)

    return (
        upd_s_long.clamp(s_min, s_max),
        upd_s_short.clamp(s_min, s_max),
        upd_d.clamp(D_MIN, D_MAX),
    )


def step(delta_t, rating, s_long, s_short, d, w, s_min, s_max=S_MAX):
    """One FSRS-7 transition, mirroring the reference ``FSRS7.step``: select the
    first-review init when the incoming state is all-zeros (per element), else the update.

    ``delta_t`` and ``rating`` are tensors of shape ``[batch]``; the state tensors match.
    Returns the new ``(s_long, s_short, d)``."""
    is_first = (s_long == 0) & (s_short == 0) & (d == 0)

    init_s_long, init_s_short, init_d_ = init_state(rating, w, s_min)
    upd_s_long, upd_s_short, upd_d = update_state(
        delta_t, rating, s_long, s_short, d, w, s_min, s_max
    )

    new_s_long = torch.where(is_first, init_s_long, upd_s_long)
    new_s_short = torch.where(is_first, init_s_short, upd_s_short)
    new_d = torch.where(is_first, init_d_, upd_d)
    return new_s_long, new_s_short, new_d


# ──────────────────────────────────────────────────────────────────────────────
# Interval inversion (scheduling): find t such that forgetting_curve(t, ...) = dr.
# The dual-stability curve has no closed-form inverse, so we root-find. We use Brent's
# method (bracketing: inverse-quadratic / secant / bisection) in log(t) space, inverting
# the FULL forgetting_curve p (the same value the simulator uses to decide recall). Used
# by the FSRS-7 policies (roadmap step 2c).
#
# Why Brent and not Newton: tests/newton_steps_study.py compares both, vectorized, against
# a scipy.brentq golden over a grid of states x DR in [0.60, 0.99] x real param sets.
# Brent converges to machine precision in ~1 step for the median case and reaches
# worst-case interval error < 0.1% in 12 iterations; a safeguarded-Newton (rtsafe) variant
# needed >20 (its steps, started from the bracket midpoint, are repeatedly rejected) AND
# costs ~2x the transcendentals per iteration (it also needs dR/dt). So Brent is both
# faster-converging and cheaper here.
# ──────────────────────────────────────────────────────────────────────────────

# Shortest schedulable interval: 1 minute, in days.
MIN_INTERVAL_DAYS = 1.0 / 1440.0

# Brent iterations for the interval inversion (worst-case interval error < 0.1% over
# DR in [0.60, 0.99]; the median case is exact in 1 step). See tests/newton_steps_study.py.
INVERSE_N_ITER = 12


def forgetting_curve_inverse(
    dr,
    s_long,
    s_short,
    d,
    w,
    n_iter=INVERSE_N_ITER,
    min_t=MIN_INTERVAL_DAYS,
    max_t=S_MAX,
):
    """Interval ``t`` (days) at which the dual-stability recall probability equals ``dr``,
    via a vectorized Brent's method in ``u = log(t)`` (classic Dekker-Brent: inverse-
    quadratic / secant / bisection, all branch-free through torch.where). ``dr`` and the
    state tensors are broadcastable. The root is bracketed by the two single-component
    inverses t1* (short) and t2* (long) -- where r1, resp. r2, alone equals dr -- because
    p is their decreasing weighted mix; clamping that bracket to ``[min_t, max_t]`` also
    collapses an UNREACHABLE dr (root past a bound) onto that bound. Result in
    ``[min_t, max_t]``."""
    decay1 = -(w[23] * s_short.pow(w[33] - 0.3)).clamp(0.01, 0.95)
    factor1 = (w[25].log() * decay1.pow(-1.0)).clamp(max=60.0).exp() - 1.0
    a1 = factor1 / s_short

    decay2 = -w[24].clamp(0.01, 0.95)
    factor2 = w[26].pow(decay2.pow(-1.0)) - 1.0
    d_timescale = ((d - 5.0) * (w[32] - 0.3)).exp()
    a2 = factor2 * d_timescale / s_long

    weight1 = w[27] * s_short.pow(-w[29])
    weight2 = w[28] * s_long.pow(w[30]) * ((d - 5.0) * (w[31] - 0.5)).exp()
    wt_sum = (weight1 + weight2).clamp(min=1e-9)

    log_min = math.log(min_t)
    log_max = math.log(max_t)
    scale = 1.0 - 2e-5
    dr_t = torch.as_tensor(dr, dtype=s_long.dtype, device=s_long.device)
    tol = 1e-12

    def f_of_u(u):
        t = u.exp()
        inner1 = a1 * t + 1.0
        inner2 = a2 * t + 1.0
        p = (weight1 * inner1.pow(decay1) + weight2 * inner2.pow(decay2)) / wt_sum
        return p * scale + 1e-5 - dr_t

    # Tight initial bracket [a, b] from the single-component inverses (clamped to range).
    t1 = ((dr_t.pow(1.0 / decay1) - 1.0) / a1).clamp(min=1e-12)
    t2 = ((dr_t.pow(1.0 / decay2) - 1.0) / a2).clamp(min=1e-12)
    a = torch.minimum(t1, t2).log().clamp(log_min, log_max)
    b = torch.maximum(t1, t2).log().clamp(log_min, log_max)
    fa = f_of_u(a)
    fb = f_of_u(b)

    # Brent keeps |f(b)| <= |f(a)| (b is the running best estimate).
    sw = fa.abs() < fb.abs()
    a, b = torch.where(sw, b, a), torch.where(sw, a, b)
    fa, fb = torch.where(sw, fb, fa), torch.where(sw, fa, fb)
    c, fc = a.clone(), fa.clone()
    dd = a.clone()
    mflag = torch.ones_like(b, dtype=torch.bool)

    for _ in range(n_iter):
        use_iq = (fa != fc) & (fb != fc)
        s_iq = (
            a * fb * fc / ((fa - fb) * (fa - fc))
            + b * fa * fc / ((fb - fa) * (fb - fc))
            + c * fa * fb / ((fc - fa) * (fc - fb))
        )
        dsec = torch.where(fb - fa == 0, torch.full_like(fa, 1e-30), fb - fa)
        s_sec = b - fb * (b - a) / dsec
        s = torch.where(use_iq, s_iq, s_sec)

        # Conditions forcing a bisection step (s outside [(3a+b)/4, b], or steps stalling).
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
        # Replace the endpoint that keeps the root bracketed (f changes sign across [a, b]).
        side = fa * fs < 0.0
        a = torch.where(side, a, s)
        fa = torch.where(side, fa, fs)
        b = torch.where(side, s, b)
        fb = torch.where(side, fs, fb)
        sw = fa.abs() < fb.abs()
        a, b = torch.where(sw, b, a), torch.where(sw, a, b)
        fa, fb = torch.where(sw, fb, fa), torch.where(sw, fa, fb)

    return b.clamp(log_min, log_max).exp()
