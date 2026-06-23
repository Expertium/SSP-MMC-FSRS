"""Scheduling policies for the FSRS-7 simulator (roadmap step 2c).

Each policy is a callable

    policy(s_long, s_short, d, prev_interval, grade, ease) -> (interval, ease)

operating on equally-shaped tensors (the batch of cards being scheduled). ``interval`` is
in **days** and may be sub-day (same-day reviews are allowed for every policy that asks for
them); the simulator clamps it to ``[MIN_INTERVAL_DAYS, S_MAX]``. ``ease`` is carried for
Anki-SM-2 and ignored by the others.

Unlike the FSRS-6 policies these consume the 3-component state ``(s_long, s_short, d)`` and,
for desired-retention scheduling, invert the dual-stability forgetting curve numerically
(``fsrs7.forgetting_curve_inverse``, Brent's method) since it has no closed form. The
FSRS-6 ``s_max``-aware interval capping is dropped — it was tied to the invertible curve and
the solver grid; the simulator simply clamps the result.
"""

import torch

from . import fsrs7
from .policies import _anki_sm2_next_interval

# Canonical Memrise schedule WITH its same-day steps restored (4h, 12h, then days). Under
# FSRS-6 (no same-day modelling) the two sub-day steps were dropped; FSRS-7 predicts recall
# for them, so they belong. Source: Memrise "How does the spaced repetition system work".
MEMRISE_STEPS = [4 / 24, 12 / 24, 1.0, 6.0, 12.0, 48.0, 96.0]


def create_dr_policy(desired_retention, w, n_iter=fsrs7.NEWTON_N_ITER):
    """Fixed desired-retention: schedule the interval at which predicted recall == DR
    (may be sub-day right after learning / a lapse — that's the intended FSRS-7 behaviour).

    Uses the fast Newton inverse (``method="newton"``): the simulator only ever schedules
    realistic (recurrence-generated) states, where Newton is ~2x cheaper than Brent and
    matches it to <1e-6 (see fsrs7 module note + tests/inverse_speedup_study.py).

    ``w`` may be a shared ``(34,)`` vector or per-user ``(parallel, 34)``; the latter is
    stored as ``(34, P, 1)`` so it broadcasts against the simulator's ``(P, deck)`` state."""
    w = torch.as_tensor(w, dtype=torch.float64)
    if w.ndim == 2:  # per-user (P, 34) -> (34, P, 1)
        w = w.transpose(0, 1).unsqueeze(-1).contiguous()

    def dr_policy(s_long, s_short, d, prev_interval, grade, ease):
        wt = w.to(device=s_long.device, dtype=s_long.dtype)
        interval = fsrs7.forgetting_curve_inverse(
            desired_retention, s_long, s_short, d, wt, n_iter=n_iter, method="newton"
        )
        return interval, ease

    return dr_policy


def create_fixed_interval_policy(interval, w=None):
    """Always schedule the same interval (days)."""

    def fixed_policy(s_long, s_short, d, prev_interval, grade, ease):
        return torch.full_like(s_long, float(interval)), ease

    return fixed_policy


def make_memrise_policy(w=None):
    """Memrise: step to the next entry in MEMRISE_STEPS (by closest match to the previous
    interval); new cards and lapses (grade==1) restart at the first step (4h)."""

    def memrise_policy(s_long, s_short, d, prev_interval, grade, ease):
        steps = torch.tensor(MEMRISE_STEPS, device=s_long.device, dtype=s_long.dtype)
        dist = (prev_interval.unsqueeze(-1) - steps).abs()
        closest = dist.argmin(dim=-1)
        nxt = (closest + 1).clamp(max=len(MEMRISE_STEPS) - 1)
        interval = steps[nxt]
        restart = (prev_interval == 0) | (grade == 1)
        interval = torch.where(restart, steps[0], interval)
        return interval, ease

    return memrise_policy


def make_anki_sm2_policy(w=None):
    """Anki SM-2 baseline (day-level; same as the FSRS-6 port — its own ease ladder,
    independent of the FSRS state)."""

    def anki_sm2_policy(s_long, s_short, d, prev_interval, grade, ease):
        interval, new_ease = _anki_sm2_next_interval(
            prev_interval,
            prev_interval,  # elapsed == prev_interval (assume on-time)
            grade,
            ease,
            graduating_interval=1.0,
            easy_interval=4.0,
            easy_bonus=1.3,
            hard_interval_factor=1.2,
            ease_min=1.3,
            ease_max=5.5,
        )
        return interval, new_ease

    return anki_sm2_policy
