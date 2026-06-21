"""FSRS-7 PyTorch simulator (roadmap step 2b).

Replaces the inlined FSRS-6 memory model of ``simulation.py`` with the verified FSRS-7
recurrence (``fsrs7``) and adds **same-day reviews**: a card's ``due`` is a fractional day,
and within each calendar day an inner "same-day rounds" loop re-reviews cards whose
scheduled interval lands later the same day (learning steps, lapses, high-DR scheduling).
The short-term stability component predicts recall for these sub-day intervals.

Design (see memory step2-fsrs7):
  - State per (deck, card): ``s_long, s_short, d`` (FSRS-7), plus ``due`` / ``last_date``
    as fractional days, ``ivl``, ``ease`` (SM-2 only), ``reps``, ``lapses``.
  - Outer loop over integer days; inner loop over same-day rounds (capped at
    ``max_same_day`` reviews per card per day, default 10, so it can't blow up at 99% DR).
  - Each round, cards due ``< today + 1`` are reviewed at their own time
    ``t = max(due, today)`` (so ``delta_t = t - last_date`` is the real elapsed time),
    recall is drawn from the dual forgetting curve, the state + next interval are updated,
    and ``due`` is reset (possibly to later the same day -> another round).
  - Same-day reviews count against the shared daily budgets ``max_cost_perday`` /
    ``review_limit_perday``; ``learn_limit_perday`` caps new cards/day. Cards that don't fit
    the budget roll over to the next day (their ``due`` is left in the past).

This is the source-of-truth Python implementation (float64). It currently uses torch RNG;
the shared counter-RNG path (for bit-level Python<->Rust parity, step 2d) needs a per-round
counter dimension and will be added with the Rust port.
"""

import numpy as np
import torch
from tqdm import trange

from . import fsrs7
from .config import (
    DEFAULT_LEARN_COSTS,
    DEFAULT_REVIEW_COSTS,
    DEFAULT_FIRST_RATING_PROB,
    DEFAULT_REVIEW_RATING_PROB,
)

# FSRS-7 with --secs uses this stability floor (srs-benchmark Config.s_min); the model
# clamps S to [S_MIN_SECS, S_MAX]. Differs from SSP-MMC's solver-grid S_MIN (0.1).
S_MIN_SECS = 1e-4


@torch.inference_mode()
def simulate(
    parallel,
    w,
    policy,
    device,
    deck_size=10000,
    learn_span=365,
    max_cost_perday=86400 / 2,
    learn_limit_perday=10,
    review_limit_perday=9999,
    learn_costs=DEFAULT_LEARN_COSTS,
    review_costs=DEFAULT_REVIEW_COSTS,
    first_rating_prob=DEFAULT_FIRST_RATING_PROB,
    review_rating_prob=DEFAULT_REVIEW_RATING_PROB,
    seed=42,
    s_min=S_MIN_SECS,
    s_max=fsrs7.S_MAX,
    max_same_day=10,
    progress_desc=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dtype = torch.float64
    shape = (parallel, deck_size)

    w_t = torch.as_tensor(w, dtype=dtype, device=device)

    # Memory state. s_long == 0 marks a card that hasn't been learned yet (init_state always
    # returns s_long >= s_min > 0), mirroring the reference's all-zeros "is_first".
    s_long = torch.zeros(shape, dtype=dtype, device=device)
    s_short = torch.zeros_like(s_long)
    diff = torch.zeros_like(s_long)
    last_date = torch.zeros_like(s_long)
    due = torch.full(shape, float("inf"), dtype=dtype, device=device)
    ivl = torch.zeros_like(s_long)
    ease = torch.full_like(s_long, 2.5)
    reps = torch.zeros_like(s_long)
    lapses = torch.zeros_like(s_long)

    # First-review rating per card (same across decks, as in the FSRS-6 torch path).
    first_ratings = np.random.choice([1, 2, 3, 4], deck_size, p=first_rating_prob)
    first_rating = (
        torch.as_tensor(first_ratings, dtype=torch.int64, device=device)
        .unsqueeze(0)
        .expand(parallel, deck_size)
        .contiguous()
    )

    review_costs_t = torch.as_tensor(review_costs, dtype=dtype, device=device)
    learn_costs_t = torch.as_tensor(learn_costs, dtype=dtype, device=device)
    pass_ratings = torch.tensor([2, 3, 4], device=device)
    review_rating_prob_t = torch.as_tensor(
        review_rating_prob, dtype=dtype, device=device
    )

    review_cnt_per_day = torch.zeros((parallel, learn_span), dtype=dtype, device=device)
    learn_cnt_per_day = torch.zeros_like(review_cnt_per_day)
    memorized_cnt_per_day = torch.zeros_like(review_cnt_per_day)
    cost_per_day = torch.zeros_like(review_cnt_per_day)

    def clamp_state(sl, ss, dd):
        return (
            sl.clamp(s_min, s_max),
            ss.clamp(s_min, s_max),
            dd.clamp(fsrs7.D_MIN, fsrs7.D_MAX),
        )

    for today in trange(learn_span, position=1, leave=False, desc=progress_desc):
        today_f = torch.full((), float(today), dtype=dtype, device=device)

        # --- Memorized snapshot at the start of the day (over learned cards). ---
        learned = s_long > 0
        dt_snap = (today_f - last_date).clamp(min=0.0)
        sl_c, ss_c, d_c = clamp_state(s_long, s_short, diff)
        r_snap = fsrs7.forgetting_curve(dt_snap, sl_c, ss_c, d_c, w_t)
        memorized_cnt_per_day[:, today] = torch.where(
            learned, r_snap, torch.zeros_like(r_snap)
        ).sum(dim=-1)

        # --- Per-day shared budgets / counters (carried across same-day rounds). ---
        cost_used = torch.zeros((parallel, 1), dtype=dtype, device=device)
        reviews_used = torch.zeros((parallel, 1), dtype=dtype, device=device)
        learns_used = torch.zeros((parallel, 1), dtype=dtype, device=device)
        same_day_count = torch.zeros(shape, dtype=dtype, device=device)

        for rnd in range(max_same_day):
            # Review candidates: learned, due today (or overdue), under the per-card cap.
            rev_cand = learned & (due < today_f + 1.0) & (same_day_count < max_same_day)
            # New-card learning only happens in the first round of the day.
            learn_cand = (s_long == 0) if rnd == 0 else torch.zeros_like(rev_cand)

            if not (rev_cand.any() or learn_cand.any()):
                break

            # Review at the scheduled time (or now, if overdue); finite for non-candidates.
            t_review = torch.where(rev_cand, torch.maximum(due, today_f), today_f)
            dt = (t_review - last_date).clamp(min=0.0)
            sl_c, ss_c, d_c = clamp_state(s_long, s_short, diff)
            r = fsrs7.forgetting_curve(dt, sl_c, ss_c, d_c, w_t)
            forget = torch.rand(shape, dtype=dtype, device=device) > r
            pass_idx = torch.multinomial(
                review_rating_prob_t, parallel * deck_size, replacement=True
            ).view(shape)
            pass_rating = pass_ratings[pass_idx]
            rev_rating = torch.where(forget, torch.ones_like(pass_rating), pass_rating)

            rating = torch.where(rev_cand, rev_rating, first_rating)
            rev_cost = review_costs_t[rev_rating - 1]
            learn_cost = learn_costs_t[first_rating - 1]
            cand_cost = torch.where(
                rev_cand, rev_cost, torch.zeros_like(rev_cost)
            ) + torch.where(learn_cand, learn_cost, torch.zeros_like(learn_cost))

            # Budget gating (prefix in card-index order, continued from earlier rounds).
            cum_cost = cost_used + torch.cumsum(cand_cost, dim=-1)
            cum_rev = reviews_used + torch.cumsum(rev_cand.to(dtype), dim=-1)
            cum_learn = learns_used + torch.cumsum(learn_cand.to(dtype), dim=-1)
            admit = (
                (rev_cand | learn_cand)
                & (cum_cost <= max_cost_perday)
                & (~rev_cand | (cum_rev <= review_limit_perday))
                & (~learn_cand | (cum_learn <= learn_limit_perday))
            )
            admit_rev = admit & rev_cand
            admit_learn = admit & learn_cand

            if not admit.any():
                break

            # --- Apply reviews. ---
            if admit_rev.any():
                u_sl, u_ss, u_d = fsrs7.update_state(
                    dt, rating, s_long, s_short, diff, w_t, s_min, s_max
                )
                s_long = torch.where(admit_rev, u_sl, s_long)
                s_short = torch.where(admit_rev, u_ss, s_short)
                diff = torch.where(admit_rev, u_d, diff)
                last_date = torch.where(admit_rev, t_review, last_date)
                reps = reps + (admit_rev & ~forget).to(dtype)
                lapses = lapses + (admit_rev & forget).to(dtype)

            # --- Apply new-card learning. ---
            if admit_learn.any():
                i_sl, i_ss, i_d = fsrs7.init_state(first_rating, w_t, s_min)
                s_long = torch.where(admit_learn, i_sl, s_long)
                s_short = torch.where(admit_learn, i_ss, s_short)
                diff = torch.where(admit_learn, i_d, diff)
                last_date = torch.where(admit_learn, today_f, last_date)
                learned = s_long > 0

            same_day_count = torch.where(
                admit_rev, same_day_count + 1.0, same_day_count
            )

            # --- Update running budgets. ---
            cost_used = cost_used + (cand_cost * admit).sum(dim=-1, keepdim=True)
            reviews_used = reviews_used + admit_rev.sum(dim=-1, keepdim=True).to(dtype)
            learns_used = learns_used + admit_learn.sum(dim=-1, keepdim=True).to(dtype)
            review_cnt_per_day[:, today] += admit_rev.sum(dim=-1).to(dtype)
            learn_cnt_per_day[:, today] += admit_learn.sum(dim=-1).to(dtype)

            # --- Schedule the next interval/due for everything acted on this round. ---
            act = admit_rev | admit_learn
            base_time = torch.where(admit_learn, today_f, t_review)
            prev_ivl = torch.where(admit_learn, torch.zeros_like(ivl), ivl)
            # Clamp state for the policy so un-acted cards (s_long==0) can't feed NaN/inf
            # into the curve inverter; for acted cards this is a no-op (already in range).
            psl, pss, pdd = clamp_state(s_long, s_short, diff)
            new_ivl, new_ease = policy(psl, pss, pdd, prev_ivl, rating, ease)
            new_ivl = new_ivl.clamp(fsrs7.MIN_INTERVAL_DAYS, s_max)
            ivl = torch.where(act, new_ivl, ivl)
            ease = torch.where(act, new_ease, ease)
            due = torch.where(act, base_time + new_ivl, due)

        cost_per_day[:, today] = cost_used.squeeze(-1)

    return (
        review_cnt_per_day.cpu().numpy(),
        learn_cnt_per_day.cpu().numpy(),
        memorized_cnt_per_day.cpu().numpy(),
        cost_per_day.cpu().numpy(),
    )
