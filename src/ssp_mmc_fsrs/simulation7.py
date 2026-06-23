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
    ``max_same_day`` reviews per card per day, default 8 = p99 of same-day review counts
    measured across anki-revlogs-10k, so it can't blow up at 99% DR).
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
from . import shared_rng
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
    max_same_day=8,
    progress_desc=None,
    rng_kind="torch",
    gru=None,
):
    if rng_kind not in ("torch", "shared"):
        raise ValueError(f"Unknown rng_kind: {rng_kind!r} (use 'torch' or 'shared')")
    # Optional GRU pseudo-ground-truth recall predictor (roadmap step 4). When given, FSRS-7
    # still SCHEDULES (the policy is unchanged) but the GRU decides recall/forget and the
    # recorded knowledge -- instead of FSRS's own p(recall). The GRU is Markovian in its
    # hidden state h (a ``BatchedGRU`` from ssp_mmc_fsrs.gru); we carry h per card and step it
    # with each realized (dt, rating), exactly like the FSRS (s_long, s_short, d) state. Must
    # be batched over the same ``parallel`` axis and built in this sim's dtype/device.
    use_gru = gru is not None
    if use_gru and gru.parallel != parallel:
        raise ValueError(
            f"gru.parallel ({gru.parallel}) must equal parallel ({parallel})"
        )
    torch.manual_seed(seed)
    np.random.seed(seed)
    dtype = torch.float64
    shape = (parallel, deck_size)

    def _per_user(arr, k):
        """Normalize a cost/prob arg to a contiguous (parallel, k) tensor: a shared 1-D
        ``(k,)`` arg is broadcast to every user; a ``(parallel, k)`` arg is used per user."""
        t = torch.as_tensor(arr, dtype=dtype, device=device)
        if t.ndim == 1:
            if t.shape[0] != k:
                raise ValueError(f"expected length-{k} array, got {tuple(t.shape)}")
            return t.unsqueeze(0).expand(parallel, k).contiguous()
        if t.shape != (parallel, k):
            raise ValueError(f"expected ({parallel}, {k}) array, got {tuple(t.shape)}")
        return t.contiguous()

    # FSRS-7 params: shared ``(34,)`` OR per-user ``(parallel, 34)``. Carried as ``(34, P, 1)``
    # so every fsrs7 ``w[i]`` indexes to ``(P, 1)`` and broadcasts against the ``(P, deck)``
    # state (init_state gathers each user's own w[0..3]).
    w_in = torch.as_tensor(w, dtype=dtype, device=device)
    if w_in.ndim == 1:
        w_t = w_in.view(-1, 1, 1).expand(-1, parallel, 1).contiguous()
    elif w_in.ndim == 2 and w_in.shape == (parallel, w_in.shape[1]):
        w_t = w_in.transpose(0, 1).unsqueeze(-1).contiguous()
    else:
        raise ValueError(
            f"w must be (34,) or ({parallel}, 34), got {tuple(w_in.shape)}"
        )

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

    # Per-card GRU hidden state (P, deck, n_hidden), zeros = nn.GRU's default h_0. Advanced
    # with each realized review; read via gru.p_recall(h, dt) to get the GRU's p(recall).
    h_gru = gru.init_hidden(deck_size) if use_gru else None

    # Per-user cost/probability tables (a shared 1-D arg is broadcast across all users).
    review_costs_t = _per_user(review_costs, 4)  # (P, 4)
    learn_costs_t = _per_user(learn_costs, 4)  # (P, 4)
    review_rating_prob_t = _per_user(review_rating_prob, 3)  # (P, 3)
    first_rating_prob_t = _per_user(first_rating_prob, 4)  # (P, 4)
    pass_ratings = torch.tensor([2, 3, 4], device=device)

    # First-review rating per (user, card) -- per user, NOT broadcast across the parallel
    # axis: each user draws from its OWN first_rating_prob. torch path: per-row multinomial;
    # shared path: per-(deck, card) counter draw (KIND_INIT_RATING, round 0) with a per-user
    # categorical, so it matches the Rust simulator cell-for-cell.
    if rng_kind == "torch":
        first_rating = (
            torch.multinomial(first_rating_prob_t, deck_size, replacement=True) + 1
        ).to(torch.int64)
    else:
        u0 = shared_rng.uniform_block_r(
            shared_rng.KIND_INIT_RATING,
            0,
            0,
            parallel,
            deck_size,
            learn_span,
            max_same_day,
            seed,
        )
        first_rating = torch.as_tensor(
            shared_rng.categorical(u0, first_rating_prob) + 1,
            dtype=torch.int64,
            device=device,
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
        # Knowledge = sum of p(recall); GRU's p(recall) when a predictor is supplied.
        learned = s_long > 0
        dt_snap = (today_f - last_date).clamp(min=0.0)
        if use_gru:
            r_snap = gru.p_recall(h_gru, dt_snap)
        else:
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
            if use_gru:
                r = gru.p_recall(h_gru, dt)
            else:
                sl_c, ss_c, d_c = clamp_state(s_long, s_short, diff)
                r = fsrs7.forgetting_curve(dt, sl_c, ss_c, d_c, w_t)
            if rng_kind == "torch":
                rand = torch.rand(shape, dtype=dtype, device=device)
                # Per-user pass-rating draw: row p samples from its own review_rating_prob.
                pass_idx = torch.multinomial(
                    review_rating_prob_t, deck_size, replacement=True
                )
            else:  # shared: per-(day, round, deck, card) counter draws
                u_f = shared_rng.uniform_block_r(
                    shared_rng.KIND_FORGET,
                    today,
                    rnd,
                    parallel,
                    deck_size,
                    learn_span,
                    max_same_day,
                    seed,
                )
                rand = torch.as_tensor(u_f, dtype=dtype, device=device)
                u_p = shared_rng.uniform_block_r(
                    shared_rng.KIND_PASS_RATING,
                    today,
                    rnd,
                    parallel,
                    deck_size,
                    learn_span,
                    max_same_day,
                    seed,
                )
                pass_idx = torch.as_tensor(
                    shared_rng.categorical(u_p, review_rating_prob),
                    dtype=torch.int64,
                    device=device,
                )
            forget = rand > r
            pass_rating = pass_ratings[pass_idx]
            rev_rating = torch.where(forget, torch.ones_like(pass_rating), pass_rating)

            rating = torch.where(rev_cand, rev_rating, first_rating)
            # Per-user costs: gather each user's own cost for the realized rating.
            rev_cost = torch.gather(review_costs_t, 1, rev_rating - 1)
            learn_cost = torch.gather(learn_costs_t, 1, first_rating - 1)
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

            # --- Advance the GRU hidden state for everything reviewed/learned this round. ---
            # Predict-then-advance: the recall draw above used the pre-review h; now step it
            # with each realized (dt, rating). Learning is the card's first sequence element
            # (dt = 0, first_rating); a review uses its elapsed dt and realized rev_rating.
            # admit_learn and admit_rev are disjoint, so both step from the same old h.
            if use_gru:
                h_learn = gru.step(h_gru, torch.zeros_like(dt), first_rating)
                h_rev = gru.step(h_gru, dt, rev_rating)
                h_gru = torch.where(admit_learn.unsqueeze(-1), h_learn, h_gru)
                h_gru = torch.where(admit_rev.unsqueeze(-1), h_rev, h_gru)

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
