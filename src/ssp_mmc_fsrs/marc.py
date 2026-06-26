"""MARC — Maximize Accumulated Retention, Cost-adjusted (objective redesign for SSP-MMC).

The dual of SSP-MMC's "Minimize Memorization Cost". SSP-MMC's Bellman minimizes a *shaped cost
to reach S_MAX* — an instrumental proxy that the step-5 study showed is the wrong objective
(it chases stability and produces harmful heterogeneity; see memory ``ssp-mmc-dominated-by-dr``).
MARC keeps the SAME machinery (tabular value iteration over the FSRS-7 ``(D, S_long, S_short)``
grid, action = target retention ``R ∈ [0.60, 0.99]``) but fixes the **goal**:

    reward(state, R) = ∫₀^Δ p_recall(τ) dτ  −  λ · E[review cost]
                       └─ retention area ─┘     └─ price-of-time penalty ─┘

and **MAXIMIZES** the long-run discounted sum (no ``S_MAX`` terminal). ``Δ`` is the interval the
action schedules; the integral (closed-form, ``fsrs7.forgetting_curve_area``) is EXACTLY a card's
contribution to the simulator's knowledge metric, so MARC optimizes the true currency
(recall-per-second) instead of an instrumental stability goal. ``λ`` is the single price-of-time
(the Lagrange multiplier of the daily-time budget the per-card MDP can't see); sweep ``λ`` to
trace the knowledge↔workload front. This is a **subtraction (Lagrangian), not a ratio**, and
there is **one ``λ``** even though fail/success reviews cost different *time* (failure is already
penalized through lost retention area, not a second price).

Why it is NOT ADR: ADR is a low-capacity 15-coefficient closed form fit by black-box search;
MARC is the exact optimal control of the right objective — a high-capacity per-state table from
value iteration. The only shared idea is ``λ`` (intrinsic to any knowledge-vs-time tradeoff).

Build decisions (memory ``ssp-mmc-objective-redesign``):
  * Keep the current ``(S_long, S_short, D)`` grid steps; redo the grid-convergence study at
    build time (MARC's value function is smoother — no terminal — so likely same-or-coarser).
  * Lower the solver-grid ``S_max`` from 100y to ~5–20y (default 10y here): with no ``S_MAX``
    terminal the high-S tail isn't needed, and states with true ``S > S_max`` are clamped to the
    grid top at lookup (a free consequence of ``searchsorted``-clamped indexing).
  * Keep ``R ∈ [0.60, 0.99]`` (inherited from ``SSPMMCSolver7``) so MARC compares apples-to-apples
    with fixed DR (Anki's DR range), NOT widened down.

**Status: reference implementation, not yet benchmarked.** Value iteration is eager (torch ops on
the GPU, but not the fused minimize-only Triton kernels in ``solver7`` — a maximizing kernel is a
later optimization). The retention table it produces has the SAME shape/order as SSP-MMC's, so it
flows through the Rust simulator's ``policy="ssp_mmc"`` path with no Rust changes.
"""

import time

import numpy as np
import torch

from . import fsrs7
from . import solver7
from .solver7 import SSPMMCSolver7, DISCOUNT_FACTOR

# MARC drops the S_MAX terminal, so it solves on a much shorter stability horizon than SSP-MMC's
# 100-year grid. 10 years is the middle of the 5–20y build decision; states above it clamp to the
# grid top at policy lookup.
MARC_S_MAX_DEFAULT = 365 * 10


def build_marc_s_grid(
    s_max=MARC_S_MAX_DEFAULT, n_linear=5, n_log=66, skew=0.4, lin_max=0.1
):
    """A lower-``S_max`` version of ``solver7.build_hybrid_s_grid`` (same 5-linear + 66-skewed-log
    shape, 71 points), capped at ``s_max`` days instead of 36500. The base solver keeps its model
    clamp ``self.s_max = fsrs7.S_MAX``; because this grid tops out lower, any next-state stability
    above ``s_max`` is snapped to the top grid cell on lookup — exactly the "clamp S > S_max" rule.
    """
    s_min = solver7.S_MIN
    lin = np.linspace(s_min, lin_max, n_linear)
    u = np.arange(n_log + 1) / n_log
    v = u**skew
    logp = np.exp(np.log(lin_max) + v * (np.log(s_max) - np.log(lin_max)))[1:]
    s = np.concatenate([lin, logp])
    s[-1] = s_max
    return s


class MARCSolver7(SSPMMCSolver7):
    """FSRS-7 MARC solver: same 3-D grid + transitions as ``SSPMMCSolver7``, but the reward is the
    retention area minus ``λ·cost`` and value iteration MAXIMIZES with no terminal.

    ``solve(lambda_cost)`` returns ``(value_matrix, retention_matrix)`` shaped
    ``(d_size, s_size, s_size)`` — the retention table consumable by the simulator's SSP-MMC path.
    """

    def __init__(
        self,
        review_costs,
        first_rating_prob,
        review_rating_prob,
        w,
        *,
        s_max_days=MARC_S_MAX_DEFAULT,
        s_state=None,
        **kwargs,
    ):
        if s_state is None:
            s_state = build_marc_s_grid(s_max_days)
        # store_intervals=True so the base build keeps the per-action interval Δ; MARC integrates
        # the forgetting curve over it (cheap, closed-form) to get the retention-area reward.
        super().__init__(
            review_costs=review_costs,
            first_rating_prob=first_rating_prob,
            review_rating_prob=review_rating_prob,
            w=w,
            s_state=s_state,
            store_intervals=True,
            **kwargs,
        )
        self._build_area()

    # ── per-user build: retention area per (state, action) ──────────────────────────────────
    def _build_area(self):
        """Closed-form ``∫₀^Δ p_recall`` for every (state, action), reusing the intervals the base
        build already stored. Chunked over difficulty to bound the 4-D ``(d, sl, ss, r)`` temporary,
        mirroring ``_build_transitions``."""
        dev, dt, w_t = self.device, self.dtype, self._w_t
        s_g = self._s_state_t
        d_g = self._d_state_t
        s_size, r_size = self.s_size, self.r_size
        sl4 = s_g.view(1, -1, 1, 1)  # (1, s_long, 1, 1)
        ss4 = s_g.view(1, 1, -1, 1)  # (1, 1, s_short, 1)

        self._area_flat = torch.empty((self.n_states, r_size), device=dev, dtype=dt)
        interval = self._interval_flat.view(self.d_size, s_size, s_size, r_size)
        per_d = s_size * s_size * r_size
        chunk_d = max(1, 10_000_000 // per_d)
        with torch.inference_mode():
            for d0 in range(0, self.d_size, chunk_d):
                d1 = min(d0 + chunk_d, self.d_size)
                d4 = d_g[d0:d1].view(-1, 1, 1, 1)
                area = fsrs7.forgetting_curve_area(interval[d0:d1], sl4, ss4, d4, w_t)
                row0, row1 = d0 * s_size * s_size, d1 * s_size * s_size
                self._area_flat[row0:row1] = area.reshape(-1, r_size)
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ── reward = retention area − λ · expected review cost ──────────────────────────────────
    def _reward(self, lambda_cost):
        """``(n_states, r_size)`` reward: closed-form retention area minus ``λ`` times the REAL
        expected review time (no shaping hyperparameters — that proxy was SSP-MMC's whole problem).
        Expected cost = ``(1-r)·cost_again + r·Σ_g rating_prob_g·cost_g`` from the cached recall."""
        rp = self._r_pred_flat
        rc = self.review_costs  # (4,): Again, Hard, Good, Easy
        rrp = self.review_rating_prob  # (3,): Hard, Good, Easy (among successes)
        succ_cost = (
            float(rrp[0]) * float(rc[1])
            + float(rrp[1]) * float(rc[2])
            + float(rrp[2]) * float(rc[3])
        )
        exp_cost = (1.0 - rp) * float(rc[0]) + rp * succ_cost
        return self._area_flat - float(lambda_cost) * exp_cost

    def _branch_probs_max(self):
        rp = self._r_pred_flat
        rrp = self.review_rating_prob
        return [
            1.0 - rp,
            rp * float(rrp[0]),
            rp * float(rrp[1]),
            rp * float(rrp[2]),
        ]

    # ── maximizing value iteration (eager; no terminal) ─────────────────────────────────────
    def _run_vi_max(self, reward, n_iter, convergence_tol):
        """Plain value iteration maximizing ``V(s) = max_R [reward(s,R) + γ·Σ_g p_g·V(s'_g)]`` from
        ``V ≡ 0`` (γ < 1 ⇒ contraction ⇒ converges; no ``S_MAX`` terminal). ``γ = DISCOUNT_FACTOR``
        (0.97) is kept — its role is the effective horizon, NOT a convergence hack."""
        branch_probs = self._branch_probs_max()
        transitions = [t.long() for t in self._transitions]
        V = torch.zeros(self.n_states, device=self.device, dtype=self.dtype)
        it = 0
        diff = float("inf")
        while it < n_iter and diff > convergence_tol:
            it += 1
            action_value = reward.clone()
            for prob, trans in zip(branch_probs, transitions):
                action_value.addcmul_(prob, V[trans], value=DISCOUNT_FACTOR)
            v_new = action_value.amax(dim=-1)
            check_interval = 1 if it <= 15 else 3
            if it % check_interval == 0:
                diff = (v_new - V).abs().max().item()
            V = v_new
        return V, it, diff

    def _argmax_action(self, reward, V):
        """One extra backup at the converged ``V`` to recover the argmax target-retention index."""
        branch_probs = self._branch_probs_max()
        action_value = reward.clone()
        for prob, trans in zip(branch_probs, self._transitions):
            action_value.addcmul_(prob, V[trans.long()], value=DISCOUNT_FACTOR)
        return action_value.argmax(dim=-1)

    def solve(self, lambda_cost, n_iter=100_000, convergence_tol=1e-4, verbose=True):
        """Maximize the MARC objective at price-of-time ``lambda_cost``.

        Returns ``(value_matrix, retention_matrix)`` as numpy arrays of shape
        ``(d_size, s_size, s_size)`` indexed ``[d, s_long, s_short]``. The ``retention_matrix`` is
        the target-retention table; the simulator inverts the FSRS-7 curve to get the interval.

        NOTE: ``convergence_tol`` is on ``|ΔV|`` in card·day units (NOT SSP-MMC's cost units), so
        the right value depends on ``λ`` and grid; tune it alongside the build-time convergence
        study. The default is a placeholder for the reference implementation.
        """
        start = time.perf_counter()
        with torch.inference_mode():
            reward = self._reward(lambda_cost)
            V, it, diff = self._run_vi_max(reward, n_iter, convergence_tol)
            action = self._argmax_action(reward, V)
            value_flat = V.cpu().numpy()
            action_flat = action.cpu().numpy()
        if verbose:
            dt = time.perf_counter() - start
            print(
                f"MARC(FSRS-7) solve in {dt:.1f}s. iters {it}/{n_iter}, "
                f"|dV| {diff:.4g}, lambda={lambda_cost}"
            )
        value_matrix = value_flat.reshape(self.d_size, self.s_size, self.s_size)
        retention_matrix = self.r_state[action_flat].reshape(
            self.d_size, self.s_size, self.s_size
        )
        self.value_matrix = value_matrix
        self.retention_matrix = retention_matrix
        return value_matrix, retention_matrix
