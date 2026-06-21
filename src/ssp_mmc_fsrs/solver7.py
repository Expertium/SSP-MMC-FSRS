"""FSRS-7 Bellman / value-iteration solver (roadmap step 3).

Ports the FSRS-6 solver (`solver.py`, `SSPMMCSolver` + `bellman_solver`) to the FSRS-7
memory model. The state goes from 2-D `(D, S)` to **3-D `(D, S_long, S_short)`** and the
transition function uses the dual-stability FSRS-7 recurrence (`fsrs7.update_state`) with
intervals from the Brent inverter (`fsrs7.forgetting_curve_inverse`). Action = target
retention `R`; the optimal policy is the `R` minimizing long-run review cost per state.

Design (see memory `step3-bellman`):

* **Single shared S grid for BOTH S_long and S_short**: ``N_S`` (~120-150) points,
  log-spaced ``S_MIN(1e-4) -> S_MAX(36500)`` with a min increment of 1e-4 so the first
  ~12 points step linearly (1e-4, 2e-4, ...) before the log spacing takes over; the grid
  ends exactly at ``S_MAX``. (S_short was measured to span the SAME range as S_long, so one
  grid + one S_MAX is correct -- a smaller S_short max would clip real values.)
* **State** = ``(d_size, s_size, s_size)`` flattened to ``n_states`` for the value
  iteration; transitions are stored as FLAT next-state indices to keep VRAM modest.
* **Action** ``R in [0.60, 0.99]`` (FSRS-7's expanded DR range), ``r_eps=0.01``.
* **Cost hyperparameters (13: 11 free numeric + 2 categorical)** -- see `solve`.

**Build once, solve many:** the transitions / achieved-recall / branch probabilities
depend only on ``(w, grid, R)`` -- NOT on the cost hyperparameters. They are built once in
``__init__`` and kept on the device; ``solve(hyperparams)`` only recomputes the (cheap)
cost term and runs value iteration. This makes the "15 hyperparameter sets per user"
convergence sweep build the expensive structure once per user.

Everything is torch so it shares the `fsrs7` math and runs on the GPU. f32 by default
(value iteration doesn't need f64; the interior inverse round-trip is ~5e-5 in f32).
"""

import time

import numpy as np
import torch

from . import fsrs7

# Optional fused value-iteration kernel (GPU only). A custom Triton kernel computes, per
# state, min over actions of [const_cost + discount * sum_g prob_g * V[next_g]] entirely in
# registers -- it never materializes the (n_states, r_size) action-value array and reads the
# transition indices as int32, cutting per-iteration memory traffic ~3x vs the eager path.
try:
    import triton
    import triton.language as tl

    HAS_TRITON = torch.cuda.is_available()
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _bellman_step_kernel(
        state_ptr,
        const_cost_ptr,
        r_pred_ptr,
        t0_ptr,
        t1_ptr,
        t2_ptr,
        t3_ptr,
        new_state_ptr,
        n_states,
        r_size,
        rrp0,
        rrp1,
        rrp2,
        discount,
        R_BLOCK: tl.constexpr,
    ):
        """One Bellman backup for a single state (vectorized over the R_BLOCK actions).

        new_state[s] = min(state[s], min_r [ const_cost[s,r]
                          + discount * ( (1-rp)*V[t0] + rp*rrp0*V[t1]
                                       + rp*rrp1*V[t2] + rp*rrp2*V[t3] ) ])
        where rp = r_pred[s,r] and t0..t3 = transition[s,r] (int32 flat next-state indices).
        """
        s = tl.program_id(0)
        r = tl.arange(0, R_BLOCK)
        mask = r < r_size
        base = s * r_size + r
        cc = tl.load(const_cost_ptr + base, mask=mask, other=float("inf"))
        rp = tl.load(r_pred_ptr + base, mask=mask, other=0.0)
        i0 = tl.load(t0_ptr + base, mask=mask, other=0)
        i1 = tl.load(t1_ptr + base, mask=mask, other=0)
        i2 = tl.load(t2_ptr + base, mask=mask, other=0)
        i3 = tl.load(t3_ptr + base, mask=mask, other=0)
        v0 = tl.load(state_ptr + i0, mask=mask, other=0.0)
        v1 = tl.load(state_ptr + i1, mask=mask, other=0.0)
        v2 = tl.load(state_ptr + i2, mask=mask, other=0.0)
        v3 = tl.load(state_ptr + i3, mask=mask, other=0.0)
        prob0 = 1.0 - rp
        av = cc + discount * (
            prob0 * v0 + rp * rrp0 * v1 + rp * rrp1 * v2 + rp * rrp2 * v3
        )
        av = tl.where(mask, av, float("inf"))
        row_min = tl.min(av, axis=0)
        s_old = tl.load(state_ptr + s)
        tl.store(new_state_ptr + s, tl.minimum(s_old, row_min))


# ── Grid constants (FSRS-7) ──────────────────────────────────────────────────
S_MIN = 1e-4  # FSRS-7 --secs stability floor
S_MAX = (
    fsrs7.S_MAX
)  # 36500.0, the model clamp (grid now matches it; was 9125 for FSRS-6)
S_INCREMENT_MIN = 1e-4  # minimum spacing between consecutive S grid points
N_S = 135  # number of S grid points (tunable knob; 120-150 range)

D_MIN = 1.0
D_MAX = 10.0
D_EPS = 0.1  # difficulty grid spacing -> 91 points

R_MIN = 0.60  # FSRS-7 DR range expands to [0.60, 0.99]
R_MAX = 0.99
R_EPS = 0.01  # 40 candidate target-retention actions (full fidelity)

COST_MAX = 1_000_000.0
DISCOUNT_FACTOR = 0.97  # OFF-LIMITS as the convergence fix (per CLAUDE.md)


def build_s_grid(s_min=S_MIN, s_max=S_MAX, n=N_S, increment_min=S_INCREMENT_MIN):
    """Log-spaced S grid with a minimum absolute increment, ending exactly at ``s_max``.

    ``s[0]=s_min``; each subsequent point is ``max(exp(log(s_min)+i*log_step), prev+inc)``.
    For tiny S the ``prev+inc`` floor wins (linear 1e-4 steps); once the geometric step
    exceeds ``increment_min`` the log spacing takes over. The last point is pinned to
    ``s_max`` exactly so the top grid cell represents "fully learned".
    """
    log_step = (np.log(s_max) - np.log(s_min)) / (n - 1)
    s = np.empty(n, dtype=np.float64)
    s[0] = s_min
    for i in range(1, n):
        geo = np.exp(np.log(s_min) + i * log_step)
        s[i] = max(geo, s[i - 1] + increment_min)
    s[-1] = s_max  # pin the top exactly
    return s


class SSPMMCSolver7:
    """FSRS-7 SSP-MMC solver: 3-D state, dual-stability transitions, per-user inputs."""

    def __init__(
        self,
        review_costs,
        first_rating_prob,
        review_rating_prob,
        w,
        device=None,
        n_s=N_S,
        dtype=torch.float32,
        engine="auto",
    ):
        # Per-user inputs (CLAUDE.md: per-user FSRS params + per-button costs + H/G/E probs).
        self.review_costs = np.asarray(review_costs, dtype=np.float64)
        self.first_rating_prob = np.asarray(first_rating_prob, dtype=np.float64)
        self.review_rating_prob = np.asarray(review_rating_prob, dtype=np.float64)
        self.w = list(w)
        assert len(self.w) == 34, f"FSRS-7 needs 34 params, got {len(self.w)}"

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.n_s = n_s
        # Value-iteration engine: "auto" (Triton kernel on CUDA, else eager), or force
        # "triton"/"eager" -- used to cross-check the fused kernel against eager.
        assert engine in ("auto", "triton", "eager")
        self.engine = engine
        self._init_state_spaces()
        self._build_transitions()

    # ── grids ────────────────────────────────────────────────────────────────
    def _init_state_spaces(self):
        self.s_min, self.s_max = S_MIN, S_MAX
        self.s_state = build_s_grid(self.s_min, self.s_max, self.n_s)
        self.s_size = len(self.s_state)

        self.d_min, self.d_max, self.d_eps = D_MIN, D_MAX, D_EPS
        self.d_size = int(np.ceil((self.d_max - self.d_min) / self.d_eps + 1))
        self.d_state = np.linspace(self.d_min, self.d_max, self.d_size)

        self.r_min, self.r_max, self.r_eps = R_MIN, R_MAX, R_EPS
        self.r_size = int(np.ceil((self.r_max - self.r_min) / self.r_eps + 1))
        self.r_state = np.linspace(self.r_min, self.r_max, self.r_size)

        self.n_states = self.d_size * self.s_size * self.s_size

        self._s_state_t = torch.as_tensor(
            self.s_state, device=self.device, dtype=self.dtype
        )
        self._w_t = torch.as_tensor(self.w, device=self.device, dtype=self.dtype)

    # ── index maps ─────────────────────────────────────────────────────────────
    def s2i_torch(self, s):
        """Map continuous stability -> grid index (first grid point >= s), clamped."""
        idx = torch.searchsorted(self._s_state_t, s.contiguous())
        return idx.clamp_(0, self.s_size - 1)

    def d2i_torch(self, d):
        idx = torch.floor(
            (d - self.d_min) / (self.d_max - self.d_min) * self.d_size
        ).to(torch.long)
        return idx.clamp_(0, self.d_size - 1)

    def s2i(self, s):
        return np.clip(np.searchsorted(self.s_state, s), 0, self.s_size - 1)

    def d2i(self, d):
        idx = np.floor(
            (d - self.d_min) / (self.d_max - self.d_min) * self.d_size
        ).astype(int)
        return np.clip(idx, 0, self.d_size - 1)

    def _flat_index(self, d_idx, sl_idx, ss_idx):
        return (d_idx * self.s_size + sl_idx) * self.s_size + ss_idx

    # ── per-user build (hyperparameter-independent) ─────────────────────────────
    def _build_transitions(self):
        """Build the hyperparameter-independent pieces once: achieved recall ``r_pred`` and
        the 4 flat next-state index tables, kept on the device for reuse across many
        ``solve`` calls. Branch probabilities are NOT stored -- they are trivial functions
        of ``r_pred`` (fail = 1-r_pred, success_g = r_pred * review_rating_prob[g]) and are
        recomputed on the fly, which saves ~1 GB of VRAM.

        To bound VRAM, the 4-D ``(d, sl, ss, r)`` grid is processed in **chunks over
        difficulty** so the Brent inverter's temporaries never materialize for the whole
        grid at once. Transitions are stored as int32 (cast to int64 only inside the gather)
        to halve their footprint.
        """
        dev, dt, w_t = self.device, self.dtype, self._w_t
        s_g = self._s_state_t
        d_g = torch.as_tensor(self.d_state, device=dev, dtype=dt)
        # 3-D state grids (d, sl, ss) for the (state-only) cost modifiers.
        self._d3 = d_g.view(-1, 1, 1)
        self._sl3 = s_g.view(1, -1, 1)
        self._ss3 = s_g.view(1, 1, -1)

        n_states, r_size, s_size = self.n_states, self.r_size, self.s_size
        # Preallocated persistent outputs. int32 transitions: the fused Triton kernel reads
        # them directly (int32 pointer offsets), halving index bandwidth vs int64; the eager
        # fallback casts to int64 only where torch advanced indexing requires it.
        self._transitions = [
            torch.empty((n_states, r_size), device=dev, dtype=torch.int32)
            for _ in range(4)
        ]
        self._r_pred_flat = torch.empty((n_states, r_size), device=dev, dtype=dt)

        # Action grids that don't vary with the d-chunk.
        sl4 = self._sl3.unsqueeze(-1)  # (1, sl, 1, 1)
        ss4 = self._ss3.unsqueeze(-1)  # (1, 1, ss, 1)
        r4 = torch.as_tensor(self.r_state, device=dev, dtype=dt).view(1, 1, 1, -1)

        per_d = s_size * s_size * r_size
        chunk_d = max(1, 10_000_000 // per_d)  # ~10M-element 4-D chunks
        interior_max = 0.0
        with torch.inference_mode():
            for d0 in range(0, self.d_size, chunk_d):
                d1 = min(d0 + chunk_d, self.d_size)
                d4 = d_g[d0:d1].view(-1, 1, 1, 1)
                t = fsrs7.forgetting_curve_inverse(r4, sl4, ss4, d4, w_t)
                r_pred = fsrs7.forgetting_curve(t, sl4, ss4, d4, w_t)  # (cd, sl, ss, r)

                interior = (t > fsrs7.MIN_INTERVAL_DAYS * 1.001) & (
                    t < self.s_max * 0.999
                )
                if bool(interior.any()):
                    interior_max = max(
                        interior_max, float((r_pred - r4).abs()[interior].max().item())
                    )

                row0, row1 = d0 * s_size * s_size, d1 * s_size * s_size
                self._r_pred_flat[row0:row1] = r_pred.reshape(-1, r_size)
                for g in (1, 2, 3, 4):
                    rating = torch.tensor(float(g), device=dev, dtype=dt)
                    nsl, nss, nd = fsrs7.update_state(
                        t, rating, sl4, ss4, d4, w_t, self.s_min, self.s_max
                    )
                    flat = self._flat_index(
                        self.d2i_torch(nd), self.s2i_torch(nsl), self.s2i_torch(nss)
                    )
                    self._transitions[g - 1][row0:row1] = flat.reshape(-1, r_size).to(
                        torch.int32
                    )
                    del nsl, nss, nd, flat
                del t, r_pred
        self.inverse_check = interior_max

        # Initial cost-to-go: COST_MAX everywhere, 0 at the terminal (top S_long, all
        # difficulty and all S_short) -- a maximal-long-stability card is "done".
        state_init = torch.full((n_states,), COST_MAX, device=dev, dtype=dt)
        state_init = state_init.reshape(self.d_size, s_size, s_size)
        state_init[:, -1, :] = 0.0
        self._state_init = state_init.reshape(n_states)
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ── solve ──────────────────────────────────────────────────────────────────
    def _const_cost(self, hyperparams):
        """Build ``sum_k prob_k * cost_k`` (n_states, r_size) for the given cost
        hyperparameters, reusing the cached transitions/recall."""
        tf_sl = hyperparams["transform_s_long"]
        tf_ss = hyperparams["transform_s_short"]
        exp_sl = float(hyperparams["exp_s_long"])
        exp_ss = float(hyperparams["exp_s_short"])
        exp_d = float(hyperparams["exp_d"])
        base_fail = 1.0  # PINNED (kills global-scale redundancy)
        base_succ = float(hyperparams["base_succ"])
        wf_sl = float(hyperparams["w_fail_s_long"])
        wf_ss = float(hyperparams["w_fail_s_short"])
        wf_d = float(hyperparams["w_fail_d"])
        ws_sl = float(hyperparams["w_succ_s_long"])
        ws_ss = float(hyperparams["w_succ_s_short"])
        ws_d = float(hyperparams["w_succ_d"])
        w_ret = float(hyperparams.get("w_retention", 0.0))
        assert tf_sl in ("log", "no_log") and tf_ss in ("log", "no_log")
        assert exp_sl > 0 and exp_ss > 0 and exp_d > 0

        def s_ratio(s, transform):
            if transform == "log":
                return torch.log1p(s) / np.log1p(self.s_max)
            return s / self.s_max

        mod_sl = s_ratio(self._sl3, tf_sl).pow(exp_sl)
        mod_ss = s_ratio(self._ss3, tf_ss).pow(exp_ss)
        mod_d = (((self._d3 - 1.0) / 9.0).clamp(0.0, 1.0)).pow(exp_d)

        fail_state_cost = self.review_costs[0] * (
            base_fail + wf_sl * mod_sl + wf_ss * mod_ss + wf_d * mod_d
        )  # (d, sl, ss)
        fail_state_cost = fail_state_cost.reshape(self.n_states, 1)
        succ_state_cost = [
            (
                self.review_costs[g - 1]
                * (base_succ + ws_sl * mod_sl + ws_ss * mod_ss + ws_d * mod_d)
            ).reshape(self.n_states, 1)
            for g in (2, 3, 4)
        ]

        # Branch probabilities recomputed from r_pred (not stored): fail = 1 - r_pred,
        # success_g = r_pred * review_rating_prob[g].
        rp = self._r_pred_flat
        const_cost = (1.0 - rp) * fail_state_cost
        for g in (2, 3, 4):
            const_cost = const_cost + (rp * self.review_rating_prob[g - 2]) * (
                succ_state_cost[g - 2] - w_ret * rp
            )
        return const_cost

    def _branch_probs(self):
        """The 4 per-rating branch-probability arrays (fail, H, G, E) as (n_states, r_size)
        tensors, derived from the cached ``r_pred``. Precomputed once per solve so the value
        iteration doesn't recompute them every step."""
        rp = self._r_pred_flat
        rrp = self.review_rating_prob
        return [
            1.0 - rp,
            rp * float(rrp[0]),
            rp * float(rrp[1]),
            rp * float(rrp[2]),
        ]

    def _use_triton(self):
        if self.engine == "eager":
            return False
        if self.engine == "triton":
            assert HAS_TRITON and self.device == "cuda", "Triton engine unavailable"
            return True
        return HAS_TRITON and self.device == "cuda" and self.dtype == torch.float32

    def _run_iteration(self, const_cost, n_iter, convergence_tol):
        """Core value iteration. Returns ``(state, it, cost_diff)`` with ``state`` the flat
        cost-to-go (n_states,). Dispatches to the fused Triton kernel on CUDA, else eager."""
        if self._use_triton():
            return self._run_iteration_triton(const_cost, n_iter, convergence_tol)
        return self._run_iteration_eager(const_cost, n_iter, convergence_tol)

    def _run_iteration_triton(self, const_cost, n_iter, convergence_tol):
        """Fused value iteration via ``_bellman_step_kernel`` (no action-value array). Ping-
        pongs two ``state`` buffers (Jacobi backup: all reads from the previous iterate)."""
        rrp = self.review_rating_prob
        rrp0, rrp1, rrp2 = float(rrp[0]), float(rrp[1]), float(rrp[2])
        const_cost = const_cost.contiguous()
        t0, t1, t2, t3 = self._transitions
        state = self._state_init.clone()
        new_state = torch.empty_like(state)
        r_block = triton.next_power_of_2(self.r_size)
        grid = (self.n_states,)
        it = 0
        cost_diff = 1e9
        while it < n_iter and cost_diff > convergence_tol:
            it += 1
            _bellman_step_kernel[grid](
                state,
                const_cost,
                self._r_pred_flat,
                t0,
                t1,
                t2,
                t3,
                new_state,
                self.n_states,
                self.r_size,
                rrp0,
                rrp1,
                rrp2,
                DISCOUNT_FACTOR,
                R_BLOCK=r_block,
            )
            check_interval = 10 if it <= 100 else 25
            if it % check_interval == 0:
                cost_diff = (state - new_state).max().item()
            state, new_state = new_state, state
        return state, it, cost_diff

    def _run_iteration_eager(self, const_cost, n_iter, convergence_tol):
        """Eager (CPU / no-Triton) value iteration: ``addcmul_`` to fuse ``prob*gather``
        into the accumulate, ``amin`` for the action min. Casts int32 transitions to int64
        for torch advanced indexing."""
        branch_probs = self._branch_probs()
        transitions = [t.long() for t in self._transitions]
        state = self._state_init.clone()
        it = 0
        cost_diff = 1e9
        while it < n_iter and cost_diff > convergence_tol:
            it += 1
            action_value = const_cost.clone()
            for prob, trans in zip(branch_probs, transitions):
                action_value.addcmul_(prob, state[trans], value=DISCOUNT_FACTOR)
            optimal_value = torch.amin(action_value, dim=-1)
            check_interval = 10 if it <= 100 else 25
            if it % check_interval == 0:
                new_state = torch.minimum(state, optimal_value)
                cost_diff = (state - new_state).max().item()
                state = new_state
            else:
                torch.minimum(state, optimal_value, out=state)
        return state, it, cost_diff

    def _argmin_action(self, const_cost, state):
        """One extra eager backup at the converged ``state`` to recover the argmin action
        (the chosen target-retention index per state)."""
        branch_probs = self._branch_probs()
        action_value = const_cost.clone()
        for prob, trans in zip(branch_probs, self._transitions):
            action_value.addcmul_(prob, state[trans.long()], value=DISCOUNT_FACTOR)
        return action_value.argmin(dim=-1)

    def measure_convergence(self, hyperparams, n_iter=3000, convergence_tol=0.1):
        """Run value iteration and return ``(converged, frac_at_max, iters)`` WITHOUT
        building the retention matrix -- the cheap path for the convergence sweep. A user's
        set "converges" iff fewer than 1/20 of states stay pinned at the max cost (the
        FSRS-6 ``converge.py`` criterion)."""
        with torch.inference_mode():
            const_cost = self._const_cost(hyperparams)
            state, it, _ = self._run_iteration(const_cost, n_iter, convergence_tol)
            actual_max = state.max()
            frac = float((state == actual_max).sum().item()) / state.numel()
        return frac < (1.0 / 20.0), frac, it

    def solve(self, hyperparams, n_iter=100_000, convergence_tol=0.1, verbose=True):
        """Run value iteration; return ``(cost_matrix, retention_matrix)`` as numpy arrays
        of shape ``(d_size, s_size, s_size)`` indexed ``[d, s_long, s_short]``.

        Cost hyperparameters (FINAL design): two categoricals ``transform_s_long`` /
        ``transform_s_short`` in {"log","no_log"}; exponents ``exp_s_long`` /
        ``exp_s_short`` / ``exp_d`` (>0); ``base_fail`` PINNED 1.0; ``base_succ`` (free);
        failure weights ``w_fail_{s_long,s_short,d}``; success weights
        ``w_succ_{s_long,s_short,d}``; ``w_retention``.

        ``failure_cost = review_costs[0] * (1.0 + Sum w_fail_x * mod_x)``;
        ``success_cost = review_cost  * (base_succ + Sum w_succ_x * mod_x) - w_retention*R``
        with ``mod_x = ratio_x ** exp_x``, ratios ``log1p(S)/log1p(S_MAX)`` (or ``S/S_MAX``)
        per the S transforms and ``(D-1)/9`` for difficulty.
        """
        start = time.perf_counter()
        with torch.inference_mode():
            const_cost = self._const_cost(hyperparams)
            state, it, cost_diff = self._run_iteration(
                const_cost, n_iter, convergence_tol
            )
            optimal_action = self._argmin_action(const_cost, state)
            cost_flat = state.cpu().numpy()
            action_flat = optimal_action.cpu().numpy()
        if verbose:
            dt = time.perf_counter() - start
            print(
                f"Bellman(FSRS-7) done in {dt:.1f}s. "
                f"Iterations: {it}/{n_iter}, cost diff: {cost_diff:.4g}"
            )

        cost_matrix = cost_flat.reshape(self.d_size, self.s_size, self.s_size)
        # Target (desired) retention chosen per state -- the policy's decision. The
        # interval for the simulator is then fsrs7.forgetting_curve_inverse(target_R, ...).
        retention_matrix = self.r_state[action_flat].reshape(
            self.d_size, self.s_size, self.s_size
        )
        self.cost_matrix = cost_matrix
        self.retention_matrix = retention_matrix
        return cost_matrix, retention_matrix
