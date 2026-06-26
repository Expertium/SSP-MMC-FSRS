"""Cost ADR — closed-form Adaptive Desired Retention policy (roadmap step 5.5).

A faithful port of JSchoreels' **Cost ADR** (FSRS-7, Rust):
https://github.com/JSchoreels/fsrs-rs/tree/feature/fsrs-7-adr-optimization
(``src/cost_adr.rs`` :: ``CostAdrPolicyEvaluator::evaluate_retention``, ``docs/COST_ADR.MD``).

**What it is.** A single smooth function ``DR(stability, difficulty; cost_weight)`` with **15
coefficients**: it reads (long-term) stability + difficulty and outputs a desired retention,
with one knob ``cost_weight`` (the price-of-time / Lagrange λ) that slides the whole policy
along the workload↔knowledge tradeoff. ``cost_weight=0 → high DR (~0.90)``;
``cost_weight=1024 → low DR (~0.50)``. Fixed DR is the special case where all the
stability/difficulty coefficients are zero (then ``DR`` is constant).

**How it's meant to be used (step 5.5).** Unlike SSP-MMC (a per-card Bellman solve) and MARC
(a re-objectived Bellman), ADR has NO solver: you fit the 15 coefficients ONCE by an
evolutionary search directly on the simulator's knowledge-vs-time hypervolume (the true
objective), then evaluate the closed form per card. Here we provide only the **closed-form
evaluation** + a **retention-table filler**; the evolutionary fitter is a separate step.

**Integration (no Rust changes).** ``retention_table(...)`` evaluates ``DR`` over the same
``(difficulty, s_long, s_short)`` grid SSP-MMC's Bellman fills, so the resulting table flows
through the existing ``policy="ssp_mmc"`` path of the Rust simulator unchanged (that path just
looks a target retention up in the table and inverts the FSRS-7 curve). ADR's DR depends on
``(stability, difficulty)`` only — NOT on ``s_short`` separately — so the table is constant
along the ``s_short`` axis (``s_short`` mainly drives same-day dynamics, not the inter-day DR
decision; this matches the reference, whose policy inputs are long-term stability + difficulty).

NumPy throughout: the grid is tiny (``d_size × s_size`` ≈ a few thousand points), so there is
no reason to involve the GPU. See memory ``adr-cost-adr`` and CLAUDE.md step 5.5.
"""

import numpy as np

from . import fsrs7

# Reference defaults (cost_adr.rs :: CostAdrPolicy::new -> new_with_settings(coef, 0.0, 1024.0,
# 0.30, 0.995, None)). cost_weight sweeps [0, 1024]; DR is bounded to [0.30, 0.995].
COST_WEIGHT_MIN = 0.0
COST_WEIGHT_MAX = 1024.0
RETENTION_MIN = 0.30
RETENTION_MAX = 0.995

# Stability/difficulty normalization bounds. The reference normalizes with its simulation's
# S_MIN/S_MAX/D_MIN/D_MAX; we default to OUR FSRS-7 bounds so the features span our grid. The
# FSRS-7 stability floor is 1e-4 (the --secs S_MIN), the model clamp is fsrs7.S_MAX (36500 d).
S_MIN_NORM = 1e-4
S_MAX_NORM = float(fsrs7.S_MAX)
D_MIN_NORM = float(fsrs7.D_MIN)
D_MAX_NORM = float(fsrs7.D_MAX)

# JSchoreels' default/initial coefficients (cost_adr.rs). They were fit under the reference's
# OWN normalization bounds, so they are NOT a finished policy for our pipeline — use them only
# as an initialization seed for the evolutionary fit on our simulator. After fitting on our
# bounds, the coefficients adapt. Order: [base(0..5), z(5..10), z2(10..15)].
DEFAULT_ADR_COEF = np.array(
    [
        -0.202,
        9.14,
        -0.0978,
        0.226,
        -5.31,  # base   (multiplies phi)
        -7.44,
        24.1,
        -0.375,
        1.81,
        -22.9,  # z      (multiplies -softplus(.)·z)
        -5.82,
        22.3,
        1.72,
        -1.99,
        -19.4,  # z²     (multiplies -softplus(.)·z²)
    ],
    dtype=np.float64,
)

N_COEF = 15


def _softplus(x):
    """Numerically stable softplus, mirroring cost_adr.rs::softplus. (np.where evaluates both
    branches, so every exp argument is clamped to avoid spurious overflow warnings.)"""
    x = np.asarray(x, dtype=np.float64)
    small = np.exp(np.minimum(x, 0.0))  # exact where x < -20 (the only place it's used)
    mid = np.log1p(np.exp(np.minimum(x, 20.0)))  # exact where |x| <= 20
    return np.where(x > 20.0, x, np.where(x < -20.0, small, mid))


def _sigmoid(x):
    """Numerically stable logistic sigmoid (no overflow: exp only ever sees a <= 0 argument)."""
    x = np.asarray(x, dtype=np.float64)
    z = np.exp(-np.abs(x))
    return np.where(x >= 0.0, 1.0 / (1.0 + z), z / (1.0 + z))


def normalized_cost_weight(cost_weight, cw_min=COST_WEIGHT_MIN, cw_max=COST_WEIGHT_MAX):
    """z = normalized ln(1+cost_weight) in [0, 1] (cost_adr.rs::normalized_cost_weight)."""
    w = np.clip(cost_weight, cw_min, cw_max)
    lo = np.log1p(cw_min)
    hi = np.log1p(cw_max)
    return float(np.clip((np.log1p(w) - lo) / (hi - lo), 0.0, 1.0))


def desired_retention(
    stability,
    difficulty,
    cost_weight,
    coef=DEFAULT_ADR_COEF,
    *,
    s_min=S_MIN_NORM,
    s_max=S_MAX_NORM,
    d_min=D_MIN_NORM,
    d_max=D_MAX_NORM,
    retention_min=RETENTION_MIN,
    retention_max=RETENTION_MAX,
    cw_min=COST_WEIGHT_MIN,
    cw_max=COST_WEIGHT_MAX,
):
    """Cost ADR desired retention for (broadcastable) ``stability``/``difficulty`` arrays.

    Verbatim port of ``CostAdrPolicyEvaluator::evaluate_retention``::

        x_s = clamp((ln(stability) - ln(s_min)) / (ln(s_max) - ln(s_min)), 0, 1)
        x_d = clamp((difficulty - d_min) / (d_max - d_min), 0, 1)
        phi = [1, x_s, x_d, x_s·x_d, x_s²]
        base     = coef[0:5]  · phi
        z_eff    = softplus(coef[5:10]  · phi) · z
        z2_eff   = softplus(coef[10:15] · phi) · z²
        DR = retention_min + (retention_max - retention_min)·sigmoid(base - z_eff - z2_eff)

    with ``z = normalized_cost_weight(cost_weight)`` and ``z² = z·z``. ``cost_weight`` is a
    scalar (one policy per price-of-time); ``stability``/``difficulty`` may be arrays.
    """
    coef = np.asarray(coef, dtype=np.float64)
    if coef.shape != (N_COEF,):
        raise ValueError(f"ADR needs {N_COEF} coefficients, got {coef.shape}")
    stability = np.asarray(stability, dtype=np.float64)
    difficulty = np.asarray(difficulty, dtype=np.float64)

    log_s_min = np.log(s_min)
    log_s_span = np.log(s_max) - log_s_min
    x_s = np.clip((np.log(stability) - log_s_min) / log_s_span, 0.0, 1.0)
    x_d = np.clip((difficulty - d_min) / (d_max - d_min), 0.0, 1.0)

    z = normalized_cost_weight(cost_weight, cw_min, cw_max)
    z2 = z * z

    ones = np.ones_like(x_s + x_d)  # broadcast to the common shape
    phi = np.stack(
        [ones * 1.0, x_s + 0.0 * ones, x_d + 0.0 * ones, x_s * x_d, x_s * x_s]
    )

    base = np.tensordot(coef[0:5], phi, axes=(0, 0))
    z_eff = _softplus(np.tensordot(coef[5:10], phi, axes=(0, 0))) * z
    z2_eff = _softplus(np.tensordot(coef[10:15], phi, axes=(0, 0))) * z2

    span = retention_max - retention_min
    return retention_min + span * _sigmoid(base - z_eff - z2_eff)


def retention_table(
    coef,
    cost_weight,
    s_grid,
    d_grid,
    *,
    s_min=None,
    s_max=None,
    d_min=D_MIN_NORM,
    d_max=D_MAX_NORM,
    retention_min=RETENTION_MIN,
    retention_max=RETENTION_MAX,
    cw_min=COST_WEIGHT_MIN,
    cw_max=COST_WEIGHT_MAX,
):
    """Fill a ``(d_size, s_long_size, s_short_size)`` ADR retention table on the SSP-MMC grid.

    DR is evaluated at ``(s_long, difficulty)`` per ``cost_weight`` and broadcast (constant)
    across the ``s_short`` axis, so the table has the exact shape/order the Rust simulator's
    ``policy="ssp_mmc"`` path expects (flattened ``[d, s_long, s_short]``). ``s_min``/``s_max``
    default to the grid endpoints (so normalization spans the grid actually used).

    Returns a contiguous float64 array ready to hand to ``simulate_fsrs7(..., "ssp_mmc", 0.0,
    ..., retention_table, s_grid, d_grid)``.
    """
    s_grid = np.asarray(s_grid, dtype=np.float64)
    d_grid = np.asarray(d_grid, dtype=np.float64)
    s_min = float(s_grid[0]) if s_min is None else float(s_min)
    s_max = float(s_grid[-1]) if s_max is None else float(s_max)

    # DR over the (difficulty, s_long) plane.
    dd, sl = np.meshgrid(d_grid, s_grid, indexing="ij")  # (d_size, s_long_size)
    dr = desired_retention(
        sl,
        dd,
        cost_weight,
        coef,
        s_min=s_min,
        s_max=s_max,
        d_min=d_min,
        d_max=d_max,
        retention_min=retention_min,
        retention_max=retention_max,
        cw_min=cw_min,
        cw_max=cw_max,
    )  # (d_size, s_long_size)

    # Broadcast over s_short -> (d_size, s_long_size, s_short_size).
    table = np.repeat(dr[:, :, None], len(s_grid), axis=2)
    return np.ascontiguousarray(table, dtype=np.float64)
