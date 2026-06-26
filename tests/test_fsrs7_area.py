"""Parity test: fsrs7.forgetting_curve_area (closed form) must match numerical quadrature of
fsrs7.forgetting_curve.

Run with:  uv run --no-sync python tests/test_fsrs7_area.py

``forgetting_curve_area(t, ...)`` claims to be the exact definite integral
``∫₀^t forgetting_curve(τ, ...) dτ``. We check it against a fine midpoint-rule quadrature of the
curve over [0, t], for random realistic states + intervals and a few real FSRS-7 param sets, in
float64. The closed form and the quadrature should agree to ~quadrature error (a few 1e-5 rel).

This is the definition-of-done for the MARC reward's retention term (the integral that prices a
card's knowledge contribution; see ssp_mmc_fsrs.marc, ssp_mmc_fsrs.fsrs7.forgetting_curve_area).
"""

import json
import os
import sys

import numpy as np
import torch

from ssp_mmc_fsrs import fsrs7  # noqa: E402

torch.set_num_threads(1)

SRS_BENCH = r"C:\Users\Andrew\srs-benchmark"
FSRS7_PARAMS = os.path.join(SRS_BENCH, "result", "FSRS-7-short-secs-recency.jsonl")

# Default FSRS-7 params (from tests/test_solver7_smoke.py) so the test runs without the dataset.
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
S_MIN = 1e-4
S_MAX = float(fsrs7.S_MAX)


def real_param_sets(n):
    out = []
    if not os.path.exists(FSRS7_PARAMS):
        return out
    with open(FSRS7_PARAMS) as f:
        for line in f:
            rec = json.loads(line)
            params = rec.get("parameters", {}).get("0")
            if params and len(params) == 34:
                out.append((rec.get("user"), [float(x) for x in params]))
            if len(out) >= n:
                break
    return out


def random_states(batch, rng):
    """Random realistic (s_long, s_short, d, interval). s_short <= s_long (the model keeps it
    smaller); intervals log-uniform from ~1 min to ~2000 days to span sub-day .. multi-year."""
    s_long = np.exp(rng.uniform(np.log(0.01), np.log(S_MAX), size=batch))
    s_short = s_long * rng.uniform(0.2, 1.0, size=batch)
    s_short = np.clip(s_short, S_MIN, S_MAX)
    d = rng.uniform(1.0, 10.0, size=batch)
    t = np.exp(rng.uniform(np.log(1.0 / 1440.0), np.log(2000.0), size=batch))
    to_t = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
    return to_t(t), to_t(s_long), to_t(s_short), to_t(d)


def quad_area(t, s, s_short, d, w, n=8000):
    """Midpoint-rule ∫₀^t forgetting_curve dτ, vectorized over the sample batch."""
    u = (torch.arange(n, dtype=t.dtype) + 0.5) / n  # (n,)
    taus = t.unsqueeze(0) * u.unsqueeze(1)  # (n, batch)
    fc = fsrs7.forgetting_curve(
        taus, s.unsqueeze(0), s_short.unsqueeze(0), d.unsqueeze(0), w
    )
    return (t / n) * fc.sum(dim=0)


def run_one(w_list, label, rng):
    w = torch.tensor(w_list, dtype=torch.float64)
    t, s_long, s_short, d = random_states(512, rng)
    closed = fsrs7.forgetting_curve_area(t, s_long, s_short, d, w)
    quad = quad_area(t, s_long, s_short, d, w)
    a = closed.double().numpy()
    b = quad.double().numpy()
    denom = np.maximum(np.abs(b), 1e-9)
    max_abs = float(np.max(np.abs(a - b)))
    max_rel = float(np.max(np.abs(a - b) / denom))
    good = max_rel < 2e-3  # midpoint quadrature error floor at n=8000
    print(
        f"    {label:14s} {'OK ' if good else 'DIFF'} max_abs={max_abs:.2e} max_rel={max_rel:.2e}"
    )
    return good


def main():
    rng = np.random.default_rng(0)
    all_ok = True
    all_ok &= run_one(DEFAULT_W, "default", rng)
    for user, params in real_param_sets(4):
        all_ok &= run_one(params, f"user{user}", rng)
    print()
    if all_ok:
        print("PASS: forgetting_curve_area matches quadrature of forgetting_curve.")
        return 0
    print("FAIL: see diffs above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
