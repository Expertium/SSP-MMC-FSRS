"""Parity test: ssp_mmc_fsrs.fsrs7 must match the srs-benchmark FSRS-7 reference.

Run with:  uv run --no-sync python tests/test_fsrs7_parity.py

Definition of done for roadmap step 2a (port the FSRS-7 memory model into Python). We feed
identical (delta_t, rating) sequences through:
  - the live reference ``models.fsrs_v7.FSRS7`` from C:\\Users\\Andrew\\srs-benchmark, and
  - our ``ssp_mmc_fsrs.fsrs7.step`` iterated the same way,
and require the full 3-state trajectory [s_long, s_short, d] to agree. Both are f32 torch
running the same ops, so we expect bit-identical (tolerance is a tiny safety margin).

Importing the reference: ``models/__init__.py`` eagerly imports the whole model zoo (needs
sklearn, the Rust backend, ...). We sidestep it by pre-registering a lightweight ``models``
package pointing at the real directory, so ``import models.fsrs_v7`` pulls in only the FSRS
inheritance chain (which just needs pandas, a dev dep).
"""

import os
import sys
import types
import json

import numpy as np
import torch

torch.set_num_threads(1)

SRS_BENCH = r"C:\Users\Andrew\srs-benchmark"
FSRS7_PARAMS = os.path.join(SRS_BENCH, "result", "FSRS-7-short-secs-recency.jsonl")

# Our port.
from ssp_mmc_fsrs import fsrs7  # noqa: E402


def load_reference():
    """Import the real FSRS7 + its Config without running models/__init__.py."""
    if SRS_BENCH not in sys.path:
        sys.path.insert(0, SRS_BENCH)
    if "models" not in sys.modules:
        pkg = types.ModuleType("models")
        pkg.__path__ = [os.path.join(SRS_BENCH, "models")]
        sys.modules["models"] = pkg
    # importlib so the bare `config`/`models.*` imports inside the reference resolve.
    from config import load_config  # noqa
    from models.fsrs_v7 import FSRS7  # noqa

    cfg = load_config(custom_args_list=["--algo", "FSRS-7", "--short", "--secs"])
    return FSRS7, cfg


def real_param_sets(n):
    """First ``n`` per-user 34-param sets from FSRS-7-short-secs-recency.jsonl."""
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


def make_sequences(batch, seq_len, rng):
    """Random (delta_t, rating) sequences [seq_len, batch, 2], f32.

    delta_t mixes sub-day (same-day, <1) and multi-day gaps to exercise the whole curve;
    the first step's delta_t is irrelevant (first-review init path ignores it)."""
    # delta_t: log-uniform over ~[0.002, 400] days (≈3 min .. ~1 year).
    dt = np.exp(rng.uniform(np.log(0.002), np.log(400.0), size=(seq_len, batch)))
    dt[0] = 0.0
    rating = rng.integers(1, 5, size=(seq_len, batch)).astype(np.float32)
    seqs = np.stack([dt.astype(np.float32), rating], axis=-1)
    return torch.from_numpy(seqs)


def reference_trajectory(model, seqs):
    outputs, _ = model.forward(seqs)  # [seq_len, batch, 3] = [s_long, s_short, d]
    return outputs


def my_trajectory(seqs, w, s_min):
    seq_len, batch, _ = seqs.shape
    s_long = torch.zeros(batch)
    s_short = torch.zeros(batch)
    d = torch.zeros(batch)
    outs = []
    for t in range(seq_len):
        dt = seqs[t, :, 0]
        rating = seqs[t, :, 1]
        s_long, s_short, d = fsrs7.step(dt, rating, s_long, s_short, d, w, s_min)
        outs.append(torch.stack([s_long, s_short, d], dim=1))
    return torch.stack(outs)


def compare(ref, mine, label):
    ref = ref.detach().double().numpy()
    mine = mine.detach().double().numpy()
    names = ["s_long", "s_short", "d"]
    ok = True
    for i, nm in enumerate(names):
        a, b = ref[..., i], mine[..., i]
        denom = np.maximum(np.abs(a), 1e-9)
        max_abs = float(np.max(np.abs(a - b)))
        max_rel = float(np.max(np.abs(a - b) / denom))
        good = np.allclose(a, b, rtol=1e-4, atol=1e-5)
        ok = ok and good
        bit = (
            "bit-exact"
            if max_abs == 0.0
            else f"max_abs={max_abs:.2e} max_rel={max_rel:.2e}"
        )
        print(f"    {label:18s} {nm:8s} {'OK ' if good else 'DIFF'} {bit}")
    return ok


def run_one(FSRS7, cfg, w_list, label, rng):
    model = FSRS7(cfg, w=w_list)
    model.clipper(model)  # clip to valid params, exactly as training does
    w = model.w.data.clone()
    all_ok = True
    for batch, seq_len in [(64, 24), (128, 40)]:
        seqs = make_sequences(batch, seq_len, rng)
        ref = reference_trajectory(model, seqs)
        mine = my_trajectory(seqs, w, cfg.s_min)
        all_ok &= compare(ref, mine, f"{label} B{batch}xL{seq_len}")
    return all_ok


def main():
    FSRS7, cfg = load_reference()
    print(f"reference loaded: FSRS-7 s_min={cfg.s_min} device={cfg.device}")
    rng = np.random.default_rng(0)

    all_ok = True

    # 1) Default parameters.
    all_ok &= run_one(FSRS7, cfg, FSRS7.init_w, "default", rng)

    # 2) A few real per-user param sets (what we actually simulate with).
    for user, params in real_param_sets(3):
        all_ok &= run_one(FSRS7, cfg, params, f"user{user}", rng)

    # 3) Random params (clipped to valid box) to stress the whole range.
    lo = np.array(fsrs7_clip_lo(), dtype=np.float64)
    hi = np.array(fsrs7_clip_hi(), dtype=np.float64)
    for k in range(3):
        rand = (lo + rng.random(34) * (hi - lo)).tolist()
        all_ok &= run_one(FSRS7, cfg, rand, f"rand{k}", rng)

    print()
    if all_ok:
        print("PASS: ssp_mmc_fsrs.fsrs7 matches the srs-benchmark FSRS-7 reference.")
        return 0
    print("FAIL: see diffs above.")
    return 1


def fsrs7_clip_lo():
    # mirror FSRS7ParameterClipper._CLIP_LO (only used to draw valid random params)
    return [
        0.0001,
        0.0001,
        0.0001,
        0.0001,
        1.0,
        0.001,
        0.1,
        0.0,
        0.0,
        0.3,
        0.01,
        0.1,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.5,
        0.001,
        0.001,
        0.0,
        0.0,
        1.0,
        0.01,
        0.01,
        0.2,
        0.5,
        0.01,
        0.1,
        0.0,
        0.1,
        0.0,
        0.0,
        0.0,
    ]


def fsrs7_clip_hi():
    return [
        50.0,
        100.0,
        100.0,
        100.0,
        10.0,
        4.0,
        4.0,
        4.0,
        1.2,
        3.0,
        1.5,
        1.0,
        3.5,
        1.0,
        7.0,
        4.0,
        2.0,
        6.0,
        1.5,
        1.0,
        5.0,
        1.0,
        7.0,
        0.25,
        0.95,
        0.85,
        0.99,
        1.0,
        1.0,
        0.9,
        1.1,
        1.0,
        0.6,
        0.6,
    ]


if __name__ == "__main__":
    sys.exit(main())
