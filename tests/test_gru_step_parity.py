"""Parity: BatchedGRU per-step stepping == srs-benchmark GRU full-sequence forward.

Run with srs-benchmark's venv (it has the reference GRU + fsrs_optimizer):

    C:/Users/Andrew/srs-benchmark/.venv/Scripts/python.exe tests/test_gru_step_parity.py

(Our own venv can't import the reference model, so this test is cross-environment. We load
``BatchedGRU`` directly from its source file -- it only needs ``torch`` -- so the package's
heavier ``__init__`` deps are not required.)

The simulator carries the GRU hidden state ``h`` and reads p(recall) from it one review at a
time (Markovian), instead of re-running the whole history. This test proves that is *exact*:
for a batch of random review sequences, the per-step ``(w, s, d)`` and the resulting
retentions from ``ssp_mmc_fsrs.gru.BatchedGRU`` match the reference ``GRU.forward`` (run over
the full sequence) to floating-point precision.

Both sides run in float64 (the reference model is cast with ``.double()``) so the only
difference is our explicit GRUCell vs ``nn.GRU`` -- which agree to ~1e-12. The bar is 1e-6.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

SRSB = Path(r"C:/Users/Andrew/srs-benchmark")
GRU_SRC = Path(__file__).resolve().parents[1] / "src" / "ssp_mmc_fsrs" / "gru.py"
TOL = 1e-6


def _load_batched_gru():
    """Load BatchedGRU from its source file without triggering the package __init__."""
    spec = importlib.util.spec_from_file_location("ssp_mmc_fsrs_gru", GRU_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BatchedGRU


def _load_reference_gru():
    """Construct srs-benchmark's GRU (--short --secs) with its global pretrained weights."""
    sys.path.insert(0, str(SRSB))
    os.chdir(SRSB)  # so GRU.__init__ finds ./pretrain/GRU-short-secs_pretrain.pth
    from config import Config, create_parser  # type: ignore
    from models import GRU  # type: ignore

    parser = create_parser()
    args = parser.parse_args(["--algo", "GRU", "--short", "--secs"])
    config = Config(args)
    config.device = torch.device("cpu")
    model = GRU(config).to("cpu").double().eval()
    return model


def _random_sequence(T, N, seed=0):
    """A (T, N, 2) sequence of (delta_t fractional-days, rating 1..4), float64."""
    rng = np.random.default_rng(seed)
    # Mix sub-day and multi-day intervals to exercise the full curve range.
    log_dt = rng.uniform(np.log(1e-3), np.log(400.0), size=(T, N))
    dt = np.exp(log_dt)
    rating = rng.integers(1, 5, size=(T, N)).astype(np.float64)
    seq = np.stack([dt, rating], axis=-1)
    return torch.tensor(seq, dtype=torch.float64)


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return cond


@torch.no_grad()
def main():
    T, N = 24, 96
    BatchedGRU = _load_batched_gru()
    model = _load_reference_gru()
    seq = _random_sequence(T, N, seed=1)  # (T, N, 2)
    q = torch.tensor(
        np.exp(np.random.default_rng(2).uniform(np.log(1e-3), np.log(400.0), (T, N))),
        dtype=torch.float64,
    )  # independent query times for the forgetting-curve comparison

    # --- reference: full-sequence forward, then heads at every step ---
    w_ref, s_ref, d_ref = model(seq)  # each (T, N, 2)
    ret_ref = torch.stack(
        [
            model.forgetting_curve(q[t].unsqueeze(-1), w_ref[t], s_ref[t], d_ref[t])
            for t in range(T)
        ]
    )  # (T, N)

    # --- ours: BatchedGRU (P=1) stepped one review at a time ---
    gru = BatchedGRU.from_state_dicts(
        [model.state_dict()], device="cpu", dtype=torch.float64
    )
    h = gru.init_hidden(N)  # (1, N, H)
    w_m, s_m, d_m, ret_m = [], [], [], []
    for t in range(T):
        dt_t = seq[t, :, 0].unsqueeze(0)  # (1, N)
        rt_t = seq[t, :, 1].unsqueeze(0)  # (1, N)
        h = gru.step(h, dt_t, rt_t)
        wc, sc, dc = gru.curve_params(h)  # each (1, N, 2)
        w_m.append(wc[0])
        s_m.append(sc[0])
        d_m.append(dc[0])
        ret_m.append(gru.forgetting_curve(q[t].unsqueeze(0), wc, sc, dc)[0])
    w_m = torch.stack(w_m)
    s_m = torch.stack(s_m)
    d_m = torch.stack(d_m)
    ret_m = torch.stack(ret_m)

    dw = (w_m - w_ref).abs().max().item()
    ds = (s_m - s_ref).abs().max().item()
    dd = (d_m - d_ref).abs().max().item()
    dret = (ret_m - ret_ref).abs().max().item()

    ok = True
    ok &= check("w params", dw < TOL, f"max|d|={dw:.2e}")
    ok &= check("s params", ds < TOL, f"max|d|={ds:.2e}")
    ok &= check("d params", dd < TOL, f"max|d|={dd:.2e}")
    ok &= check("retention", dret < TOL, f"max|d|={dret:.2e}")
    print()
    if ok:
        print("PASS: BatchedGRU stepping matches the reference forward.")
        return 0
    print("FAIL: see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
