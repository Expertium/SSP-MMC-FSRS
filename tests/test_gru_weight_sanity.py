"""Saved per-user GRU weights reproduce the reference forward on REAL user data.

Run with srs-benchmark's venv (needs the reference GRU + features + data):

    C:/Users/Andrew/srs-benchmark/.venv/Scripts/python.exe tests/test_gru_weight_sanity.py

This is the belt-and-suspenders check for the weight round-trip: it loads a finetuned
``user_{id}.pth`` into BOTH the reference ``GRU`` and our ``BatchedGRU``, then -- on that
user's actual review histories (the per-review cumulative ``tensor`` of (delta_t, rating)) --
compares the predicted retentions. They must agree to < 1e-6. This is what catches
state_dict key/shape/normalization-buffer mistakes that the random-sequence step-parity test
might miss (e.g. the 0-d input_mean buffer).
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SRSB = Path(r"C:/Users/Andrew/srs-benchmark")
GRU_SRC = REPO / "src" / "ssp_mmc_fsrs" / "gru.py"
WDIR = REPO / "outputs" / "gru_weights" / "GRU-short-secs"
USER_ID = 1
SAMPLE = 300  # number of (card, review) items to compare
TOL = 1e-6


def _load_batched_gru():
    spec = importlib.util.spec_from_file_location("ssp_mmc_fsrs_gru", GRU_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BatchedGRU


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return cond


@torch.no_grad()
def main():
    pth = WDIR / f"user_{USER_ID}.pth"
    if not pth.exists():
        print(f"SKIP: {pth} not found; run experiments/train_gru_per_user.py first.")
        return 0

    BatchedGRU = _load_batched_gru()
    sys.path.insert(0, str(SRSB))
    os.chdir(SRSB)
    from config import Config, create_parser  # type: ignore
    from features import create_features  # type: ignore
    from models import GRU  # type: ignore
    import pandas as pd

    parser = create_parser()
    args = parser.parse_args(["--algo", "GRU", "--short", "--secs"])
    config = Config(args)
    config.device = torch.device("cpu")

    state = torch.load(pth, weights_only=True, map_location="cpu")
    model = GRU(config)
    model.load_state_dict(state)
    model = model.double().eval()

    # Real user data: each row's `tensor` is the cumulative (delta_t, rating) history of
    # prior reviews; `delta_t` is the current review's elapsed interval (the query).
    data = Path(r"C:/Users/Andrew/anki-revlogs-10k")
    df = create_features(
        pd.read_parquet(data / "revlogs" / f"{USER_ID=}"), config=config
    )
    df = df[df["tensor"].map(len) > 0].reset_index(drop=True)
    n = min(SAMPLE, len(df))
    rows = df.iloc[np.linspace(0, len(df) - 1, n).astype(int)]

    gru = BatchedGRU.from_state_dicts([state], device="cpu", dtype=torch.float64)

    diffs = []
    for _, row in rows.iterrows():
        seq = torch.as_tensor(np.asarray(row["tensor"]), dtype=torch.float64)  # (L, 2)
        L = seq.shape[0]
        delta_n = torch.tensor([float(row["delta_t"])], dtype=torch.float64)  # (1,)

        # reference: forward over the full history, read the last step, eval the curve
        w_ref, s_ref, d_ref = model(seq.unsqueeze(1))  # (L, 1, 2) each
        ret_ref = model.forgetting_curve(
            delta_n.unsqueeze(-1), w_ref[L - 1], s_ref[L - 1], d_ref[L - 1]
        )  # (1,)

        # ours: step the history one review at a time, then p_recall at delta_n
        h = gru.init_hidden(1)  # (1, 1, H)
        for t in range(L):
            dt_t = seq[t, 0].reshape(1, 1)
            rt_t = seq[t, 1].reshape(1, 1)
            h = gru.step(h, dt_t, rt_t)
        ret_m = gru.p_recall(h, delta_n.reshape(1, 1))  # (1, 1)

        diffs.append(abs(float(ret_m) - float(ret_ref)))

    max_diff = max(diffs)
    print(f"compared {n} (card, review) items from user {USER_ID}")
    ok = check("retention vs reference", max_diff < TOL, f"max|d|={max_diff:.2e}")
    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
