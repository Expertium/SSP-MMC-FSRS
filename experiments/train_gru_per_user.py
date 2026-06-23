"""Finetune a per-user GRU recall predictor for N users (roadmap step 4, part A).

For each user we start from srs-benchmark's *global* pretrained GRU
(``pretrain/GRU-short-secs_pretrain.pth`` + its inner-optimizer state) and run the same
``reptile_trainer_gru.finetune`` srs-benchmark uses to produce a per-user GRU -- but on the
user's **full review history** (no train/test split), because we want the best-possible
per-user fit to serve as pseudo-ground-truth p(recall) inside the simulator. The resulting
``state_dict`` (weights + ``input_mean``/``input_std`` buffers, all self-contained) is saved
to ``outputs/gru_weights/GRU-short-secs/user_{id}.pth`` in THIS repo.

This reuses srs-benchmark's verified machinery **in place** (it has every dependency), so it
must run with srs-benchmark's venv:

    C:/Users/Andrew/srs-benchmark/.venv/Scripts/python.exe experiments/train_gru_per_user.py --users 1-3 --eval
    C:/Users/Andrew/srs-benchmark/.venv/Scripts/python.exe experiments/train_gru_per_user.py --users 1-1000

The .pth files are read later by ``ssp_mmc_fsrs.gru.BatchedGRU`` (candle reads the same .pth
for the eventual Rust port).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
SRSB = Path(r"C:/Users/Andrew/srs-benchmark")
DATA = Path(r"C:/Users/Andrew/anki-revlogs-10k")
OUT = REPO / "outputs" / "gru_weights" / "GRU-short-secs"

# srs-benchmark imports rely on its package root being importable and being the cwd (so the
# GRU constructor finds ./pretrain/...). Compute absolute paths above BEFORE we chdir.
sys.path.insert(0, str(SRSB))
os.chdir(SRSB)

import pandas as pd  # noqa: E402
import torch.nn as nn  # noqa: E402
import reptile_trainer_gru as rt  # type: ignore  # noqa: E402
from config import Config, create_parser  # type: ignore  # noqa: E402
from features import create_features  # type: ignore  # noqa: E402
from models import GRU  # type: ignore  # noqa: E402
from reptile_trainer_gru import BATCH_SIZE, MAX_SEQ_LEN, finetune, get_inner_opt  # type: ignore  # noqa: E402
from fsrs_optimizer import BatchDataset, BatchLoader  # type: ignore  # noqa: E402
from utils import batch_process_wrapper  # type: ignore  # noqa: E402


def build_config() -> Config:
    parser = create_parser()
    args = parser.parse_args(
        ["--algo", "GRU", "--short", "--secs", "--data", str(DATA), "--processes", "1"]
    )
    return Config(args)


CONFIG = build_config()
DEVICE = rt.DEVICE  # may be overridden by --device in main()
OPT_PRETRAIN = f"./pretrain/{CONFIG.get_optimizer_file_name()}_pretrain.pth"


def set_device(device_str: str) -> None:
    """Force the finetune + eval onto a device (e.g. 'cpu' for CPU-sharded throughput)."""
    global DEVICE
    dev = torch.device(device_str)
    CONFIG.device = dev
    rt.DEVICE = dev  # finetune() reads this module global for its data loaders
    DEVICE = dev


def load_user_df(user_id: int) -> pd.DataFrame:
    df = pd.read_parquet(DATA / "revlogs" / f"{user_id=}")
    return create_features(df, config=CONFIG)


def mean_logloss(model, df) -> float:
    """Unweighted mean BCE over the whole df (sanity metric; lower = better fit)."""
    ds = BatchDataset(
        df.copy(),
        BATCH_SIZE,
        sort_by_length=False,
        max_seq_len=MAX_SEQ_LEN,
        device=DEVICE,
    )
    loader = BatchLoader(ds, shuffle=False)
    loss_fn = nn.BCELoss(reduction="sum")
    total, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            r = batch_process_wrapper(model, batch)
            total += loss_fn(r["retentions"], r["labels"]).item()
            n += int(r["labels"].shape[0])
    return total / max(n, 1)


def finetune_user(user_id: int, do_eval: bool) -> dict:
    """Finetune the global pretrained GRU on one user's full history. Returns a report."""
    t0 = time.perf_counter()
    df = load_user_df(user_id)
    n_reviews = len(df)
    t_load = time.perf_counter() - t0

    model = GRU(CONFIG).to(
        DEVICE
    )  # constructor auto-loads the global pretrained weights
    inner_opt = get_inner_opt(model.parameters(), path=OPT_PRETRAIN)

    before = mean_logloss(model, df) if do_eval else None

    t1 = time.perf_counter()
    learner = finetune(df, model, inner_opt.state_dict())
    t_fit = time.perf_counter() - t1

    after = mean_logloss(learner, df) if do_eval else None

    OUT.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.detach().cpu() for k, v in learner.state_dict().items()}
    torch.save(cpu_state, OUT / f"user_{user_id}.pth")

    return {
        "user": user_id,
        "n_reviews": n_reviews,
        "logloss_before": before,
        "logloss_after": after,
        "t_load": t_load,
        "t_fit": t_fit,
        "t_total": time.perf_counter() - t0,
    }


def parse_users(spec: str) -> list[int]:
    """'1-1000' -> [1..1000]; '1,5,9' -> [1,5,9]; '1-3,10' -> [1,2,3,10]."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--users", default="1-1000", help="e.g. '1-1000' or '1,2,3' or '1-3,10'"
    )
    ap.add_argument(
        "--eval", action="store_true", help="report before/after mean logloss"
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="re-finetune users with a saved .pth"
    )
    ap.add_argument(
        "--device", default=None, help="force device, e.g. 'cpu' (default: auto)"
    )
    ap.add_argument(
        "--shard",
        default=None,
        help="'k/n': keep only every n-th user (0-based offset k) for parallel shards",
    )
    ap.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="torch intra-op threads per process (set low when running many CPU shards)",
    )
    args = ap.parse_args()

    if args.device:
        set_device(args.device)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    users = parse_users(args.users)
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        users = [u for i, u in enumerate(users) if i % n == k]
    print(f"Device: {DEVICE}; output: {OUT}")
    print(f"Users requested: {len(users)} ({users[0]}..{users[-1]})")

    done = 0
    skipped = 0
    t_start = time.perf_counter()
    for i, uid in enumerate(users, 1):
        out_path = OUT / f"user_{uid}.pth"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            rep = finetune_user(uid, do_eval=args.eval)
        except Exception as e:  # keep going; record which users failed
            print(f"[{i}/{len(users)}] user {uid}: FAILED ({type(e).__name__}: {e})")
            continue
        done += 1
        msg = (
            f"[{i}/{len(users)}] user {uid}: n={rep['n_reviews']} "
            f"load={rep['t_load']:.2f}s fit={rep['t_fit']:.2f}s total={rep['t_total']:.2f}s"
        )
        if args.eval:
            msg += (
                f" | logloss {rep['logloss_before']:.4f} -> {rep['logloss_after']:.4f}"
            )
        print(msg, flush=True)

    elapsed = time.perf_counter() - t_start
    print(
        f"\nDone: finetuned {done}, skipped {skipped} (already present) in {elapsed:.1f}s"
        + (f" ({elapsed / done:.2f}s/user)" if done else "")
    )


if __name__ == "__main__":
    main()
