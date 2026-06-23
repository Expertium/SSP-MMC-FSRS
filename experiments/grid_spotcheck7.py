"""225-vs-300 time-convergence spot-check on the aggressive cells from grid_accuracy7.

The grid-accuracy run showed mean study-time/day is very grid-sensitive for aggressive
(expensive) policies. Question: has time CONVERGED by the 225-pt S grid, or is it still
moving? We recompute just the worst cells at 300 pts and compare to the stored 135/225
values: if |300-225|/300 << |225-135|/225, time is converging and 225 is close.

300 pts exceeds 12 GB VRAM (the solver's int64 argmin copy), so it spills to host RAM --
slow but fine on 64 GB. Resumable (incremental save per cell) because spilling near the
VRAM limit is crash-prone; pair with a retry supervisor.
"""

import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "experiments", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ssp_mmc_fsrs.solver7 import SSPMMCSolver7, COST_MAX  # noqa: E402
from ssp_mmc_fsrs.simulation7 import simulate  # noqa: E402
from converge7 import (  # noqa: E402
    load_jsonl_by_user,
    make_hyperparam_sets,
    DEFAULT_PARAMS,
    DEFAULT_BUTTON_USAGE,
)
from grid_accuracy7 import make_ssp_mmc_policy, _normalize  # noqa: E402

# Aggressive (user, set) cells: the largest 135->225 time differences.
CELLS = [(2, 1), (1, 1), (2, 0), (4, 0), (6, 0), (5, 0), (4, 1)]
FINE = 300
RESULTS = ROOT / "outputs" / "checkpoints" / "grid_spotcheck7.json"
MAIN = ROOT / "outputs" / "checkpoints" / "grid_accuracy7.json"


def main():
    torch.set_num_threads(1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    params = load_jsonl_by_user(DEFAULT_PARAMS)
    usage = load_jsonl_by_user(DEFAULT_BUTTON_USAGE)
    hp_sets = make_hyperparam_sets(3, seed=0)

    fine = {}
    if RESULTS.exists():
        try:
            fine = dict(json.loads(RESULTS.read_text(encoding="utf-8")).get("fine", {}))
            print(f"resuming: {len(fine)}/{len(CELLS)} fine cells done", flush=True)
        except (json.JSONDecodeError, OSError):
            pass

    def save(complete=False):
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        tmp = RESULTS.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "fine_n_s": FINE,
                    "cells": [list(c) for c in CELLS],
                    "fine": fine,
                    "complete": complete,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(RESULTS)

    by_user = defaultdict(list)
    for u, s in CELLS:
        by_user[u].append(s)

    for u, sets in by_user.items():
        if all(f"{u}|{s}" in fine for s in sets):
            print(f"user {u}: all fine cells done, skip", flush=True)
            continue
        w = params[u]["parameters"]["0"]
        us = usage[u]
        frp = _normalize(us["first_rating_prob"])
        rrp = _normalize(us["review_rating_prob"])
        t0 = time.perf_counter()
        solver = SSPMMCSolver7(
            review_costs=us["review_costs"],
            first_rating_prob=frp,
            review_rating_prob=rrp,
            w=w,
            device=dev,
            n_s=FINE,
        )
        for s in sets:
            if f"{u}|{s}" in fine:
                continue
            cm, rm = solver.solve(hp_sets[s], n_iter=2000, verbose=False)
            frac = float((cm >= COST_MAX * 0.99).mean())
            pol = make_ssp_mmc_policy(solver, rm, w, dev)
            _, _, mem, cost = simulate(
                parallel=8,
                w=w,
                policy=pol,
                device=dev,
                deck_size=10000,
                learn_span=365,
                learn_costs=us["learn_costs"],
                review_costs=us["review_costs"],
                first_rating_prob=frp,
                review_rating_prob=rrp,
                seed=42,
            )
            fine[f"{u}|{s}"] = {
                "memorized": float(mem.mean()),
                "time": float(cost.mean()),
                "frac_at_max": frac,
            }
            print(
                f"  FINE {FINE} user {u} set {s}: memorized={float(mem.mean()):.1f} "
                f"time/day={float(cost.mean()) / 60:.2f} min frac@max={frac:.2%} "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
            save()
            del pol, rm, cm, mem, cost
        del solver
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ---- compare 135 -> 225 -> 300 ----
    md = json.loads(MAIN.read_text(encoding="utf-8"))
    R, ref, co = md["results"], md["ref_n_s"], md["coarse_n_s"]
    print("\n==== study-time convergence: 135 -> 225 -> 300 (aggressive cells) ====")
    print(
        f"{'user':>4} {'set':>3} | {'t135':>7} {'t225':>7} {'t300':>7} | "
        f"{'|225-135|/225':>13} {'|300-225|/300':>13}"
    )
    for u, s in CELLS:
        t135 = R[f"{co}|{u}|{s}"]["time"] / 60
        t225 = R[f"{ref}|{u}|{s}"]["time"] / 60
        t300 = fine[f"{u}|{s}"]["time"] / 60
        d1 = abs(t225 - t135) / t225 * 100
        d2 = abs(t300 - t225) / t300 * 100
        print(
            f"{u:>4} {s:>3} | {t135:>7.2f} {t225:>7.2f} {t300:>7.2f} | "
            f"{d1:>12.1f}% {d2:>12.1f}%"
        )
    save(complete=True)
    print("SPOTCHECK COMPLETE")


if __name__ == "__main__":
    main()
