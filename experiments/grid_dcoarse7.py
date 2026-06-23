"""Coarse-D grid experiment: does a ~2x-smaller, non-uniform difficulty grid hurt memorized?

New D grid: [1,2]@0.1 + [2,9]@0.25 + [9,10]@0.1 = 49 points -- fine at the extremes (where the
optimal policy is most D-sensitive), coarse in the flat middle -- vs the uniform 91. S stays
at 135. We solve+sim the SAME 10 users x 3 seed-0 hyperparameter sets as grid_accuracy7 and
compare MEMORIZED to the stored 135 (uniform-D, S=135) and 225 (uniform-D, S=225) runs.

Memorized only: the 225-vs-300 spot-check showed study-time is grid-unreliable for aggressive
policies, so a time comparison wouldn't mean much. memorized is grid-robust, so it's the
trustworthy axis for judging whether the coarse D grid is acceptable.

n_states = 49 * 135^2 ~= 0.89M (about half the uniform-135 run), so VRAM is comfortable.
Resumable (incremental save per cell).
"""

import argparse
import gc
import json
import sys
import time
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

S_N = 135
RESULTS = ROOT / "outputs" / "checkpoints" / "grid_dcoarse7.json"
MAIN = ROOT / "outputs" / "checkpoints" / "grid_accuracy7.json"


def build_coarse_d_grid():
    """[1,2]@0.1 + [2,9]@0.25 + [9,10]@0.1, deduped -> 49 non-uniform points."""
    seg1 = np.round(np.arange(1.0, 2.0 + 1e-9, 0.1), 4)
    seg2 = np.round(np.arange(2.0, 9.0 + 1e-9, 0.25), 4)
    seg3 = np.round(np.arange(9.0, 10.0 + 1e-9, 0.1), 4)
    return np.unique(np.concatenate([seg1, seg2, seg3]))


def parse_args():
    p = argparse.ArgumentParser(description="Coarse-D grid memorized experiment.")
    p.add_argument(
        "--n-users", type=int, default=None, help="Limit users (default all)."
    )
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--deck-size", type=int, default=10_000)
    p.add_argument("--learn-span", type=int, default=365)
    p.add_argument("--sim-seed", type=int, default=42)
    p.add_argument("--solve-iter", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    params = load_jsonl_by_user(DEFAULT_PARAMS)
    usage = load_jsonl_by_user(DEFAULT_BUTTON_USAGE)
    hp_sets = make_hyperparam_sets(3, seed=0)
    n_sets = len(hp_sets)
    md = json.loads(MAIN.read_text(encoding="utf-8"))
    users = md["users"]
    if args.n_users:
        users = users[: args.n_users]

    d_grid = build_coarse_d_grid()
    print(f"coarse D grid: {len(d_grid)} points (uniform was 91)", flush=True)
    print(f"  {d_grid.tolist()}", flush=True)
    print(f"users: {users}  | S={S_N}  | sim seed={args.sim_seed}", flush=True)

    res = {}
    if RESULTS.exists():
        try:
            res = dict(
                json.loads(RESULTS.read_text(encoding="utf-8")).get("results", {})
            )
            print(f"resuming: {len(res)} cells done", flush=True)
        except (json.JSONDecodeError, OSError):
            pass

    def save(complete=False):
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        tmp = RESULTS.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "s_n": S_N,
                    "d_size": len(d_grid),
                    "d_grid": d_grid.tolist(),
                    "users": users,
                    "hyperparam_sets": hp_sets,
                    "results": res,
                    "complete": complete,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(RESULTS)

    for u in users:
        pend = [s for s in range(n_sets) if f"{u}|{s}" not in res]
        if not pend:
            print(f"user {u}: done, skip", flush=True)
            continue
        w = params[u]["parameters"]["0"]
        us = usage[u]
        frp = _normalize(us["first_rating_prob"])
        rrp = _normalize(us["review_rating_prob"])
        tb = time.perf_counter()
        solver = SSPMMCSolver7(
            review_costs=us["review_costs"],
            first_rating_prob=frp,
            review_rating_prob=rrp,
            w=w,
            device=dev,
            n_s=S_N,
            d_state=d_grid,
        )
        peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
        for s in pend:
            cm, rm = solver.solve(hp_sets[s], n_iter=args.solve_iter, verbose=False)
            frac = float((cm >= COST_MAX * 0.99).mean())
            pol = make_ssp_mmc_policy(solver, rm, w, dev)
            _, _, mem, cost = simulate(
                parallel=args.parallel,
                w=w,
                policy=pol,
                device=dev,
                deck_size=args.deck_size,
                learn_span=args.learn_span,
                learn_costs=us["learn_costs"],
                review_costs=us["review_costs"],
                first_rating_prob=frp,
                review_rating_prob=rrp,
                seed=args.sim_seed,
            )
            res[f"{u}|{s}"] = {
                "memorized": float(mem.mean()),
                "time": float(cost.mean()),
                "frac_at_max": frac,
            }
            print(
                f"  coarseD user {u} set {s}: memorized={float(mem.mean()):.1f} "
                f"frac@max={frac:.2%}",
                flush=True,
            )
            save()
            del pol, rm, cm, mem, cost
        del solver
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
        print(f"user {u}: build peak VRAM {peak:.2f} GB", flush=True)

    # ---- memorized comparison: 225 (ref) vs 135 vs coarse-D(49), all relative to 225 ----
    R, ref, co = md["results"], md["ref_n_s"], md["coarse_n_s"]
    print(
        f"\n==== memorized: {ref}(ref) vs {co} vs coarse-D({len(d_grid)}) [S={S_N}] ===="
    )
    print(
        f"{'user':>4} {'set':>3} | {'m225':>8} {'m135':>8} {'mDcoarse':>9} | "
        f"{'135v225':>8} {'Dcv225':>8} {'Dcv135':>8}"
    )
    a135, aD, aD135 = [], [], []
    for u in users:
        for s in range(n_sets):
            m225 = R[f"{ref}|{u}|{s}"]["memorized"]
            m135 = R[f"{co}|{u}|{s}"]["memorized"]
            mD = res[f"{u}|{s}"]["memorized"]
            d135 = abs(m135 - m225) / m225 * 100
            dD = abs(mD - m225) / m225 * 100
            dD135 = abs(mD - m135) / m135 * 100
            a135.append(d135)
            aD.append(dD)
            aD135.append(dD135)
            print(
                f"{u:>4} {s:>3} | {m225:>8.1f} {m135:>8.1f} {mD:>9.1f} | "
                f"{d135:>7.2f}% {dD:>7.2f}% {dD135:>7.2f}%"
            )
    print(
        f"\n135 vs 225 (S only):       mean {np.mean(a135):.2f}%  max {np.max(a135):.2f}%"
    )
    print(f"coarse-D vs 225 (D+S):     mean {np.mean(aD):.2f}%  max {np.max(aD):.2f}%")
    print(
        f"coarse-D vs 135 (D only):  mean {np.mean(aD135):.2f}%  max {np.max(aD135):.2f}%"
    )
    save(complete=True)
    print("DCOARSE COMPLETE")


if __name__ == "__main__":
    main()
