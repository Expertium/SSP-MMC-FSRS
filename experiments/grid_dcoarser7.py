"""Final D-grid test: even-coarser middle (0.5 steps) D grid with the chosen hybrid S grid.

D grid: [1,2]@0.1 + [2,9]@0.5 + [9,10]@0.1 = 35 pts (vs coarse-D's 49; middle 0.5 not 0.25,
edges still 0.1). S grid: the chosen hybrid lin5_log66_sk0.4 (71 pts). Compares MEMORIZED to
the 225 reference and to the current best (hybrid-71 + coarse-D49, = 1.27% vs 225). If the
error vs 225 doesn't grow much, this becomes the permanent production grid
(35 * 71^2 ~= 176k states). Resumable; --users/--results partitioning for parallel runs.
"""

import argparse
import gc
import json
import sys
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
from grid_hybrid7 import build_hybrid_s_grid  # noqa: E402

RESULTS = ROOT / "outputs" / "checkpoints" / "grid_dcoarser7.json"
MAIN = ROOT / "outputs" / "checkpoints" / "grid_accuracy7.json"


def build_coarser_d_grid():
    """[1,2]@0.1 + [2,9]@0.5 + [9,10]@0.1, deduped -> 35 non-uniform points."""
    seg1 = np.round(np.arange(1.0, 2.0 + 1e-9, 0.1), 4)
    seg2 = np.round(np.arange(2.0, 9.0 + 1e-9, 0.5), 4)
    seg3 = np.round(np.arange(9.0, 10.0 + 1e-9, 0.1), 4)
    return np.unique(np.concatenate([seg1, seg2, seg3]))


def parse_args():
    p = argparse.ArgumentParser(description="Even-coarser-D grid test (hybrid S).")
    p.add_argument("--users", type=str, default=None)
    p.add_argument("--results", type=Path, default=RESULTS)
    p.add_argument("--n-users", type=int, default=None)
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
    if args.users:
        users = [int(x) for x in args.users.split(",")]
    elif args.n_users:
        users = users[: args.n_users]

    d_grid = build_coarser_d_grid()
    s_grid = build_hybrid_s_grid(5, 66, 0.4)
    print(
        f"coarser-D: {len(d_grid)} pts; hybrid-S: {len(s_grid)} pts; "
        f"states={len(d_grid) * len(s_grid) ** 2:,}; users={users}",
        flush=True,
    )
    rp = args.results
    res = {}
    if rp.exists():
        try:
            res = dict(json.loads(rp.read_text(encoding="utf-8")).get("results", {}))
            print(f"resuming: {len(res)} cells done", flush=True)
        except (json.JSONDecodeError, OSError):
            pass

    def save(complete=False):
        rp.parent.mkdir(parents=True, exist_ok=True)
        tmp = rp.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "d_size": len(d_grid),
                    "s_size": len(s_grid),
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
        tmp.replace(rp)

    for u in users:
        pend = [s for s in range(n_sets) if f"{u}|{s}" not in res]
        if not pend:
            continue
        w = params[u]["parameters"]["0"]
        us = usage[u]
        frp = _normalize(us["first_rating_prob"])
        rrp = _normalize(us["review_rating_prob"])
        solver = SSPMMCSolver7(
            review_costs=us["review_costs"],
            first_rating_prob=frp,
            review_rating_prob=rrp,
            w=w,
            device=dev,
            d_state=d_grid,
            s_state=s_grid,
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
                f"  coarserD user {u} set {s}: memorized={float(mem.mean()):.1f} "
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

    save(complete=True)
    print("DCOARSER COMPLETE")


if __name__ == "__main__":
    main()
