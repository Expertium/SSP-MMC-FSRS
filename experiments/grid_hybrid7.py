"""Hybrid S-grid experiments: LINEAR low-S region + SKEWED-LOG high-S region, all coarse-D.

Motivation: a pure log grid over [1e-4, 36500] over-samples the sub-day region (and the
prior S-grid sweep showed packing low-S hurts). Here we instead put a few LINEAR steps over
S in [1e-4, 0.1] days (sub-~2.4h, the same-day/relearn region) and spend the rest of the
budget on a SKEWED-LOG region over [0.1, 36500], varying:
  * number of linear steps  (n_linear)
  * number of log steps      (n_log)
  * skew of the log region   (skew<1 packs HIGH S, =1 log-uniform, >1 packs low S)

Compared on MEMORIZED to the saved 225 (uniform-D, S=225) and 135 (uniform-D, S=135) runs.
Resumable; supports --users/--results partitioning for parallel runs.
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

from ssp_mmc_fsrs.solver7 import (  # noqa: E402
    SSPMMCSolver7,
    COST_MAX,
    S_MIN,
    S_MAX,
)
from ssp_mmc_fsrs.simulation7 import simulate  # noqa: E402
from converge7 import (  # noqa: E402
    load_jsonl_by_user,
    make_hyperparam_sets,
    DEFAULT_PARAMS,
    DEFAULT_BUTTON_USAGE,
)
from grid_accuracy7 import make_ssp_mmc_policy, _normalize  # noqa: E402
from grid_dcoarse7 import build_coarse_d_grid  # noqa: E402

LIN_MAX = 0.1  # split point: linear over [S_MIN, 0.1], skewed-log over [0.1, S_MAX]
# (label, n_linear, n_log, skew). total points = n_linear + n_log - 1 (shared 0.1).
HYBRID_CONFIGS = [
    ("hyb_lin5_log46_sk1.0", 5, 46, 1.0),  # 50 total, log-uniform high part
    ("hyb_lin5_log46_sk0.6", 5, 46, 0.6),  # 50, moderate high-skew
    ("hyb_lin5_log46_sk0.4", 5, 46, 0.4),  # 50, heavy high-skew
    ("hyb_lin3_log48_sk0.6", 3, 48, 0.6),  # 50, fewer linear steps
    ("hyb_lin10_log41_sk0.6", 10, 41, 0.6),  # 50, more linear steps
    ("hyb_lin5_log71_sk0.6", 5, 71, 0.6),  # 75 total, scale up
    # --- round 2: push to match coarseD_S135 (1.61%); winner was lin5_log46_sk0.4 @1.80% ---
    ("hyb_lin5_log46_sk0.3", 5, 46, 0.3),  # 51, heavier skew at same budget
    ("hyb_lin5_log56_sk0.4", 5, 56, 0.4),  # 60, more log pts at best skew
    ("hyb_lin5_log66_sk0.4", 5, 66, 0.4),  # 70, more log pts
    ("hyb_lin5_log71_sk0.4", 5, 71, 0.4),  # 76, best skew (vs 1.66% at sk0.6)
    # --- round 3: more LINEAR steps with the log part held fixed (46 pts, skew 0.4) ---
    ("hyb_lin10_log46_sk0.4", 10, 46, 0.4),  # 55
    ("hyb_lin15_log46_sk0.4", 15, 46, 0.4),  # 60
    ("hyb_lin20_log46_sk0.4", 20, 46, 0.4),  # 65
    ("hyb_lin25_log46_sk0.4", 25, 46, 0.4),  # 70
]
RESULTS = ROOT / "outputs" / "checkpoints" / "grid_hybrid7.json"
MAIN = ROOT / "outputs" / "checkpoints" / "grid_accuracy7.json"


def build_hybrid_s_grid(n_linear, n_log, skew=1.0, lin_max=LIN_MAX):
    """Linear [S_MIN, lin_max] + power-skewed log [lin_max, S_MAX], deduped at the boundary."""
    lin = np.linspace(S_MIN, lin_max, n_linear)
    u = np.arange(n_log) / (n_log - 1)
    v = u**skew
    logp = np.exp(np.log(lin_max) + v * (np.log(S_MAX) - np.log(lin_max)))
    s = np.unique(np.concatenate([lin, logp]))  # sorted + strictly increasing
    s[-1] = S_MAX
    return s


def parse_args():
    p = argparse.ArgumentParser(description="Hybrid linear+log S-grid experiments.")
    p.add_argument("--n-users", type=int, default=None)
    p.add_argument("--users", type=str, default=None, help="Comma list of user ids.")
    p.add_argument("--results", type=Path, default=RESULTS)
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
    d_grid = build_coarse_d_grid()
    results_path = args.results

    res = {}
    if results_path.exists():
        try:
            res = dict(
                json.loads(results_path.read_text(encoding="utf-8")).get("results", {})
            )
            print(f"resuming: {len(res)} cells done", flush=True)
        except (json.JSONDecodeError, OSError):
            pass

    def save(complete=False):
        results_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = results_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "configs": [list(c) for c in HYBRID_CONFIGS],
                    "lin_max": LIN_MAX,
                    "d_size": len(d_grid),
                    "users": users,
                    "hyperparam_sets": hp_sets,
                    "results": res,
                    "complete": complete,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(results_path)

    for label, n_lin, n_log, skew in HYBRID_CONFIGS:
        s_grid = build_hybrid_s_grid(n_lin, n_log, skew)
        print(
            f"\n## {label}: n_lin={n_lin} n_log={n_log} skew={skew} "
            f"-> {len(s_grid)} S pts (states={len(d_grid) * len(s_grid) ** 2:,})",
            flush=True,
        )
        for u in users:
            pend = [s for s in range(n_sets) if f"{label}|{u}|{s}" not in res]
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
                res[f"{label}|{u}|{s}"] = {
                    "memorized": float(mem.mean()),
                    "time": float(cost.mean()),
                    "frac_at_max": frac,
                }
                print(
                    f"  {label} user {u} set {s}: memorized={float(mem.mean()):.1f} "
                    f"frac@max={frac:.2%}",
                    flush=True,
                )
                save()
                del pol, rm, cm, mem, cost
            del solver
            gc.collect()
            if dev == "cuda":
                torch.cuda.empty_cache()

    # ---- summary vs 225 and 135 (both uniform-D) ----
    R, ref, co = md["results"], md["ref_n_s"], md["coarse_n_s"]

    def agg(label):
        v225, v135 = [], []
        for u in users:
            for s in range(n_sets):
                cell = res.get(f"{label}|{u}|{s}")
                if cell is None:
                    return None
                m = cell["memorized"]
                m225 = R[f"{ref}|{u}|{s}"]["memorized"]
                m135 = R[f"{co}|{u}|{s}"]["memorized"]
                v225.append(abs(m - m225) / m225 * 100)
                v135.append(abs(m - m135) / m135 * 100)
        return (np.mean(v225), np.max(v225), np.mean(v135), np.max(v135))

    print("\n==== hybrid memorized rel-diff vs 225 and vs 135 (both uniform-D) ====")
    print(
        f"{'config':<24} {'Spts':>5} | {'vs225 mean':>10} {'vs225 max':>9} | "
        f"{'vs135 mean':>10} {'vs135 max':>9}"
    )
    for label, n_lin, n_log, skew in HYBRID_CONFIGS:
        a = agg(label)
        if a:
            npts = len(build_hybrid_s_grid(n_lin, n_log, skew))
            print(
                f"{label:<24} {npts:>5} | {a[0]:>9.2f}% {a[1]:>8.2f}% | "
                f"{a[2]:>9.2f}% {a[3]:>8.2f}%"
            )
    save(complete=True)
    print("HYBRID COMPLETE")


if __name__ == "__main__":
    main()
