"""S-grid fineness/shape experiments, all with the coarse-D(49) grid.

Builds several S grids and measures MEMORIZED vs the saved 225 (uniform-D, S=225) and 135
(uniform-D, S=135) reference runs. Configs:
  * S75_uniform  - 75 log-uniform points
  * S50_uniform  - 50 log-uniform points
  * S50_low      - 50 points, log steps skewed to CONCENTRATE at LOW S (fine low / coarse high)
  * S50_high     - 50 points, log steps skewed to CONCENTRATE at HIGH S (coarse low / fine high)

The skew uses log(S) = log(s_min) + (i/(n-1))**skew * log(s_max/s_min): skew=1 -> log-uniform
(reproduces build_s_grid); skew>1 packs points at low S; skew<1 packs them at high S.

Memorized only -- study-time is grid-unreliable (see the 225-vs-300 spot-check). Resumable.
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
    S_INCREMENT_MIN,
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

# (label, n_s, skew). skew=1 log-uniform; skew<1 packs HIGH S; skew>1 packs LOW S.
CONFIGS = [
    ("S75_uniform", 75, 1.0),
    ("S50_uniform", 50, 1.0),
    ("S50_low", 50, 1.5),  # concentrate at LOW S
    ("S50_high", 50, 0.67),  # concentrate at HIGH S
    # --- second round: an intermediate point count + 3 levels of HIGH-S skew at 50 ---
    ("S100_uniform", 100, 1.0),
    ("S50_high_slight", 50, 0.8),
    ("S50_high_mod", 50, 0.6),
    ("S50_high_heavy", 50, 0.4),
]
RESULTS = ROOT / "outputs" / "checkpoints" / "grid_sgrid7.json"
MAIN = ROOT / "outputs" / "checkpoints" / "grid_accuracy7.json"
DCOARSE = ROOT / "outputs" / "checkpoints" / "grid_dcoarse7.json"


def build_skewed_s_grid(n, skew=1.0):
    """Log-spaced S grid with a power skew on the log axis. skew=1 == build_s_grid."""
    u = np.arange(n) / (n - 1)
    v = u**skew
    s = np.exp(np.log(S_MIN) + v * (np.log(S_MAX) - np.log(S_MIN)))
    s[0] = S_MIN
    for i in range(1, n):
        s[i] = max(s[i], s[i - 1] + S_INCREMENT_MIN)
    s[-1] = S_MAX
    return s


def parse_args():
    p = argparse.ArgumentParser(description="S-grid shape experiments (coarse-D).")
    p.add_argument("--n-users", type=int, default=None)
    p.add_argument(
        "--users", type=str, default=None, help="Comma list of user ids (partition)."
    )
    p.add_argument(
        "--results", type=Path, default=RESULTS, help="Output path (partition)."
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
                    "configs": [list(c) for c in CONFIGS],
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

    for label, n_s, skew in CONFIGS:
        s_grid = build_skewed_s_grid(n_s, skew)
        print(
            f"\n## config {label}: n_s={n_s} skew={skew} "
            f"(states={len(d_grid) * n_s * n_s:,})",
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

    # ---- summary: memorized rel-diff vs 225 and vs 135 (both uniform-D) ----
    R, ref, co = md["results"], md["ref_n_s"], md["coarse_n_s"]

    def agg(getter):
        v225, v135 = [], []
        for u in users:
            for s in range(n_sets):
                m = getter(u, s)
                if m is None:
                    return None
                m225 = R[f"{ref}|{u}|{s}"]["memorized"]
                m135 = R[f"{co}|{u}|{s}"]["memorized"]
                v225.append(abs(m - m225) / m225 * 100)
                v135.append(abs(m - m135) / m135 * 100)
        return (np.mean(v225), np.max(v225), np.mean(v135), np.max(v135))

    print("\n==== memorized rel-diff vs 225 and vs 135 (both uniform-D) ====")
    print(
        f"{'config':<16} {'states':>9} | {'vs225 mean':>10} {'vs225 max':>9} | "
        f"{'vs135 mean':>10} {'vs135 max':>9}"
    )
    # baseline: coarse-D at S=135 (from grid_dcoarse7), if present
    if DCOARSE.exists():
        dc = json.loads(DCOARSE.read_text(encoding="utf-8")).get("results", {})
        a = agg(lambda u, s: dc.get(f"{u}|{s}", {}).get("memorized"))
        if a:
            print(
                f"{'coarseD_S135':<16} {len(d_grid) * 135 * 135:>9,} | "
                f"{a[0]:>9.2f}% {a[1]:>8.2f}% | {a[2]:>9.2f}% {a[3]:>8.2f}%"
            )
    for label, n_s, skew in CONFIGS:
        a = agg(lambda u, s: res.get(f"{label}|{u}|{s}", {}).get("memorized"))
        if a:
            print(
                f"{label:<16} {len(d_grid) * n_s * n_s:>9,} | "
                f"{a[0]:>9.2f}% {a[1]:>8.2f}% | {a[2]:>9.2f}% {a[3]:>8.2f}%"
            )
    save(complete=True)
    print("SGRID COMPLETE")


if __name__ == "__main__":
    main()
