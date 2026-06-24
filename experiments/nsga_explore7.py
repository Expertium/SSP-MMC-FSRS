"""NSGA-II exploration pre-run: harvest SSP-MMC seeds for the step-5 ax run.

Cheap multi-objective search (MAXIMIZE knowledge, MINIMIZE time/day) over the 13-D SSP-MMC
cost-hyperparameter space on a small user set, to find decent hyperparameter sets (non-dominated;
beating or at least matching fixed DR) to hand the full 1k ax run as manual seed candidates.

Why NSGA-II (pymoo) and not ax here: it's gradient-free, natively multi-objective (returns a
diverse non-dominated set = exactly the seeds we want), and has no GP cost (ax's GP is O(n^3),
brutal at ~1000 evals). It evolves the population toward the front, so for the same eval budget
the seeds are near-optimal rather than best-of-random.

Plus a free `w_retention` sweep: from the diagnostic we know high w_ret -> near-constant-R policy
== fixed DR, so a handful of solves gives DR-matching seeds directly (no search budget spent).

Reuses the optimizer's eval pipeline verbatim (same solve+sim, same SIM_SEED, same grids) so the
seeds transfer cleanly.

Run:
    uv run --no-sync python -m experiments.nsga_explore7 --n-users 30 --pop 50 --gen 20 --seed 42
    (smoke: --n-users 3 --pop 4 --gen 2 --skip-dr)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import experiments.hyperparameter_optimizer7 as opt  # noqa: E402

from pymoo.algorithms.moo.nsga2 import NSGA2  # noqa: E402
from pymoo.core.callback import Callback  # noqa: E402
from pymoo.core.mixed import (  # noqa: E402
    MixedVariableDuplicateElimination,
    MixedVariableMating,
    MixedVariableSampling,
)
from pymoo.core.problem import ElementwiseProblem  # noqa: E402
from pymoo.core.variable import Choice, Real  # noqa: E402
from pymoo.optimize import minimize  # noqa: E402

# exp_* are searched in LOG space (the optimizer marks them log_scale=True).
LOG_PARAMS = {"exp_s_long", "exp_s_short", "exp_d"}


def build_vars():
    """pymoo mixed-variable spec from the optimizer's PARAMETERS (log_scale -> log-space Real)."""
    vars_ = {}
    for spec in opt.PARAMETERS:
        name = spec["name"]
        if spec["type"] == "choice":
            vars_[name] = Choice(options=list(spec["values"]))
        else:
            lo, hi = spec["bounds"]
            if name in LOG_PARAMS:
                vars_[name] = Real(bounds=(float(np.log(lo)), float(np.log(hi))))
            else:
                vars_[name] = Real(bounds=(float(lo), float(hi)))
    return vars_


def x_to_hp(x):
    """pymoo decision dict -> solver hyperparameter dict (undo log transform, round to 2 digits)."""
    hp = {}
    for spec in opt.PARAMETERS:
        name = spec["name"]
        v = x[name]
        if spec["type"] == "choice":
            hp[name] = v
        elif name in LOG_PARAMS:
            hp[name] = round(float(np.exp(v)), 2)
        else:
            hp[name] = round(float(v), 2)
    return hp


class SSPProblem(ElementwiseProblem):
    """Minimize (-knowledge, time/day): NSGA-II maximizes knowledge, minimizes daily time."""

    def __init__(self, users):
        super().__init__(vars=build_vars(), n_obj=2, n_ieq_constr=0)
        self.users = users
        self.count = 0

    def _evaluate(self, x, out, *args, **kwargs):
        hp = x_to_hp(x)
        k, t = opt.evaluate_hp(hp, self.users)
        self.count += 1
        print(f"[eval {self.count}] knowledge={k:.1f} time={t:.2f}", flush=True)
        out["F"] = [-k, t]


class FrontSaver(Callback):
    """Dump the current non-dominated set each generation (crash insurance + live monitoring)."""

    def __init__(self, path):
        super().__init__()
        self.path = path

    def notify(self, algorithm):
        front = sorted(
            (
                {
                    "params": x_to_hp(ind.X),
                    "knowledge": -float(ind.F[0]),
                    "time_per_day_min": float(ind.F[1]),
                }
                for ind in algorithm.opt
            ),
            key=lambda d: d["knowledge"],
        )
        json.dump(
            {
                "generation": int(algorithm.n_gen),
                "n_evals": int(algorithm.evaluator.n_eval),
                "front": front,
            },
            open(self.path, "w"),
            indent=2,
        )


def run_nsga(users, pop, gen, seed, partial_path=None):
    algo = NSGA2(
        pop_size=pop,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(
            eliminate_duplicates=MixedVariableDuplicateElimination()
        ),
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )
    res = minimize(
        SSPProblem(users),
        algo,
        ("n_gen", gen),
        seed=seed,
        verbose=True,
        callback=FrontSaver(partial_path) if partial_path else None,
    )
    X, F = res.X, res.F
    if isinstance(X, dict):  # single non-dominated solution
        X, F = [X], [F]
    front = [(x_to_hp(x), -float(f[0]), float(f[1])) for x, f in zip(X, F)]
    return sorted(front, key=lambda p: p[1])  # by knowledge


def _neutral_hp(w_ret):
    return {
        "transform_s_long": "no_log",
        "transform_s_short": "no_log",
        "exp_s_long": 1.0,
        "exp_s_short": 1.0,
        "exp_d": 1.0,
        "base_succ": 1.0,
        "w_fail_s_long": 0.0,
        "w_fail_s_short": 0.0,
        "w_fail_d": 0.0,
        "w_succ_s_long": 0.0,
        "w_succ_s_short": 0.0,
        "w_succ_d": 0.0,
        "w_retention": w_ret,
    }


def w_ret_sweep(users, weights=(0.5, 1, 2, 4, 8, 16, 32, 64)):
    """Free DR-matching seeds: high w_ret -> near-constant-R policy == a fixed-DR level."""
    out = []
    for w in weights:
        hp = _neutral_hp(float(w))
        rt = opt._solve_all(hp, users, progress_every=0)
        mem, cost = opt._simulate(users, "ssp_mmc", 0.0, rt)
        k, t = opt._objectives_from(mem, cost)
        print(f"[w_ret sweep] w_ret={w:<5} knowledge={k:.1f} time={t:.2f}", flush=True)
        out.append((hp, k, t))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="NSGA-II SSP-MMC seed exploration (step 5)."
    )
    ap.add_argument("--n-users", type=int, default=30)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--gen", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-dr", action="store_true", help="skip DR baseline + comparison"
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    uids = list(range(1, args.n_users + 1))
    out_dir = args.out or (opt.lib.CHECKPOINTS_DIR / f"users_1-{args.n_users}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"NSGA-II explore: users={len(uids)} pop={args.pop} gen={args.gen} "
        f"(~{args.pop * args.gen} evals) | out={out_dir}",
        flush=True,
    )

    users = opt.Users(uids)
    t0 = time.perf_counter()
    front = run_nsga(
        users, args.pop, args.gen, args.seed, out_dir / "nsga_seeds_partial.json"
    )
    print(f"\nNSGA-II done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    sweep = w_ret_sweep(users)

    dr_points = None
    if not args.skip_dr:
        dr_baseline = opt.load_or_make_dr_baseline(users, out_dir / "dr_baseline7.json")
        dr_points = opt._dr_points(dr_baseline)
        # Report the union (NSGA front + sweep) against DR.
        union = [(hp, k, t) for hp, k, t in (front + sweep)]
        comparisons = opt._compare_to_dr(union, dr_points)
        n_beat = sum(1 for c in comparisons if c["beats"])
        print(
            f"\n=== {n_beat}/{len(comparisons)} points beat the DR front ===",
            flush=True,
        )
        for c in sorted(comparisons, key=lambda c: c["knowledge"]):
            print(
                f"  k={c['knowledge']:.1f} t={c['time_per_day_min']:.2f}  "
                f"closestDR={c['closest_dr'] * 100:.0f}% beats={c['beats']} "
                f"reduction={c['workload_reduction_pct']:.1f}%"
            )

    seeds = [hp for hp, _, _ in front] + [hp for hp, _, _ in sweep]
    payload = {
        "n_users": args.n_users,
        "nsga_front": [
            {"params": hp, "knowledge": k, "time_per_day_min": t} for hp, k, t in front
        ],
        "w_ret_sweep": [
            {"params": hp, "knowledge": k, "time_per_day_min": t} for hp, k, t in sweep
        ],
        "seeds": seeds,
    }
    if dr_points is not None:
        payload["dr_front"] = [
            {"dr": d, "knowledge": k, "time_per_day_min": t} for d, k, t in dr_points
        ]
    path = out_dir / "nsga_seeds.json"
    json.dump(payload, open(path, "w"), indent=2)
    print(
        f"\nSaved {len(seeds)} seeds ({len(front)} NSGA + {len(sweep)} sweep) to {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
