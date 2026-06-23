"""Run one pipeline variant over the 60 (hp, user) datapoints and dump results to JSON.

Invoked as a subprocess by run_iteration7.py:
    python -m bench7.bench_variant7 <pyd_dir> <out_json>

`<pyd_dir>` holds that variant's ssp_mmc_rust.pyd; we put it first on sys.path so the
variant under test wins over any installed copy. The Python package (ssp_mmc_fsrs,
solver7, gru) is editable, so its on-disk state (which run_iteration7 git-stashes for the
"before" build) is what we import here.

Per datapoint = full pipeline: build+solve the Bellman (Python/GPU SSPMMCSolver7) with the
hp -> SSP-MMC retention table -> Rust simulate_fsrs7 with the per-user GRU. We record the
MIN over REPS of the whole-pipeline time AND the sim-only time, plus the (deterministic)
total knowledge (final memorized) and time_spent (total study cost).
"""

import json
import sys
import time

import numpy as np


def main():
    pyd_dir, out_json = sys.argv[1], sys.argv[2]
    sys.path.insert(0, pyd_dir)  # this variant's pyd wins over the installed copy

    import torch

    import ssp_mmc_rust
    from ssp_mmc_fsrs.gru import BatchedGRU
    from ssp_mmc_fsrs.solver7 import (
        SSPMMCSolver7,
        build_hybrid_s_grid,
        build_production_d_grid,
    )
    from bench7._common7 import (
        DECK,
        HP_SETS,
        LEARN_LIMIT,
        MAX_COST,
        MAX_SAME_DAY,
        N_ITER,
        REPS,
        REVIEW_LIMIT,
        S_MAX,
        S_MIN,
        SEED,
        SPAN,
        WDIR,
        datapoint_order,
        load_users,
    )

    S_GRID = build_hybrid_s_grid()
    D_GRID = build_production_d_grid()
    CUDA = torch.cuda.is_available()
    users = load_users()
    by_uid = {u["uid"]: u for u in users}

    def r2(x):
        return np.ascontiguousarray(np.asarray(x, np.float64).reshape(1, -1))

    # GRU flat weights are hp-independent -> load once per user.
    gflat_cache = {}
    for u in users:
        gru = BatchedGRU.from_pth_paths(
            [str(WDIR / f"user_{u['uid']}.pth")], device="cpu", dtype=torch.float64
        )
        gflat_cache[u["uid"]] = np.ascontiguousarray(gru.flat_weights().numpy(), np.float64)

    times, sim_times, knowledge, time_spent = [], [], [], []
    for hi, uid in datapoint_order():
        u = by_uid[uid]
        hp = HP_SETS[hi]
        gflat = gflat_cache[uid]
        best_tot, best_sim = float("inf"), float("inf")
        know = tsp = 0.0
        for _ in range(REPS):
            if CUDA:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            solver = SSPMMCSolver7(
                review_costs=u["rc"],
                first_rating_prob=u["frp"],
                review_rating_prob=u["rrp"],
                w=u["w"],
                s_state=S_GRID,
                d_state=D_GRID,
            )
            _, rm = solver.solve(hp, verbose=False)
            if CUDA:
                torch.cuda.synchronize()
            t_bellman = time.perf_counter() - t0

            ret_flat = np.ascontiguousarray(rm.reshape(1, -1))
            s_grid = np.ascontiguousarray(solver.s_state, np.float64)
            d_grid = np.ascontiguousarray(solver.d_state, np.float64)

            ts0 = time.perf_counter()
            out = ssp_mmc_rust.simulate_fsrs7(
                1,
                DECK,
                SPAN,
                SEED,
                r2(u["w"]),
                r2(u["lc"]),
                r2(u["rc"]),
                r2(u["frp"]),
                r2(u["rrp"]),
                MAX_COST,
                LEARN_LIMIT,
                REVIEW_LIMIT,
                float(S_MIN),
                float(S_MAX),
                MAX_SAME_DAY,
                N_ITER,
                "ssp_mmc",
                0.0,
                gflat,
                ret_flat,
                s_grid,
                d_grid,
            )
            t_sim = time.perf_counter() - ts0

            best_tot = min(best_tot, t_bellman + t_sim)
            best_sim = min(best_sim, t_sim)
            mem = np.asarray(out[2])
            cost = np.asarray(out[3])
            know = float(mem[:, -1].sum())
            tsp = float(cost.sum())

        times.append(best_tot)
        sim_times.append(best_sim)
        knowledge.append(know)
        time_spent.append(tsp)

    json.dump(
        {
            "pyd": getattr(ssp_mmc_rust, "__file__", pyd_dir),
            "order": datapoint_order(),
            "times": times,
            "sim_times": sim_times,
            "knowledge": knowledge,
            "time_spent": time_spent,
        },
        open(out_json, "w"),
    )


if __name__ == "__main__":
    main()
