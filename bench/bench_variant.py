"""Run one simulator variant over the 150 (user, DR) pairs and dump results to JSON.

Invoked as a subprocess by run_iteration.py:
    python -m bench.bench_variant <pyd_dir> <out_json>

`<pyd_dir>` holds that variant's ssp_mmc_rust.pyd; we put it first on sys.path so the
variant under test wins over any installed copy. Single-threaded (the Rust simulator is
scalar; run_iteration sets thread env vars). Records, per pair, the MIN of REPS timings
plus the (deterministic) memorized/time_spent outputs.
"""

import json
import sys
import time

import numpy as np

from bench._common import load_users, make_args, pair_order, DRS, REPS


def main():
    pyd_dir, out_json = sys.argv[1], sys.argv[2]
    sys.path.insert(0, pyd_dir)
    import ssp_mmc_rust

    users = load_users()
    times, memorized, time_spent = [], [], []
    for dr in DRS:
        for u in users:
            args = make_args(u, dr)
            best = float("inf")
            res = None
            for _ in range(REPS):
                t0 = time.perf_counter()
                res = ssp_mmc_rust.simulate(*args)
                dt = time.perf_counter() - t0
                best = min(best, dt)
            times.append(best)
            memorized.append(float(np.asarray(res[2]).mean()))
            time_spent.append(float(np.asarray(res[3]).sum()))

    json.dump(
        {
            "pyd": getattr(ssp_mmc_rust, "__file__", pyd_dir),
            "order": pair_order(users),
            "times": times,
            "memorized": memorized,
            "time_spent": time_spent,
        },
        open(out_json, "w"),
    )


if __name__ == "__main__":
    main()
