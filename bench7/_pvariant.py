"""Worker for validate_parallel.py: run ONE batched (parallel=N) simulate_fsrs7 and dump.

    python -m bench7._pvariant <pyd_dir> <stacked_npz> <out_json> <reps>

Loads this variant's ssp_mmc_rust.pyd (first on sys.path) + the pre-stacked per-user inputs
(w/costs/probs/gru/retention for all N datapoints, built once by the driver), runs the sim
with parallel=N (all users in one call), and records the min-of-reps wall time plus the
full per-user output arrays (so the driver can check serial==parallel bit-exact).
"""

import json
import sys
import time

import numpy as np


def main():
    pyd_dir, npz_path, out_json, reps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    sys.path.insert(0, pyd_dir)
    import ssp_mmc_rust

    z = np.load(npz_path)
    N = int(z["N"])
    deck = int(z["deck"])
    span = int(z["span"])
    seed = int(z["seed"])
    args = (
        N,
        deck,
        span,
        seed,
        z["w"],
        z["lc"],
        z["rc"],
        z["frp"],
        z["rrp"],
        float(z["max_cost"]),
        int(z["learn_limit"]),
        int(z["review_limit"]),
        float(z["s_min"]),
        float(z["s_max"]),
        int(z["max_same_day"]),
        int(z["n_iter"]),
        "ssp_mmc",
        0.0,
        z["gru"],
        z["ret"],
        z["s_grid"],
        z["d_grid"],
    )

    best = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = ssp_mmc_rust.simulate_fsrs7(*args)
        best = min(best, time.perf_counter() - t0)

    review, learn, mem, cost, t_fsrs, t_gru = out
    json.dump(
        {
            "time": best,
            "t_fsrs": float(t_fsrs),
            "t_gru": float(t_gru),
            # full arrays for the bit-exact serial-vs-parallel check
            "review": np.asarray(review).tolist(),
            "learn": np.asarray(learn).tolist(),
            "memorized": np.asarray(mem).tolist(),
            "cost": np.asarray(cost).tolist(),
        },
        open(out_json, "w"),
    )


if __name__ == "__main__":
    main()
