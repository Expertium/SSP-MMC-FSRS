"""Granular profiler for the full per-user pipeline (roadmap steps 3-5 wiring).

Pipeline per user: Bellman solve (FSRS-7 SSP-MMC, GPU) -> SSP-MMC scheduling policy ->
Rust simulator (deck x span) with the per-user GRU as the recall predictor. We time, for
10 users on average, the buckets we'll later optimize:

  * Bellman build  : build per-user transitions (Python/GPU, SSPMMCSolver7.__init__)
  * Bellman solve  : value iteration to V* (Python/GPU, .solve)
  * Sim wall       : the Rust simulate_fsrs7 call (CPU)
      - FSRS math   : update_state + curve + curve-inverse + SSP lookup (Rust Instant)
      - GRU infer   : p_recall + step                                   (Rust Instant)

Run with:  uv run --no-sync python experiments/profile_pipeline.py [--users 1-10] [--deck 10000] [--span 1825]

(Requires the RELEASE Rust build for representative timing:
 VIRTUAL_ENV=.../.venv CARGO_BUILD_JOBS=2 maturin develop --release --uv -m rust/Cargo.toml)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "experiments")
import lib  # type: ignore  # noqa: E402

import ssp_mmc_rust  # noqa: E402
from ssp_mmc_fsrs import fsrs7  # noqa: E402
from ssp_mmc_fsrs.gru import BatchedGRU  # noqa: E402
from ssp_mmc_fsrs.simulation7 import simulate, S_MIN_SECS  # noqa: E402
from ssp_mmc_fsrs.solver7 import (  # noqa: E402
    DISCOUNT_FACTOR,
    SSPMMCSolver7,
    build_hybrid_s_grid,
    build_production_d_grid,
)

REPO = Path(__file__).resolve().parents[1]
WDIR = REPO / "outputs" / "gru_weights" / "GRU-short-secs"
FSRS7 = "../srs-benchmark/result/FSRS-7-short-secs-recency.jsonl"
BU = "../Anki-button-usage/button_usage.jsonl"

S_GRID = build_hybrid_s_grid()
D_GRID = build_production_d_grid()
HP = {
    "transform_s_long": "log",
    "transform_s_short": "log",
    "exp_s_long": 1.0,
    "exp_s_short": 1.0,
    "exp_d": 1.0,
    "base_succ": 1.0,
    "w_fail_s_long": 0.5,
    "w_fail_s_short": 0.5,
    "w_fail_d": 0.5,
    "w_succ_s_long": 0.5,
    "w_succ_s_short": 0.5,
    "w_succ_d": 0.5,
    "w_retention": 1.0,
}

MAX_COST = 86400 / 2
LEARN_LIMIT = 10
REVIEW_LIMIT = 9999
MAX_SAME_DAY = 8
# Scheduling inverse iterations passed to the Rust simulator. The Rust sim now schedules
# with the fast Newton inverse, so this is the Newton count (matches the Python policies).
N_ITER = fsrs7.NEWTON_N_ITER
S_MAX = fsrs7.S_MAX
S_MIN = S_MIN_SECS
CUDA = torch.cuda.is_available()


def load_user(uid):
    w, _, _ = lib.load_fsrs_weights(FSRS7, uid)
    u = lib.normalize_button_usage(lib.load_button_usage_config(BU, uid))
    return {
        "w": np.asarray(w, np.float64),
        "lc": np.asarray(u["learn_costs"], np.float64),
        "rc": np.asarray(u["review_costs"], np.float64),
        "frp": np.asarray(u["first_rating_prob"], np.float64),
        "rrp": np.asarray(u["review_rating_prob"], np.float64),
    }


def make_ssp_policy(solver, rm, w):
    """Python SSP-MMC policy callable (for parity): state -> target retention -> interval."""
    R = torch.as_tensor(np.ascontiguousarray(rm), dtype=torch.float64)
    s_state = torch.as_tensor(solver.s_state, dtype=torch.float64)
    d_state = torch.as_tensor(solver.d_state, dtype=torch.float64)
    s_size, d_size = len(solver.s_state), len(solver.d_state)
    wt = torch.as_tensor(w, dtype=torch.float64).reshape(-1)

    def pol(s_long, s_short, d, prev_ivl, rating, ease):
        sl = torch.searchsorted(s_state, s_long.contiguous()).clamp(0, s_size - 1)
        ss = torch.searchsorted(s_state, s_short.contiguous()).clamp(0, s_size - 1)
        hi = torch.searchsorted(d_state, d.contiguous()).clamp(0, d_size - 1)
        lo = (hi - 1).clamp(0, d_size - 1)
        pick_hi = (d_state[hi] - d).abs() <= (d - d_state[lo]).abs()
        di = torch.where(pick_hi, hi, lo)
        target = R[di, sl, ss]
        interval = fsrs7.forgetting_curve_inverse(
            target, s_long, s_short, d, wt, method="newton"
        )
        return interval, ease

    return pol


def _r2(x):
    return np.ascontiguousarray(np.asarray(x, np.float64).reshape(1, -1))


def solve_bellman(u):
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
    if CUDA:
        torch.cuda.synchronize()
    t_build = time.perf_counter() - t0
    t1 = time.perf_counter()
    _, rm = solver.solve(HP, verbose=False)
    if CUDA:
        torch.cuda.synchronize()
    t_solve = time.perf_counter() - t1
    return solver, rm, t_build, t_solve


def ssp_parity(uid=1, deck=500, span=120):
    """Confirm the Rust ssp_mmc policy matches the Python SSP-MMC callable (gru off)."""
    u = load_user(uid)
    solver, rm, _, _ = solve_bellman(u)
    pol = make_ssp_policy(solver, rm, u["w"])
    py = simulate(
        parallel=1,
        w=_r2(u["w"]),
        policy=pol,
        device="cpu",
        deck_size=deck,
        learn_span=span,
        max_cost_perday=MAX_COST,
        learn_limit_perday=LEARN_LIMIT,
        review_limit_perday=REVIEW_LIMIT,
        learn_costs=_r2(u["lc"]),
        review_costs=_r2(u["rc"]),
        first_rating_prob=_r2(u["frp"]),
        review_rating_prob=_r2(u["rrp"]),
        seed=42,
        s_min=S_MIN,
        s_max=S_MAX,
        max_same_day=MAX_SAME_DAY,
        rng_kind="shared",
    )
    rs = ssp_mmc_rust.simulate_fsrs7(
        1,
        deck,
        span,
        42,
        _r2(u["w"]),
        _r2(u["lc"]),
        _r2(u["rc"]),
        _r2(u["frp"]),
        _r2(u["rrp"]),
        MAX_COST,
        LEARN_LIMIT,
        REVIEW_LIMIT,
        float(S_MIN),
        float(S_MAX),
        MAX_SAME_DAY,
        N_ITER,
        "ssp_mmc",
        0.0,
        None,
        np.ascontiguousarray(rm.reshape(1, -1)),
        np.ascontiguousarray(solver.s_state, np.float64),
        np.ascontiguousarray(solver.d_state, np.float64),
    )
    names = ["review", "learn", "memorized", "cost"]
    ok = True
    for n, a, b in zip(names, py, rs):
        a, b = np.asarray(a), np.asarray(b)
        if n in ("review", "learn"):
            good = np.array_equal(a, b)
        else:
            good = np.allclose(a, b, rtol=1e-4, atol=1e-4)
        ok = ok and good
        print(f"    {n:10s} {'OK' if good else 'DIFF'}")
    print(f"  SSP-MMC Python<->Rust parity: {'PASS' if ok else 'FAIL'}")
    return ok


def run_one(uid, deck, span):
    u = load_user(uid)
    solver, rm, t_build, t_solve = solve_bellman(u)
    gru = BatchedGRU.from_pth_paths(
        [WDIR / f"user_{uid}.pth"], device="cpu", dtype=torch.float64
    )
    gflat = np.ascontiguousarray(gru.flat_weights().numpy(), np.float64)
    ret_flat = np.ascontiguousarray(rm.reshape(1, -1))
    s_grid = np.ascontiguousarray(solver.s_state, np.float64)
    d_grid = np.ascontiguousarray(solver.d_state, np.float64)

    t2 = time.perf_counter()
    out = ssp_mmc_rust.simulate_fsrs7(
        1,
        deck,
        span,
        42,
        _r2(u["w"]),
        _r2(u["lc"]),
        _r2(u["rc"]),
        _r2(u["frp"]),
        _r2(u["rrp"]),
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
    t_sim = time.perf_counter() - t2
    review, _, mem, _, t_fsrs, t_gru = out
    return {
        "build": t_build,
        "solve": t_solve,
        "sim": t_sim,
        "fsrs": t_fsrs,
        "gru": t_gru,
        "reviews": float(review.sum()),
        "know": float(mem[:, -1].sum()),
    }


def bellman_cpu(u, solver):
    """Run the Rust CPU Bellman on the SAME grids/costs/hp as the Python GPU solver."""
    return ssp_mmc_rust.bellman_fsrs7(
        np.ascontiguousarray(u["w"], np.float64),
        np.ascontiguousarray(u["rc"], np.float64),
        np.ascontiguousarray(u["rrp"], np.float64),
        np.ascontiguousarray(solver.s_state, np.float64),
        np.ascontiguousarray(solver.d_state, np.float64),
        np.ascontiguousarray(solver.r_state, np.float64),
        HP["transform_s_long"] == "log",
        HP["transform_s_short"] == "log",
        HP["exp_s_long"],
        HP["exp_s_short"],
        HP["exp_d"],
        HP["base_succ"],
        HP["w_fail_s_long"],
        HP["w_fail_s_short"],
        HP["w_fail_d"],
        HP["w_succ_s_long"],
        HP["w_succ_s_short"],
        HP["w_succ_d"],
        HP["w_retention"],
        float(S_MIN),
        float(S_MAX),
        float(DISCOUNT_FACTOR),
        100_000,
        0.1,
        20,
        N_ITER,
    )


def bellman_compare(users):
    """Bellman GPU (Python solver7) vs CPU (Rust), on byte-identical inputs."""
    print(f"\nBellman GPU (Python/torch) vs CPU (Rust), {len(users)} users:")
    print(
        f"{'user':>5} {'gpu_build':>10} {'gpu_solve':>10} {'gpu_tot':>9} "
        f"{'cpu_build':>10} {'cpu_solve':>10} {'cpu_tot':>9} {'CPU/GPU':>8} {'R_maxdiff':>10}"
    )
    g_tot = c_tot = 0.0
    for uid in users:
        u = load_user(uid)
        solver, rm_gpu, gb, gs = solve_bellman(u)
        ret_cpu, _it, cb, cs = bellman_cpu(u, solver)
        rm_cpu = np.asarray(ret_cpu).reshape(rm_gpu.shape)
        maxdiff = float(np.abs(rm_cpu - rm_gpu).max())
        gt, ct = gb + gs, cb + cs
        g_tot += gt
        c_tot += ct
        print(
            f"{uid:>5} {gb:>9.3f}s {gs:>9.3f}s {gt:>8.3f}s "
            f"{cb:>9.3f}s {cs:>9.3f}s {ct:>8.3f}s {ct / gt:>7.1f}x {maxdiff:>10.4f}"
        )
    n = len(users)
    print(
        f"\n  GPU avg: build {g_tot / n:.3f}s total/user "
        f"| CPU(Rust) avg: {c_tot / n:.3f}s total/user | CPU/GPU = {c_tot / g_tot:.1f}x"
    )
    print(
        "  (R_maxdiff = max |target-retention_cpu - target-retention_gpu| over all states)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="1-10")
    ap.add_argument("--deck", type=int, default=10000)
    ap.add_argument("--span", type=int, default=365 * 5)
    ap.add_argument(
        "--bellman", action="store_true", help="Bellman GPU-vs-CPU comparison only"
    )
    args = ap.parse_args()
    a, b = (args.users.split("-") + [args.users])[:2]
    users = list(range(int(a), int(b) + 1))

    if args.bellman:
        bellman_compare(users)
        return

    print(
        f"Device: {'cuda' if CUDA else 'cpu'} | grid S={len(S_GRID)} D={len(D_GRID)} "
        f"states={len(S_GRID) ** 2 * len(D_GRID)} | deck={args.deck} span={args.span}d"
    )
    print("\nSSP-MMC correctness check (user 1, small):")
    ssp_parity()

    print(f"\nProfiling {len(users)} users (deck={args.deck}, span={args.span}d)...")
    print(
        f"{'user':>5} {'build':>7} {'solve':>7} {'sim':>8} {'fsrs':>8} {'gru':>8} "
        f"{'reviews':>9} {'know':>8}"
    )
    rows = []
    for uid in users:
        r = run_one(uid, args.deck, args.span)
        rows.append(r)
        print(
            f"{uid:>5} {r['build']:>6.2f}s {r['solve']:>6.2f}s {r['sim']:>7.2f}s "
            f"{r['fsrs']:>7.2f}s {r['gru']:>7.2f}s {r['reviews']:>9.0f} {r['know']:>8.0f}"
        )

    def mean(k):
        return float(np.mean([r[k] for r in rows]))

    mb, ms, msim, mf, mg = (
        mean("build"),
        mean("solve"),
        mean("sim"),
        mean("fsrs"),
        mean("gru"),
    )
    total = mb + ms + msim
    print("\n--- averages over", len(users), "users ---")
    print(f"  Bellman build : {mb:6.2f}s  ({100 * mb / total:4.1f}%)")
    print(f"  Bellman solve : {ms:6.2f}s  ({100 * ms / total:4.1f}%)")
    print(f"  Sim wall      : {msim:6.2f}s  ({100 * msim / total:4.1f}%)")
    print(f"     FSRS math  : {mf:6.2f}s  (of sim: {100 * mf / msim:4.1f}%)")
    print(f"     GRU infer  : {mg:6.2f}s  (of sim: {100 * mg / msim:4.1f}%)")
    print(f"     other/sim  : {msim - mf - mg:6.2f}s  (RNG, budget, bookkeeping)")
    print(f"  TOTAL/user    : {total:6.2f}s")


if __name__ == "__main__":
    main()
