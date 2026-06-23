"""Validate + benchmark across-user rayon in simulate_fsrs7 (the parallel=N path).

    uv run --no-sync python -m bench7.validate_parallel

The 60-datapoint per-datapoint protocol (run_iteration7) runs parallel=1, so it can't see
across-user parallelism. This validates that change differently:

  1. Build the per-user inputs ONCE for all 60 datapoints (Bellman solve on GPU -> retention
     table, GRU weights, w/costs/probs), stacked into (60, ...) arrays.
  2. Build `before` (champion HEAD, serial user loop) and `after` (working tree, rayon) as
     standalone pyds (git-stash rust/+src/), same as run_iteration7.
  3. Run EACH variant with parallel=60 (all users in one call). rayon only reorders the
     independent per-user work and the RNG is keyed by the global cell index, so the outputs
     must be **bit-exact** between serial and parallel.
  4. Report: bit-exactness (review/learn exact, memorized/cost ~0) and speedup =
     time_serial / time_parallel.

ACCEPT iff outputs are bit-exact AND speedup > 1 (we expect a large multiple on the 5950X).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from bench7 import _common7

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
BUILD = ROOT / "bench7" / "_build"
DLL = ROOT / "rust" / "target" / "release" / "ssp_mmc_rust.dll"
REPS = 1 if _common7.SMOKE else 2


def _run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def cargo_build():
    env = os.environ.copy()
    env["PYO3_PYTHON"] = str(VENV_PY)
    env["CARGO_BUILD_JOBS"] = "2"
    r = subprocess.run(["cargo", "build", "--release"], cwd=ROOT / "rust", env=env)
    if r.returncode != 0:
        sys.exit(f"cargo build failed (exit {r.returncode})")


def build_to(dirpath):
    cargo_build()
    dirpath.mkdir(parents=True, exist_ok=True)
    shutil.copy(DLL, dirpath / "ssp_mmc_rust.pyd")


def build_variants():
    before_dir, after_dir = BUILD / "before", BUILD / "after"
    changed = _run(["git", "status", "--porcelain", "--", "rust", "src"]).stdout.strip() != ""
    if not changed:
        print("rust/ + src/ unchanged -> before == after (baseline; speedup ~1 expected)")
        build_to(after_dir)
        return after_dir, after_dir
    print("stashing rust/ + src/ to build `before` (serial) from HEAD ...")
    if _run(["git", "stash", "push", "--include-untracked", "--", "rust", "src"]).returncode != 0:
        sys.exit("git stash failed")
    try:
        build_to(before_dir)
    finally:
        if _run(["git", "stash", "pop"]).returncode != 0:
            sys.exit("git stash pop failed -- resolve manually")
    print("building `after` (rayon) from working tree ...")
    build_to(after_dir)
    return before_dir, after_dir


def build_stacked_inputs(npz_path):
    """Solve the Bellman (GPU) for all 60 datapoints and stack the per-user sim inputs."""
    import torch

    from ssp_mmc_fsrs.gru import BatchedGRU
    from ssp_mmc_fsrs.solver7 import (
        SSPMMCSolver7,
        build_hybrid_s_grid,
        build_production_d_grid,
    )

    S_GRID = build_hybrid_s_grid()
    D_GRID = build_production_d_grid()
    users = _common7.load_users()
    by_uid = {u["uid"]: u for u in users}
    order = _common7.datapoint_order()
    print(f"building stacked inputs for {len(order)} datapoints (Bellman solve on GPU) ...")

    W, LC, RC, FRP, RRP, GRU, RET = [], [], [], [], [], [], []
    gru_cache = {}
    for hi, uid in order:
        u = by_uid[uid]
        hp = _common7.HP_SETS[hi]
        solver = SSPMMCSolver7(
            review_costs=u["rc"],
            first_rating_prob=u["frp"],
            review_rating_prob=u["rrp"],
            w=u["w"],
            s_state=S_GRID,
            d_state=D_GRID,
        )
        _, rm = solver.solve(hp, verbose=False)
        if uid not in gru_cache:
            g = BatchedGRU.from_pth_paths(
                [str(_common7.WDIR / f"user_{uid}.pth")], device="cpu", dtype=torch.float64
            )
            gru_cache[uid] = g.flat_weights().numpy().reshape(-1)  # (1,505) -> (505,)
        W.append(u["w"])
        LC.append(u["lc"])
        RC.append(u["rc"])
        FRP.append(u["frp"])
        RRP.append(u["rrp"])
        GRU.append(gru_cache[uid])
        RET.append(np.asarray(rm).reshape(-1))

    np.savez(
        npz_path,
        N=len(order),
        deck=_common7.DECK,
        span=_common7.SPAN,
        seed=_common7.SEED,
        w=np.ascontiguousarray(np.array(W, np.float64)),
        lc=np.ascontiguousarray(np.array(LC, np.float64)),
        rc=np.ascontiguousarray(np.array(RC, np.float64)),
        frp=np.ascontiguousarray(np.array(FRP, np.float64)),
        rrp=np.ascontiguousarray(np.array(RRP, np.float64)),
        gru=np.ascontiguousarray(np.array(GRU, np.float64)),
        ret=np.ascontiguousarray(np.array(RET, np.float64)),
        s_grid=np.ascontiguousarray(np.asarray(S_GRID, np.float64)),
        d_grid=np.ascontiguousarray(np.asarray(D_GRID, np.float64)),
        max_cost=_common7.MAX_COST,
        learn_limit=_common7.LEARN_LIMIT,
        review_limit=_common7.REVIEW_LIMIT,
        s_min=_common7.S_MIN,
        s_max=_common7.S_MAX,
        max_same_day=_common7.MAX_SAME_DAY,
        n_iter=_common7.N_ITER,
    )


def run_worker(pyd_dir, npz_path, out, tag):
    print(f"running `{tag}` (parallel={_common7.N_USERS * len(_common7.HP_SETS)}) ...")
    r = subprocess.run(
        [str(VENV_PY), "-m", "bench7._pvariant", str(pyd_dir), str(npz_path), str(out), str(REPS)],
        cwd=ROOT,
    )
    if r.returncode:
        sys.exit(f"`{tag}` run failed (exit {r.returncode})")
    return json.load(open(out))


def main():
    before_dir, after_dir = build_variants()
    npz = BUILD / "stacked_inputs.npz"
    BUILD.mkdir(parents=True, exist_ok=True)
    build_stacked_inputs(npz)

    # SEQUENTIAL: serial first, then rayon (no GPU here, but keep them off each other).
    before = run_worker(before_dir, npz, BUILD / "p_before.json", "before (serial)")
    after = run_worker(after_dir, npz, BUILD / "p_after.json", "after (rayon)")

    def arr(d, k):
        return np.asarray(d[k], dtype=np.float64)

    rev_exact = np.array_equal(arr(before, "review"), arr(after, "review"))
    lrn_exact = np.array_equal(arr(before, "learn"), arr(after, "learn"))
    mem_max = float(np.abs(arr(before, "memorized") - arr(after, "memorized")).max())
    cost_max = float(np.abs(arr(before, "cost") - arr(after, "cost")).max())
    bit_exact = rev_exact and lrn_exact and mem_max == 0.0 and cost_max == 0.0

    speedup = before["time"] / after["time"]
    n = _common7.N_USERS * len(_common7.HP_SETS)
    print("\n================ across-user rayon validation ================")
    print(f"batched parallel={n}  (deck={_common7.DECK}, span={_common7.SPAN}, reps={REPS})")
    print(f"serial   time: {before['time']:.3f}s")
    print(f"rayon    time: {after['time']:.3f}s")
    print(f"SPEEDUP (serial/rayon): {speedup:.2f}x")
    print("\nbit-exactness (serial vs rayon, same RNG cells):")
    print(f"  review exact : {rev_exact}")
    print(f"  learn  exact : {lrn_exact}")
    print(f"  memorized max|diff| : {mem_max:.2e}")
    print(f"  cost      max|diff| : {cost_max:.2e}")
    ok = bit_exact and speedup > 1.0
    print(f"\n=> {'PASS (bit-exact + faster)' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
