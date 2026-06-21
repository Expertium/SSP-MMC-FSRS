"""Speedup iteration driver: benchmark `before` (HEAD) vs `after` (working tree).

    uv run --no-sync python -m bench.run_iteration --iter N --comment "what changed"

Steps:
  1. Build `before` from clean HEAD (git-stash the rust/ change, build, unstash) and
     `after` from the working tree -- both as standalone ssp_mmc_rust.pyd files.
     (If rust/ is unchanged, before == after; used for the iteration-0 baseline.)
  2. Run both variants as two concurrent single-threaded processes (so external load
     hits both equally), each producing 150 (user, DR) results.
  3. Stats: median of the 150 (time_before/time_after) ratios; one-sided Wilcoxon
     (before > after). Drift: memorized & time_spent of `after` vs the iteration-0
     reference (established on the first run / --iter 0).
  4. Accept iff ALL: (C1) median ratio > 1; (C2) Wilcoxon p < 0.01; (C3) avg |drift|
     <= 0.15% for both memorized and time_spent; (C4) max |drift| <= 1% for both.
  5. Append a row to bench/SPEEDUP_LOG.md (skipped in BENCH_SMOKE mode).

The script does NOT commit/revert; it reports ACCEPT/REJECT so the caller decides.
"""

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from bench import _common

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
BUILD = ROOT / "bench" / "_build"
LOG = ROOT / "bench" / "SPEEDUP_LOG.md"
REF = ROOT / "bench" / ("_build/smoke_ref.json" if _common.SMOKE else "iter0_reference.json")
DLL = ROOT / "rust" / "target" / "release" / "ssp_mmc_rust.dll"


def _run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def cargo_build():
    env = os.environ.copy()
    env["PYO3_PYTHON"] = str(VENV_PY)
    env["CARGO_BUILD_JOBS"] = "1"
    # cwd=rust/ so cargo picks up rust/.cargo/config.toml (config is discovered from the
    # working dir upward, not the manifest dir).
    r = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=ROOT / "rust",
        env=env,
    )
    if r.returncode != 0:
        sys.exit(f"cargo build failed (exit {r.returncode})")


def build_to(dirpath):
    cargo_build()
    dirpath.mkdir(parents=True, exist_ok=True)
    shutil.copy(DLL, dirpath / "ssp_mmc_rust.pyd")


def build_variants():
    """Return (before_dir, after_dir). before=HEAD via git stash; after=working tree."""
    before_dir, after_dir = BUILD / "before", BUILD / "after"
    # porcelain catches untracked files too (e.g. a new rust/.cargo/config.toml), which
    # `git diff` would miss; gitignored paths (rust/target) are excluded by default.
    changed = _run(["git", "status", "--porcelain", "--", "rust"]).stdout.strip() != ""
    if not changed:
        print("rust/ unchanged -> before == after (baseline run)")
        build_to(after_dir)
        return after_dir, after_dir
    print("stashing rust/ to build `before` from HEAD ...")
    if _run(["git", "stash", "push", "--include-untracked", "--", "rust"]).returncode != 0:
        sys.exit("git stash failed")
    try:
        build_to(before_dir)
    finally:
        if _run(["git", "stash", "pop"]).returncode != 0:
            sys.exit("git stash pop failed -- resolve manually")
    print("building `after` from working tree ...")
    build_to(after_dir)
    return before_dir, after_dir


def run_both(before_dir, after_dir):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    out_b, out_a = BUILD / "before.json", BUILD / "after.json"

    def launch(pyd_dir, out):
        return subprocess.Popen(
            [str(VENV_PY), "-m", "bench.bench_variant", str(pyd_dir), str(out)],
            cwd=ROOT,
            env=env,
        )

    pb, pa = launch(before_dir, out_b), launch(after_dir, out_a)  # simultaneous
    rb, ra = pb.wait(), pa.wait()
    if rb or ra:
        sys.exit(f"variant run failed (before={rb}, after={ra})")
    return json.load(open(out_b)), json.load(open(out_a))


def drift_pct(after_vals, ref_vals):
    a, r = np.asarray(after_vals), np.asarray(ref_vals)
    pct = np.abs(a - r) / np.maximum(np.abs(r), 1e-12) * 100.0
    return float(pct.mean()), float(pct.max())


def fmt_p(p):
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--comment", default="")
    args = ap.parse_args()

    before_dir, after_dir = build_variants()
    before, after = run_both(before_dir, after_dir)

    tb, ta = np.asarray(before["times"]), np.asarray(after["times"])
    ratios = tb / ta
    median_ratio = float(np.median(ratios))
    try:
        pval = float(wilcoxon(tb, ta, alternative="greater").pvalue)
    except ValueError:
        pval = 1.0  # all-zero differences (identical) -> no significance

    baseline = args.iter == 0 or not REF.exists()
    if baseline:
        json.dump(
            {
                "order": after["order"],
                "memorized": after["memorized"],
                "time_spent": after["time_spent"],
            },
            open(REF, "w"),
        )
    ref = json.load(open(REF))
    mem_avg, mem_max = drift_pct(after["memorized"], ref["memorized"])
    tsp_avg, tsp_max = drift_pct(after["time_spent"], ref["time_spent"])

    c1 = median_ratio > 1.0
    c2 = pval < 0.01
    c3 = mem_avg <= 0.15 and tsp_avg <= 0.15
    c4 = mem_max <= 1.0 and tsp_max <= 1.0
    accept = c1 and c2 and c3 and c4

    ck = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    print("\n================ iteration", args.iter, "================")
    print(f"median speedup (before/after): {median_ratio:.4f}x")
    print(f"wilcoxon p (before>after)    : {fmt_p(pval)}")
    print(f"memorized drift vs iter0     : avg {mem_avg:.4f}%  max {mem_max:.4f}%")
    print(f"time_spent drift vs iter0    : avg {tsp_avg:.4f}%  max {tsp_max:.4f}%")
    print(f"C1 ratio>1: {ck(c1)}  C2 p<0.01: {ck(c2)}  C3 avg<=0.15%: {ck(c3)}  C4 max<=1%: {ck(c4)}")
    print("BASELINE (iter0 reference established)" if baseline else f"DECISION: {'ACCEPT' if accept else 'REJECT'}")

    if _common.SMOKE:
        print("[SMOKE mode: not logging]")
        return

    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not LOG.exists():
        LOG.write_text(
            "# Speedup iteration log\n\n"
            f"Workload: {_common.N_USERS} users x DR {list(_common.DRS)}, parallel=1, "
            f"deck={_common.DECK_SIZE}, span={_common.LEARN_SPAN}, seed={_common.SEED}; "
            "per pair = min of 3 timed reps; before=HEAD vs after=candidate, run "
            "simultaneously (1 thread each). Accept iff C1 median(before/after)>1, "
            "C2 Wilcoxon p<0.01, C3 avg drift<=0.15%, C4 max drift<=1% (memorized & time_spent vs iter0).\n\n"
            "| timestamp | iter | speedup (med) | wilcoxon p | mem drift avg/max % | time drift avg/max % | C1 | C2 | C3 | C4 | accept | comment |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        )
    decision = "baseline" if baseline else ("ACCEPT" if accept else "REJECT")
    row = (
        f"| {ts} | {args.iter} | {median_ratio:.3f}x | {fmt_p(pval)} | "
        f"{mem_avg:.3f}/{mem_max:.3f} | {tsp_avg:.3f}/{tsp_max:.3f} | "
        f"{ck(c1)} | {ck(c2)} | {ck(c3)} | {ck(c4)} | {decision} | {args.comment} |\n"
    )
    with open(LOG, "a") as f:
        f.write(row)
    print(f"logged to {LOG}")


if __name__ == "__main__":
    main()
