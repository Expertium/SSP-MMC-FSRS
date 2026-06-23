"""FSRS-7 whole-sim speedup iteration driver: `before` (champion HEAD) vs `after` (tree).

    uv run --no-sync python -m bench7.run_iteration7 --iter N --comment "what changed"

Protocol (see memory sim7-speedup-protocol):
  1. Build `before` from champion HEAD (git-stash tracked changes in rust/ + src/, build,
     unstash) and `after` from the working tree -- each a standalone ssp_mmc_rust.pyd. The
     pure-Python package is editable, so stashing src/ is what makes `before` use HEAD's
     Python; the pyd rebuild is for Rust changes. (No change -> before == after = baseline.)
  2. Run the two variants SEQUENTIALLY (not in parallel): the GPU is ~100% utilised by a
     single user's Bellman, so concurrent runs would contend and corrupt timing. Each
     produces 60 (hp, user) datapoints.
  3. Speed: median of the 60 (time_before/time_after) ratios; one-sided Wilcoxon
     (before > after). Correctness: knowledge & time_spent of `after` vs the ITER-0
     reference (not the champion), so cumulative drift stays bounded.
  4. Accept iff ALL: (C1) median ratio > 1; (C2) Wilcoxon p < 0.01; and the 4 correctness
     checks -- avg |drift| <= 0.5% and max |drift| <= 2.5% for BOTH knowledge and
     time_spent. Canary: report if time_spent blows its threshold while knowledge does not.
  5. Append a row to bench7/SPEEDUP_LOG7.md (skipped in BENCH7_SMOKE mode).

Reports ACCEPT/REJECT; the caller commits (accept) or reverts (reject).
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

from bench7 import _common7

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
BUILD = ROOT / "bench7" / "_build"
LOG = ROOT / "bench7" / "SPEEDUP_LOG7.md"
REF = ROOT / "bench7" / ("_build/smoke_ref.json" if _common7.SMOKE else "iter0_reference.json")
DLL = ROOT / "rust" / "target" / "release" / "ssp_mmc_rust.dll"

AVG_THRESH = 0.5  # % avg drift, knowledge & time_spent
MAX_THRESH = 2.5  # % max drift


def _run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def cargo_build():
    env = os.environ.copy()
    env["PYO3_PYTHON"] = str(VENV_PY)
    env["CARGO_BUILD_JOBS"] = "2"
    # cwd=rust/ so cargo picks up rust/.cargo/config.toml (discovered from the working dir).
    r = subprocess.run(["cargo", "build", "--release"], cwd=ROOT / "rust", env=env)
    if r.returncode != 0:
        sys.exit(f"cargo build failed (exit {r.returncode})")


def build_to(dirpath):
    cargo_build()
    dirpath.mkdir(parents=True, exist_ok=True)
    shutil.copy(DLL, dirpath / "ssp_mmc_rust.pyd")


def build_variants():
    """Return (before_dir, after_dir). before=HEAD via git stash of rust/+src/; after=tree."""
    before_dir, after_dir = BUILD / "before", BUILD / "after"
    changed = _run(["git", "status", "--porcelain", "--", "rust", "src"]).stdout.strip() != ""
    if not changed:
        print("rust/ + src/ unchanged -> before == after (baseline run)")
        build_to(after_dir)
        return after_dir, after_dir
    print("stashing rust/ + src/ to build `before` from HEAD ...")
    if _run(["git", "stash", "push", "--include-untracked", "--", "rust", "src"]).returncode != 0:
        sys.exit("git stash failed")
    try:
        build_to(before_dir)
    finally:
        if _run(["git", "stash", "pop"]).returncode != 0:
            sys.exit("git stash pop failed -- resolve manually")
    print("building `after` from working tree ...")
    build_to(after_dir)
    return before_dir, after_dir


def run_variant(pyd_dir, out, tag):
    """Run one variant to completion (sequential -- never concurrent with the other)."""
    env = os.environ.copy()
    print(f"running `{tag}` ...")
    r = subprocess.run(
        [str(VENV_PY), "-m", "bench7.bench_variant7", str(pyd_dir), str(out)],
        cwd=ROOT,
        env=env,
    )
    if r.returncode:
        sys.exit(f"`{tag}` variant run failed (exit {r.returncode})")
    return json.load(open(out))


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
    # SEQUENTIAL: before first, then after (no GPU contention).
    before = run_variant(before_dir, BUILD / "before.json", "before")
    after = run_variant(after_dir, BUILD / "after.json", "after")

    tb, ta = np.asarray(before["times"]), np.asarray(after["times"])
    sb, sa = np.asarray(before["sim_times"]), np.asarray(after["sim_times"])
    ratios = tb / ta
    median_ratio = float(np.median(ratios))
    sim_median_ratio = float(np.median(sb / sa))
    try:
        pval = float(wilcoxon(tb, ta, alternative="greater").pvalue)
    except ValueError:
        pval = 1.0  # all-zero differences (identical) -> no significance

    baseline = args.iter == 0 or not REF.exists()
    if baseline:
        json.dump(
            {
                "order": after["order"],
                "knowledge": after["knowledge"],
                "time_spent": after["time_spent"],
            },
            open(REF, "w"),
        )
    ref = json.load(open(REF))
    know_avg, know_max = drift_pct(after["knowledge"], ref["knowledge"])
    tsp_avg, tsp_max = drift_pct(after["time_spent"], ref["time_spent"])

    c1 = median_ratio > 1.0
    c2 = pval < 0.01
    know_ok = know_avg <= AVG_THRESH and know_max <= MAX_THRESH
    tsp_ok = tsp_avg <= AVG_THRESH and tsp_max <= MAX_THRESH
    accept = c1 and c2 and know_ok and tsp_ok
    # canary: time_spent drifts past threshold while knowledge stays within.
    canary = (not tsp_ok) and know_ok

    ck = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    print("\n================ iteration", args.iter, "================")
    print(f"datapoints: {len(tb)}  (deck={_common7.DECK}, span={_common7.SPAN}, reps={_common7.REPS})")
    print(f"median speedup whole-pipeline (before/after): {median_ratio:.4f}x")
    print(f"median speedup sim-only                     : {sim_median_ratio:.4f}x")
    print(f"wilcoxon p (before>after)                   : {fmt_p(pval)}")
    print(f"knowledge  drift vs iter0 : avg {know_avg:.4f}%  max {know_max:.4f}%  (<= {AVG_THRESH}/{MAX_THRESH})")
    print(f"time_spent drift vs iter0 : avg {tsp_avg:.4f}%  max {tsp_max:.4f}%  (<= {AVG_THRESH}/{MAX_THRESH})")
    print(
        f"C1 ratio>1: {ck(c1)}  C2 p<0.01: {ck(c2)}  "
        f"knowledge<thr: {ck(know_ok)}  time_spent<thr: {ck(tsp_ok)}"
    )
    if canary:
        print("CANARY: time_spent blew its threshold while knowledge stayed within "
              "-> the schedule's workload drifted (knowledge saturates, so it hid it).")
    print("BASELINE (iter0 reference established)" if baseline else f"DECISION: {'ACCEPT' if accept else 'REJECT'}")

    if _common7.SMOKE:
        print("[SMOKE mode: not logging]")
        return

    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not LOG.exists():
        LOG.write_text(
            "# FSRS-7 whole-sim speedup log\n\n"
            f"Workload: {_common7.N_USERS} users x {len(_common7.HP_SETS)} hp = "
            f"{_common7.N_USERS * len(_common7.HP_SETS)} datapoints, deck={_common7.DECK}, "
            f"span={_common7.SPAN}, seed={_common7.SEED}; per datapoint = min of "
            f"{_common7.REPS} reps; before=champion HEAD vs after=candidate, run SEQUENTIALLY. "
            "Accept iff C1 median(before/after)>1, C2 Wilcoxon p<0.01, and avg drift<=0.5% & "
            "max drift<=2.5% for BOTH knowledge and time_spent vs iter0.\n\n"
            "| timestamp | iter | speedup (med) | sim-only | wilcoxon p | know drift avg/max % | time drift avg/max % | C1 | C2 | know | time | accept | comment |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        )
    decision = "baseline" if baseline else ("ACCEPT" if accept else "REJECT")
    if canary and not baseline:
        decision += " (CANARY)"
    row = (
        f"| {ts} | {args.iter} | {median_ratio:.3f}x | {sim_median_ratio:.3f}x | {fmt_p(pval)} | "
        f"{know_avg:.3f}/{know_max:.3f} | {tsp_avg:.3f}/{tsp_max:.3f} | "
        f"{ck(c1)} | {ck(c2)} | {ck(know_ok)} | {ck(tsp_ok)} | {decision} | {args.comment} |\n"
    )
    with open(LOG, "a") as f:
        f.write(row)
    print(f"logged to {LOG}")


if __name__ == "__main__":
    main()
