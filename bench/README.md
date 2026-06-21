# Simulator speedup benchmark protocol

Rigorous, repeatable protocol for accepting (or rejecting) a performance change to the
Rust simulator (`rust/src/simulate.rs`). The goal is to confirm a change is **genuinely
faster** *and* **doesn't change the simulation results** beyond a tight tolerance.

> **Reuse note:** this same protocol is run twice — once for the current **FSRS-6**
> simulator, and again after step 2 once the **FSRS-7** simulator exists. See
> [Re-running for FSRS-7](#re-running-for-fsrs-7) at the bottom.

## Workload (fixed across all iterations)

So that drift checks stay comparable across iterations, the workload never changes within
a campaign:

- **First 50 users** (ids 1–50), each with their **real per-user params**:
  - FSRS params: `srs-benchmark/result/FSRS-6-short-recency.jsonl` (21 params, integer-day
    intervals; FSRS-6 was trained without `--secs`).
  - costs & rating-probs: `Anki-button-usage/button_usage.jsonl`.
- **Policy:** fixed desired retention, **DR ∈ {0.70, 0.90, 0.99}**.
- `parallel=1` per user (so each user yields its **own** wall-clock time),
  `deck_size=10000`, `learn_span=365*5`, `seed=42`.

→ **150 (user, DR) pairs** (3 DR × 50 users).

## Measurement

- **before** = clean `HEAD`; **after** = the working-tree candidate. The harness
  git-stashes `rust/` to build `before`, then builds `after` from the working tree. Both
  are compiled with plain `cargo build --release` and loaded as standalone
  `ssp_mmc_rust.pyd` files.
- The two variants run as **two simultaneous single-threaded processes**, so any external
  factor (thermal, background load) affects both equally.
- CPU frequency is locked (reduces variance). Pinning each process to a dedicated core
  would reduce variance further but is deemed not worth it.
- Per (user, DR) pair: run **3 reps**, keep the **min** time → 150 `before` times and 150
  `after` times. Also record the (deterministic) outputs **memorized**
  (`memorized_cnt.mean()`) and **time_spent** (`cost.sum()`).

## Accept criteria

A change is accepted **iff ALL four** hold:

1. **C1 — speed:** median over the 150 pairs of `time_before / time_after` **> 1**.
2. **C2 — significance:** one-sided Wilcoxon signed-rank test (H1: before > after) over the
   150 paired times, **p < 0.01**.
3. **C3 — avg correctness:** the average absolute drift of **memorized** is **≤ 0.15%**,
   and the same for **time_spent**.
4. **C4 — worst-case correctness:** the max absolute drift (over any single user×DR) of
   **memorized** is **≤ 1%**, and the same for **time_spent**.

**Drift is always measured against iteration 0** (`bench/iter0_reference.json`), never the
previous iteration. This bounds *cumulative* drift across many accepted changes to the
thresholds above (we tolerate ≤0.15% avg / ≤1% worst vs the original simulator, not per
step).

## Running

```bash
# fast pipeline check (tiny workload, no logging, no reference write)
BENCH_SMOKE=1 uv run --no-sync python -m bench.run_iteration --iter 0 --comment smoke

# establish the iteration-0 golden reference (run once, with rust/ at HEAD)
uv run --no-sync python -m bench.run_iteration --iter 0 --comment "baseline"

# each speedup change: edit rust/, then
uv run --no-sync python -m bench.run_iteration --iter N --comment "what changed"
```

The driver prints the verdict and appends a row to `bench/SPEEDUP_LOG.md`
(timestamp, iter#, median speedup, Wilcoxon p — scientific notation if tiny, C1–C4
pass/fail, drift avg/max, comment).

## Workflow per change

1. Edit `rust/src/simulate.rs` (the candidate optimization).
2. `run_iteration.py --iter N --comment "..."`.
3. **ACCEPT** (all of C1–C4): commit the change (so `HEAD` advances and becomes the next
   `before`).
4. **REJECT**: `git checkout -- rust/` to discard the candidate.

## Files

- `_common.py` — fixed workload + per-user data loading.
- `bench_variant.py` — runs one variant over the 150 pairs; dumps JSON.
- `run_iteration.py` — builds before/after, runs them simultaneously, computes stats +
  drift, applies C1–C4, logs.
- `iter0_reference.json` — golden memorized/time_spent reference (committed).
- `SPEEDUP_LOG.md` — the iteration log (committed).
- `_build/` — throwaway build outputs (gitignored).

## Re-running for FSRS-7

After step 2 (FSRS-7 simulator), run the **same** protocol and **same** statistical
criteria on the FSRS-7 simulator, changing only what's model-specific:

- Params file → `srs-benchmark/result/FSRS-7-short-secs-recency.jsonl` (**34** params,
  all **10000** users present; FSRS-6's file is missing user 4371).
- FSRS-7 uses seconds (`--secs`) and models same-day reviews, and memory state gains a
  third variable (short- vs long-term stability) — the simulator and its `make_args`
  change accordingly.
- **Re-establish a fresh iteration-0 reference** (it is specific to the FSRS-7 simulator;
  do not compare FSRS-7 results against the FSRS-6 reference).
