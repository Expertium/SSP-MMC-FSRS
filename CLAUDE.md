# CLAUDE.md — SSP-MMC-FSRS project guide

This file orients an AI agent (and Andrew) working in this repo. It captures **what
the project is**, **where the important code lives**, the **roadmap** we're working
through, and the **project rules** (tooling, commits) every change must follow.

## What this project is

SSP-MMC-FSRS finds review schedules that minimize study time for a target amount of
knowledge. It treats scheduling as a stochastic shortest-path / Markov Decision Process:
given FSRS's memory-state transition function (stability, difficulty), a **Bellman
solver** computes, for every (stability, difficulty) state, the next interval that
minimizes long-run review cost. We then **simulate** review histories under different
scheduling policies (SSP-MMC, fixed desired retention, fixed intervals, Memrise,
Anki-SM-2) and compare them on knowledge-vs-workload tradeoffs.

See [README.md](README.md) for the policy descriptions, the results table, and the
current caveats (e.g. "FSRS is assumed perfectly accurate at predicting recall" — a
caveat this roadmap is specifically designed to relax via a learned p(recall) predictor).

## Repo orientation (key files)

- `src/ssp_mmc_fsrs/simulation.py` — `simulate()`: the PyTorch-vectorized simulator.
  Runs `parallel` independent decks at once on `device`. **Currently inlines the
  FSRS-6 memory model** (`w[0..20]`: `stability_after_success/failure`,
  `stability_short_term`, difficulty updates, `power_forgetting_curve`). This is the
  thing to port to Rust (step 1) and upgrade to FSRS-7 (step 2).
- `src/ssp_mmc_fsrs/solver.py` — `SSPMMCSolver`: the Bellman/value-iteration solver.
  Holds `discount_factor=0.97`. Convergence behavior here is the subject of step 3.
- `src/ssp_mmc_fsrs/policies.py` — scheduling policies (SSP-MMC, DR, intervals, Memrise, SM-2).
- `src/ssp_mmc_fsrs/core.py` — shared FSRS math (forgetting curve, etc.).
- `src/ssp_mmc_fsrs/{config,io}.py` — defaults (costs, rating probs, `S_MIN`) and IO.
- `experiments/simulate.py` — main simulation runner (per-user, writes plots + JSON).
- `experiments/hyperparameter_optimizer.py` — SSP-MMC hyperparameter search. **Already
  uses `ax-platform`** (BoTorch/Bayesian) — this is the Bayesian tuner step 5 builds on.
- `experiments/generate_ssp_mmc_policies.py` — builds SSP-MMC policies + surface plots.
- `experiments/converge.py` — convergence checks (step 3). Defaults read FSRS params
  from `../srs-benchmark/result/FSRS-6-recency.jsonl` and button usage from
  `../Anki-button-usage/button_usage.jsonl`; writes `unconverged_users.json`.
- `outputs/` — generated plots, simulations, policies, checkpoints (gitignored work).

## Roadmap

Worked roughly in order; each step is a milestone. **Always verify a re-implementation
matches the reference before moving on.**

### 1. Re-implement the simulator in Rust (for speed)
Port `simulate()` from `simulation.py` to Rust. **Verify it matches the Python
simulator** — same seed/inputs must produce the same per-day review/learn/memorized/cost
arrays (within floating-point tolerance). The Python version is the source of truth for
correctness; Rust is purely for speed.

### 2. Implement FSRS-7
Replace the inlined FSRS-6 memory model with FSRS-7 (in both the Python and Rust
simulators).

**FSRS-7 is a bigger model:** 34 parameters (vs FSRS-6's 21) and — critically — it tracks
**short-term and long-term stability separately, so memory state has a third variable**
(short-term stability, long-term stability, difficulty) instead of two. In the simulator
that's just extra state to carry, but it blows up the Bellman solver's state space (step 3).

References:
- **Python FSRS implementation:** `C:\Users\Andrew\srs-benchmark`
- **Rust FSRS implementation:** https://github.com/Expertium/fsrs-rs-speed-autoresearch

**Same-day reviews:** the simulator currently does **not** model same-day reviews —
intervals are forced to be at least one day (README caveat #2). With **FSRS-7, same-day
reviews should be simulated.** FSRS-7's short-term memory model handles within-day
repetitions, so the simulator must be extended to allow multiple reviews of a card on the
same day rather than clamping every interval to ≥ 1 day.

### 3. Investigate Bellman solver convergence under FSRS-7
Using `experiments/converge.py`, measure **for how many users the Bellman solver fails
to converge** with FSRS-7. Then see whether the unconverged count can be reduced **by
means other than lowering `discount_factor` (currently 0.97 in `solver.py`)** — that
knob is off-limits as the fix; we want to understand and address the root cause.

**State-space blowup:** FSRS-7's third state variable (short- vs long-term stability)
turns the solver's 2-D `(stability, difficulty)` grid into a **3-D** grid — far more
expensive in time and memory, which compounds the convergence problem and limits how fine
the grid can be.

### 4. Add GRU + LSTM pseudo-ground-truth recall predictors
Grab the **GRU** and **LSTM** model architectures + **global pretrained weights** from
https://github.com/open-spaced-repetition/srs-benchmark (`pretrain/`). **Per-user params
are not published** — we re-generate them ourselves from the pretrained weights (GRU on
1k users for step 5, LSTM on 10k for step 6); see the "Per-user optimized/trained
parameters" section below.

Key architectural decision: **split the scheduler from the recall predictor.**
- **Scheduler:** FSRS-7 (decides intervals).
- **Pseudo-ground-truth p(recall):** a learned model (GRU/LSTM) decides, inside the
  simulator, whether a card was actually recalled or forgotten — instead of assuming
  FSRS's own predicted recall is ground truth (relaxes README caveat #1).

GRU is used during hyperparameter tuning (step 5); LSTM during final evaluation (step 6).

### 5. Optimize SSP-MMC hyperparameters (Pareto front vs fixed DR)
Find **several hyperparameter sets that Pareto-beat fixed desired retention** on the
**total-knowledge vs workload** tradeoff. Requires:
- A **Bayesian hyperparameter tuner** (build on the existing `ax-platform` setup).
- **Many simulations on 1k users**, with **GRU** as the pseudo-ground-truth p(recall).
  Raw data: `C:\Users\Andrew\anki-revlogs-10k`.
- **Serious performance work** so this finishes in reasonable wall-clock time. Preferred:
  **parallelize the simulations on CUDA** (see https://github.com/Expertium/fsrs-autoresearch
  for inspiration). If full CUDA isn't feasible, the **Rust** simulator (step 1) should
  at least give a large speedup over Python.

### 6. Full evaluation on 10k users
Evaluate the good hyperparameter sets from step 5 on **10k users**, using **LSTM** as the
pseudo-ground-truth p(recall) predictor.

## Experiment design summary

| Phase | Users | Scheduler | Pseudo-GT p(recall) predictor | Tuner |
| --- | --- | --- | --- | --- |
| Hyperparameter optimization (step 5) | 1k | FSRS-7 | **GRU** | Bayesian (ax-platform) |
| Full evaluation (step 6) | 10k | FSRS-7 | **LSTM** | — (evaluate only) |

Rationale: tune on a cheaper 1k-user set with GRU, then validate the surviving Pareto
candidates on the full 10k-user set with a *different* predictor (LSTM) to guard against
overfitting the tuner to one predictor's quirks.

## Simulator parallelization (the `parallel` axis = users)

Design decision: repurpose `simulate()`'s `parallel` dimension from "N identical
Monte-Carlo replicas of ONE user" to "N **different users** simulated at once", each with
their own FSRS params + per-user costs/rating-probs. Per-user estimates get noisier (1
deck instead of `PARALLEL` decks per user), but step 5's objective is the **mean**
memorized/workload over 1k users, so the per-user noise averages out across users.

Implications:
- `simulate()` (Python + Rust) must accept **per-deck (per-user) parameter arrays**:
  `w` → `(parallel, n_w)`, costs → `(parallel, 4)`, probs → `(parallel, 4)`/`(parallel, 3)`.
  The step-1 Rust port is built to take per-deck params from the start; its parity test
  just broadcasts one user across all decks so it still matches the current shared-param
  Python.
- The **hard part is parallelizing the Bellman solve** (`solver.py`) across users — each
  user needs their own SSP-MMC policy table. The simulator only *consumes* per-user
  policies; batched value-iteration across a `users` axis is the open performance problem
  for steps 3/5, not something the step-1 simulator port solves.
- VRAM: state tensors are `parallel × deck_size`. 1k users × 10k cards fits the 4070; 10k
  users likely needs batching (e.g. 1k-user chunks).

## Key paths & external resources

- Raw Anki review logs (10k users): `C:\Users\Andrew\anki-revlogs-10k`
- Python FSRS / GRU / LSTM reference + benchmark: `C:\Users\Andrew\srs-benchmark`
  and https://github.com/open-spaced-repetition/srs-benchmark
- Rust FSRS reference: https://github.com/Expertium/fsrs-rs-speed-autoresearch
- CUDA-parallel simulation inspiration: https://github.com/Expertium/fsrs-autoresearch
- FSRS params used by `converge.py`: `../srs-benchmark/result/FSRS-6-recency.jsonl`
  (after step 2 this becomes the FSRS-7 params file — use
  `../srs-benchmark/result/FSRS-7-short-secs-recency.jsonl`)

### Per-user optimized/trained parameters
- **FSRS-7** (per-user optimized parameters): already optimized — **we do NOT train
  these ourselves.** Use the param files in `C:\Users\Andrew\srs-benchmark\result` (the
  local clone of https://github.com/open-spaced-repetition/srs-benchmark/tree/main/result),
  specifically the params from **`FSRS-7-short-secs-recency.jsonl`** in that folder.
  Schema: one JSON record per line, keyed by `user` (ids 1–10000); the params are under
  `parameters["0"]` and there are **34** of them — note this differs from FSRS-6's 21
  (`w[0..20]`), which matters for step 2.
- **Per-user simulator inputs (costs & rating probabilities):** already computed for all
  10k users in `../Anki-button-usage/button_usage.jsonl` (keyed by `user`, ids 1–10000):
  per-user `learn_costs` (4), `review_costs` (4), `first_rating_prob` (4),
  `review_rating_prob` (3), plus offsets/session-lens and transition matrices. **No need
  to recompute from raw revlogs and risk being off** — these are the official values, and
  `converge.py`/`lib.py` already default to this path. The producer is `script.py` in that
  repo (reads `../anki-revlogs-10k/revlogs`; to regenerate, run with `--max-workers 1`);
  `analysis.ipynb` is only a cross-user summary, **not** the producer.
- **GRU / LSTM**: per-user params are **NOT published anywhere** — the only thing
  available is the *global pretrained* weights at
  https://github.com/open-spaced-repetition/srs-benchmark/tree/main/pretrain (NOT in
  `result/`). So we must **re-generate user-specific params ourselves**, starting from
  those pretrained weights: train **GRU on 1k users** (step 5) and **LSTM on 10k users**
  (step 6). This is real training work, not a download.

## Working conventions

### Project rules
1. Manage the project environment with **`uv`**: `uv sync` (base) /
   `uv sync --extra experiments` (experiments need the extra dependencies).
2. When code changes affect the README examples, **update `README.md`**.
3. Before each commit, run **`uv run ruff format`**.
4. Commit messages must follow **Conventional Commits** (see below).

### Conventional Commits
Structure a commit message as:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

- **`fix:`** — patches a bug (SemVer `PATCH`).
- **`feat:`** — introduces a new feature (SemVer `MINOR`).
- **`BREAKING CHANGE:`** — a footer `BREAKING CHANGE:`, or a `!` after the type/scope,
  signals a breaking change (SemVer `MAJOR`). Can accompany any type.
- Other types are allowed and have no SemVer effect on their own — e.g. `build:`,
  `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`.
- A **scope** may be added in parentheses for context, e.g. `feat(parser): add array parsing`.
- Footers other than `BREAKING CHANGE: <description>` follow git trailer format.

### Git remotes
- `origin` = your fork (`Expertium/SSP-MMC-FSRS`) — push your work here.
- `upstream` = `open-spaced-repetition/SSP-MMC-FSRS` — pull updates from here.

### Definition of done for re-implementations
When a re-implementation is meant to match a reference (Rust↔Python, FSRS-7↔srs-benchmark),
treat **numerical agreement as the definition of done** — write the comparison and run it.
