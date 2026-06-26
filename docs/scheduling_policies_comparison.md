# SSP-MMC vs ADR vs MARC

Three ways to decide **how long to wait before showing a card again**. All three sit on top of
the *same* FSRS-7 memory model and are compared in the *same* simulator against the *same*
baseline (fixed desired retention). They differ in **what objective they optimize** and **how
much freedom they have to shape the schedule**.

This document explains each one twice: first in plain terms, then technically. It assumes you've
seen the project's framing — we judge a policy by the **knowledge-vs-workload tradeoff**:

- **Knowledge** = the average, over every day of the simulation, of `Σ_cards p(recall)` (how many
  cards you'd remember if tested right now). Higher is better.
- **Workload** = average minutes of study per day. Lower is better.

A policy is "good" if, at the same knowledge, it needs *less* study time than fixed desired
retention — or equivalently, more knowledge for the same time.

---

## TL;DR

| | **Fixed DR** (baseline) | **SSP-MMC** | **Cost ADR** | **MARC** |
| --- | --- | --- | --- | --- |
| **One-line** | Always aim for the same recall % | Bellman-optimal schedule for a *proxy* cost | A small formula for the recall % to aim for | Bellman-optimal schedule for the *real* objective |
| **What it decides** | A single retention target R, same for every card | A target R **per memory state** | A target R **per (stability, difficulty)** | A target R **per memory state** |
| **How it's computed** | Pick one number | Value iteration: *minimize* cost-to-reach-S_MAX | Fit 15 coefficients by evolutionary search | Value iteration: *maximize* `recall − λ·time` |
| **Objective it optimizes** | — (you set R by hand) | A **proxy** (cost to reach max stability) | The **real** knowledge-vs-time metric | The **real** knowledge-vs-time metric |
| **Capacity** (how flexible) | Lowest (1 number) | Highest (full table) | Low (smooth 15-coef surface) | Highest (full table) |
| **The workload knob** | R (≈0.70–0.99) | 13 cost hyperparameters | `cost_weight` = price of time λ | `λ` = price of time |
| **Mesa/meta aligned?** | n/a | **No** — optimizes a surrogate | **Yes** — fit on the metric itself | **Yes** — solves the metric itself |
| **Ships in Anki?** | Yes (already) | If it wins (Rust-portable table/solve) | Easily (evaluate a formula) | If it wins (Rust-portable table/solve) |
| **Status in this repo** | baseline | **done — it failed** (≈ fixed DR at best) | step 5.5 (reference code in `adr.py`) | candidate (reference code in `marc.py`) |

The big picture: **SSP-MMC and MARC are the same machine pointed at different targets**; **ADR and
MARC chase the same target with very different tools** (a tiny formula vs a full optimal-control
table). Fixed DR is the special case all three contain.

---

## The shared foundation

Every policy ultimately outputs an **interval** (how many days until the next review). They all do
it the same way: pick a **target retention** `R` (the probability of recall you want at review
time), then ask FSRS-7 "at what interval does this card's predicted recall drop to `R`?" and
schedule that. So the entire job of a policy is: **choose `R`** — possibly depending on the card's
current memory state.

FSRS-7's memory state has three numbers: long-term stability `S_long`, short-term stability
`S_short`, and difficulty `D`. The "interval for a target `R`" is computed by inverting FSRS-7's
forgetting curve (`fsrs7.forgetting_curve_inverse`).

In code, all three policies are consumed by the simulator the **same way**: they produce a
**retention table** over the `(D, S_long, S_short)` grid (or a formula evaluated to fill one), and
the Rust simulator's `policy="ssp_mmc"` path looks `R` up in that table and inverts the curve. So
ADR and MARC need **no new simulator code** — they're just new ways to fill the table.

---

## Fixed desired retention (the baseline)

**Simple:** You pick one number, like "I want to remember 90% of my cards at review time." Every
card, regardless of how strong or hard it is, is scheduled to that same 90%. Want less work? Pick a
lower number (you'll forget more but review less). This is what Anki does today.

**Technical:** `R(state) = R₀`, a constant. The workload knob is `R₀` itself; sweeping it (≈0.70 to
0.99) traces a knowledge-vs-time curve. It's the simplest possible policy — zero state-dependence —
and it's the bar the other three must clear. (Code: `policies7.create_dr_policy`.) Note: the curve
**folds back** at very high `R₀` — past ~0.94, daily time explodes toward the budget cap while
knowledge actually *drops*, because cramming reviews starves the learning of new cards.

---

## SSP-MMC

**Simple:** SSP-MMC treats scheduling as a long-term planning problem: "from this card's current
state, what sequence of intervals gets it to 'permanently learned' for the least total study time?"
It solves that plan exactly for every possible memory state, producing a custom target retention for
each state. The catch: "permanently learned" (reaching maximum stability) is a **stand-in goal**, not
the thing we actually care about (knowledge per minute). Optimizing the stand-in turns out to pull the
schedule in the wrong direction.

**Technical:** A stochastic-shortest-path MDP solved by value iteration (`solver7.SSPMMCSolver7`).
`V(state)` = minimum expected discounted **cost to reach `S_MAX`** (the terminal). Each review costs a
*shaped* cost — a function of state with **13 cost hyperparameters** (`transform_*`, `exp_*`,
`w_fail_*`, `w_succ_*`, `w_retention`, …) that the step-5 tuner searched. The policy is
`R(state) = argmin_R [ shaped_cost(state, R) + γ·E[V(next state)] ]`, `γ = 0.97`.

**The problem (proven in step 5):** `S_MAX` is *instrumental*, not the measured metric. Minimizing
cost-to-`S_MAX` makes the policy front-load reviews to push cards to maximum stability, which is
anti-aligned with "knowledge per minute across a budgeted deck." This is a **mesa/meta
misalignment**: the inner objective (the proxy cost) ≠ the outer objective (the knowledge-vs-time
metric). No amount of tuning the 13 hyperparameters fixes it, because *every* setting is
"Bellman-optimal for some shaping of the wrong cost." Empirically: on 1000 users, only 2 of 36
Pareto points beat fixed DR, and only by ~1% (noise); the rest lose. **Verdict: SSP-MMC ≈ fixed DR
at best.** (See memory `ssp-mmc-dominated-by-dr`.)

---

## Cost ADR (Adaptive Desired Retention)

**Simple:** ADR says "don't aim for the same retention on every card — aim a little higher on cards
that are cheap to keep and lower on cards that aren't, using a simple formula." That formula reads
the card's stability and difficulty and spits out a target retention. The whole formula has one big
dial, `cost_weight`, that means "how much do I value my time?" Turn it up and every target drops
(less studying, less knowledge); turn it down and they rise. Crucially, the formula's shape is
**learned directly by trying it in the simulator and keeping what scores best on the real
knowledge-vs-time tradeoff** — so it's aimed at the right target from the start. The price: it's a
*smooth, simple* surface, so it can't express very intricate state-dependence.

**Technical:** A closed form `DR(stability, difficulty; cost_weight)` with **15 coefficients**
(`adr.desired_retention`, ported verbatim from JSchoreels' `cost_adr.rs`):

```
x_s = normalized ln(stability),  x_d = normalized difficulty,  z = normalized ln(1+cost_weight)
phi = [1, x_s, x_d, x_s·x_d, x_s²]
base   = coef[0:5]·phi
z_eff  = softplus(coef[5:10]·phi)·z
z2_eff = softplus(coef[10:15]·phi)·z²
DR = retention_min + (retention_max − retention_min)·sigmoid(base − z_eff − z2_eff)
```

`cost_weight` is the **price-of-time / Lagrange multiplier λ**; sweeping it 0→1024 slides average
DR ~0.90→0.50 and traces the Pareto front *by construction* (softplus makes the dependence
monotone). The 15 coefficients are fit **once** by an **evolutionary search on the simulator's
knowledge-vs-time hypervolume** — the true objective, with budget caps and the finite horizon baked
in. There is **no per-card solve and no proxy**: policy and objective are the same, so ADR has **no
mesa/meta misalignment**. Trade-off vs SSP-MMC/MARC: **low capacity** (a smooth degree-2 surface in
`(x_s, x_d)`), so it can't represent sharp, non-smooth optimal schedules — *if* any exist.

DR depends on **long-term stability + difficulty only** (not `S_short`), so the retention table is
constant along the `S_short` axis (`adr.retention_table`). Fixed DR is the exact special case:
all-zero coefficients ⇒ constant DR. So ADR is **≥ fixed DR by construction** (it contains it). It's
trivially Anki-portable (evaluate a formula; no table, no solve). (Code: `ssp_mmc_fsrs/adr.py`;
memory `adr-cost-adr`.)

---

## MARC (Maximize Accumulated Retention, Cost-adjusted)

**Simple:** MARC keeps SSP-MMC's powerful planning machine but **points it at the right target**.
Instead of "minimize the time to push this card to maximum stability," MARC says "maximize the
actual knowledge this card delivers, minus the price of the time it costs." Knowledge delivered over
an interval is literally the area under the card's forgetting curve — exactly what the simulator
measures. Because it still does a full per-state plan, MARC can express schedules far more intricate
than ADR's simple formula — *and* it's now optimizing the thing we actually care about. The open
question is whether that extra flexibility actually buys anything over ADR's tidy formula, or whether
the simple formula already captures all the gain.

**Technical:** Same MDP and value iteration as SSP-MMC (same `(D, S_long, S_short)` grid, same
action set `R ∈ [0.60, 0.99]`), but:

- **Reward** per `(state, R)`: `reward = ∫₀^Δ p_recall(τ) dτ − λ · E[review cost]`, where `Δ` is the
  interval the action schedules. The integral is the card's exact contribution to the knowledge
  metric and is **closed-form** (`fsrs7.forgetting_curve_area`): the FSRS-7 curve is a fixed convex
  combination of two power curves `(1+a·τ)^b`, and `∫₀^Δ (1+a·τ)^b dτ = ((1+aΔ)^(b+1)−1)/(a(b+1))`,
  so no quadrature is needed. `E[review cost]` is the *real* expected review time — **no shaping
  hyperparameters** (those were SSP-MMC's whole problem).
- **Maximize**, no terminal: `V(state) = max_R [ reward(state, R) + γ·E[V(next state)] ]`, from
  `V ≡ 0`. There is **no `S_MAX` terminal** — stability is no longer a goal, just a state variable.
- **One knob:** `λ` = price of time (the shadow price of the daily-time budget the per-card MDP can't
  see). It's a **subtraction (Lagrangian), not a ratio**, and there is **one** `λ` even though
  fail/success reviews cost different time — failure is already penalized through the lost retention
  area, not a second price. Sweep `λ` to trace the front.

This is **exact optimal control of the right objective** — high capacity (a full per-state table)
*and* aligned (the inner objective *is* the measured metric). That makes it genuinely different from
**both** SSP-MMC (high capacity, wrong objective) and ADR (right objective, low capacity).

**Why it's NOT just ADR:** ADR is a low-capacity closed form fit by black-box search; MARC is a
high-capacity tabular policy from value iteration. The only shared idea is `λ`, which is intrinsic to
*any* knowledge-vs-time tradeoff — not an ADR-specific concept. A clean three-way contrast is:

> exact optimal control of the right objective (**MARC**) · a fitted closed form (**ADR**) · fixed DR.

**Build decisions** (memory `ssp-mmc-objective-redesign`): keep the current grid steps (redo the
convergence study at build time — MARC's value function should be smoother without a terminal); lower
the solver grid's `S_max` from 100y to ~5–20y (default 10y) since the high-stability tail is no longer
needed (states above `S_max` clamp to the grid top at lookup); and **keep `R ∈ [0.60, 0.99]`** so MARC
compares apples-to-apples with fixed DR's range. (Code: `ssp_mmc_fsrs/marc.py`.)

**Caveat:** MARC might still converge to a near-constant DR — the step-5 diagnostics hint fixed DR may
be close to optimal for this idealized objective. If so, that's a clean, informative result ("DR is
near-optimal here"), not a failure of the method.

---

## How they relate

```
                         optimizes the REAL knowledge-vs-time objective?
                          NO (proxy)                 YES (the metric itself)
                     ┌───────────────────────┬───────────────────────────────┐
   high capacity     │                       │                               │
   (per-state table) │      SSP-MMC          │            MARC               │
                     │  (wrong target —      │  (right target + full         │
                     │   failed in step 5)   │   flexibility — candidate)    │
                     ├───────────────────────┼───────────────────────────────┤
   low capacity      │                       │      Cost ADR (15 coefs)       │
   (closed form)     │   (fixed DR = 1 num)  │   ⊃ fixed DR as a special case │
                     └───────────────────────┴───────────────────────────────┘
```

- **Fixed DR** is the floor and a special case of **ADR** (zero coefficients) and reachable by
  **SSP-MMC/MARC** (constant table).
- **SSP-MMC → MARC** = keep the machine, fix the objective (proxy cost-to-`S_MAX` → real
  recall-minus-price-of-time; minimize → maximize; drop the terminal).
- **ADR → MARC** = keep the objective, swap the tool (a smooth 15-coef formula → a full
  value-iteration table). Same `λ` idea; very different capacity.
- The research question for step 5.5 / step 6: does MARC's extra capacity Pareto-beat ADR's compact
  closed form, or does the closed form already capture ~all the achievable gain over fixed DR?

## Where each is used in the roadmap

- **Step 5 (done):** tuned SSP-MMC on 1k users with the GRU predictor → it does not beat fixed DR.
- **Step 5.5:** implement Cost ADR, benchmark fixed DR vs SSP-MMC vs ADR on 10k users with the LSTM
  predictor.
- **Step 6:** full 6-policy × 7-setup evaluation on 10k users with LSTM (ADR and SSP-MMC are two of
  the policies); MARC is a candidate addition if step 5.5 motivates it.

## Code map

| Piece | Location |
| --- | --- |
| Fixed-DR policy | `src/ssp_mmc_fsrs/policies7.py` (`create_dr_policy`) |
| SSP-MMC solver | `src/ssp_mmc_fsrs/solver7.py` (`SSPMMCSolver7`) |
| Cost ADR closed form + table | `src/ssp_mmc_fsrs/adr.py` |
| MARC solver | `src/ssp_mmc_fsrs/marc.py` (`MARCSolver7`) |
| Closed-form retention area (MARC reward) | `src/ssp_mmc_fsrs/fsrs7.py` (`forgetting_curve_area`) |
| FSRS-7 forgetting curve + inverse | `src/ssp_mmc_fsrs/fsrs7.py` |
| Simulator (consumes any retention table) | `src/ssp_mmc_fsrs/simulation7.py` + Rust `simulate_fsrs7` |
| Tests (ready to run) | `tests/test_fsrs7_area.py`, `tests/test_adr_marc_smoke.py` |
