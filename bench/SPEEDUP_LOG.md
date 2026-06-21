# Speedup iteration log

Workload: 50 users x DR [0.7, 0.9, 0.99], parallel=1, deck=10000, span=1825, seed=42; per pair = min of 3 timed reps; before=HEAD vs after=candidate, run simultaneously (1 thread each). Accept iff C1 median(before/after)>1, C2 Wilcoxon p<0.01, C3 avg drift<=0.15%, C4 max drift<=1% (memorized & time_spent vs iter0).

| timestamp | iter | speedup (med) | wilcoxon p | mem drift avg/max % | time drift avg/max % | C1 | C2 | C3 | C4 | accept | comment |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-21 19:09:40 | 0 | 0.999x | 0.8015 | 0.000/0.000 | 0.000/0.000 | FAIL | FAIL | PASS | PASS | baseline | baseline: current Rust simulator (commit 58aa055), FSRS-6 DR policy |
| 2026-06-21 19:20:51 | 1 | 1.220x | 1.17e-26 | 0.000/0.000 | 0.000/0.000 | PASS | PASS | PASS | PASS | ACCEPT | skip forget/pass RNG draws for non-due cards (counter-based -> bit-identical) |
| 2026-06-21 19:39:05 | 2 | 1.274x | 3.72e-23 | 0.000/0.001 | 0.000/0.038 | PASS | PASS | PASS | PASS | ACCEPT | fast approximate transcendentals (exp2/log2-based pow/exp/ln) on the per-card hot path |
