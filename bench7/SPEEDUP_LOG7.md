# FSRS-7 whole-sim speedup log

Workload: 20 users x 3 hp = 60 datapoints, deck=3000, span=1095, seed=42; per datapoint = min of 2 reps; before=champion HEAD vs after=candidate, run SEQUENTIALLY. Accept iff C1 median(before/after)>1, C2 Wilcoxon p<0.01, and avg drift<=0.5% & max drift<=2.5% for BOTH knowledge and time_spent vs iter0.

| timestamp | iter | speedup (med) | sim-only | wilcoxon p | know drift avg/max % | time drift avg/max % | C1 | C2 | know | time | accept | comment |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-24 00:26:40 | 0 | 0.997x | 0.996x | 0.9607 | 0.000/0.000 | 0.000/0.000 | FAIL | FAIL | PASS | PASS | baseline | baseline (post-Newton-7 HEAD) |
| 2026-06-24 00:43:20 | 1 | 1.418x | 1.573x | 8.15e-12 | 0.000/0.000 | 0.000/0.000 | PASS | PASS | PASS | PASS | ACCEPT | GRU: cache curve(h) per card, refresh only on step (daily snapshot no longer reruns GRU heads) |
| 2026-06-24 01:05:53 | 2 | 1.020x | 1.027x | 1.15e-08 | 0.000/0.000 | 0.000/0.000 | PASS | PASS | PASS | PASS | ACCEPT | FSRS: dedupe short_component_recall in update_state (computed twice for identical args) |
