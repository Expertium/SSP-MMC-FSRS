# Speedup iteration log

Workload: 50 users x DR [0.7, 0.9, 0.99], parallel=1, deck=10000, span=1825, seed=42; per pair = min of 3 timed reps; before=HEAD vs after=candidate, run simultaneously (1 thread each). Accept iff C1 median(before/after)>1, C2 Wilcoxon p<0.01, C3 avg drift<=0.15%, C4 max drift<=1% (memorized & time_spent vs iter0).

| timestamp | iter | speedup (med) | wilcoxon p | mem drift avg/max % | time drift avg/max % | C1 | C2 | C3 | C4 | accept | comment |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-21 19:09:40 | 0 | 0.999x | 0.8015 | 0.000/0.000 | 0.000/0.000 | FAIL | FAIL | PASS | PASS | baseline | baseline: current Rust simulator (commit 58aa055), FSRS-6 DR policy |
| 2026-06-21 19:20:51 | 1 | 1.220x | 1.17e-26 | 0.000/0.000 | 0.000/0.000 | PASS | PASS | PASS | PASS | ACCEPT | skip forget/pass RNG draws for non-due cards (counter-based -> bit-identical) |
| 2026-06-21 19:39:05 | 2 | 1.274x | 3.72e-23 | 0.000/0.001 | 0.000/0.038 | PASS | PASS | PASS | PASS | ACCEPT | fast approximate transcendentals (exp2/log2-based pow/exp/ln) on the per-card hot path |
| 2026-06-21 19:53:40 | 3 | 1.362x | 1.15e-26 | 0.000/0.001 | 0.000/0.038 | PASS | PASS | PASS | PASS | ACCEPT | target-cpu=native (AVX2+FMA via rust/.cargo/config.toml) |
| 2026-06-21 20:07:03 | 4 | 1.173x | 1.15e-26 | 0.008/0.350 | 0.021/0.214 | PASS | PASS | PASS | PASS | ACCEPT | cruder fastmath polynomials (exp2 degree-5, log2 3-term) |
| 2026-06-21 20:19:22 | 5 | 0.604x | 1.0000 | 0.008/0.350 | 0.021/0.214 | FAIL | FAIL | PASS | PASS | REJECT | fuse retrievability into the draw pass (one fewer full-deck pass, bit-identical) |
| 2026-06-21 20:34:32 | 6 | 1.140x | 1.49e-26 | 0.010/0.416 | 0.024/0.304 | PASS | PASS | PASS | PASS | ACCEPT | f32 hot path (stability/retrievability/updates/policy in f32 for 8-wide AVX2) |
