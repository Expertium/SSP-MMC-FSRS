//! Rust port of `simulate()` (roadmap step 1), f32 hot path.
//!
//! Reference / source of truth: `src/ssp_mmc_fsrs/simulation.py` with `rng_kind="shared"`
//! and the policies in `src/ssp_mmc_fsrs/policies.py`.
//!
//! The simulator state and per-card math are **f32** (stability, retrievability,
//! difficulty, intervals, costs) so the hot loops vectorize 8-wide on AVX2 via the f32
//! `fastmath` approximations. This diverges from the f64 Python reference at the ~f32
//! level, kept within the speedup protocol's drift budget. The RNG stream is still f64
//! (counter-based), the memorized sum accumulates in f64, and per-deck constants are
//! computed in f64 then cast to f32. Anki-SM-2 interval math was already f32.
//!
//! Params are **per-deck** (the `parallel` axis = users): `w` is `(parallel, n_w)`,
//! costs/probs/offsets are `(parallel, k)`. The parity test broadcasts one user across
//! all decks so it matches the current shared-param Python.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::fastmath;
use crate::rng;

const S_MIN: f32 = 0.1;
const STABILITY_INIT: f32 = 1e-10;
const MEMRISE_SEQ: [f32; 6] = [1.0, 6.0, 12.0, 48.0, 96.0, 180.0];

enum Policy {
    Fixed,
    Dr,
    Memrise,
    Sm2,
}

/// First index `i` with `u < cum[i]`, clipped to `cum.len()-1`. Operates on the f64 RNG
/// uniforms. Matches `shared_rng.categorical`.
#[inline]
fn categorical(u: f64, cum: &[f64]) -> usize {
    let mut k = 0;
    while k + 1 < cum.len() && u >= cum[k] {
        k += 1;
    }
    k
}

/// Anki round-half-up then floor on minimums, in f32 (matches `_constrain_passing_interval_tensor`).
#[inline]
fn constrain_f32(x: f32, minimum: f32) -> f32 {
    (x + 0.5).floor().max(minimum).max(1.0)
}

/// Memrise: rung after the one closest to `prev` (np.argmin keeps the first minimum).
#[inline]
fn memrise_next_rung(prev: f32) -> f32 {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for (i, &v) in MEMRISE_SEQ.iter().enumerate() {
        let dist = (prev - v).abs();
        if dist < best_dist {
            best_dist = dist;
            best = i;
        }
    }
    MEMRISE_SEQ[(best + 1).min(MEMRISE_SEQ.len() - 1)]
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn simulate<'py>(
    py: Python<'py>,
    parallel: usize,
    deck_size: usize,
    learn_span: usize,
    seed: u64,
    w: PyReadonlyArray2<'py, f64>,
    learn_costs: PyReadonlyArray2<'py, f64>,
    review_costs: PyReadonlyArray2<'py, f64>,
    first_rating_prob: PyReadonlyArray2<'py, f64>,
    review_rating_prob: PyReadonlyArray2<'py, f64>,
    first_rating_offset: PyReadonlyArray2<'py, f64>,
    forget_rating_offset: PyReadonlyArray2<'py, f64>,
    max_cost_perday: f64,
    learn_limit_perday: i64,
    review_limit_perday: i64,
    sim_s_max: f64,
    policy_s_max: f64,
    policy: &str,
    policy_param: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
)> {
    let pol = match policy {
        "fixed" => Policy::Fixed,
        "dr" => Policy::Dr,
        "memrise" => Policy::Memrise,
        "sm2" => Policy::Sm2,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown policy {other:?} (use 'fixed', 'dr', 'memrise', or 'sm2')"
            )));
        }
    };

    let w = w.as_array();
    let learn_costs = learn_costs.as_array();
    let review_costs = review_costs.as_array();
    let first_rating_prob = first_rating_prob.as_array();
    let review_rating_prob = review_rating_prob.as_array();
    let first_rating_offset = first_rating_offset.as_array();
    let forget_rating_offset = forget_rating_offset.as_array();

    let mut review_cnt = Array2::<f32>::zeros((parallel, learn_span));
    let mut learn_cnt = Array2::<f32>::zeros((parallel, learn_span));
    let mut memorized = Array2::<f32>::zeros((parallel, learn_span));
    let mut cost_day = Array2::<f32>::zeros((parallel, learn_span));

    let cells = (parallel as u64) * (deck_size as u64);
    let span = learn_span as u64;

    let max_cost_perday = max_cost_perday as f32;
    let policy_param_f = policy_param as f32;
    let policy_s_max_f = policy_s_max as f32;
    let sim_s_max_f = sim_s_max as f32;

    for p in 0..parallel {
        // --- per-deck constants (computed in f64, cast to f32) ---
        let decay = (-w[[p, 20]]) as f32;
        let fc_factor = (0.9_f64.powf(1.0 / -w[[p, 20]]) - 1.0) as f32;
        let exp_w8 = w[[p, 8]].exp() as f32;
        let w9 = w[[p, 9]] as f32;
        let w10 = w[[p, 10]] as f32;
        let w11 = w[[p, 11]] as f32;
        let w12 = w[[p, 12]] as f32;
        let w13 = w[[p, 13]] as f32;
        let w14 = w[[p, 14]] as f32;
        let w15 = w[[p, 15]] as f32;
        let w16 = w[[p, 16]] as f32;
        let w4 = w[[p, 4]] as f32;
        let w5 = w[[p, 5]] as f32;
        let w6 = w[[p, 6]] as f32;
        let w7 = w[[p, 7]] as f32;
        let init_d4 = (w[[p, 4]] - (3.0 * w[[p, 5]]).exp() + 1.0) as f32;
        let fail_div = (w[[p, 17]] * w[[p, 18]]).exp() as f32;
        let init_s = [
            w[[p, 0]] as f32,
            w[[p, 1]] as f32,
            w[[p, 2]] as f32,
            w[[p, 3]] as f32,
        ];
        let lc = [
            learn_costs[[p, 0]] as f32,
            learn_costs[[p, 1]] as f32,
            learn_costs[[p, 2]] as f32,
            learn_costs[[p, 3]] as f32,
        ];
        let rc = [
            review_costs[[p, 0]] as f32,
            review_costs[[p, 1]] as f32,
            review_costs[[p, 2]] as f32,
            review_costs[[p, 3]] as f32,
        ];
        let fro = [
            first_rating_offset[[p, 0]] as f32,
            first_rating_offset[[p, 1]] as f32,
            first_rating_offset[[p, 2]] as f32,
            first_rating_offset[[p, 3]] as f32,
        ];
        let forget_off = forget_rating_offset[[p, 0]] as f32;

        // categorical tables stay f64 (used with the f64 RNG uniforms)
        let mut cum_first = [0.0f64; 4];
        let mut acc = 0.0;
        for i in 0..4 {
            acc += first_rating_prob[[p, i]];
            cum_first[i] = acc;
        }
        let mut cum_review = [0.0f64; 3];
        acc = 0.0;
        for i in 0..3 {
            acc += review_rating_prob[[p, i]];
            cum_review[i] = acc;
        }

        // interval to (just past) policy_s_max at rating 3 -> ceil; shared by all policies
        let int_req = |s: f32, d: f32| -> f32 {
            let c_coef = exp_w8 * (11.0 - d) * fastmath::pow_f32(s, -w9);
            let s_next = policy_s_max_f + 1e-3;
            let max_r =
                (1.0 - fastmath::ln_f32((s_next / s - 1.0) / c_coef + 1.0) / w10).max(0.01);
            let ivl_raw = s / fc_factor * (fastmath::pow_f32(max_r, 1.0 / decay) - 1.0);
            ivl_raw.ceil().max(1.0)
        };
        // DR base interval: interval to reach desired retention -> floor
        let ni_floor = |s: f32, r: f32| -> f32 {
            (s / fc_factor * (fastmath::pow_f32(r, 1.0 / decay) - 1.0))
                .floor()
                .max(1.0)
        };

        // --- per-card state ---
        let mut due = vec![learn_span as f32; deck_size];
        let mut difficulty = vec![STABILITY_INIT; deck_size];
        let mut stability = vec![STABILITY_INIT; deck_size];
        let mut ease = vec![2.5f32; deck_size];
        let mut retriev = vec![0.0f32; deck_size];
        let mut last_date = vec![0.0f32; deck_size];
        let mut ivl = vec![0.0f32; deck_size];
        let mut rating = vec![0i32; deck_size];
        // scratch (reused each day)
        let mut cost = vec![0.0f32; deck_size];
        let mut need_review = vec![false; deck_size];
        let mut need_learn = vec![false; deck_size];
        let mut forget = vec![false; deck_size];
        let mut true_review = vec![false; deck_size];
        let mut true_learn = vec![false; deck_size];

        // initial rating: KIND_INIT_RATING=0, day=0 -> counter = cell index p*deck_size+c
        for c in 0..deck_size {
            let cell = (p as u64) * (deck_size as u64) + (c as u64);
            let u = rng::uniform(cell, seed);
            rating[c] = (categorical(u, &cum_first) + 1) as i32;
        }

        for today in 0..learn_span {
            let today_f = today as f32;
            let base_forget = (span + today as u64) * cells; // KIND_FORGET=1
            let base_pass = (2 * span + today as u64) * cells; // KIND_PASS_RATING=2

            // retrievability (uses end-of-previous-day stability)
            for c in 0..deck_size {
                if stability[c] > STABILITY_INIT {
                    let dt = today_f - last_date[c];
                    retriev[c] = fastmath::pow_f32(1.0 + fc_factor * dt / stability[c], decay);
                }
            }

            // need_review, forget/rating draws (only for due cards), review cost
            for c in 0..deck_size {
                cost[c] = 0.0;
                let nr = due[c] <= today_f;
                need_review[c] = nr;
                if nr {
                    let cell = (p as u64) * (deck_size as u64) + (c as u64);
                    let f = rng::uniform(base_forget + cell, seed) > retriev[c] as f64;
                    forget[c] = f;
                    rating[c] = if f {
                        1
                    } else {
                        (categorical(rng::uniform(base_pass + cell, seed), &cum_review) + 2) as i32
                    };
                    cost[c] = rc[(rating[c] - 1) as usize];
                }
            }

            // true_review: per-day cost/count budget, cumulative along card index
            let mut cum_cost = 0.0f32;
            let mut cum_cnt = 0i64;
            for c in 0..deck_size {
                cum_cost += cost[c];
                if need_review[c] {
                    cum_cnt += 1;
                }
                true_review[c] = need_review[c]
                    && (cum_cost <= max_cost_perday)
                    && (cum_cnt <= review_limit_perday);
            }

            // apply review updates (stability uses OLD difficulty; difficulty via next_d)
            for c in 0..deck_size {
                if !true_review[c] {
                    continue;
                }
                last_date[c] = today_f;
                let s = stability[c];
                let r = retriev[c];
                let d = difficulty[c];
                let rt = rating[c];
                let s_new = if forget[c] {
                    let t1 = w11
                        * fastmath::pow_f32(d, -w12)
                        * (fastmath::pow_f32(s + 1.0, w13) - 1.0)
                        * fastmath::exp_f32((1.0 - r) * w14);
                    t1.min(s / fail_div)
                } else {
                    let hp = if rt == 2 { w15 } else { 1.0 };
                    let eb = if rt == 4 { w16 } else { 1.0 };
                    s * (1.0
                        + exp_w8
                            * (11.0 - d)
                            * fastmath::pow_f32(s, -w9)
                            * (fastmath::exp_f32((1.0 - r) * w10) - 1.0)
                            * hp
                            * eb)
                };
                stability[c] = s_new.clamp(S_MIN, sim_s_max_f);

                let delta_d = -w6 * (rt as f32 - 3.0);
                let mut nd = d + delta_d * (10.0 - d) / 9.0;
                nd = w7 * init_d4 + (1.0 - w7) * nd;
                let mut d_new = nd.clamp(1.0, 10.0);
                if forget[c] {
                    d_new = (d_new - w6 * forget_off).clamp(1.0, 10.0);
                }
                difficulty[c] = d_new;
            }

            // need_learn + learn cost (cost array now holds review AND learn costs)
            for c in 0..deck_size {
                need_learn[c] = stability[c] == STABILITY_INIT;
                if need_learn[c] {
                    cost[c] = lc[(rating[c] - 1) as usize];
                }
            }

            // true_learn: fresh cumulative budget over the full cost array
            cum_cost = 0.0;
            cum_cnt = 0;
            for c in 0..deck_size {
                cum_cost += cost[c];
                if need_learn[c] {
                    cum_cnt += 1;
                }
                true_learn[c] = need_learn[c]
                    && (cum_cost <= max_cost_perday)
                    && (cum_cnt <= learn_limit_perday);
            }

            // apply learn updates
            for c in 0..deck_size {
                if !true_learn[c] {
                    continue;
                }
                last_date[c] = today_f;
                let rt = rating[c];
                stability[c] = init_s[(rt - 1) as usize].clamp(S_MIN, sim_s_max_f);
                let id = w4 - fastmath::exp_f32(w5 * (rt as f32 - 1.0)) + 1.0;
                difficulty[c] = (id - w6 * fro[(rt - 1) as usize]).clamp(1.0, 10.0);
            }

            // policy: new interval (and ease for SM-2) for reviewed-or-learned cards
            for c in 0..deck_size {
                if !(true_review[c] || true_learn[c]) {
                    continue;
                }
                let s = stability[c];
                let d = difficulty[c];
                let is_learn = true_learn[c];
                let grade = rating[c];

                let new_ivl: f32 = match pol {
                    Policy::Fixed => {
                        if s > policy_s_max_f {
                            1e9
                        } else {
                            policy_param_f.min(int_req(s, d))
                        }
                    }
                    Policy::Dr => {
                        if s > policy_s_max_f {
                            1e9
                        } else {
                            ni_floor(s, policy_param_f).min(int_req(s, d))
                        }
                    }
                    Policy::Memrise => {
                        let prev = if is_learn { 0.0 } else { ivl[c] };
                        if prev == 0.0 || grade == 1 {
                            1.0
                        } else if s > policy_s_max_f {
                            1e9
                        } else {
                            memrise_next_rung(prev).min(int_req(s, d))
                        }
                    }
                    Policy::Sm2 => {
                        let prev = if is_learn { 0.0f32 } else { ivl[c] };
                        let ease_in = if is_learn { 2.5f32 } else { ease[c] };
                        let ease_c = ease_in.clamp(1.3, 5.5);
                        let mut interval: f32 = if prev == 0.0 {
                            if grade < 4 {
                                1.0
                            } else {
                                4.0
                            }
                        } else {
                            let current = prev.max(1.0);
                            let hard = constrain_f32(current * 1.2, current + 1.0);
                            let good = constrain_f32(current * ease_c, hard + 1.0);
                            let easy = constrain_f32(current * ease_c * 1.3, good + 1.0);
                            match grade {
                                2 => hard,
                                4 => easy,
                                _ => good,
                            }
                        };
                        if grade == 1 {
                            interval = 1.0;
                        }
                        let mut ne = ease_c;
                        if grade == 1 {
                            ne -= 0.2;
                        }
                        if grade == 2 {
                            ne -= 0.15;
                        }
                        if grade == 4 {
                            ne += 0.15;
                        }
                        ease[c] = ne.clamp(1.3, 5.5);
                        let capped = if s > policy_s_max_f {
                            1e9
                        } else {
                            interval.min(int_req(s, d))
                        };
                        capped.round_ties_even()
                    }
                };
                ivl[c] = new_ivl;
                due[c] = today_f + ivl[c];
            }

            // per-day aggregates (memorized accumulates in f64 for an accurate sum)
            let mut rev = 0u32;
            let mut lrn = 0u32;
            let mut mem = 0.0f64;
            let mut cst = 0.0f64;
            for c in 0..deck_size {
                if true_review[c] {
                    rev += 1;
                }
                if true_learn[c] {
                    lrn += 1;
                }
                mem += retriev[c] as f64;
                if true_review[c] || true_learn[c] {
                    cst += cost[c] as f64;
                }
            }
            review_cnt[[p, today]] = rev as f32;
            learn_cnt[[p, today]] = lrn as f32;
            memorized[[p, today]] = mem as f32;
            cost_day[[p, today]] = cst as f32;
        }
    }

    Ok((
        review_cnt.into_pyarray(py),
        learn_cnt.into_pyarray(py),
        memorized.into_pyarray(py),
        cost_day.into_pyarray(py),
    ))
}
