//! Rust port of the FSRS-7 simulator (roadmap step 2d), f64.
//!
//! Reference / source of truth: `src/ssp_mmc_fsrs/simulation7.py` with `rng_kind="shared"`
//! and the policies in `src/ssp_mmc_fsrs/policies7.py`. This is the **f64** port (matches
//! the f64 Python reference closely; only cross-language libm differences remain). The f32
//! hot-path conversion is a later speedup campaign, mirroring the FSRS-6 port.
//!
//! Differences from the FSRS-6 `simulate.rs`:
//!   - 3-component memory state (s_long, s_short, d) and the dual-stability forgetting
//!     curve (`fsrs7`).
//!   - Same-day reviews: `due` is a fractional day and an inner same-day-rounds loop
//!     re-reviews cards whose interval lands later the same day (capped at `max_same_day`
//!     per card per day). The shared RNG gains a per-round counter dimension.
//!   - The DR policy inverts the dual curve numerically (Brent's method).
//!
//! Params are per-deck (the `parallel` axis = users): `w` is `(parallel, 34)`.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::rng;

const S_MAX: f64 = 36500.0;
const D_MIN: f64 = 1.0;
const D_MAX: f64 = 10.0;
const MIN_IVL: f64 = 1.0 / 1440.0; // 1 minute in days
const MEMRISE7: [f64; 7] = [4.0 / 24.0, 12.0 / 24.0, 1.0, 6.0, 12.0, 48.0, 96.0];

enum Policy {
    Fixed,
    Dr,
    Memrise,
    Sm2,
}

#[inline]
fn categorical(u: f64, cum: &[f64]) -> usize {
    let mut k = 0;
    while k + 1 < cum.len() && u >= cum[k] {
        k += 1;
    }
    k
}

#[inline]
fn constrain_f64(x: f64, minimum: f64) -> f64 {
    (x + 0.5).floor().max(minimum).max(1.0)
}

#[inline]
fn memrise_next_rung(prev: f64) -> f64 {
    let mut best = 0usize;
    let mut best_dist = f64::INFINITY;
    for (i, &v) in MEMRISE7.iter().enumerate() {
        let dist = (prev - v).abs();
        if dist < best_dist {
            best_dist = dist;
            best = i;
        }
    }
    MEMRISE7[(best + 1).min(MEMRISE7.len() - 1)]
}

// ── FSRS-7 memory model (f64 port of fsrs7.py) ──────────────────────────────────

#[inline]
fn short_component_recall(t: f64, s_short: f64, w: &[f64]) -> f64 {
    let t = t.max(0.0);
    let mag = (w[23] * s_short.powf(w[33] - 0.3)).clamp(0.01, 0.95);
    let decay1 = -mag;
    let factor1 = ((w[25].ln() / decay1).min(60.0)).exp() - 1.0;
    (t / s_short * factor1 + 1.0).powf(decay1)
}

#[inline]
fn forgetting_curve(t: f64, s: f64, s_short: f64, d: f64, w: &[f64]) -> f64 {
    let t = t.max(0.0);
    let r1 = short_component_recall(t, s_short, w);
    let decay2 = -(w[24].clamp(0.01, 0.95));
    let factor2 = w[26].powf(1.0 / decay2) - 1.0;
    let d_ts = ((d - 5.0) * (w[32] - 0.3)).exp();
    let r2 = (t / s * factor2 * d_ts + 1.0).powf(decay2);
    let weight1 = w[27] * s_short.powf(-w[29]);
    let weight2 = w[28] * s.powf(w[30]) * ((d - 5.0) * (w[31] - 0.5)).exp();
    let retention = (weight1 * r1 + weight2 * r2) / (weight1 + weight2);
    retention * (1.0 - 2e-5) + 1e-5
}

#[inline]
fn next_stability(last_s: f64, last_d: f64, r: f64, rating: i64, start: usize, w: &[f64]) -> f64 {
    let hard = if rating == 2 { w[start + 6] } else { 1.0 };
    let easy = if rating == 4 { w[start + 7] } else { 1.0 };
    let new_s_fail =
        w[start + 3] * ((last_s + 1.0).powf(w[start + 4]) - 1.0) * ((1.0 - r) * w[start + 5]).exp();
    let pls = last_s.min(new_s_fail);
    let sinc = (w[start] - 1.5).exp()
        * (11.0 - last_d)
        * last_s.powf(-w[start + 1])
        * (((1.0 - r) * w[start + 2]).exp() - 1.0)
        * hard
        * easy
        + 1.0;
    let new_s_success = pls.max(last_s * sinc);
    if rating > 1 {
        new_s_success
    } else {
        pls
    }
}

#[inline]
fn init_d(rating: i64, w: &[f64]) -> f64 {
    w[4] - (w[5] * (rating as f64 - 1.0)).exp() + 1.0
}

#[inline]
fn next_difficulty(last_d: f64, rating: i64, retention: f64, w: &[f64]) -> f64 {
    let delta_d0 = -w[6] * (rating as f64 - 3.0);
    let delta_d = if rating == 1 {
        delta_d0 * (retention + 0.1)
    } else {
        delta_d0
    };
    let new_d = last_d + delta_d * (10.0 - last_d) / 9.0;
    let reverted = 0.01 * init_d(4, w) + 0.99 * new_d;
    reverted.clamp(D_MIN, D_MAX)
}

#[inline]
fn init_state(rating: i64, w: &[f64], s_min: f64) -> (f64, f64, f64) {
    let idx = (rating.clamp(1, 4) - 1) as usize;
    let s_long = w[idx].clamp(s_min, S_MAX);
    let s_short = (0.8 * w[idx]).clamp(s_min, S_MAX);
    let d = init_d(rating, w).clamp(D_MIN, D_MAX);
    (s_long, s_short, d)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn update_state(
    dt: f64,
    rating: i64,
    s_long: f64,
    s_short: f64,
    d: f64,
    w: &[f64],
    s_min: f64,
    s_max: f64,
) -> (f64, f64, f64) {
    let last_s = s_long.clamp(s_min, s_max);
    let last_s_short = s_short.clamp(s_min, s_max);
    let last_d = d.clamp(D_MIN, D_MAX);
    let r = forgetting_curve(dt, last_s, last_s_short, last_d, w);
    let upd_s_long = next_stability(last_s, last_d, r, rating, 7, w);
    let r1 = short_component_recall(dt, last_s_short, w);
    let upd_s_short_raw = next_stability(last_s_short, last_d, r1, rating, 15, w);
    let upd_s_short = if rating == 1 {
        upd_s_short_raw.min(0.8 * upd_s_long)
    } else {
        upd_s_short_raw
    };
    let upd_d = next_difficulty(last_d, rating, r, w);
    (
        upd_s_long.clamp(s_min, s_max),
        upd_s_short.clamp(s_min, s_max),
        upd_d.clamp(D_MIN, D_MAX),
    )
}

/// Interval at which recall == `dr`, by Brent's method in log(t) (f64 port of
/// fsrs7.forgetting_curve_inverse). Result in [min_t, max_t].
#[allow(clippy::too_many_arguments)]
fn forgetting_curve_inverse(
    dr: f64,
    s_long: f64,
    s_short: f64,
    d: f64,
    w: &[f64],
    n_iter: usize,
    min_t: f64,
    max_t: f64,
) -> f64 {
    let mag = (w[23] * s_short.powf(w[33] - 0.3)).clamp(0.01, 0.95);
    let decay1 = -mag;
    let factor1 = ((w[25].ln() / decay1).min(60.0)).exp() - 1.0;
    let a1 = factor1 / s_short;
    let decay2 = -(w[24].clamp(0.01, 0.95));
    let factor2 = w[26].powf(1.0 / decay2) - 1.0;
    let d_ts = ((d - 5.0) * (w[32] - 0.3)).exp();
    let a2 = factor2 * d_ts / s_long;
    let weight1 = w[27] * s_short.powf(-w[29]);
    let weight2 = w[28] * s_long.powf(w[30]) * ((d - 5.0) * (w[31] - 0.5)).exp();
    let wt_sum = (weight1 + weight2).max(1e-9);
    let scale = 1.0 - 2e-5;
    let log_min = min_t.ln();
    let log_max = max_t.ln();
    let tol = 1e-12;

    let f_of_u = |u: f64| -> f64 {
        let t = u.exp();
        let p = (weight1 * (a1 * t + 1.0).powf(decay1) + weight2 * (a2 * t + 1.0).powf(decay2))
            / wt_sum
            * scale
            + 1e-5;
        p - dr
    };

    let t1 = ((dr.powf(1.0 / decay1) - 1.0) / a1).max(1e-12);
    let t2 = ((dr.powf(1.0 / decay2) - 1.0) / a2).max(1e-12);
    let mut a = t1.min(t2).ln().clamp(log_min, log_max);
    let mut b = t1.max(t2).ln().clamp(log_min, log_max);
    let mut fa = f_of_u(a);
    let mut fb = f_of_u(b);
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut c = a;
    let mut fc = fa;
    let mut dd = a;
    let mut mflag = true;

    for _ in 0..n_iter {
        let s_interp = if fa != fc && fb != fc {
            a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
        } else {
            let dsec = if fb - fa == 0.0 { 1e-30 } else { fb - fa };
            b - fb * (b - a) / dsec
        };
        let lo_b = (3.0 * a + b) / 4.0;
        let not_between = (s_interp - lo_b) * (s_interp - b) >= 0.0;
        let cond2 = mflag && (s_interp - b).abs() >= (b - c).abs() / 2.0;
        let cond3 = !mflag && (s_interp - b).abs() >= (c - dd).abs() / 2.0;
        let cond4 = mflag && (b - c).abs() < tol;
        let cond5 = !mflag && (c - dd).abs() < tol;
        let bis = not_between || cond2 || cond3 || cond4 || cond5;
        let s = if bis { 0.5 * (a + b) } else { s_interp };
        mflag = bis;

        let fs = f_of_u(s);
        dd = c;
        c = b;
        fc = fb;
        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }
    b.clamp(log_min, log_max).exp()
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn simulate_fsrs7<'py>(
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
    max_cost_perday: f64,
    learn_limit_perday: i64,
    review_limit_perday: i64,
    s_min: f64,
    s_max: f64,
    max_same_day: usize,
    n_iter: usize,
    policy: &str,
    policy_param: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
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

    let mut review_cnt = Array2::<f64>::zeros((parallel, learn_span));
    let mut learn_cnt = Array2::<f64>::zeros((parallel, learn_span));
    let mut memorized = Array2::<f64>::zeros((parallel, learn_span));
    let mut cost_day = Array2::<f64>::zeros((parallel, learn_span));

    let cells = (parallel as u64) * (deck_size as u64);
    let span = learn_span as u64;
    let mr = max_same_day as u64;

    for p in 0..parallel {
        let wd: Vec<f64> = (0..w.shape()[1]).map(|i| w[[p, i]]).collect();
        let lc = [
            learn_costs[[p, 0]],
            learn_costs[[p, 1]],
            learn_costs[[p, 2]],
            learn_costs[[p, 3]],
        ];
        let rc = [
            review_costs[[p, 0]],
            review_costs[[p, 1]],
            review_costs[[p, 2]],
            review_costs[[p, 3]],
        ];
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

        // per-card state (s_long == 0 marks not-yet-learned; due == +inf = unscheduled)
        let mut s_long = vec![0.0f64; deck_size];
        let mut s_short = vec![0.0f64; deck_size];
        let mut diff = vec![0.0f64; deck_size];
        let mut last_date = vec![0.0f64; deck_size];
        let mut due = vec![f64::INFINITY; deck_size];
        let mut ivl = vec![0.0f64; deck_size];
        let mut ease = vec![2.5f64; deck_size];
        let mut same_day = vec![0.0f64; deck_size];

        // first-review rating: KIND_INIT_RATING=0, day 0, round 0 -> counter = cell index
        let mut first_rating = vec![0i64; deck_size];
        for c in 0..deck_size {
            let cell = (p * deck_size + c) as u64;
            first_rating[c] = (categorical(rng::uniform(cell, seed), &cum_first) + 1) as i64;
        }

        for today in 0..learn_span {
            let today_f = today as f64;

            // memorized snapshot (start of day, learned cards)
            let mut mem = 0.0f64;
            for c in 0..deck_size {
                if s_long[c] > 0.0 {
                    let dt = (today_f - last_date[c]).max(0.0);
                    mem += forgetting_curve(
                        dt,
                        s_long[c].clamp(s_min, s_max),
                        s_short[c].clamp(s_min, s_max),
                        diff[c].clamp(D_MIN, D_MAX),
                        &wd,
                    );
                }
            }
            memorized[[p, today]] = mem;

            // per-day shared budgets, carried across same-day rounds
            let mut cost_used = 0.0f64;
            let mut reviews_used = 0i64;
            let mut learns_used = 0i64;
            for c in 0..deck_size {
                same_day[c] = 0.0;
            }
            let mut day_reviews = 0u32;
            let mut day_learns = 0u32;

            for rnd in 0..max_same_day {
                let base_forget = ((span + today as u64) * mr + rnd as u64) * cells; // KIND_FORGET=1
                let base_pass = ((2 * span + today as u64) * mr + rnd as u64) * cells; // KIND_PASS_RATING=2

                // cumulative budgets over the whole deck (prefix in card-index order),
                // continued from earlier rounds via cost_used / reviews_used / learns_used.
                let mut cc = cost_used;
                let mut cr = reviews_used;
                let mut cl = learns_used;
                let mut any_cand = false;
                let mut any_admit = false;
                let mut round_cost = 0.0f64;

                for c in 0..deck_size {
                    let rev_cand =
                        s_long[c] > 0.0 && due[c] < today_f + 1.0 && same_day[c] < max_same_day as f64;
                    let learn_cand = rnd == 0 && s_long[c] == 0.0;
                    if !(rev_cand || learn_cand) {
                        continue;
                    }
                    any_cand = true;

                    // resolve rating + cost + (for reviews) recall and elapsed time. A lapse
                    // is fully encoded by rating==1 (update_state / next_difficulty handle it).
                    let cell = (p * deck_size + c) as u64;
                    let (rating, cnd_cost, t_review, dt);
                    if rev_cand {
                        t_review = due[c].max(today_f);
                        dt = (t_review - last_date[c]).max(0.0);
                        let r = forgetting_curve(
                            dt,
                            s_long[c].clamp(s_min, s_max),
                            s_short[c].clamp(s_min, s_max),
                            diff[c].clamp(D_MIN, D_MAX),
                            &wd,
                        );
                        let recalled = rng::uniform(base_forget + cell, seed) <= r;
                        rating = if recalled {
                            (categorical(rng::uniform(base_pass + cell, seed), &cum_review) + 2)
                                as i64
                        } else {
                            1
                        };
                        cnd_cost = rc[(rating - 1) as usize];
                    } else {
                        rating = first_rating[c];
                        cnd_cost = lc[(rating - 1) as usize];
                        t_review = today_f;
                        dt = 0.0;
                    }

                    // gating (cumsum includes every candidate, matching the Python cumsum)
                    cc += cnd_cost;
                    if rev_cand {
                        cr += 1;
                    }
                    if learn_cand {
                        cl += 1;
                    }
                    let admit = cc <= max_cost_perday
                        && (!rev_cand || cr <= review_limit_perday)
                        && (!learn_cand || cl <= learn_limit_perday);
                    if !admit {
                        continue;
                    }
                    any_admit = true;
                    round_cost += cnd_cost;

                    // apply the memory update / init
                    if rev_cand {
                        let (nsl, nss, nd) =
                            update_state(dt, rating, s_long[c], s_short[c], diff[c], &wd, s_min, s_max);
                        s_long[c] = nsl;
                        s_short[c] = nss;
                        diff[c] = nd;
                        last_date[c] = t_review;
                        same_day[c] += 1.0;
                        day_reviews += 1;
                    } else {
                        let (isl, iss, idd) = init_state(rating, &wd, s_min);
                        s_long[c] = isl;
                        s_short[c] = iss;
                        diff[c] = idd;
                        last_date[c] = today_f;
                        day_learns += 1;
                    }

                    // schedule the next interval / due
                    let is_learn = learn_cand;
                    let prev = if is_learn { 0.0 } else { ivl[c] };
                    let base_time = if is_learn { today_f } else { t_review };
                    let new_ivl: f64 = match pol {
                        Policy::Fixed => policy_param,
                        Policy::Dr => forgetting_curve_inverse(
                            policy_param,
                            s_long[c],
                            s_short[c],
                            diff[c],
                            &wd,
                            n_iter,
                            MIN_IVL,
                            S_MAX,
                        ),
                        Policy::Memrise => {
                            if prev == 0.0 || rating == 1 {
                                MEMRISE7[0]
                            } else {
                                memrise_next_rung(prev)
                            }
                        }
                        Policy::Sm2 => {
                            let ease_in = if is_learn { 2.5 } else { ease[c] };
                            let ease_c = ease_in.clamp(1.3, 5.5);
                            let mut interval = if prev == 0.0 {
                                if rating < 4 {
                                    1.0
                                } else {
                                    4.0
                                }
                            } else {
                                let current = prev.max(1.0);
                                let hard = constrain_f64(current * 1.2, current + 1.0);
                                let good = constrain_f64(current * ease_c, hard + 1.0);
                                let easy = constrain_f64(current * ease_c * 1.3, good + 1.0);
                                match rating {
                                    2 => hard,
                                    4 => easy,
                                    _ => good,
                                }
                            };
                            if rating == 1 {
                                interval = 1.0;
                            }
                            let mut ne = ease_c;
                            if rating == 1 {
                                ne -= 0.2;
                            }
                            if rating == 2 {
                                ne -= 0.15;
                            }
                            if rating == 4 {
                                ne += 0.15;
                            }
                            ease[c] = ne.clamp(1.3, 5.5);
                            interval
                        }
                    };
                    let new_ivl = new_ivl.clamp(MIN_IVL, s_max);
                    ivl[c] = new_ivl;
                    due[c] = base_time + new_ivl;
                }

                // carry the admitted totals into the next same-day round
                cost_used += round_cost;
                reviews_used = day_reviews as i64;
                learns_used = day_learns as i64;

                if !any_cand || !any_admit {
                    break;
                }
            }

            review_cnt[[p, today]] = day_reviews as f64;
            learn_cnt[[p, today]] = day_learns as f64;
            cost_day[[p, today]] = cost_used;
        }
    }

    Ok((
        review_cnt.into_pyarray(py),
        learn_cnt.into_pyarray(py),
        memorized.into_pyarray(py),
        cost_day.into_pyarray(py),
    ))
}
