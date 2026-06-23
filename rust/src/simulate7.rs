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

use std::time::Instant;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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
    SspMmc,
}

#[inline]
fn categorical(u: f64, cum: &[f64]) -> usize {
    let mut k = 0;
    while k + 1 < cum.len() && u >= cum[k] {
        k += 1;
    }
    k
}

/// Stability -> grid index: first index with grid[i] >= s, clamped (matches solver7.s2i,
/// np.searchsorted 'left').
#[inline]
pub(crate) fn s2i(grid: &[f64], s: f64) -> usize {
    grid.partition_point(|&g| g < s).min(grid.len() - 1)
}

/// Difficulty -> grid index: nearest grid point (matches solver7.d2i non-uniform branch).
#[inline]
pub(crate) fn d2i(grid: &[f64], d: f64) -> usize {
    let hi = grid.partition_point(|&g| g < d).min(grid.len() - 1);
    let lo = hi.saturating_sub(1);
    if (grid[hi] - d).abs() <= (d - grid[lo]).abs() {
        hi
    } else {
        lo
    }
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
pub(crate) fn forgetting_curve(t: f64, s: f64, s_short: f64, d: f64, w: &[f64]) -> f64 {
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
pub(crate) fn update_state(
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
pub(crate) fn forgetting_curve_inverse(
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

// ── GRU pseudo-ground-truth recall predictor (f64 port of ssp_mmc_fsrs/gru.py) ──────
// Per-user weights arrive as one flat row (BatchedGRU.FLAT_ORDER / FLAT_LEN). The GRU is
// tiny (7 hidden) so a scalar f64 implementation is fastest, and it matches the Python
// BatchedGRU to ~1e-15 (identical math). Only nn.GRU is recurrent, so we carry h per card.
const GRU_FLAT_LEN: usize = 505;

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
#[inline]
fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}
#[inline]
fn layernorm7(x: &[f64; 7], gamma: &[f64; 7]) -> [f64; 7] {
    let mean = x.iter().sum::<f64>() / 7.0;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 7.0;
    let inv = 1.0 / (var + 1e-5).sqrt();
    let mut o = [0.0; 7];
    for i in 0..7 {
        o[i] = (x[i] - mean) * inv * gamma[i];
    }
    o
}
#[inline]
fn rd<const N: usize>(r: &[f64], o: &mut usize) -> [f64; N] {
    let mut a = [0.0; N];
    a.copy_from_slice(&r[*o..*o + N]);
    *o += N;
    a
}

struct GruW {
    pre_w: [f64; 35], pre_b: [f64; 7], ln_pre: [f64; 7],
    w_ih: [f64; 147], w_hh: [f64; 147], b_ih: [f64; 21], b_hh: [f64; 21],
    ln_post1: [f64; 7], post_w: [f64; 49], post_b: [f64; 7], ln_post2: [f64; 7],
    w_fc_w: [f64; 14], w_fc_b: [f64; 2], s_fc_w: [f64; 14], s_fc_b: [f64; 2],
    d_fc_w: [f64; 14], d_fc_b: [f64; 2], input_mean: f64, input_std: f64,
}

impl GruW {
    fn from_row(r: &[f64]) -> GruW {
        // Fields are evaluated top-to-bottom, so the shared offset reads them in order.
        let mut o = 0usize;
        GruW {
            pre_w: rd::<35>(r, &mut o), pre_b: rd::<7>(r, &mut o), ln_pre: rd::<7>(r, &mut o),
            w_ih: rd::<147>(r, &mut o), w_hh: rd::<147>(r, &mut o),
            b_ih: rd::<21>(r, &mut o), b_hh: rd::<21>(r, &mut o),
            ln_post1: rd::<7>(r, &mut o), post_w: rd::<49>(r, &mut o),
            post_b: rd::<7>(r, &mut o), ln_post2: rd::<7>(r, &mut o),
            w_fc_w: rd::<14>(r, &mut o), w_fc_b: rd::<2>(r, &mut o),
            s_fc_w: rd::<14>(r, &mut o), s_fc_b: rd::<2>(r, &mut o),
            d_fc_w: rd::<14>(r, &mut o), d_fc_b: rd::<2>(r, &mut o),
            input_mean: {
                let v = r[o];
                o += 1;
                v
            },
            input_std: r[o],
        }
    }

    #[inline]
    fn features(&self, dt: f64, rating: i64) -> [f64; 5] {
        let mut x = [0.0; 5];
        x[0] = ((1e-5 + dt).ln() - self.input_mean) / self.input_std;
        x[1 + (rating.clamp(1, 4) - 1) as usize] = 1.0; // rating one-hot
        x
    }

    /// Advance the hidden state by one review (dt, rating). Mirrors BatchedGRU.step.
    #[inline]
    fn step(&self, h: &[f64; 7], dt: f64, rating: i64) -> [f64; 7] {
        let x = self.features(dt, rating);
        let mut a = [0.0; 7];
        for o in 0..7 {
            let mut s = self.pre_b[o];
            for i in 0..5 {
                s += self.pre_w[o * 5 + i] * x[i];
            }
            a[o] = silu(s);
        }
        let c = layernorm7(&a, &self.ln_pre);
        let mut hn = [0.0; 7];
        for i in 0..7 {
            // GRU gate order: reset (0..7), update (7..14), new (14..21).
            let (mut gir, mut giz, mut gin) = (self.b_ih[i], self.b_ih[7 + i], self.b_ih[14 + i]);
            let (mut ghr, mut ghz, mut ghn) = (self.b_hh[i], self.b_hh[7 + i], self.b_hh[14 + i]);
            for k in 0..7 {
                gir += self.w_ih[i * 7 + k] * c[k];
                giz += self.w_ih[(7 + i) * 7 + k] * c[k];
                gin += self.w_ih[(14 + i) * 7 + k] * c[k];
                ghr += self.w_hh[i * 7 + k] * h[k];
                ghz += self.w_hh[(7 + i) * 7 + k] * h[k];
                ghn += self.w_hh[(14 + i) * 7 + k] * h[k];
            }
            let rg = sigmoid(gir + ghr);
            let zg = sigmoid(giz + ghz);
            let ng = (gin + rg * ghn).tanh();
            hn[i] = (1.0 - zg) * ng + zg * h[i];
        }
        hn
    }

    /// 2-curve forgetting-curve params (w, s, d) from the hidden state.
    #[inline]
    fn curve(&self, h: &[f64; 7]) -> ([f64; 2], [f64; 2], [f64; 2]) {
        let g1 = layernorm7(h, &self.ln_post1);
        let mut e = [0.0; 7];
        for o in 0..7 {
            let mut s = self.post_b[o];
            for i in 0..7 {
                s += self.post_w[o * 7 + i] * g1[i];
            }
            e[o] = silu(s);
        }
        let gg = layernorm7(&e, &self.ln_post2);
        let (mut wl, mut sl, mut dl) = ([0.0; 2], [0.0; 2], [0.0; 2]);
        for o in 0..2 {
            let (mut sw, mut ss, mut sd) = (self.w_fc_b[o], self.s_fc_b[o], self.d_fc_b[o]);
            for i in 0..7 {
                sw += self.w_fc_w[o * 7 + i] * gg[i];
                ss += self.s_fc_w[o * 7 + i] * gg[i];
                sd += self.d_fc_w[o * 7 + i] * gg[i];
            }
            wl[o] = sw;
            sl[o] = ss;
            dl[o] = sd;
        }
        let m = wl[0].max(wl[1]); // stable softmax over the 2 curves
        let (e0, e1) = ((wl[0] - m).exp(), (wl[1] - m).exp());
        let zz = e0 + e1;
        (
            [e0 / zz, e1 / zz],
            [sl[0].clamp(-25.0, 25.0).exp(), sl[1].clamp(-25.0, 25.0).exp()],
            [dl[0].clamp(-25.0, 25.0).exp(), dl[1].clamp(-25.0, 25.0).exp()],
        )
    }

    /// p(recall) after elapsed dt, given hidden state h. Mirrors BatchedGRU.p_recall.
    #[inline]
    fn p_recall(&self, h: &[f64; 7], dt: f64) -> f64 {
        let (wc, sc, dc) = self.curve(h);
        let mut r = 0.0;
        for k in 0..2 {
            r += wc[k] * (1.0 + dt / (1e-7 + sc[k])).powf(-dc[k]);
        }
        (1.0 - 1e-7) * r
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    parallel, deck_size, learn_span, seed, w, learn_costs, review_costs,
    first_rating_prob, review_rating_prob, max_cost_perday, learn_limit_perday,
    review_limit_perday, s_min, s_max, max_same_day, n_iter, policy, policy_param,
    gru_weights=None, retention_table=None, s_grid=None, d_grid=None,
))]
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
    gru_weights: Option<PyReadonlyArray2<'py, f64>>,
    // SSP-MMC policy inputs (policy == "ssp_mmc"): per-user target-retention table flattened
    // [d, s_long, s_short] -> (parallel, d_size*s_size*s_size), plus the solver's s/d grids.
    retention_table: Option<PyReadonlyArray2<'py, f64>>,
    s_grid: Option<PyReadonlyArray1<'py, f64>>,
    d_grid: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64, // t_fsrs: seconds in FSRS-7 math (update_state, curve, curve-inverse, SSP lookup)
    f64, // t_gru: seconds in GRU inference (p_recall, step)
)> {
    let pol = match policy {
        "fixed" => Policy::Fixed,
        "dr" => Policy::Dr,
        "memrise" => Policy::Memrise,
        "sm2" => Policy::Sm2,
        "ssp_mmc" => Policy::SspMmc,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown policy {other:?} (use fixed/dr/memrise/sm2/ssp_mmc)"
            )));
        }
    };

    let w = w.as_array();
    let learn_costs = learn_costs.as_array();
    let review_costs = review_costs.as_array();
    let first_rating_prob = first_rating_prob.as_array();
    let review_rating_prob = review_rating_prob.as_array();

    // Optional GRU pseudo-ground-truth recall predictor: when supplied, the GRU decides
    // recall + the recorded knowledge; FSRS-7 still schedules. Per-user weights are one
    // flat (parallel, GRU_FLAT_LEN) row each (BatchedGRU.flat_weights()).
    let gru_view = gru_weights.as_ref().map(|g| g.as_array());
    if let Some(gv) = &gru_view {
        if gv.shape() != [parallel, GRU_FLAT_LEN] {
            return Err(PyValueError::new_err(format!(
                "gru_weights must be ({parallel}, {GRU_FLAT_LEN}), got {:?}",
                gv.shape()
            )));
        }
    }

    // SSP-MMC: grids (shared across users) + per-user retention table.
    let sgrid_v: Option<Vec<f64>> = s_grid.as_ref().map(|g| g.as_array().to_vec());
    let dgrid_v: Option<Vec<f64>> = d_grid.as_ref().map(|g| g.as_array().to_vec());
    let ret_view = retention_table.as_ref().map(|r| r.as_array());
    if matches!(pol, Policy::SspMmc) {
        let ok = match (&sgrid_v, &dgrid_v, &ret_view) {
            (Some(sg), Some(dg), Some(rv)) => {
                rv.shape() == [parallel, sg.len() * sg.len() * dg.len()]
            }
            _ => false,
        };
        if !ok {
            return Err(PyValueError::new_err(
                "policy 'ssp_mmc' requires retention_table (parallel, d*s*s), s_grid, d_grid",
            ));
        }
    }
    let mut t_fsrs_ns: u128 = 0;
    let mut t_gru_ns: u128 = 0;

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

        // GRU predictor state for this user (if enabled): per-user weights + per-card hidden h.
        let gw = gru_view.as_ref().map(|gv| {
            let row: Vec<f64> = (0..GRU_FLAT_LEN).map(|i| gv[[p, i]]).collect();
            GruW::from_row(&row)
        });
        let mut hgru: Vec<[f64; 7]> = if gw.is_some() {
            vec![[0.0; 7]; deck_size]
        } else {
            Vec::new()
        };
        // SSP-MMC per-user target-retention table, flattened [d, s_long, s_short].
        let ret_row: Option<Vec<f64>> = ret_view
            .as_ref()
            .map(|rv| (0..rv.shape()[1]).map(|i| rv[[p, i]]).collect());

        for today in 0..learn_span {
            let today_f = today as f64;

            // memorized snapshot (start of day, learned cards). The whole loop is one op
            // class -- GRU p_recall when enabled, else FSRS forgetting_curve -- so block-time
            // it (one Instant/day) to keep profiler overhead negligible.
            let t_snap = Instant::now();
            let mut mem = 0.0f64;
            for c in 0..deck_size {
                if s_long[c] > 0.0 {
                    let dt = (today_f - last_date[c]).max(0.0);
                    mem += match &gw {
                        Some(g) => g.p_recall(&hgru[c], dt),
                        None => forgetting_curve(
                            dt,
                            s_long[c].clamp(s_min, s_max),
                            s_short[c].clamp(s_min, s_max),
                            diff[c].clamp(D_MIN, D_MAX),
                            &wd,
                        ),
                    };
                }
            }
            if gw.is_some() {
                t_gru_ns += t_snap.elapsed().as_nanos();
            } else {
                t_fsrs_ns += t_snap.elapsed().as_nanos();
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
                        let t_r = Instant::now();
                        let r = match &gw {
                            Some(g) => g.p_recall(&hgru[c], dt),
                            None => forgetting_curve(
                                dt,
                                s_long[c].clamp(s_min, s_max),
                                s_short[c].clamp(s_min, s_max),
                                diff[c].clamp(D_MIN, D_MAX),
                                &wd,
                            ),
                        };
                        if gw.is_some() {
                            t_gru_ns += t_r.elapsed().as_nanos();
                        } else {
                            t_fsrs_ns += t_r.elapsed().as_nanos();
                        }
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
                        let t_us = Instant::now();
                        let (nsl, nss, nd) =
                            update_state(dt, rating, s_long[c], s_short[c], diff[c], &wd, s_min, s_max);
                        t_fsrs_ns += t_us.elapsed().as_nanos();
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

                    // advance the GRU hidden state with the realized (dt, rating): learning
                    // uses (dt = 0, first_rating); a review uses its elapsed dt and rev rating.
                    if let Some(g) = &gw {
                        let t_st = Instant::now();
                        hgru[c] = g.step(&hgru[c], dt, rating);
                        t_gru_ns += t_st.elapsed().as_nanos();
                    }

                    // schedule the next interval / due
                    let is_learn = learn_cand;
                    let prev = if is_learn { 0.0 } else { ivl[c] };
                    let base_time = if is_learn { today_f } else { t_review };
                    let new_ivl: f64 = match pol {
                        Policy::Fixed => policy_param,
                        Policy::Dr => {
                            let t_inv = Instant::now();
                            let v = forgetting_curve_inverse(
                                policy_param, s_long[c], s_short[c], diff[c], &wd, n_iter, MIN_IVL,
                                S_MAX,
                            );
                            t_fsrs_ns += t_inv.elapsed().as_nanos();
                            v
                        }
                        Policy::SspMmc => {
                            // SSP-MMC: look up the optimal target retention for this state
                            // (nearest grid cell), then invert the curve to an interval.
                            let t_inv = Instant::now();
                            let sg = sgrid_v.as_ref().unwrap();
                            let dg = dgrid_v.as_ref().unwrap();
                            let rt = ret_row.as_ref().unwrap();
                            let ssz = sg.len();
                            let sl_i = s2i(sg, s_long[c]);
                            let ss_i = s2i(sg, s_short[c]);
                            let d_i = d2i(dg, diff[c]);
                            let target_r = rt[(d_i * ssz + sl_i) * ssz + ss_i];
                            let v = forgetting_curve_inverse(
                                target_r, s_long[c], s_short[c], diff[c], &wd, n_iter, MIN_IVL,
                                S_MAX,
                            );
                            t_fsrs_ns += t_inv.elapsed().as_nanos();
                            v
                        }
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
        (t_fsrs_ns as f64) / 1e9,
        (t_gru_ns as f64) / 1e9,
    ))
}
