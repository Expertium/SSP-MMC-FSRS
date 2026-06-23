//! Rust port of the FSRS-7 SSP-MMC Bellman solver (CPU), f64.
//!
//! Reference / source of truth: `src/ssp_mmc_fsrs/solver7.py` (`SSPMMCSolver7`). This is the
//! CPU implementation that would ship in Anki (the solver is per-user and Anki-bound; GPU is
//! experiment-only). It mirrors the eager Modified-Policy-Iteration path: build the
//! hyperparameter-independent transitions (target-R -> interval -> next state per rating) +
//! achieved recall once, then value-iterate to the unique V* with `mpi_eval` cheap
//! fixed-policy sweeps per greedy backup. Returns the per-state target-retention table (the
//! policy) plus split build/solve timings so we can compare against the Python GPU solver.
//!
//! Grids, costs and hyperparameters are passed in from Python so the CPU and GPU solvers run
//! on byte-identical inputs (fair timing + a clean correctness cross-check).

use std::time::Instant;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::simulate7::{d2i, forgetting_curve, forgetting_curve_inverse, s2i, update_state};

const MIN_IVL: f64 = 1.0 / 1440.0; // 1 minute in days (matches fsrs7.MIN_INTERVAL_DAYS)

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn bellman_fsrs7<'py>(
    py: Python<'py>,
    w: PyReadonlyArray1<'py, f64>,
    review_costs: PyReadonlyArray1<'py, f64>,
    review_rating_prob: PyReadonlyArray1<'py, f64>, // [H, G, E]
    s_state: PyReadonlyArray1<'py, f64>,
    d_state: PyReadonlyArray1<'py, f64>,
    r_state: PyReadonlyArray1<'py, f64>,
    log_sl: bool,
    log_ss: bool,
    exp_sl: f64,
    exp_ss: f64,
    exp_d: f64,
    base_succ: f64,
    wf_sl: f64,
    wf_ss: f64,
    wf_d: f64,
    ws_sl: f64,
    ws_ss: f64,
    ws_d: f64,
    w_ret: f64,
    s_min: f64,
    s_max: f64,
    discount: f64,
    n_iter: usize,
    convergence_tol: f64,
    mpi_eval: usize,
    inv_n_iter: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize, f64, f64)> {
    let w: Vec<f64> = w.as_array().to_vec();
    let rc: Vec<f64> = review_costs.as_array().to_vec();
    let rrp: Vec<f64> = review_rating_prob.as_array().to_vec();
    let sg: Vec<f64> = s_state.as_array().to_vec();
    let dg: Vec<f64> = d_state.as_array().to_vec();
    let rg: Vec<f64> = r_state.as_array().to_vec();
    let (s_size, d_size, r_size) = (sg.len(), dg.len(), rg.len());
    let n_states = d_size * s_size * s_size;
    let flat = |d: usize, sl: usize, ss: usize| (d * s_size + sl) * s_size + ss;

    // ── build transitions + achieved recall (hyperparameter-INDEPENDENT) ──
    let t_build0 = Instant::now();
    let mut r_pred = vec![0.0f64; n_states * r_size];
    let mut trans: [Vec<u32>; 4] = [
        vec![0u32; n_states * r_size],
        vec![0u32; n_states * r_size],
        vec![0u32; n_states * r_size],
        vec![0u32; n_states * r_size],
    ];
    for d in 0..d_size {
        for sl in 0..s_size {
            for ss in 0..s_size {
                let st = flat(d, sl, ss);
                let (sl_v, ss_v, d_v) = (sg[sl], sg[ss], dg[d]);
                for (ri, &r) in rg.iter().enumerate() {
                    // target retention r -> interval -> achieved recall + next state per rating
                    let t =
                        forgetting_curve_inverse(r, sl_v, ss_v, d_v, &w, inv_n_iter, MIN_IVL, s_max);
                    r_pred[st * r_size + ri] = forgetting_curve(t, sl_v, ss_v, d_v, &w);
                    for g in 1..=4i64 {
                        let (nsl, nss, nd) = update_state(t, g, sl_v, ss_v, d_v, &w, s_min, s_max);
                        trans[(g - 1) as usize][st * r_size + ri] =
                            flat(d2i(&dg, nd), s2i(&sg, nsl), s2i(&sg, nss)) as u32;
                    }
                }
            }
        }
    }
    let t_build = t_build0.elapsed().as_secs_f64();

    // ── const_cost = sum_k prob_k * cost_k (hyperparameter-DEPENDENT) ──
    let t_solve0 = Instant::now();
    let s_ratio = |s: f64, log: bool| {
        if log {
            (1.0 + s).ln() / (1.0 + s_max).ln()
        } else {
            s / s_max
        }
    };
    let mut const_cost = vec![0.0f64; n_states * r_size];
    let mut cc_max = 0.0f64;
    for d in 0..d_size {
        let mod_d = (((dg[d] - 1.0) / 9.0).clamp(0.0, 1.0)).powf(exp_d);
        for sl in 0..s_size {
            let mod_sl = s_ratio(sg[sl], log_sl).powf(exp_sl);
            for ss in 0..s_size {
                let mod_ss = s_ratio(sg[ss], log_ss).powf(exp_ss);
                let st = flat(d, sl, ss);
                let fail_cost = rc[0] * (1.0 + wf_sl * mod_sl + wf_ss * mod_ss + wf_d * mod_d);
                let succ_mod = base_succ + ws_sl * mod_sl + ws_ss * mod_ss + ws_d * mod_d;
                let succ = [rc[1] * succ_mod, rc[2] * succ_mod, rc[3] * succ_mod];
                for ri in 0..r_size {
                    let rp = r_pred[st * r_size + ri];
                    let mut cc = (1.0 - rp) * fail_cost;
                    for j in 0..3 {
                        cc += (rp * rrp[j]) * (succ[j] - w_ret * rp);
                    }
                    const_cost[st * r_size + ri] = cc;
                    if cc > cc_max {
                        cc_max = cc;
                    }
                }
            }
        }
    }

    // ── Modified Policy Iteration with a uniform upper-bound init (== Python eager MPI) ──
    let upper = cc_max / (1.0 - discount);
    // terminal: s_long index == s_size-1 (a maximal-long-stability card is "done"); for
    // st = (d*s_size+sl)*s_size+ss, sl = (st / s_size) % s_size.
    let is_terminal = |st: usize| (st / s_size) % s_size == s_size - 1;
    let mut state = vec![upper; n_states];
    for st in 0..n_states {
        if is_terminal(st) {
            state[st] = 0.0;
        }
    }
    let mut buf = vec![0.0f64; n_states]; // optimal/ev scratch (Jacobi)
    let mut policy = vec![0usize; n_states];

    // greedy action value at a state, reading the current `state` (Jacobi).
    let action_value = |state: &[f64], base: usize, ri: usize| -> f64 {
        let off = base + ri;
        let rp = r_pred[off];
        const_cost[off]
            + discount
                * ((1.0 - rp) * state[trans[0][off] as usize]
                    + (rp * rrp[0]) * state[trans[1][off] as usize]
                    + (rp * rrp[1]) * state[trans[2][off] as usize]
                    + (rp * rrp[2]) * state[trans[3][off] as usize])
    };

    let mut it = 0usize;
    let mut cost_diff = 1e9f64;
    while it < n_iter && cost_diff > convergence_tol {
        it += 1;
        // greedy backup: optimal value + argmin policy per state (from old state)
        for st in 0..n_states {
            let base = st * r_size;
            let mut best_v = f64::INFINITY;
            let mut best_r = 0usize;
            for ri in 0..r_size {
                let av = action_value(&state, base, ri);
                if av < best_v {
                    best_v = av;
                    best_r = ri;
                }
            }
            buf[st] = best_v;
            policy[st] = best_r;
        }
        cost_diff = 0.0;
        for st in 0..n_states {
            if buf[st] < state[st] {
                let dd = state[st] - buf[st];
                if dd > cost_diff {
                    cost_diff = dd;
                }
                state[st] = buf[st];
            }
        }
        if cost_diff <= convergence_tol {
            break;
        }
        // cheap fixed-policy evaluation sweeps (1 action/state)
        for _ in 0..mpi_eval {
            for st in 0..n_states {
                buf[st] = action_value(&state, st * r_size, policy[st]);
            }
            for st in 0..n_states {
                if buf[st] < state[st] {
                    state[st] = buf[st];
                }
            }
        }
    }
    let t_solve = t_solve0.elapsed().as_secs_f64();

    // ── recover the argmin action at V* -> per-state target retention (the policy) ──
    let mut retention = vec![0.0f64; n_states];
    for st in 0..n_states {
        let base = st * r_size;
        let mut best_v = f64::INFINITY;
        let mut best_r = 0usize;
        for ri in 0..r_size {
            let av = action_value(&state, base, ri);
            if av < best_v {
                best_v = av;
                best_r = ri;
            }
        }
        retention[st] = rg[best_r];
    }

    Ok((retention.into_pyarray(py), it, t_build, t_solve))
}
