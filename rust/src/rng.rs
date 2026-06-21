//! Shared deterministic counter-based RNG (splitmix64 finalizer).
//!
//! This MUST stay bit-identical to the Python implementation in
//! `src/ssp_mmc_fsrs/shared_rng.py`. Both sides use u64 wrapping arithmetic and the
//! same float conversion, so the same `(counter, key)` always maps to the same `f64`.
//!
//! It is *counter-based*: there is no sequential state. The random value for a draw is a
//! pure function of a counter (derived from draw kind + day + deck + card) and the seed.
//! That makes it order-independent — numpy can vectorize it, Rust can compute it per
//! cell, and a future CUDA port can compute it per thread — all producing identical
//! streams from the same seed.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

const M1: u64 = 0xbf58476d1ce4e5b9;
const M2: u64 = 0x94d049bb133111eb;
/// 2^53, the number of representable mantissa steps in [0, 1) for an f64.
const TWO_POW_53: f64 = 9007199254740992.0;

/// splitmix64 finalizer (a.k.a. its mixing function). Strong diffusion, u64 -> u64.
#[inline]
pub fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(M1);
    z = (z ^ (z >> 27)).wrapping_mul(M2);
    z ^ (z >> 31)
}

/// Hash a counter under a key (the seed). Mixing the key first decorrelates seeds, and
/// the outer mix decorrelates adjacent counters (which are often structured/sequential).
#[inline]
pub fn hash_counter(counter: u64, key: u64) -> u64 {
    mix(counter.wrapping_add(mix(key)))
}

/// Uniform double in [0, 1): take the top 53 bits and divide by 2^53.
#[inline]
pub fn uniform(counter: u64, key: u64) -> f64 {
    ((hash_counter(counter, key) >> 11) as f64) * (1.0 / TWO_POW_53)
}

/// Parity helper exposed to Python: map an array of counters to uniforms in [0, 1).
/// Used by `tests/test_rng_parity.py` to prove bit-for-bit agreement with numpy.
#[pyfunction]
pub fn rng_uniforms<'py>(
    py: Python<'py>,
    counters: PyReadonlyArray1<'py, u64>,
    key: u64,
) -> Bound<'py, PyArray1<f64>> {
    let counters = counters.as_array();
    let out: Vec<f64> = counters.iter().map(|&c| uniform(c, key)).collect();
    out.into_pyarray(py)
}
