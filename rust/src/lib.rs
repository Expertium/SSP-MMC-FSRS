//! Rust port of the SSP-MMC-FSRS simulator.
//!
//! Reference (source of truth for correctness): `src/ssp_mmc_fsrs/simulation.py`.
//! The `simulate()` port itself is not here yet; this currently exposes the shared RNG
//! primitive (see `rng.rs`) plus trivial smoke-test functions.

use pyo3::prelude::*;

mod fastmath;
mod rng;
mod simulate;

/// Return the crate version. Trivial smoke-test that the module imports and calls.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Add two floats. Trivial smoke-test for numeric arg passing across the FFI boundary.
#[pyfunction]
fn add(a: f64, b: f64) -> f64 {
    a + b
}

#[pymodule]
fn ssp_mmc_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(rng::rng_uniforms, m)?)?;
    m.add_function(wrap_pyfunction!(simulate::simulate, m)?)?;
    Ok(())
}
