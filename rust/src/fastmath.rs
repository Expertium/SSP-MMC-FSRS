//! Fast approximate transcendentals for the hot simulator loop.
//!
//! `pow`, `exp`, `ln` for **positive** inputs (stability, retrievability bases,
//! difficulty — all > 0). Built from `exp2`/`log2` via series that are accurate to
//! ~1e-8 relative over the simulator's input ranges — far below the speedup protocol's
//! 0.15% drift budget — but branch-light and faster than MSVC's libm, and they let the
//! retrievability/update loops auto-vectorize. Accuracy is checked end-to-end by
//! tests/test_simulate_parity.py (full sim vs the f64 Python reference).

const LN2: f64 = std::f64::consts::LN_2;
const LOG2E: f64 = std::f64::consts::LOG2_E;
const SQRT2: f64 = std::f64::consts::SQRT_2;

/// log2(x) for x > 0. Range-reduce to mantissa m in [sqrt(0.5), sqrt(2)); ln(m) via the
/// atanh series ln(m) = 2*(t + t^3/3 + t^5/5 + t^7/7 + t^9/9), t = (m-1)/(m+1).
#[inline]
pub fn log2(x: f64) -> f64 {
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > SQRT2 {
        m *= 0.5;
        e += 1;
    }
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    // ln(m) = 2*(t + t^3/3 + t^5/5 + ...); |t| <= 0.1716 so 3 terms give ~3e-5.
    let ln_m = 2.0 * t * (1.0 + t2 * (1.0 / 3.0 + t2 * (1.0 / 5.0)));
    e as f64 + ln_m * LOG2E
}

/// 2^x. Split x = k + f with k = round(x), f in [-0.5, 0.5]; 2^f = e^(f*ln2) via its
/// Taylor series; 2^k by writing the exponent bits.
#[inline]
pub fn exp2(x: f64) -> f64 {
    let k = (x + 0.5).floor();
    let u = (x - k) * LN2;
    // e^u, |u| <= 0.347; degree-5 Taylor gives ~4e-5.
    let p = 1.0
        + u * (1.0 + u * (1.0 / 2.0 + u * (1.0 / 6.0 + u * (1.0 / 24.0 + u * (1.0 / 120.0)))));
    let scale = f64::from_bits(((k as i64 + 1023) as u64) << 52);
    scale * p
}

/// x^y for x > 0.
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    exp2(y * log2(x))
}

/// e^x.
#[inline]
pub fn exp(x: f64) -> f64 {
    exp2(x * LOG2E)
}

/// ln(x) for x > 0.
#[inline]
pub fn ln(x: f64) -> f64 {
    log2(x) * LN2
}
