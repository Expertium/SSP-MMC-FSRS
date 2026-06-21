//! Fast approximate transcendentals for the hot simulator loop.
//!
//! f32 `pow`/`exp`/`ln` for **positive** inputs (stability, retrievability bases,
//! difficulty — all > 0). Built from `exp2`/`log2` via short series (~1e-5 over the
//! simulator's input ranges) — well within the speedup protocol's drift budget — and
//! branch-light. f32<->i32 conversions are AVX2-native (f64<->i64 needs AVX-512), so the
//! hot loops can vectorize. Accuracy is checked end-to-end by tests/test_simulate_parity.py.

const LN2: f32 = std::f32::consts::LN_2;
const LOG2E: f32 = std::f32::consts::LOG2_E;
const SQRT2: f32 = std::f32::consts::SQRT_2;

/// log2(x) for x > 0. Range-reduce to mantissa m in [sqrt(0.5), sqrt(2)); ln(m) via the
/// atanh series ln(m) = 2*(t + t^3/3 + t^5/5), t = (m-1)/(m+1).
#[inline]
pub fn log2_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let mut e = ((bits >> 23) & 0xff) as i32 - 127;
    let mut m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    if m > SQRT2 {
        m *= 0.5;
        e += 1;
    }
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let ln_m = 2.0 * t * (1.0 + t2 * (1.0 / 3.0 + t2 * (1.0 / 5.0)));
    e as f32 + ln_m * LOG2E
}

/// 2^x. Split x = k + f with k = round(x); 2^f = e^(f*ln2) (degree-5 Taylor); 2^k by
/// writing the exponent bits.
#[inline]
pub fn exp2_f32(x: f32) -> f32 {
    let k = (x + 0.5).floor();
    let u = (x - k) * LN2;
    let p = 1.0 + u * (1.0 + u * (0.5 + u * (1.0 / 6.0 + u * (1.0 / 24.0 + u * (1.0 / 120.0)))));
    let scale = f32::from_bits(((k as i32 + 127) as u32) << 23);
    scale * p
}

/// x^y for x > 0.
#[inline]
pub fn pow_f32(x: f32, y: f32) -> f32 {
    exp2_f32(y * log2_f32(x))
}

/// e^x.
#[inline]
pub fn exp_f32(x: f32) -> f32 {
    exp2_f32(x * LOG2E)
}

/// ln(x) for x > 0.
#[inline]
pub fn ln_f32(x: f32) -> f32 {
    log2_f32(x) * LN2
}
