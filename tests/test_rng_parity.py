"""Parity test: the shared RNG must produce bit-identical uniforms in Python and Rust.

Run with:  uv run --no-sync python tests/test_rng_parity.py

This is the first verification sub-milestone of the Rust port (step 1): before porting the
simulator loop, prove the random-number primitive both sides will rely on agrees exactly.
Both implementations derive each float64 from the same 64-bit hash, so we demand exact
equality (==), not just closeness.
"""

import sys

import numpy as np

import ssp_mmc_rust
from ssp_mmc_fsrs import shared_rng


def main() -> int:
    # A dense sweep of small counters plus 64-bit edge cases (overflow / boundary bits).
    edges = [
        0,
        1,
        2,
        2**16,
        2**32 - 1,
        2**32,
        2**52,
        2**53,
        2**53 + 1,
        2**63 - 1,
        2**63,
        2**64 - 1,
    ]
    counters = np.concatenate(
        [
            np.arange(0, 100_000, dtype=np.uint64),
            np.array(edges, dtype=np.uint64),
        ]
    )

    keys = [0, 1, 42, 2**63, 2**64 - 1]

    all_ok = True
    for key in keys:
        py = shared_rng.uniform(counters, key)
        rs = ssp_mmc_rust.rng_uniforms(counters, np.uint64(key))

        exact = np.array_equal(py, rs)
        in_range = bool(np.all((rs >= 0.0) & (rs < 1.0)))
        if exact and in_range:
            print(f"key={key:<20} OK  (n={counters.size}, all in [0,1))")
        else:
            all_ok = False
            max_diff = float(np.max(np.abs(py - rs)))
            n_mismatch = int(np.count_nonzero(py != rs))
            print(
                f"key={key:<20} FAIL  mismatches={n_mismatch}  "
                f"max_abs_diff={max_diff:.3e}  in_range={in_range}"
            )
            bad = np.flatnonzero(py != rs)[:5]
            for i in bad:
                print(f"    counter={int(counters[i])}  py={py[i]!r}  rs={rs[i]!r}")

    print()
    if all_ok:
        print("PASS: Python and Rust shared RNG agree bit-for-bit.")
        return 0
    print("FAIL: shared RNG diverges between Python and Rust.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
