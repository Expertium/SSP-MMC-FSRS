"""Shared deterministic counter-based RNG (splitmix64 finalizer).

This MUST stay bit-identical to the Rust implementation in `rust/src/rng.rs`. Both sides
use 64-bit wrapping arithmetic and the same float conversion, so the same ``(counter,
key)`` always maps to the same ``float64``.

It is *counter-based*: there is no sequential state. The random value for a draw is a
pure function of a counter (derived from draw kind + day + deck + card) and the seed.
That makes it order-independent, so numpy can compute a whole ``(parallel, deck_size)``
block of draws at once with plain array ops.

numpy footgun avoided here: ``uint64`` mixed with a Python ``int`` silently promotes to
``float64`` and loses precision. Every constant and shift amount below is therefore a
``np.uint64``, so all arithmetic stays in wrapping ``uint64``.
"""

from __future__ import annotations

import numpy as np

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30 = np.uint64(30)
_S27 = np.uint64(27)
_S31 = np.uint64(31)
_S11 = np.uint64(11)
# 1 / 2^53. 2^53 is the number of representable mantissa steps in [0, 1) for a float64.
_INV_TWO_POW_53 = 1.0 / 9007199254740992.0


def _mix(z: np.ndarray | np.uint64) -> np.ndarray | np.uint64:
    """splitmix64 finalizer; matches `mix()` in rng.rs. Operates elementwise on uint64."""
    z = (z ^ (z >> _S30)) * _M1
    z = (z ^ (z >> _S27)) * _M2
    return z ^ (z >> _S31)


def _hash_counter(counter: np.ndarray, key: int) -> np.ndarray:
    """Hash counters under a key (the seed); matches `hash_counter()` in rng.rs.

    The key is mixed via a 1-element array (not a uint64 *scalar*) so the wraparound
    goes through numpy's array ufunc, which wraps silently. A uint64 scalar multiply
    instead raises a spurious ``RuntimeWarning: overflow`` even though the wrap is wanted.
    """
    mixed_key = _mix(np.array([key], dtype=np.uint64))[0]
    return _mix(counter + mixed_key)


def uniform(counter, key) -> np.ndarray:
    """Uniform float64 in [0, 1) for each counter; matches `uniform()` in rng.rs."""
    counter = np.asarray(counter, dtype=np.uint64)
    hashed = _hash_counter(counter, key)
    return (hashed >> _S11).astype(np.float64) * _INV_TWO_POW_53


# --- Simulator draw helpers -------------------------------------------------
# Draw-kind ids for the simulator's three random draws. Keep these (and the counter
# layout + categorical rule below) in sync with the Rust simulator so the streams match.
KIND_INIT_RATING = 0
KIND_FORGET = 1
KIND_PASS_RATING = 2


def counters(kind, day, parallel, deck_size, learn_span):
    """(parallel, deck_size) uint64 counters, unique across (kind, day, deck, card).

    Layout: ``(kind * learn_span + day) * (parallel * deck_size) + (deck * deck_size + card)``.
    Order-independent, so Python and Rust derive identical counters for the same cell.
    """
    cells = parallel * deck_size
    base = np.uint64((kind * learn_span + day) * cells)
    idx = np.arange(cells, dtype=np.uint64).reshape(parallel, deck_size)
    return base + idx


def uniform_block(kind, day, parallel, deck_size, learn_span, key):
    """Uniforms in [0, 1) for a whole (parallel, deck_size) block of one draw kind."""
    return uniform(counters(kind, day, parallel, deck_size, learn_span), key)


def counters_r(kind, day, rnd, parallel, deck_size, learn_span, max_rounds):
    """(parallel, deck_size) counters with an extra same-day **round** dimension, for the
    FSRS-7 simulator (which can review a card several times a day).

    Layout: ``((kind * learn_span + day) * max_rounds + rnd) * cells + cell``. With
    ``rnd = 0`` for the once-per-card initial rating this stays disjoint across kinds, and
    the Rust simulator derives the identical counter per (day, round, deck, card) cell.
    """
    cells = parallel * deck_size
    base = np.uint64(((kind * learn_span + day) * max_rounds + rnd) * cells)
    idx = np.arange(cells, dtype=np.uint64).reshape(parallel, deck_size)
    return base + idx


def uniform_block_r(kind, day, rnd, parallel, deck_size, learn_span, max_rounds, key):
    """Round-aware ``uniform_block`` for one (parallel, deck_size) block (FSRS-7)."""
    return uniform(
        counters_r(kind, day, rnd, parallel, deck_size, learn_span, max_rounds), key
    )


def categorical(uniforms, probs):
    """Category indices: first i with ``u < cumsum(probs)[i]`` (clipped to k-1).

    Matches the linear scan in the Rust simulator. ``probs`` need not be perfectly
    normalized; any leftover mass falls into the last category via the clip.
    """
    probs = np.asarray(probs, dtype=np.float64)
    cum = np.cumsum(probs)
    idx = np.searchsorted(cum, uniforms, side="right")
    return np.minimum(idx, probs.shape[-1] - 1)
