"""bank8 computed on a fixed-shape input: prefix-determinism by construction (same on both tracks).

`bank8p(x, g)` places the probe sequence x (T x D, 1 <= T <= L_MAX) in a fixed L_MAX x D float32 block whose remaining
rows are zeros, runs the unchanged bank8 pipeline on it (the base scorer from the raw inputs, `t1_bank._BASE`, then
`t1_bank8.post8`) and returns the first T rows. Why: bank8's formulas read frames <= t only, but in the unpadded
computation some floating-point associations depend on the sequence length (the HCP memory's normaliser sums whole
zero-padded triangle rows with numpy's pairwise summation; batch products over the time axis may take shape-dependent
kernels; bank8 review round 3, 2026-10-11). With the fixed block, every piece of arithmetic that contributes to an
emitted row has a shape fixed by L_MAX or by causal quantities of frames <= t (the jump J, detected at frame J; the
suffix and segment computations that depend on J are consumed only by rows t >= J), so the arithmetic of row t cannot
depend on how many real frames follow it. The zero rows enter no row < T mathematically (a jump detected inside the
padding lies at or after T). L_MAX = 4096 exceeds every released sequence (T <= ~2,750); longer sequences are rejected
rather than computed with another shape.
`bank8p_ref` is the same wrapper around `t1_bank8.bank8_ref` (the builder's bit-identity reference).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import t1_bank8 as B8  # noqa: E402

L_MAX = 4096


def pad(x):
    """(L_MAX x D float32 block with x in its first T rows and zeros below, T)."""
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or not 1 <= x.shape[0] <= L_MAX:
        raise ValueError(f"bank8p needs a (T x D) array with 1 <= T <= {L_MAX}, got shape {x.shape}")
    T = x.shape[0]
    p = np.zeros((L_MAX, x.shape[1]), np.float32)
    p[:T] = x
    return p, T


def bank8p(x, g):
    p, T = pad(x)
    return B8.bank8(p, g)[:T]


def bank8p_ref(x, g):
    p, T = pad(x)
    return B8.bank8_ref(p, g)[:T]


def outputs_fn(x, g):
    return {"bank8p": bank8p_ref(x, g)}
