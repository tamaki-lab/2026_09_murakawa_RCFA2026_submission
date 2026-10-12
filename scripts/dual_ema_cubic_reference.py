"""Synthetic-only finite-memory cubic reference; NOT a submission processor.

This checks the fixed dual-EMA kernel recurrence against an explicit tiny
tensor. It has no roundoff screen, exact cancellation fallback, challenge
reader, builder registration, or permission to generate an artifact.
"""

import numpy as np

from scripts.cubic_kernel_reference import NumericalUnresolved

SLOW_ALPHA = 0.022
FAST_ALPHA = 0.10
FAST_WEIGHT = (0.03 - SLOW_ALPHA) / (FAST_ALPHA - SLOW_ALPHA)
FLOOR = 1e-12


def _check(projected, gallery):
    z = np.asarray(projected, dtype=np.float64)
    g = np.asarray(gallery, dtype=np.float64)
    if (z.ndim != 2 or g.ndim != 2 or z.shape[0] < 1 or g.shape[0] < 2
            or z.shape[1] != g.shape[1] or not np.isfinite(z).all()
            or not np.isfinite(g).all()):
        raise ValueError("Expected finite prepared observations and gallery")
    return z, g


def _read_raw(numerator, squared_norm):
    if not np.isfinite(squared_norm) or squared_norm < 0:
        raise NumericalUnresolved("Finite-memory cubic norm unresolved")
    if squared_norm == 0 and np.any(numerator != 0):
        raise NumericalUnresolved("Zero norm with nonzero gallery numerator")
    raw = numerator / max(np.sqrt(squared_norm), FLOOR)
    if not np.isfinite(raw).all():
        raise NumericalUnresolved("Finite-memory cubic raw score nonfinite")
    return raw


def kernel_dual_ema_cubic_raw(projected, gallery):
    """O(T^2 r + TK r) raw scores via two EMA norms and their cross term."""
    z, g = _check(projected, gallery)
    units = np.zeros_like(z)
    mass = np.zeros(len(z), dtype=np.float64)
    slow_weights = np.empty(0, dtype=np.float64)
    fast_weights = np.empty(0, dtype=np.float64)
    ns = nf = np.zeros(len(g), dtype=np.float64)
    qss = qff = qsf = 0.0
    raw_rows, norms = [], []

    for t, row in enumerate(z):
        a = float(np.linalg.norm(row))
        u = row / a if a else np.zeros_like(row)
        observed = a * np.einsum("ij,j->i", g, u, optimize=False) ** 3
        self_term = a * a * float(np.dot(u, u)) ** 3
        if t == 0:
            ns = nf = observed.copy()
            qss = qff = qsf = self_term
            slow_weights = fast_weights = np.ones(1, dtype=np.float64)
        else:
            kernel = np.einsum("ij,j->i", units[:t], u, optimize=False) ** 3
            cross_s = a * float(np.dot(slow_weights * mass[:t], kernel))
            cross_f = a * float(np.dot(fast_weights * mass[:t], kernel))
            bs, bf = 1 - SLOW_ALPHA, 1 - FAST_ALPHA
            next_qss = bs * bs * qss + 2 * bs * SLOW_ALPHA * cross_s + SLOW_ALPHA**2 * self_term
            next_qff = bf * bf * qff + 2 * bf * FAST_ALPHA * cross_f + FAST_ALPHA**2 * self_term
            next_qsf = (bs * bf * qsf + bs * FAST_ALPHA * cross_s
                        + SLOW_ALPHA * bf * cross_f + SLOW_ALPHA * FAST_ALPHA * self_term)
            qss, qff, qsf = next_qss, next_qff, next_qsf
            ns = bs * ns + SLOW_ALPHA * observed
            nf = bf * nf + FAST_ALPHA * observed
            slow_weights = np.append(bs * slow_weights, SLOW_ALPHA)
            fast_weights = np.append(bf * fast_weights, FAST_ALPHA)
        units[t], mass[t] = u, a
        weight = FAST_WEIGHT
        q = (1 - weight)**2 * qss + 2 * weight * (1 - weight) * qsf + weight**2 * qff
        numerator = (1 - weight) * ns + weight * nf
        raw_rows.append(_read_raw(numerator, q))
        norms.append(q)
    return np.asarray(raw_rows), np.asarray(norms)


def explicit_dual_ema_cubic_raw(projected, gallery):
    """Independent small-rank tensor oracle for synthetic tests only."""
    z, g = _check(projected, gallery)
    if z.shape[1] > 8:
        raise ValueError("Explicit oracle is limited to rank <= 8")
    gf = np.einsum("ki,kj,kl->kijl", g, g, g, optimize=False)
    raw_rows, norms = [], []
    for t, row in enumerate(z):
        a = float(np.linalg.norm(row))
        u = row / a if a else np.zeros_like(row)
        observed = a * np.einsum("i,j,k->ijk", u, u, u, optimize=False)
        if t == 0:
            slow = fast = observed.copy()
        else:
            slow = (1 - SLOW_ALPHA) * slow + SLOW_ALPHA * observed
            fast = (1 - FAST_ALPHA) * fast + FAST_ALPHA * observed
        state = (1 - FAST_WEIGHT) * slow + FAST_WEIGHT * fast
        q = float(np.sum(state * state))
        numerator = np.einsum("ijk,aijk->a", state, gf, optimize=False)
        raw_rows.append(_read_raw(numerator, q))
        norms.append(q)
    return np.asarray(raw_rows), np.asarray(norms)
