"""Fast float64 re-implementation of the 935112/941755 cubic history for research.

Not the submission path: no roundoff certification. Used only to explore variants
on the proxy; any promoted candidate is rebuilt with a validated implementation.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cubic_kernel_reference import prepare  # noqa: E402
from rcfa.methods.gallery_discriminative_readout import discriminative_mapping  # noqa: E402
from rcfa.methods.gallery_cohort_centering import cohort_center_scores  # noqa: E402


def _prep(x, g):
    z, coords, _ = prepare(np.asarray(x, np.float32), np.asarray(g, np.float32))
    z = np.asarray(z, np.float64)
    a = np.linalg.norm(z, axis=1)
    u = z / np.maximum(a, 1e-300)[:, None]
    return u, a, np.asarray(coords, np.float64)


def cumulative_cubic_raw(u, a, coords, gamma=1.0):
    """raw_t = sum_k w_k (u_k.c)^3 / sqrt(sum_kl w_k w_l (u_k.u_l)^3), w=a^gamma."""
    w = a ** gamma
    num = np.cumsum(w[:, None] * (u @ coords.T) ** 3, axis=0)
    K = (u @ u.T) ** 3
    M = np.tril(K * w[:, None] * w[None, :], -1)
    q = np.cumsum(2 * M.sum(1) + w * w * np.diag(K))
    return num / np.sqrt(np.maximum(q, 1e-300))[:, None]


def ema_cubic_raw(u, a, coords, alpha, gamma=1.0):
    """Exponentially weighted cubic state; alpha may be a per-frame array."""
    n = len(u)
    w = a ** gamma
    al = np.broadcast_to(np.asarray(alpha, np.float64), (n,)).copy()
    al[0] = 1.0
    K = (u @ u.T) ** 3
    G = (u @ coords.T) ** 3
    coef = np.zeros(n)  # weight of frame k in current state
    out = np.empty((n, coords.shape[0]))
    num = np.zeros(coords.shape[0])
    q = 0.0
    for t in range(n):
        c = 1.0 - al[t]
        cross = w[t] * np.dot(coef[:t], K[:t, t])
        q = c * c * q + 2 * c * al[t] * cross + (al[t] * w[t]) ** 2 * K[t, t]
        coef[:t] *= c
        coef[t] = al[t] * w[t]
        num = c * num + al[t] * w[t] * G[t]
        out[t] = num / np.sqrt(max(q, 1e-300))
    return out


def readout(raw, coords):
    transform, bias, _ = discriminative_mapping(coords)
    return cohort_center_scores(np.tanh(raw @ transform.T + bias).astype(np.float32))


def cubic_scores(x, g):
    u, a, c = _prep(x, g)
    return cohort_center_scores(cumulative_cubic_raw(u, a, c).astype(np.float32))


def discrim_scores(x, g, gamma=1.0):
    u, a, c = _prep(x, g)
    return readout(cumulative_cubic_raw(u, a, c, gamma), c)
