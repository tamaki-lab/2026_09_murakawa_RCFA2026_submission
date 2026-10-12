"""LOCAL synthetic reference only; not a registered submission processor.

Roundoff estimates below are diagnostics, NOT outward-rounded certificates.
No fallback, clipping, adaptive floor or challenge-data reader is provided.
"""
from __future__ import annotations

import numpy as np

from example import normalize
from rcfa.methods.gallery_cohort_centering import cohort_center_scores
from rcfa.methods.gallery_subspace_residual import GallerySubspace
from rcfa.embedding import _validate_features

UNIT_ROUNDOFF = np.finfo(np.float64).eps / 2
RAW_ERROR_LIMIT = 1e-7


class NumericalUnresolved(FloatingPointError):
    """History arithmetic cannot resolve this readout; not a performance FAIL."""


def gamma_bound(operations):
    value = operations * UNIT_ROUNDOFF
    if value >= 1:
        raise NumericalUnresolved("Operation error model exhausted")
    return value / (1 - value)


def power_with_error(dot, absolute_dot_sum, degree, dimension):
    """Forward-error model for stored constants, excluding preprocessing."""
    error = gamma_bound(dimension) * absolute_dot_sum
    value = dot ** degree
    if degree == 1:
        return value, error
    # Expanded difference avoids subtracting nearly equal cubic quantities.
    a = np.abs(dot)
    propagated = error * (3 * a * a + 3 * a * error + error * error)
    return value, propagated + gamma_bound(2) * (a + error) ** 3


def readout(numerator, q, frames):
    """Frozen prefix-mean floor and two-rounding readout for independent oracle."""
    if not np.isfinite(q) or q < 0:
        raise NumericalUnresolved("Invalid squared feature norm")
    raw = (numerator / max(np.sqrt(q), frames * 1e-12)).astype(np.float32)
    return cohort_center_scores(raw[None, :])[0]


class KernelHistory:
    """Finite-capacity causal reference, supporting ONLY degree1 and degree3."""

    def __init__(self, gallery_coordinates, degree, capacity):
        g = np.asarray(gallery_coordinates, dtype=np.float64)
        if (degree not in (1, 3) or capacity < 1 or g.ndim != 2
                or g.shape[0] < 2 or g.shape[1] < 1 or not np.isfinite(g).all()):
            raise ValueError("Invalid degree, capacity or gallery coordinates")
        self.gallery = g.copy()
        self.degree = degree
        self.units = np.zeros((capacity, g.shape[1]))
        self.weights = np.zeros(capacity)
        self.frames = 0
        self.numerator = np.zeros(len(g))
        self.numerator_error = np.zeros(len(g))
        self.q = self.q_error = self.mass = 0.0
        self.max_raw_error = 0.0

    def step(self, z):
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (self.gallery.shape[1],) or not np.isfinite(z).all():
            raise ValueError("Expected one finite projected vector")
        if self.frames == len(self.weights):
            raise ValueError("History capacity exceeded")
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            return self._step(z)

    def _step(self, z):
        a = float(np.linalg.norm(z))
        u = z / a if a != 0 else np.zeros_like(z)
        n = self.frames
        history, weights = self.units[:n], self.weights[:n]
        dot = np.einsum("ij,j->i", history, u, optimize=False)
        absdot = np.einsum("ij,j->i", np.abs(history), np.abs(u), optimize=False)
        kernel, ke = power_with_error(dot, absdot, self.degree, len(u))
        terms = 2 * a * weights * kernel
        term_errors = 2 * a * weights * ke + gamma_bound(3) * np.abs(terms)
        cross = float(np.sum(terms))
        cross_error = float(np.sum(term_errors) + gamma_bound(max(1, n)) * np.sum(np.abs(terms)))
        dotself = float(np.dot(u, u))
        diag, diag_error = power_with_error(dotself, dotself, self.degree, len(u))
        diag_error = a * a * diag_error + gamma_bound(2) * abs(a * a * diag)
        diag = a * a * diag
        new_q = self.q + cross + diag
        self.q_error += cross_error + diag_error + gamma_bound(2) * (abs(self.q) + abs(cross) + abs(diag))
        self.q = new_q
        gd = np.einsum("ij,j->i", self.gallery, u, optimize=False)
        ga = np.einsum("ij,j->i", np.abs(self.gallery), np.abs(u), optimize=False)
        k, err = power_with_error(gd, ga, self.degree, len(u))
        increment = a * k
        self.numerator_error += (a * err + gamma_bound(1) * np.abs(increment)
                                 + gamma_bound(1) * (np.abs(self.numerator) + np.abs(increment)))
        self.numerator += increment
        self.units[n], self.weights[n] = u, a
        self.frames += 1
        self.mass += a
        if not (np.isfinite(self.q) and np.isfinite(self.q_error) and np.isfinite(self.mass)
                and np.isfinite(self.numerator).all() and np.isfinite(self.numerator_error).all()):
            raise NumericalUnresolved("Nonfinite accumulated state/error estimate")
        return self._readout()

    def _readout(self):
        if self.mass == 0:
            return np.zeros(len(self.gallery), dtype=np.float32)
        floor = self.frames * 1e-12
        lo, hi = self.q - self.q_error, self.q + self.q_error
        if hi < 0:
            raise NumericalUnresolved("Negative Q outside roundoff estimate")
        if hi <= floor * floor:
            denominator = lower = upper = floor
        elif lo <= 0:
            raise NumericalUnresolved("Q cancellation unresolved; no zero clamp")
        else:
            denominator = max(np.sqrt(self.q), floor)
            lower, upper = max(np.sqrt(lo), floor), max(np.sqrt(hi), floor)
        raw = self.numerator / denominator
        diagnostic = (self.numerator_error / lower
                      + np.abs(self.numerator) * (1 / lower - 1 / upper)
                      + gamma_bound(2) * np.abs(raw))
        worst = float(np.max(diagnostic))
        self.max_raw_error = max(self.max_raw_error, worst)
        if worst > RAW_ERROR_LIMIT:
            raise NumericalUnresolved("Estimated raw-score uncertainty exceeds fixed local limit")
        return cohort_center_scores(raw.astype(np.float32)[None, :])[0]


def prepare(input_features, gallery):
    x, g = _validate_features(input_features), _validate_features(gallery)
    if x.shape[1] != g.shape[1]:
        raise ValueError("Probe/gallery dimension mismatch")
    subspace = GallerySubspace(g)
    if subspace.rank != len(g):
        raise ValueError("Full row rank required")
    with np.errstate(over="raise", invalid="raise", under="raise"):
        # Deliberately preserve native float32 norm/weighting, before projection.
        h = x * np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        z = np.einsum("td,rd->tr", h, subspace.basis, optimize=False)
        coords = np.einsum("kd,rd->kr", normalize(g), subspace.basis, optimize=False)
    return z, coords, subspace.basis


def history_scores(input_features, gallery, degree):
    z, coords, basis = prepare(input_features, gallery)
    state = KernelHistory(coords, degree, len(z))
    scores = np.array([state.step(row) for row in z], dtype=np.float32)
    return scores, dict(q=state.q, q_error_estimate=state.q_error,
        max_raw_error_estimate=state.max_raw_error,
        history_bytes=state.units.nbytes + state.weights.nbytes,
        basis_orthogonality_error=float(np.max(np.abs(basis @ basis.T - np.eye(len(basis))))))


def explicit_post_aggregation_cubic_scores(projected, gallery_coordinates):
    """Independent mean-then-cube oracle in the prepared row-space coordinates."""
    z = np.asarray(projected, dtype=np.float64)
    g = np.asarray(gallery_coordinates, dtype=np.float64)
    if (z.ndim != 2 or g.ndim != 2 or z.shape[1] != g.shape[1]
            or g.shape[0] < 2 or not np.isfinite(z).all()
            or not np.isfinite(g).all()):
        raise ValueError("Expected finite projected observations and gallery coordinates")
    total = np.zeros(z.shape[1], dtype=np.float64)
    scores = []
    for frames, row in enumerate(z, 1):
        total += row
        q = float(np.dot(total, total))
        if q == 0:
            scores.append(np.zeros(len(g), dtype=np.float32))
            continue
        norm = np.sqrt(q)
        cosine = np.einsum("ij,j->i", g, total, optimize=False) / norm
        attenuation = norm / max(norm, frames * 1e-12)
        raw = (cosine ** 3 * attenuation).astype(np.float32)
        scores.append(cohort_center_scores(raw[None, :])[0])
    return np.array(scores, dtype=np.float32)


def explicit_tensor_scores(projected, gallery_coordinates, degree):
    """Independent tiny oracle only; no automatic fallback from KernelHistory."""
    z = np.asarray(projected, dtype=np.float64)
    g = np.asarray(gallery_coordinates, dtype=np.float64)
    if degree not in (1, 3) or z.shape[1] > 8:
        raise ValueError("Tiny degree1/3 oracle only")
    tensor = np.zeros((z.shape[1],) * degree)
    gf = g if degree == 1 else np.einsum("ki,kj,kl->kijl", g, g, g)
    scores, norms = [], []
    for n, row in enumerate(z, 1):
        a = np.linalg.norm(row)
        u = row / a if a else np.zeros_like(row)
        tensor += a * (u if degree == 1 else np.einsum("i,j,k->ijk", u, u, u))
        q = float(np.sum(tensor * tensor))
        numerator = np.array([np.sum(tensor * v) for v in gf])
        scores.append(readout(numerator, q, n))
        norms.append(q)
    return np.array(scores), np.array(norms)
