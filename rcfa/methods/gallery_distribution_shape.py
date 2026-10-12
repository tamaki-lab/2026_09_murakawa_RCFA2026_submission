"""Fixed causal distribution-shape metric with the native927687 query state.

Local full-rank processor, not registered with the submission builder.
Additional covariance is descriptive dispersion, not learned uncertainty or
an authenticity signal. See docs/experiments/20260917_distribution_shape_local.md.
"""
from __future__ import annotations

import numpy as np

from example import normalize
from rcfa.methods.gallery_cumulative_norm_p2 import process_gallery_cumulative_norm_p2
from rcfa.methods.gallery_row_space_projection import project_gallery_row_space
from rcfa.methods.gallery_subspace_residual import GallerySubspace
from rcfa.embedding import _validate_features


def _length(value: np.ndarray) -> float:
    """Scaled Euclidean norm avoids squaring a tiny nonzero vector to zero."""
    scale = float(np.max(np.abs(value)))
    if scale == 0:
        return 0.0
    unit = value / scale
    result = scale * float(np.sqrt(np.sum(unit * unit)))
    if not np.isfinite(result):
        raise FloatingPointError("nonfinite vector norm")
    return result


def _bounded_float32(values: np.ndarray) -> np.ndarray:
    # Native-normalized float32 gallery rows need not have exact unit norm.
    if (not np.isfinite(values).all()
            or np.max(np.abs(values)) > 1 + 4 * np.finfo(np.float32).eps):
        raise FloatingPointError("distribution-shape score contract violation")
    return np.clip(values, -1, 1).astype(np.float32)


def _metric_scores(
    projected: np.ndarray, covariance: np.ndarray, basis: np.ndarray,
    gamma_unit: np.ndarray, gallery_lengths: np.ndarray, *, identity: bool = False,
) -> np.ndarray:
    query = np.einsum("d,kd->k", projected, basis, optimize=False)
    scale = float(np.max(np.abs(query)))
    if scale == 0:
        return np.zeros(len(gamma_unit), dtype=np.float32)
    query = query / scale
    rank = len(basis)
    metric = np.eye(rank)
    if not identity:
        metric += rank * covariance
        # Checking only the queried quadratic forms is not a global SPD test.
        # Fail on an invalid metric; never rescue it by eigenvalue clipping.
        np.linalg.cholesky(metric)
    solved = np.linalg.solve(metric, np.column_stack((query, gamma_unit.T)))
    probe_energy = float(np.sum(query * solved[:, 0]))
    gallery_energy = np.einsum("kr,rk->k", gamma_unit, solved[:, 1:], optimize=False)
    if probe_energy <= 0 or np.any(gallery_energy <= 0):
        raise FloatingPointError("nonpositive metric energy")
    numerator = np.einsum("kr,r->k", gamma_unit, solved[:, 0], optimize=False)
    cosine = numerator / np.sqrt(probe_energy * gallery_energy)
    attenuation = min(_length(projected) / 1e-12, 1.0)
    return _bounded_float32(attenuation * gallery_lengths * cosine)


class DistributionShapeState:
    """One fresh sequence. The gallery determines geometry, never Track routing."""

    def __init__(self, gallery: np.ndarray):
        gallery = _validate_features(gallery)
        if gallery.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("float32 or float64 gallery required")
        self.subspace = GallerySubspace(gallery)
        self.rank = self.subspace.rank
        self.dimension = self.subspace.dimension
        if self.rank != len(gallery):
            raise ValueError("full gallery row rank required, as in927687")
        with np.errstate(over="raise", invalid="raise"):
            self.gallery = normalize(gallery.copy()).astype(np.float64)
        self.gallery_lengths = np.array([_length(row) for row in self.gallery])
        gamma = np.einsum("nd,kd->nk", self.gallery, self.subspace.basis, optimize=False)
        lengths = np.array([_length(row) for row in gamma])
        if np.any(lengths == 0):
            raise ValueError("nonzero projected gallery rows required")
        # Cosine is invariant to this positive per-row scaling; the original
        # native gallery lengths are separately retained in the output.
        self.gamma_unit = gamma / lengths[:, None]
        self.count = 0
        self.accumulator = None
        self.parent_mean = np.zeros(self.dimension)
        self.parent_embedding = np.zeros(self.dimension)
        self.mass = 0.0
        self.covariance_mean = np.zeros(self.rank)
        self.scatter = np.zeros((self.rank, self.rank))

    @property
    def covariance(self) -> np.ndarray:
        if not self.mass:
            return np.zeros_like(self.scatter)
        c = self.scatter / self.mass
        return (c + c.T) / 2

    def identity_metric_scores(self) -> np.ndarray:
        """Diagnostic SAME-score-kernel counterfactual, not a second candidate."""
        return _metric_scores(self.parent_embedding, self.covariance,
                              self.subspace.basis, self.gamma_unit,
                              self.gallery_lengths, identity=True)

    def step(self, sample: np.ndarray) -> np.ndarray:
        row = np.asarray(sample)
        if row.shape != (self.dimension,):
            raise ValueError("sample dimension mismatch")
        _validate_features(row[None, :])
        if row.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("float32 or float64 probe required")
        if self.accumulator is not None and row.dtype != self.accumulator.dtype:
            raise ValueError("probe dtype must be constant within sequence")
        with np.errstate(over="raise", invalid="raise"):
            native_norm = np.linalg.norm(row[None, :], axis=1, keepdims=True)
            h_native = row[None, :] * np.maximum(native_norm, 1e-12)
            old = np.zeros_like(h_native) if self.accumulator is None else self.accumulator
            accumulator = old + h_native
            count = self.count + 1
            parent = accumulator / np.array([[count]], dtype=np.int64)
        projected = project_gallery_row_space(parent, self.subspace)[0]
        if not np.isfinite(projected).all():
            raise FloatingPointError("nonfinite parent projection")

        h = h_native[0].astype(np.float64)
        weight = _length(h)
        mass, mean, scatter = self.mass, self.covariance_mean, self.scatter
        if weight:
            z = np.einsum("d,kd->k", h / weight, self.subspace.basis, optimize=False)
            mass = self.mass + weight
            if not np.isfinite(mass):
                raise FloatingPointError("cumulative covariance weight overflow")
            delta = z - self.covariance_mean
            mean = self.covariance_mean + (weight / mass) * delta
            with np.errstate(over="raise", invalid="raise"):
                scatter = self.scatter + weight * np.outer(delta, z - mean)
        covariance = scatter / mass if mass else np.zeros_like(scatter)
        covariance = (covariance + covariance.T) / 2
        if not np.isfinite(covariance).all():
            raise FloatingPointError("nonfinite covariance")
        if not np.any(covariance):
            # Identity action, including first/zero/constant input. No warmup.
            unit = normalize(projected[None, :])[0]
            score = _bounded_float32(np.einsum("d,kd->k", unit, self.gallery, optimize=False))
        else:
            score = _metric_scores(projected, covariance, self.subspace.basis,
                                   self.gamma_unit, self.gallery_lengths)
        # Commit only after the numerical checks; a failed step is not a frame.
        self.count, self.accumulator = count, accumulator
        self.parent_mean, self.parent_embedding = parent[0], projected
        self.mass, self.covariance_mean, self.scatter = mass, mean, scatter
        return score


def distribution_shape_scores(input_features: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Fixed full-rank per-frame score path, identical batch/stream operation order."""
    features = _validate_features(input_features)
    state = DistributionShapeState(gallery)
    return np.stack([state.step(row) for row in features])


def distribution_shape_control_scores(input_features: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Exact927687 dispatch; do not confuse this with the internal A=I check."""
    embeddings, _ = process_gallery_cumulative_norm_p2(input_features, gallery)
    return (normalize(embeddings) @ normalize(gallery).T).astype(np.float32)
