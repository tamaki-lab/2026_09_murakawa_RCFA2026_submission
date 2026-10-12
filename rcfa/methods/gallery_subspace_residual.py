"""Gallery-row-subspace decomposition for aggregate-only RCFA diagnostics."""

from __future__ import annotations

import numpy as np

from example import normalize
from rcfa.embedding import _validate_features


class GallerySubspace:
    """Fixed projector derived only from the provided gallery embeddings."""

    def __init__(self, gallery: np.ndarray):
        gallery = np.asarray(gallery)
        if (
            gallery.ndim != 2
            or gallery.shape[0] == 0
            or gallery.shape[1] == 0
            or not np.issubdtype(gallery.dtype, np.floating)
            or not np.isfinite(gallery).all()
        ):
            raise ValueError("Expected a nonempty finite floating gallery matrix")
        normalized = normalize(gallery.astype(np.float64, copy=False))
        _, singular, vh = np.linalg.svd(normalized, full_matrices=False)
        tolerance = (
            singular[0] * max(normalized.shape) * np.finfo(np.float64).eps
            if singular.size and singular[0] > 0.0
            else 0.0
        )
        self.basis = vh[singular > tolerance].copy()
        self.dimension = normalized.shape[1]

    @property
    def rank(self) -> int:
        return len(self.basis)

    def decompose(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return row-space component, orthogonal residual and energy fraction."""
        x = _validate_features(features).astype(np.float64, copy=False)
        if x.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {x.shape[1]} != gallery dimension {self.dimension}"
            )
        coefficients = np.einsum("nd,kd->nk", x, self.basis, optimize=False)
        projected = np.einsum("nk,kd->nd", coefficients, self.basis, optimize=False)
        residual = x - projected
        energy = np.einsum("nd,nd->n", x, x, optimize=False)
        residual_energy = np.einsum("nd,nd->n", residual, residual, optimize=False)
        fraction = residual_energy / np.maximum(energy, 1e-24)
        return projected, residual, np.clip(fraction, 0.0, 1.0)


def residual_change(residual: np.ndarray) -> np.ndarray:
    """Return causal angular displacement between consecutive residuals."""
    residual = _validate_features(residual).astype(np.float64, copy=False)
    unit = normalize(residual)
    norm = np.linalg.norm(residual, axis=1)
    change = np.zeros(len(residual), dtype=np.float64)
    if len(residual) > 1:
        change[1:] = 1.0 - np.sum(unit[1:] * unit[:-1], axis=1)
        both_zero = (norm[1:] <= 1e-12) & (norm[:-1] <= 1e-12)
        change[1:][both_zero] = 0.0
    change[np.abs(change) < 1e-15] = 0.0
    return np.clip(change, 0.0, 2.0)
