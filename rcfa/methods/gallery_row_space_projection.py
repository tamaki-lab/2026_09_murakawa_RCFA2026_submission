"""Canonical gallery-row-space projection of the frozen 922934 state."""

from __future__ import annotations

import numpy as np

from rcfa.methods.gallery_subspace_residual import GallerySubspace
from rcfa.embedding import _validate_features, dual_ema_norm_p2


def project_gallery_row_space(
    values: np.ndarray, subspace: GallerySubspace
) -> np.ndarray:
    """Apply the orthogonal row-space projector with fixed contractions."""
    x = _validate_features(values).astype(np.float64, copy=False)
    if x.shape[1] != subspace.dimension:
        raise ValueError(
            f"Embedding dimension {x.shape[1]} != gallery dimension "
            f"{subspace.dimension}"
        )
    coefficients = np.einsum("nd,kd->nk", x, subspace.basis, optimize=False)
    return np.einsum("nk,kd->nd", coefficients, subspace.basis, optimize=False)


def process_gallery_row_space_projection(
    input_features: np.ndarray, gallery: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    """Project submission 922934's unchanged causal state onto row(G)."""
    features = _validate_features(input_features)
    gallery_values = _validate_features(gallery)
    if features.shape[1] != gallery_values.shape[1]:
        raise ValueError(
            f"Embedding dimension {features.shape[1]} != gallery dimension "
            f"{gallery_values.shape[1]}"
        )
    subspace = GallerySubspace(gallery_values)
    if subspace.rank != len(gallery_values):
        raise ValueError(
            f"Gallery row rank {subspace.rank} != row count {len(gallery_values)}"
        )
    parent = dual_ema_norm_p2(features)
    output = project_gallery_row_space(parent, subspace)
    if not np.isfinite(output).all():
        raise FloatingPointError("nonfinite gallery-row-space projection")
    return output, {"parent": parent, "rank": subspace.rank}
