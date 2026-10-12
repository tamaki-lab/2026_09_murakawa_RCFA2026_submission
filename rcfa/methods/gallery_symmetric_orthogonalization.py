"""Canonical symmetric gallery reconditioning for RCFA diagnostics/candidate."""

from __future__ import annotations

import numpy as np

from example import normalize
from rcfa.embedding import _validate_features
from rcfa.embedding import dual_ema_norm_p2


def symmetric_gallery_map(gallery: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Return (G.T G)^(+1/2) and its nonzero spectral diagnostics."""
    # Challenge embeddings are float32.  The inverse fractional power must be
    # constructed in float64 or its symmetry and row-permutation equivariance
    # errors are orders of magnitude above the frozen mechanical tolerances.
    g = normalize(_validate_features(gallery).astype(np.float64, copy=False))
    gram = g @ g.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    largest = float(eigenvalues[-1])
    tolerance = largest * max(gram.shape) * np.finfo(eigenvalues.dtype).eps
    positive = eigenvalues > tolerance
    if not np.any(positive):
        raise ValueError("gallery Gram matrix has no positive eigenvalues")
    inverse_three_halves = np.zeros_like(eigenvalues)
    inverse_three_halves[positive] = eigenvalues[positive] ** -1.5
    gram_inverse_three_halves = (
        eigenvectors * inverse_three_halves[np.newaxis, :]
    ) @ eigenvectors.T
    transform = g.T @ gram_inverse_three_halves @ g
    nonzero = eigenvalues[positive]
    condition = float(nonzero[-1] / nonzero[0])
    return transform, {
        "rank": int(np.count_nonzero(positive)),
        "tolerance": float(tolerance),
        "gram_condition": condition,
        "response_condition": float(np.sqrt(condition)),
        "symmetry_max_abs_error": float(np.max(np.abs(transform - transform.T))),
    }


def apply_symmetric_gallery_map(
    states: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    """Apply the fixed map to row-vector states without final normalization."""
    values = _validate_features(states)
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (values.shape[1], values.shape[1]):
        raise ValueError("transform shape does not match feature dimension")
    if not np.isfinite(matrix).all():
        raise ValueError("transform must be finite")
    # Fix the contraction order per row so evaluating a prefix cannot select a
    # different BLAS kernel and change earlier outputs at roundoff scale.
    result = np.einsum("nd,df->nf", values, matrix, optimize=False)
    if not np.isfinite(result).all():
        raise ValueError("transformed states must be finite")
    return result


def score_profiles(states: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Raw cosine profiles used by the unchanged submission builder."""
    return normalize(_validate_features(states)) @ normalize(
        _validate_features(gallery)
    ).T


def process_gallery_symmetric_orthogonalization(
    input_features: np.ndarray, gallery: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply the frozen half-inverse map once to submission 922934's state."""
    features = _validate_features(input_features)
    gallery_values = _validate_features(gallery)
    if features.shape[1] != gallery_values.shape[1]:
        raise ValueError(
            f"Embedding dimension {features.shape[1]} != gallery dimension "
            f"{gallery_values.shape[1]}"
        )
    parent = dual_ema_norm_p2(features)
    transform, spectrum = symmetric_gallery_map(gallery_values)
    output = apply_symmetric_gallery_map(parent, transform)
    if not np.isfinite(output).all():
        raise FloatingPointError("nonfinite symmetric-gallery output")
    return output, {
        "parent": parent,
        "transform": transform,
        "spectrum": spectrum,
    }
