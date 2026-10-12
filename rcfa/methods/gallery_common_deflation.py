"""Static gallery common-direction deflation for the frozen p2 incumbent."""

from __future__ import annotations

import numpy as np

from example import normalize
from rcfa.embedding import _validate_features, dual_ema_norm_p2


def normalized_gallery_and_common_direction(
    gallery: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized gallery rows and their normalized mean direction."""
    rows = np.asarray(gallery)
    if (
        rows.ndim != 2
        or rows.shape[0] == 0
        or rows.shape[1] == 0
        or not np.issubdtype(rows.dtype, np.floating)
        or not np.isfinite(rows).all()
    ):
        raise ValueError("Expected a nonempty finite floating gallery matrix")
    normalized = normalize(rows.astype(np.float64, copy=False))
    common = normalize(normalized.mean(axis=0, keepdims=True))[0]
    return normalized, common


def deflate_direction(
    states: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove one fixed direction and fall back only at the existing norm floor."""
    x = _validate_features(states).astype(np.float64, copy=False)
    u = np.asarray(direction, dtype=np.float64)
    if u.shape != (x.shape[1],) or not np.isfinite(u).all():
        raise ValueError("Direction must be one finite embedding vector")

    coefficient = np.einsum("nd,d->n", x, u, optimize=False)
    output = x - coefficient[:, None] * u[None, :]
    output_norm = np.linalg.norm(output, axis=1)
    fallback = output_norm < 1e-12
    if fallback.any():
        output[fallback] = x[fallback]
        output_norm[fallback] = np.linalg.norm(output[fallback], axis=1)
    return output, fallback, output_norm


def process_gallery_common_deflation(
    input_features: np.ndarray, gallery: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Deflate the supplied-gallery mean from submission 922934's exact state."""
    features = _validate_features(input_features)
    normalized_gallery, common = normalized_gallery_and_common_direction(gallery)
    if normalized_gallery.shape[1] != features.shape[1]:
        raise ValueError(
            f"Embedding dimension {features.shape[1]} != gallery dimension "
            f"{normalized_gallery.shape[1]}"
        )

    parent = dual_ema_norm_p2(features)
    output, fallback, output_norm = deflate_direction(parent, common)
    parent_norm = np.linalg.norm(parent, axis=1)
    ratio = output_norm / np.maximum(parent_norm, 1e-12)
    if not np.isfinite(output).all() or not np.isfinite(ratio).all():
        raise FloatingPointError("Nonfinite gallery-common deflation output")
    return output, {
        "common_direction": common,
        "fallback": fallback,
        "norm_ratio": ratio,
        "parent": parent,
    }
