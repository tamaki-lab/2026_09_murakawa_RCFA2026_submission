"""Fixed per-frame mean centering of the unchanged927687 float32 scores.

Local processor only: no builder registration or extra temporal state. This
tests cross-frame calibration, not forgery detection or column-rank correction.
"""
from __future__ import annotations

import numpy as np

from example import normalize
from rcfa.methods.gallery_cumulative_norm_p2 import process_gallery_cumulative_norm_p2


def cohort_center_scores(scores: np.ndarray) -> np.ndarray:
    """Return float32 `(score - full_row_mean) / 2`, without clipping.

    Float64 arithmetic follows the parent's final float32 rounding in the
    end-to-end processor. A shared affine transform preserves within-row order
    in real arithmetic; casting can create ties, particularly near subnormals.
    """
    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("Expected nonempty(T,K) scores with K >= 2")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError("Scores must be floating point")
    if not np.isfinite(values).all() or np.any(np.abs(values) > 1.0):
        raise ValueError("Scores must be finite and within[-1,1]")
    working = values.astype(np.float64, copy=False)
    mean = np.mean(working, axis=1, keepdims=True)
    result = ((working - mean) / 2.0).astype(np.float32)
    if not np.isfinite(result).all() or np.any(np.abs(result) > 1.0):
        raise FloatingPointError("Centered scores violate the score contract")
    return result


def cumulative_cohort_centered_scores(
    input_features: np.ndarray, gallery: np.ndarray
) -> np.ndarray:
    """Reuse927687 including its pre-centering float32 score rounding.

    Input features/gallery keep their native dtype on the parent path. Real
    builder inputs are float32; do not change accumulation or raw-gallery SVD.
    Fresh parent state on every call, no Track parameter, current/past only.
    """
    embeddings, _ = process_gallery_cumulative_norm_p2(input_features, gallery)
    parent_scores = (normalize(embeddings) @ normalize(gallery).T).astype(np.float32)
    return cohort_center_scores(parent_scores)
