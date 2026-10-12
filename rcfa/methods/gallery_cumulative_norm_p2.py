"""Fixed raw-norm p2 transfer on the unchanged cumulative-projection parent."""
from __future__ import annotations

import numpy as np

from rcfa.methods.gallery_cumulative_projection import process_gallery_cumulative_projection
from rcfa.embedding import _validate_features


def _process_cumulative_norm_power(
    input_features: np.ndarray, gallery: np.ndarray, norm_power: int
) -> tuple[np.ndarray, dict[str, object]]:
    """Internal p1 control/p2 candidate only; not a tunable processor API.

    Norm, floor and multiplication match the existing dual-EMA p2 input
    transformation, without an input cast. Float32 inputs therefore retain
    float32 norm/scaling/cumsum, followed by the parent's int64-count division
    and float64 projection. The trace's ``parent`` is the weighted cumulative
    mean, not submission927326's unweighted raw-input cumulative mean.
    """
    if norm_power == 1:
        return process_gallery_cumulative_projection(input_features, gallery)
    if norm_power != 2:
        raise ValueError("Only the p1 control and fixed p2 candidate are supported")
    features = _validate_features(input_features)
    # Fail rather than clip/reweight differently when finite input overflows.
    with np.errstate(over="raise", invalid="raise"):
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        scaled = features * np.maximum(norms, 1e-12)
    return process_gallery_cumulative_projection(scaled, gallery)


def process_gallery_cumulative_norm_p2(
    input_features: np.ndarray, gallery: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply fixed p2 before cumulative mean and the existing gallery projector.

    Returns embeddings, not scores. No Track argument, persistent state,
    pairwise evidence or builder registration is introduced here.
    """
    return _process_cumulative_norm_power(input_features, gallery, 2)
