"""Fixed row-L2 postprocessing of the final float32 scores of parent941755.

No new temporal state, floor, centering, clipping or tuning parameter. The
saved float32 parent values, not an idealized pre-rounding score, are the input.
"""

import numpy as np

from rcfa.methods.gallery_discriminative_readout import gallery_discriminative_scores


def row_l2_scores(parent_scores):
    """Normalize each score row with a fixed left-to-right float64 reduction.

    Accept any positive gallery count for synthetic tests (officially 200).
    Empty sequences are allowed; empty galleries and non-float32 scores are not.
    All-zero rows become positive zero. No input array is mutated.
    """
    scores = np.asarray(parent_scores)
    if scores.dtype != np.float32:
        raise TypeError("Row-L2 requires the parent's final float32 scores")
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("Expected two-dimensional scores with a nonempty gallery")
    if not np.isfinite(scores).all() or np.any(np.abs(scores) > 1):
        raise ValueError("Parent scores must be finite and within [-1, 1]")

    values = scores.astype(np.float64)
    squared_norm = np.zeros(len(values), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        # Explicit column order, independent of strides and number of frames.
        for column in range(values.shape[1]):
            squared_norm += values[:, column] * values[:, column]
        norm = np.sqrt(squared_norm)
        normalized = np.zeros_like(values)
        np.divide(values, norm[:, None], out=normalized, where=norm[:, None] != 0)
        result = normalized.astype(np.float32)
    if not np.isfinite(result).all() or np.any(np.abs(result) > 1):
        raise FloatingPointError("Row-L2 score contract failed")
    return result


def gallery_discriminative_row_l2_scores(input_features, gallery):
    """Use the unmodified parent's final output, then apply the fixed readout."""
    return row_l2_scores(gallery_discriminative_scores(input_features, gallery))
