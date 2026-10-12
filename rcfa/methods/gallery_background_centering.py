"""Fixed gallery-column background centering on native parent927871.

No builder registration, learned coefficient, temporal state or Track argument.
The background reference is a gallery Gram matrix, NOT a gallery time sequence.
"""
from __future__ import annotations

import numpy as np

from example import normalize
from rcfa.methods.gallery_cohort_centering import cohort_center_scores
from rcfa.methods.gallery_cumulative_norm_p2 import process_gallery_cumulative_norm_p2
from rcfa.embedding import _validate_features


def gallery_background_offsets(gallery: np.ndarray) -> np.ndarray:
    """Return centered leave-self-out column means in parent score units.

    Gallery-only reference: float64 normalization and fixed Gram contraction,
    float32 score rounding, then the unchanged parent cohort-center function.
    Self comparisons participate in ROW centering, but not in COLUMN background
    means. This is not a raw-cosine offset or a separate virtual-probe cumsum.
    """
    values = _validate_features(gallery)
    if values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("Expected at least two nonempty gallery rows")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        unit = normalize(values.astype(np.float64, copy=False))
        gram = np.einsum("id,jd->ij", unit, unit, optimize=False)
        reference = cohort_center_scores(gram.astype(np.float32)).astype(np.float64)
        count = len(values)
        background = (np.sum(reference, axis=0) - np.diag(reference)) / (count - 1)
        beta = background - np.mean(background)
    if not np.isfinite(beta).all() or np.any(np.abs(beta) > 2.0):
        raise FloatingPointError("Invalid gallery background offsets")
    return beta


def apply_gallery_background(
    parent_scores: np.ndarray, beta: np.ndarray, attenuation: np.ndarray
) -> np.ndarray:
    """Apply (native parent score - eta * beta) / 4, returning float32.

    eta is the parent's existing normalization-floor attenuation, not confidence
    or a fitted gate. The fixed factor1/4 ensures the bounded-score contract.
    No clipping, strength switch, standard-deviation scaling or rank selection.
    """
    scores = np.asarray(parent_scores)
    offsets = np.asarray(beta)
    eta = np.asarray(attenuation)
    if (scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2
            or scores.dtype != np.float32 or not np.isfinite(scores).all()
            or np.any(np.abs(scores) > 1.0)):
        raise ValueError("Expected nonempty finite float32(T,K) parent scores in[-1,1]")
    if (offsets.shape != (scores.shape[1],)
            or not np.issubdtype(offsets.dtype, np.floating)
            or not np.isfinite(offsets).all() or np.any(np.abs(offsets) > 2.0)):
        raise ValueError("Expected finite floating(K,) offsets in[-2,2]")
    if (eta.shape != (len(scores),) or not np.issubdtype(eta.dtype, np.floating)
            or not np.isfinite(eta).all() or np.any(eta < 0.0) or np.any(eta > 1.0)):
        raise ValueError("Expected finite floating(T,) attenuation in[0,1]")
    result = ((scores.astype(np.float64) - eta.astype(np.float64)[:, None]
               * offsets.astype(np.float64)[None, :]) / 4.0).astype(np.float32)
    if not np.isfinite(result).all() or np.any(np.abs(result) > 1.0):
        raise FloatingPointError("Background-centered scores violate the contract")
    return result


def cumulative_gallery_background_scores(
    input_features: np.ndarray, gallery: np.ndarray
) -> np.ndarray:
    """Reuse927871 through final float32 centering, then subtract fixed beta.

    Native input norm weighting/cumsum, raw-gallery projector and gallery scoring
    are unchanged. Fresh sequence state per call; only gallery-derived beta is
    time independent. Both Tracks must use this identical function.
    """
    embeddings, _ = process_gallery_cumulative_norm_p2(input_features, gallery)
    b32 = (normalize(embeddings) @ normalize(gallery).T).astype(np.float32)
    c32 = cohort_center_scores(b32)
    norms = np.linalg.norm(embeddings, axis=1)
    eta = norms / np.maximum(norms, 1e-12)
    beta = gallery_background_offsets(gallery)
    return apply_gallery_background(c32, beta, eta)
