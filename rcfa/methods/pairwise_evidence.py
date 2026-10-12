"""Frozen causal pairwise-evidence scores on the unchanged927326 parent.

This is score-level fusion, not an embedding transform or a calibrated test.
Formula and precision path match the completed8a42387 shadow diagnostic.
"""
from __future__ import annotations

import numpy as np

from example import normalize
from rcfa.methods.gallery_row_space_projection import project_gallery_row_space
from rcfa.methods.gallery_subspace_residual import GallerySubspace
from rcfa.embedding import _validate_features, cumulative_mean


def pairwise_evidence_scores(input_features: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Return float32(T,K) scores; fresh pair states per call, no Track argument.

    Keep the audit's float32 cumsum and rounded gallery, float64 projection,
    score contraction, evidence and emission; cast only the final output.
    """
    features = _validate_features(input_features)
    gallery_raw = _validate_features(gallery)
    if features.shape[1] != gallery_raw.shape[1]:
        raise ValueError("probe/gallery dimension mismatch")
    subspace = GallerySubspace(gallery_raw)
    if subspace.rank != len(gallery_raw):
        raise ValueError("full row rank required, as in parent927326")
    g = normalize(gallery_raw.astype(np.float32)).astype(np.float64)
    distance = np.linalg.norm(g[None, :, :] - g[:, None, :], axis=2)
    inverse = np.divide(1., distance, out=np.zeros_like(distance), where=distance > 0)
    parent = project_gallery_row_space(cumulative_mean(features), subspace)
    current = project_gallery_row_space(features, subspace)
    base_scores = normalize(parent) @ g.T
    current_scores = normalize(current) @ g.T
    valid = np.linalg.norm(current, axis=1) > 1e-12
    evidence = np.zeros_like(distance)
    output = np.empty_like(base_scores)
    for t, (base, observation) in enumerate(zip(base_scores, current_scores)):
        if not valid[t]:
            output[t] = base
            continue
        increment = np.clip((observation[None, :] - observation[:, None]) * inverse, -1., 1.)
        updated = np.clip(evidence + increment, 0., 1.)
        weight = np.max(evidence * updated, axis=1)
        evidence = updated
        output[t] = base - weight * np.maximum(base - observation, 0.)
    result = output.astype(np.float32)
    if not np.isfinite(result).all() or np.any(np.abs(result) > 1.):
        raise FloatingPointError("pairwise evidence output violates score contract")
    return result
