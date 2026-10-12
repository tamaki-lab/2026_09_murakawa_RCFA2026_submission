"""Full gallery-row-space projection of the unchanged 912085 raw mean.

Local-stage processor only: no new memory, norm-power or spectral parameters.
The final normalization/scoring convention remains that of the shared builder.
"""

from __future__ import annotations

import numpy as np

from rcfa.methods.gallery_row_space_projection import project_gallery_row_space
from rcfa.methods.gallery_subspace_residual import GallerySubspace
from rcfa.embedding import _validate_features, cumulative_mean


def process_gallery_cumulative_projection(
    input_features: np.ndarray, gallery: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    """Project the raw cumulative parent; do not normalize inputs before it.

    Preserve the parent's accumulation dtype/order, including float32 cumsum
    for float32 probes. Casting inputs to float64 first would change 912085.
    Each call starts a fresh sequence and has no Track-dependent configuration.
    """
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
    parent = cumulative_mean(features)
    output = project_gallery_row_space(parent, subspace)
    if not np.isfinite(output).all():
        raise FloatingPointError("nonfinite cumulative gallery-row-space projection")
    return output, {"parent": parent, "rank": subspace.rank}
