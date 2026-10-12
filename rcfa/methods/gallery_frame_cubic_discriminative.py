"""Local fixed memoryless cubic-discriminative + row-L2 candidate.

Each observation uses the *existing singleton* stable degree-three/readout
path. Gallery geometry is shared, but no observation state crosses frames.
Registered as a direct-score builder method. This module has no file/ZIP I/O.
"""

import numpy as np

from rcfa.methods.gallery_discriminative_readout import (
    FINAL_ERROR_LIMIT,
    apply_discriminative_readout,
    discriminative_mapping,
)
from rcfa.methods.gallery_discriminative_row_l2 import row_l2_scores
from scripts.cubic_kernel_reference import prepare
from scripts.cubic_kernel_stable import stable_raw_prepared_scores


def frame_cubic_discriminative_centered_prepared(projected, gallery_coordinates):
    """Return the frozen singleton readout BEFORE final row normalization.

    The inherited error screen covers only this discriminative path, not any
    downstream normalization. Keep the one-row kernel and affine operations.
    """
    z = np.asarray(projected, dtype=np.float64)
    coordinates = np.asarray(gallery_coordinates, dtype=np.float64)
    if (z.ndim != 2 or coordinates.ndim != 2 or len(z) == 0
            or coordinates.shape[0] < 2 or coordinates.shape[1] < 1
            or z.shape[1] != coordinates.shape[1]
            or not np.isfinite(z).all() or not np.isfinite(coordinates).all()):
        raise ValueError("Expected nonempty finite prepared rows and gallery")
    transform, bias, mapping = discriminative_mapping(coordinates)
    amplification = float(np.max(np.sum(np.abs(transform), axis=1)))
    raw_error_limit = FINAL_ERROR_LIMIT / (2 * max(1.0, amplification))
    scores = np.empty((len(z), len(coordinates)), dtype=np.float32)
    info = dict(
        frames=len(z), recomputations=0, recompute_seconds=0.0,
        max_exact_support=0, max_raw_error_estimate=0.0,
        raw_error_limit=raw_error_limit, map_amplification=amplification,
        map_error=mapping["map_error"], projected_shape=list(z.shape),
        max_history_frames=1, zero_centered_rows=0,
        min_centered_row_norm=float("inf"),
    )
    for i in range(len(z)):
        # Keep a one-row shape through the affine map as well: changing GEMV
        # into batched GEMM can change rounding/ties after tanh and row L2.
        raw, diagnostics = stable_raw_prepared_scores(
            z[i:i + 1], coordinates, 3, raw_error_limit)
        centered = apply_discriminative_readout(raw, transform, bias)
        if (centered.dtype != np.float32 or centered.shape != (1, len(coordinates))
                or not np.isfinite(centered).all() or np.any(np.abs(centered) > 1)):
            raise ValueError("Invalid singleton centered scores")
        scores[i] = centered[0]
        norm = float(np.linalg.norm(centered.astype(np.float64)))
        info["zero_centered_rows"] += int(norm == 0)
        info["min_centered_row_norm"] = min(info["min_centered_row_norm"], norm)
        for name in ("recomputations", "recompute_seconds"):
            info[name] += diagnostics[name]
        for name in ("max_exact_support", "max_raw_error_estimate"):
            info[name] = max(info[name], diagnostics[name])
    if not np.isfinite(scores).all() or np.any(np.abs(scores) > 1):
        raise FloatingPointError("Memoryless final score contract failed")
    return scores, info


def frame_cubic_discriminative_row_l2_prepared(projected, gallery_coordinates):
    """Existing944672 readout; fixed column reductions are independent of batch size.

    The inherited raw error screen is NOT a final row-L2 certificate.
    Independent final-output validation remains required near zero rows.
    """
    centered, info = frame_cubic_discriminative_centered_prepared(
        projected, gallery_coordinates)
    return row_l2_scores(centered), info


def frame_cubic_discriminative_row_l2_diagnostics(input_features, gallery):
    projected, coordinates, _ = prepare(input_features, gallery)
    return frame_cubic_discriminative_row_l2_prepared(projected, coordinates)


def frame_cubic_discriminative_row_l2_scores(input_features, gallery):
    return frame_cubic_discriminative_row_l2_diagnostics(input_features, gallery)[0]
