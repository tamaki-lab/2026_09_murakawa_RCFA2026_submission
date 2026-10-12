"""Fixed max-prefix F-RMS candidate, registered with the direct-score builder.

Parent948462 singleton cubic discriminative centered float32 scores are
unchanged. Only the prefix RMS reference becomes its historical maximum.
"""
from rcfa.methods.gallery_frame_cubic_discriminative import (
    frame_cubic_discriminative_centered_prepared,
)
from rcfa.methods.max_prefix_rms_normalization import max_prefix_rms_scores
from scripts.cubic_kernel_reference import prepare


def frame_max_prefix_rms_prepared(projected, gallery_coordinates):
    """A complete sequence, fresh state; raw estimates are not final guarantees."""
    centered, info = frame_cubic_discriminative_centered_prepared(
        projected, gallery_coordinates)
    info = dict(info, normalization="max_prefix_rms",
                normalization_history_frames=len(centered))
    return max_prefix_rms_scores(centered), info


def frame_max_prefix_rms_diagnostics(input_features, gallery):
    projected, coordinates, _ = prepare(input_features, gallery)
    return frame_max_prefix_rms_prepared(projected, coordinates)


def frame_max_prefix_rms_scores(input_features, gallery):
    return frame_max_prefix_rms_diagnostics(input_features, gallery)[0]
