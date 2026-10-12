"""Fixed two-sided F-RMS candidate; registration is not performance promotion.

Parent948462 singleton cubic discriminative centered float32 scores are
unchanged. Only the high-energy RMS readout differs; no identity accumulation.
"""
from rcfa.methods.gallery_frame_cubic_discriminative import (
    frame_cubic_discriminative_centered_prepared,
)
from rcfa.methods.two_sided_rms_normalization import two_sided_rms_scores
from scripts.cubic_kernel_reference import prepare


def frame_two_sided_rms_prepared(projected, gallery_coordinates):
    """A complete sequence, fresh state; raw estimates are not final guarantees."""
    centered, info = frame_cubic_discriminative_centered_prepared(
        projected, gallery_coordinates)
    info = dict(info, normalization="two_sided_rms",
                normalization_history_frames=len(centered))
    return two_sided_rms_scores(centered), info


def frame_two_sided_rms_diagnostics(input_features, gallery):
    projected, coordinates, _ = prepare(input_features, gallery)
    return frame_two_sided_rms_prepared(projected, coordinates)


def frame_two_sided_rms_scores(input_features, gallery):
    return frame_two_sided_rms_diagnostics(input_features, gallery)[0]
