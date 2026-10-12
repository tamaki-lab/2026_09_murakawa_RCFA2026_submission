"""Fixed local F-RMS: singleton cubic discriminative readout + prefix RMS.

Structural parent944672, normalization helper944151. No min fusion, new floor,
coefficient, recentering or file I/O. Not registered with the artifact builder.
Only score-row energy persists: identity evidence is still current-frame only.
"""

from rcfa.methods.gallery_frame_cubic_discriminative import (
    frame_cubic_discriminative_centered_prepared,
)
from rcfa.methods.prefix_rms_normalization import prefix_rms_scores
from scripts.cubic_kernel_reference import prepare


def frame_prefix_rms_prepared(projected, gallery_coordinates):
    """One complete sequence, fresh RMS state; return final float32 and diagnostics.

    Raw error estimates are not a certificate after RMS normalization. Check
    independent FINAL outputs, including zero/tiny rows, before production use.
    """
    centered, info = frame_cubic_discriminative_centered_prepared(
        projected, gallery_coordinates)
    info = dict(info, normalization='prefix_rms',
                normalization_history_frames=len(centered))
    return prefix_rms_scores(centered), info


def frame_prefix_rms_diagnostics(input_features, gallery):
    projected, coordinates, _ = prepare(input_features, gallery)
    return frame_prefix_rms_prepared(projected, coordinates)


def frame_prefix_rms_scores(input_features, gallery):
    return frame_prefix_rms_diagnostics(input_features, gallery)[0]
