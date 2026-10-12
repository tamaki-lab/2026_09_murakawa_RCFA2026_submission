"""Fixed two-sided prefix RMS; energy is measured before normalization.

No epsilon, clipping, extra centering or parameters. Low branch keeps the
948462 arithmetic. High branch attenuates before a single float32 rounding.
This is a numerical implementation, not an interval error certificate.
"""
import numpy as np


def _readout(values, norm, reference):
    """One final float32 rounding, including on the high branch."""
    if (not np.isfinite(norm).all() or not np.isfinite(reference).all()
            or np.any(norm < 0) or np.any(reference < 0)
            or np.any((norm > 0) & (reference == 0))):
        raise FloatingPointError("Unresolved two-sided RMS denominator")
    denominator = np.maximum(norm, reference)
    normalized = np.zeros_like(values)
    np.divide(values, denominator[:, None], out=normalized,
              where=denominator[:, None] != 0)
    high = norm > reference
    normalized[high] *= (reference[high] / norm[high])[:, None]
    result = normalized.astype(np.float32)
    if not np.isfinite(result).all() or np.any(np.abs(result) > 1):
        raise FloatingPointError("Two-sided RMS output contract failed")
    return result


class TwoSidedRMSNormalizer:
    """Chronological, atomic chunk state with the fixed parent energy arithmetic."""

    def __init__(self):
        self._frames = 0
        self._energy = np.float64(0)
        self._columns = None

    def process(self, parent_scores):
        scores = np.asarray(parent_scores)
        if scores.dtype != np.float32:
            raise TypeError("Expected float32 pre-row-L2 scores")
        if scores.ndim != 2 or scores.shape[1] == 0:
            raise ValueError("Expected nonempty gallery dimension")
        if self._columns is not None and scores.shape[1] != self._columns:
            raise ValueError("Gallery column count changed")
        if not np.isfinite(scores).all() or np.any(np.abs(scores) > 1):
            raise ValueError("Input must be finite and within [-1,1]")
        if self._frames + len(scores) > 2**53:
            raise ValueError("Frame count not exactly representable")
        if (not isinstance(self._frames, int) or self._frames < 0
                or not np.isfinite(self._energy) or self._energy < 0):
            raise FloatingPointError("Invalid two-sided RMS state")
        values = scores.astype(np.float64)
        squared_norm = np.zeros(len(values), dtype=np.float64)
        energy, frames = self._energy, self._frames
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for column in range(values.shape[1]):
                squared_norm += values[:, column] * values[:, column]
            norm = np.sqrt(squared_norm)
            reference = np.empty(len(values), dtype=np.float64)
            for index, current_energy in enumerate(squared_norm):
                energy = np.float64(energy + current_energy)
                frames += 1
                reference[index] = np.sqrt(energy / np.float64(frames))
            result = _readout(values, norm, reference)
        if not np.isfinite(energy):
            raise FloatingPointError("Nonfinite energy")
        self._energy, self._frames = energy, frames
        if len(scores):
            self._columns = scores.shape[1]
        return result



def two_sided_rms_scores(parent_scores):
    """Normalize one complete sequence with fresh state."""
    return TwoSidedRMSNormalizer().process(parent_scores)
