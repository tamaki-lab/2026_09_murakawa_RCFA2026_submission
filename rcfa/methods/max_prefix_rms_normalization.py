"""Fixed max-prefix-RMS readout; parent energy order, one peak scalar.

No floor, clipping, forgetting, recentering or tunable parameters.
Initial high reference can permanently attenuate genuine observations.
"""
import numpy as np


def _readout(values, norm, peak):
    if (not np.isfinite(norm).all() or not np.isfinite(peak).all()
            or np.any(norm < 0) or np.any(peak < 0)
            or np.any((norm > 0) & (peak == 0))):
        raise FloatingPointError("Unresolved max-prefix-RMS denominator")
    denominator = np.maximum(norm, peak)
    out = np.zeros_like(values)
    np.divide(values, denominator[:, None], out=out,
              where=denominator[:, None] != 0)
    result = out.astype(np.float32)
    if not np.isfinite(result).all() or np.any(np.abs(result) > 1):
        raise FloatingPointError("Max-prefix-RMS output contract failed")
    return result


class MaxPrefixRMSNormalizer:
    """Atomic chunk state; parent reduction order plus one float64 peak."""

    def __init__(self):
        self._frames = 0
        self._energy = np.float64(0)
        self._peak = np.float64(0)
        self._columns = None

    def process(self, parent_scores):
        scores = np.asarray(parent_scores)
        if scores.dtype != np.float32:
            raise TypeError("Expected float32 pre-row-L2 scores")
        if scores.ndim != 2 or scores.shape[1] == 0:
            raise ValueError("Expected two-dimensional scores with nonempty gallery")
        if self._columns is not None and scores.shape[1] != self._columns:
            raise ValueError("Gallery column count changed")
        if not np.isfinite(scores).all() or np.any(np.abs(scores) > 1):
            raise ValueError("Input must be finite and within [-1,1]")
        if (not isinstance(self._frames, int) or self._frames < 0
                or not np.isfinite(self._energy) or self._energy < 0
                or not np.isfinite(self._peak) or self._peak < 0):
            raise FloatingPointError("Invalid max-prefix-RMS state")
        if self._frames + len(scores) > 2**53:
            raise ValueError("Frame count not exactly representable")
        values = scores.astype(np.float64)
        squared_norm = np.zeros(len(values), dtype=np.float64)
        energy, frames, peak = self._energy, self._frames, self._peak
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            # Keep the parent's column and chronological reduction order.
            for column in range(values.shape[1]):
                squared_norm += values[:, column] * values[:, column]
            norm = np.sqrt(squared_norm)
            peaks = np.empty(len(values), dtype=np.float64)
            for index, current_energy in enumerate(squared_norm):
                energy = np.float64(energy + current_energy)
                frames += 1  # Including all-zero rows.
                reference = np.sqrt(energy / np.float64(frames))
                peak = np.maximum(peak, reference)
                peaks[index] = peak
            result = _readout(values, norm, peaks)
        if not np.isfinite(energy) or not np.isfinite(peak):
            raise FloatingPointError("Nonfinite max-prefix-RMS state")
        self._energy, self._frames, self._peak = energy, frames, peak
        if len(scores):
            self._columns = scores.shape[1]
        return result



def max_prefix_rms_scores(parent_scores):
    """One complete sequence with fresh state."""
    return MaxPrefixRMSNormalizer().process(parent_scores)
