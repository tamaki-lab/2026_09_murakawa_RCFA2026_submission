"""Causal prefix-RMS normalization of final float32 parent scores.

One fixed readout: y_t / max(||y_t||, sqrt(sum_{i<=t} ||y_i||^2 / t)).
The reference changes with every prefix; it is not a sequence-wide constant
or an authenticity estimate. No floor, clipping, recentering or tuning knobs.
"""
import numpy as np


class PrefixRMSNormalizer:
    """Fresh state per sequence; chunk boundaries do not change arithmetic."""

    def __init__(self):
        self._frames = 0
        self._energy = np.float64(0)
        self._columns = None

    def process(self, parent_scores):
        scores = np.asarray(parent_scores)
        if scores.dtype != np.float32:
            raise TypeError("Prefix-RMS requires the parent's final float32 scores")
        if scores.ndim != 2 or scores.shape[1] == 0:
            raise ValueError("Expected two-dimensional scores with a nonempty gallery")
        if self._columns is not None and scores.shape[1] != self._columns:
            raise ValueError("Gallery column count changed within a sequence")
        if not np.isfinite(scores).all() or np.any(np.abs(scores) > 1):
            raise ValueError("Parent scores must be finite and within [-1, 1]")
        if self._frames + len(scores) > 2**53:
            raise ValueError("Frame count is not exactly representable in float64")

        values = scores.astype(np.float64)
        squared_norm = np.zeros(len(values), dtype=np.float64)
        energy, frames = self._energy, self._frames
        # Commit state only after the entire chunk passes, including output.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            # Identical row-energy reduction order to row_l2_scores.
            for column in range(values.shape[1]):
                squared_norm += values[:, column] * values[:, column]
            norm = np.sqrt(squared_norm)
            denominator = np.empty(len(values), dtype=np.float64)
            for index, current_energy in enumerate(squared_norm):
                energy = np.float64(energy + current_energy)
                frames += 1  # Zero rows are observations too.
                reference = np.sqrt(energy / np.float64(frames))
                denominator[index] = max(norm[index], reference)
            normalized = np.zeros_like(values)
            np.divide(values, denominator[:, None], out=normalized,
                      where=denominator[:, None] != 0)
            result = normalized.astype(np.float32)
        if (not np.isfinite(energy) or not np.isfinite(result).all()
                or np.any(np.abs(result) > 1)):
            raise FloatingPointError("Prefix-RMS score contract failed")
        self._energy, self._frames = energy, frames
        if len(scores):
            self._columns = scores.shape[1]
        return result


def prefix_rms_scores(parent_scores):
    """Process one complete sequence with fresh state, without mutating input."""
    return PrefixRMSNormalizer().process(parent_scores)
