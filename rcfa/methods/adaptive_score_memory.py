"""Track-agnostic online score-history adaptation; no labels or track IDs."""
from __future__ import annotations

import numpy as np

from example import normalize
from rcfa.embedding import (
    SHORT_TAIL_DUAL_EMA_SLOW_ALPHA as SLOW,
    SHORT_TAIL_DUAL_EMA_FAST_ALPHA as FAST,
    SHORT_TAIL_DUAL_EMA_FAST_WEIGHT as WEIGHT,
)

HISTORY_ALPHA = 0.10
SCORE_SCALE = 0.20


class AdaptiveScoreMemory:
    """One independent state per sequence, using only current/past scores.

    The short-tail reference state is never frozen. Persistent directional score
    residuals blend its embedding toward the current embedding before raw cosine
    scoring. The gate is neither a liveness probability nor a score penalty.
    """

    def __init__(self, gallery: np.ndarray):
        gallery = np.asarray(gallery, dtype=np.float64)
        if gallery.ndim != 2 or min(gallery.shape) < 1 or not np.isfinite(gallery).all():
            raise ValueError("Expected finite nonempty gallery matrix")
        # Caller supplies the same normalized gallery used for final scoring.
        self.gallery = gallery.copy()
        self.slow = None
        self.fast = None
        self.residual_mean = np.zeros(gallery.shape[0], dtype=np.float64)
        self.residual_energy = 0.0
        self.gate = 0.0

    def step(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float64)
        if frame.shape != (self.gallery.shape[1],) or not np.isfinite(frame).all():
            raise ValueError("Expected finite frame with gallery embedding dimension")
        if self.slow is None:
            self.slow = frame.copy()
            self.fast = frame.copy()
            return frame.copy()
        self.slow = SLOW * frame + (1.0 - SLOW) * self.slow
        self.fast = FAST * frame + (1.0 - FAST) * self.fast
        base = (1.0 - WEIGHT) * self.slow + WEIGHT * self.fast
        residual = self.gallery @ (normalize(frame) - normalize(base))
        # Remove a gallery-wide offset; use relative score changes, not identity IDs.
        residual -= residual.mean()
        beta = HISTORY_ALPHA
        self.residual_mean = (1.0 - beta) * self.residual_mean + beta * residual
        self.residual_energy = (
            (1.0 - beta) * self.residual_energy + beta * float(residual @ residual)
        )
        coherence = float(self.residual_mean @ self.residual_mean) / max(
            self.residual_energy, 1e-12
        )
        magnitude = float(np.max(np.abs(self.residual_mean))) / SCORE_SCALE
        self.gate = float(np.clip(coherence, 0.0, 1.0) * np.clip(magnitude, 0.0, 1.0))
        return (1.0 - self.gate) * base + self.gate * frame


def process_adaptive(features: np.ndarray, gallery: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("Expected nonempty feature matrix")
    state = AdaptiveScoreMemory(gallery)
    output = np.empty(features.shape, dtype=np.float64)
    gates = np.empty(features.shape[0], dtype=np.float64)
    for index, frame in enumerate(features):
        output[index] = state.step(frame)
        gates[index] = state.gate
    return output, gates
