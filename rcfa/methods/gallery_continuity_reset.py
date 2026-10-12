"""Fixed R1 continuity-reset candidate with direct-score builder integration.

Detector constants are exploratory modeling choices, not calibrated probabilities.
Existing cubic history, gallery map and row-L2 readout are unchanged. Integer
accumulation only accelerates summation of stored float32 detector contributions.
"""
import math

import numpy as np

from rcfa.methods.gallery_discriminative_readout import (
    FINAL_ERROR_LIMIT, apply_discriminative_readout, discriminative_mapping,
)
from rcfa.methods.gallery_discriminative_row_l2 import row_l2_scores
from scripts.cubic_kernel_reference import prepare
from scripts.cubic_kernel_stable import StableRawKernelHistory

EPS = 1e-12
CONFIRM = 3


def _unit(row):
    values = np.asarray(row, dtype=np.float64)
    norm = math.hypot(*values)
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite detector norm")
    return values / norm if norm else np.zeros_like(values), norm


def _gallery_threshold(gallery):
    units = []
    for row in gallery:
        direction, norm = _unit(row)
        if norm <= EPS:
            raise ValueError("Detector gallery direction unresolved")
        units.append(direction)
    pairs = sorted(math.fsum(float(a) * float(b) for a, b in zip(units[i], units[j]))
                   for i in range(len(units)) for j in range(i))
    return pairs[(99 * len(pairs) + 99) // 100 - 1]


class _Float32Sum:
    """Exact dyadic sum in units of2**-149; round once when exposing float64.

    Storage is O(D log T), rather than rescanning all accepted frames. For
    float32 inputs this yields the correctly rounded exact sum also targeted by
    math.fsum. It does not certify the subsequent cosine or threshold comparison.
    """
    def __init__(self, dimension):
        self.totals = [0] * dimension

    def add(self, row):
        values = np.asarray(row)
        if values.dtype != np.float32 or values.shape != (len(self.totals),) or not np.isfinite(values).all():
            raise ValueError("Expected one finite native float32 contribution")
        for d, word in enumerate(np.ascontiguousarray(values).view(np.uint32).tolist()):
            exponent = (word >> 23) & 255
            mantissa = word & ((1 << 23) - 1)
            integer = ((mantissa | (1 << 23)) << (exponent - 1)) if exponent else mantissa
            self.totals[d] += -integer if word >> 31 else integer

    def value(self):
        return np.array([math.ldexp(float(total), -149) for total in self.totals])


class ContinuityDetector:
    """Online detector independent of scoring; fixed q99 / three valid lows."""
    def __init__(self, gallery):
        g = np.asarray(gallery)
        if (g.ndim != 2 or min(g.shape) < 2 or not np.issubdtype(g.dtype, np.floating)
                or not np.isfinite(g).all()):
            raise ValueError("Finite floating gallery with at least two rows/dimensions required")
        self.dimension = g.shape[1]
        self.theta = _gallery_threshold(g)
        self.accepted = _Float32Sum(self.dimension)
        self.pending = []
        self.first_low = None
        self.tau = self.frames = 0

    def step(self, frame):
        x = np.asarray(frame)
        if x.dtype != np.float32 or x.shape != (self.dimension,) or not np.isfinite(x).all():
            raise ValueError("Expected one finite native float32 frame")
        t = self.frames
        direction, norm = _unit(x)
        self.frames += 1
        if norm <= EPS:
            return self.tau, False
        with np.errstate(over="raise", invalid="raise", under="raise"):
            h = (x[None] * np.maximum(np.linalg.norm(x[None], axis=1, keepdims=True), EPS))[0]
        reference, length = _unit(self.accepted.value())
        cosine = (math.fsum(float(a) * float(b) for a, b in zip(direction, reference))
                  if length else None)
        if length == 0 or cosine >= self.theta:
            self.pending.clear()
            self.first_low = None
            self.accepted.add(h)
            return self.tau, False
        if not self.pending:
            self.first_low = t
        self.pending.append(h.copy())
        if len(self.pending) < CONFIRM:
            return self.tau, False
        self.tau = self.first_low
        self.accepted = _Float32Sum(self.dimension)
        for contribution in self.pending:
            self.accepted.add(contribution)
        self.pending.clear()
        self.first_low = None
        return self.tau, True


class ContinuityResetState:
    """Streaming prepared parent state; capacity is a storage bound only.

    step_raw returns the unchanged parent's uncentered kernel score. Apply the
    fixed readout to a row or a batch; never revise an already emitted row.
    """
    def __init__(self, gallery, capacity):
        g = np.asarray(gallery)
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("Positive integer capacity required")
        self.detector = ContinuityDetector(g)
        _, self.coordinates, self.basis = prepare(np.zeros((1, g.shape[1]), np.float32), g)
        self.transform, self.bias, _ = discriminative_mapping(self.coordinates)
        amplification = float(np.max(np.sum(np.abs(self.transform), axis=1)))
        self.raw_error_limit = FINAL_ERROR_LIMIT / (2 * max(1.0, amplification))
        self.capacity = capacity
        self.history = self._new_history()
        self.projected = []
        self.tau = self.frames = 0
        self.events = []
        self.replayed_frames = self.completed_recomputations = 0

    def _new_history(self):
        return StableRawKernelHistory(self.coordinates, 3, self.capacity, self.raw_error_limit)

    def step_raw(self, frame):
        x = np.asarray(frame)
        if x.dtype != np.float32 or x.shape != (self.basis.shape[1],) or not np.isfinite(x).all():
            raise ValueError("Expected one finite native float32 frame")
        if self.frames >= self.capacity:
            raise ValueError("History storage capacity exceeded")
        with np.errstate(over="raise", invalid="raise", under="raise"):
            h = x[None] * np.maximum(np.linalg.norm(x[None], axis=1, keepdims=True), EPS)
            z = np.einsum("td,rd->tr", h, self.basis, optimize=False)[0]
        start, event = self.detector.step(x)
        self.projected.append(z.copy())
        if event:
            suffix = self.projected[start - self.tau:]
            self.completed_recomputations += self.history.recomputations
            self.history = self._new_history()
            for row in suffix:
                raw = self.history.step(row)
            self.replayed_frames += len(suffix)
            self.projected = suffix
            self.tau = start
            self.events.append(self.frames)
        else:
            raw = self.history.step(z)
        self.frames += 1
        return raw

    def readout(self, raw):
        return row_l2_scores(apply_discriminative_readout(raw, self.transform, self.bias))


def gallery_continuity_reset_diagnostics(input_features, gallery):
    x = np.asarray(input_features)
    if x.dtype != np.float32 or x.ndim != 2 or len(x) < 1 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite float32 sequence")
    state = ContinuityResetState(gallery, len(x))
    raw = np.array([state.step_raw(row) for row in x], dtype=np.float64)
    scores = state.readout(raw)
    return scores, dict(theta=state.detector.theta, reset_frames=state.events,
                        replayed_frames=state.replayed_frames,
                        recomputations=state.completed_recomputations + state.history.recomputations)


def gallery_continuity_reset_scores(input_features, gallery):
    return gallery_continuity_reset_diagnostics(input_features, gallery)[0]
