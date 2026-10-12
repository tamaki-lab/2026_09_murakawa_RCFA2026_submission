"""Prequential linear dynamics of residuals; scores are NOT authenticity."""
import numpy as np
from example import normalize
from rcfa.methods.gallery_subspace_residual import GallerySubspace

METHODS = ("persistence", "constant_velocity", "learned_ar2", "learned_mean")
WARMUP = 128
FORGET = .99
VAR_FLOOR = 1e-6


class OnlineDynamics:
    def __init__(self, dimension):
        self.dimension = dimension
        self.weights = np.zeros((dimension, 2 * dimension + 1))
        self.inverse = np.eye(2 * dimension + 1) * 100.
        self.variance = np.ones((4, dimension)) * .01
        self.mean_sum = np.zeros(dimension)
        self.mean_weight = .01
        self.previous = []
        self.count = 0

    def step(self, vector):
        y = np.asarray(vector, dtype=np.float64)
        if y.shape != (self.dimension,) or not np.isfinite(y).all():
            raise ValueError("Invalid dynamics observation")
        self.count += 1
        if len(self.previous) < 2:
            self.previous.append(y.copy())
            return np.zeros((4, 3)), False
        older, last = self.previous
        phi = np.r_[last, older, 1.]
        predicted = np.einsum("ij,j->i", self.weights, phi, optimize=False)
        errors = y - np.stack([last, 2 * last - older, predicted,
                               self.mean_sum / self.mean_weight])
        squared = errors * errors
        variance = np.maximum(self.variance, VAR_FLOOR)
        # MSE, past-variance standardized error, diagonal Gaussian NLL / coordinate.
        metrics = np.stack([squared.mean(axis=1), (squared / variance).mean(axis=1),
                            .5 * (np.log(2 * np.pi * variance) + squared / variance).mean(axis=1)], axis=1)
        # All fitting is AFTER scoring y, without authentic/fake or track labels.
        pphi = np.einsum("ij,j->i", self.inverse, phi, optimize=False)
        gain = pphi / (FORGET + float(np.sum(phi * pphi)))
        self.weights += np.outer(errors[2], gain)
        self.inverse = (self.inverse - np.outer(gain, pphi)) / FORGET
        self.inverse = .5 * (self.inverse + self.inverse.T)
        self.variance = FORGET * self.variance + (1 - FORGET) * squared
        self.mean_sum = FORGET * self.mean_sum + y
        self.mean_weight = FORGET * self.mean_weight + 1
        self.previous = [last.copy(), y.copy()]
        if not np.isfinite(metrics).all() or not np.isfinite(self.weights).all():
            raise FloatingPointError("Nonfinite dynamics state")
        return metrics, self.count > WARMUP


def dynamics_signals(features, gallery):
    x = normalize(np.asarray(features, dtype=np.float64))
    residual = GallerySubspace(gallery).decompose(x)[1]
    dimension = min(16, x.shape[1])
    # Fixed data-independent projection. Compression, not a trained normal space.
    rng = np.random.default_rng(20260909)
    basis, _ = np.linalg.qr(rng.normal(size=(x.shape[1], dimension)))
    projected = np.einsum("nd,dk->nk", residual, basis, optimize=False)
    state = OnlineDynamics(dimension)
    rows = [state.step(y) for y in projected]
    return np.stack([r[0] for r in rows]), np.array([r[1] for r in rows])
