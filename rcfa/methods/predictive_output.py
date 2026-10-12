"""Single shared causal predictive-error output mixture, no authenticity claim."""
from collections import deque
import numpy as np
from example import normalize
from rcfa.methods.online_dynamics import dynamics_signals
from rcfa.embedding import _validate_features, dual_ema_short_tail

HISTORY = 128
QUANTILE = .95
EPSILON = 1e-12


def mixture_weights(errors, valid):
    errors, valid = np.asarray(errors), np.asarray(valid)
    if errors.ndim != 1 or valid.shape != errors.shape or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("Expected finite nonnegative errors and aligned validity")
    history = deque(maxlen=HISTORY)
    weights = np.zeros(len(errors))
    thresholds = np.zeros(len(errors))
    for t, error in enumerate(errors):
        if not valid[t]:
            continue
        if len(history) == HISTORY:
            threshold = max(float(np.quantile(history, QUANTILE)), EPSILON)
            thresholds[t] = threshold
            weights[t] = max(0., 1. - threshold / max(float(error), EPSILON))
        history.append(float(error))  # Never include current error in its threshold.
    return weights, thresholds


def process_predictive(input_features, gallery):
    x = _validate_features(input_features)
    metrics, valid = dynamics_signals(x, gallery)
    errors = metrics[:, 2, 1]  # learned AR2 error / strictly previous variance.
    weight, threshold = mixture_weights(errors, valid)
    base = dual_ema_short_tail(x)  # Original raw-input EMA, updated unconditionally.
    output = base.copy()
    active = weight > 0
    if active.any():
        output[active] = ((1 - weight[active, None]) * normalize(base[active]) +
                          weight[active, None] * normalize(x[active]))
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite mixed embedding")
    return output, dict(weight=weight, threshold=threshold, error=errors, valid=valid)
