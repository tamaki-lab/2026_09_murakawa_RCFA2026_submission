"""Matched control: previous-vector error instead of learned AR2 error.

Keep the full signal computation, including warmup and variance updates, to
isolate the signal choice. This is not an optimized implementation or a fake
detector. No track label or prelearned evaluation-data statistics are used.
"""
import numpy as np
from example import normalize
from rcfa.methods.online_dynamics import dynamics_signals
from rcfa.methods.predictive_output import mixture_weights
from rcfa.embedding import _validate_features, dual_ema_short_tail


def process_persistence(input_features, gallery):
    x = _validate_features(input_features)
    metrics, valid = dynamics_signals(x, gallery)
    errors = metrics[:, 0, 1]  # Previous-vector error / strictly previous variance.
    weight, threshold = mixture_weights(errors, valid)
    base = dual_ema_short_tail(x)
    output = base.copy()
    active = weight > 0
    if active.any():
        output[active] = ((1 - weight[active, None]) * normalize(base[active]) +
                          weight[active, None] * normalize(x[active]))
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite mixed embedding")
    return output, dict(weight=weight, threshold=threshold, error=errors, valid=valid)
