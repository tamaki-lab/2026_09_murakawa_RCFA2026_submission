"""Feed the frozen predictive-output weight into the slow EMA update rate."""

from __future__ import annotations

import numpy as np

from rcfa.methods.online_dynamics import dynamics_signals
from rcfa.methods.predictive_output import mixture_weights
from rcfa.embedding import (
    SHORT_TAIL_DUAL_EMA_FAST_ALPHA,
    SHORT_TAIL_DUAL_EMA_FAST_WEIGHT,
    SHORT_TAIL_DUAL_EMA_SLOW_ALPHA,
    _validate_features,
)


def process_predictive_state_feedback(input_features, gallery):
    """Return one shared, strictly causal predictive-gain state trajectory.

    The AR2 error and weight are exactly the frozen predictive-output signal.
    Unlike predictive_output, the weight changes the slow raw-embedding EMA
    update rate.  Raw inputs are preserved so an all-zero weight trace exactly
    reproduces dual_ema_short_tail before the builder's final normalization.
    """
    features = _validate_features(input_features)
    metrics, valid = dynamics_signals(features, gallery)
    errors = metrics[:, 2, 1]
    weight, threshold = mixture_weights(errors, valid)

    slow_alpha = (
        SHORT_TAIL_DUAL_EMA_SLOW_ALPHA
        + (SHORT_TAIL_DUAL_EMA_FAST_ALPHA - SHORT_TAIL_DUAL_EMA_SLOW_ALPHA)
        * weight
    )
    slow = features[0].astype(np.float64, copy=True)
    fast = features[0].astype(np.float64, copy=True)
    output = np.empty(features.shape, dtype=np.float64)
    output[0] = (
        (1.0 - SHORT_TAIL_DUAL_EMA_FAST_WEIGHT) * slow
        + SHORT_TAIL_DUAL_EMA_FAST_WEIGHT * fast
    )

    for index in range(1, len(features)):
        # Preserve the baseline recurrence bit-for-bit before the first active
        # event. Computing ``slow + (fast-slow)*0`` can otherwise introduce a
        # tiny floating representation difference from the literal constant.
        alpha = (
            SHORT_TAIL_DUAL_EMA_SLOW_ALPHA
            if weight[index] == 0.0
            else slow_alpha[index]
        )
        slow = alpha * features[index] + (1.0 - alpha) * slow
        fast = (
            SHORT_TAIL_DUAL_EMA_FAST_ALPHA * features[index]
            + (1.0 - SHORT_TAIL_DUAL_EMA_FAST_ALPHA) * fast
        )
        output[index] = (
            (1.0 - SHORT_TAIL_DUAL_EMA_FAST_WEIGHT) * slow
            + SHORT_TAIL_DUAL_EMA_FAST_WEIGHT * fast
        )

    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite predictive state-feedback embedding")
    return output, {
        "weight": weight,
        "threshold": threshold,
        "error": errors,
        "valid": valid,
        "slow_alpha": slow_alpha,
    }
