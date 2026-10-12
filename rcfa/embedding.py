"""Small, strictly causal embedding processors for RCFA experiments."""

from __future__ import annotations

import numpy as np


DUAL_EMA_SLOW_ALPHA = 0.01
DUAL_EMA_FAST_ALPHA = 0.10
DUAL_EMA_TARGET_INITIAL_ALPHA = 0.03
DUAL_EMA_FAST_WEIGHT = (
    (DUAL_EMA_TARGET_INITIAL_ALPHA - DUAL_EMA_SLOW_ALPHA)
    / (DUAL_EMA_FAST_ALPHA - DUAL_EMA_SLOW_ALPHA)
)

SHORT_TAIL_DUAL_EMA_SLOW_ALPHA = 0.015
SHORT_TAIL_DUAL_EMA_FAST_ALPHA = 0.10
SHORT_TAIL_DUAL_EMA_TARGET_INITIAL_ALPHA = 0.03
SHORT_TAIL_DUAL_EMA_FAST_WEIGHT = (
    (SHORT_TAIL_DUAL_EMA_TARGET_INITIAL_ALPHA - SHORT_TAIL_DUAL_EMA_SLOW_ALPHA)
    / (SHORT_TAIL_DUAL_EMA_FAST_ALPHA - SHORT_TAIL_DUAL_EMA_SLOW_ALPHA)
)

SLOW_0022_DUAL_EMA_SLOW_ALPHA = 0.022
SLOW_0022_DUAL_EMA_FAST_ALPHA = 0.10
SLOW_0022_DUAL_EMA_TARGET_INITIAL_ALPHA = 0.03
SLOW_0022_DUAL_EMA_FAST_WEIGHT = (
    (SLOW_0022_DUAL_EMA_TARGET_INITIAL_ALPHA - SLOW_0022_DUAL_EMA_SLOW_ALPHA)
    / (SLOW_0022_DUAL_EMA_FAST_ALPHA - SLOW_0022_DUAL_EMA_SLOW_ALPHA)
)

ANCHOR_0198_DUAL_EMA_SLOW_ALPHA = 0.0198
ANCHOR_0198_DUAL_EMA_FAST_ALPHA = 0.10
ANCHOR_0198_DUAL_EMA_TARGET_INITIAL_ALPHA = 0.03
ANCHOR_0198_DUAL_EMA_FAST_WEIGHT = (
    (ANCHOR_0198_DUAL_EMA_TARGET_INITIAL_ALPHA - ANCHOR_0198_DUAL_EMA_SLOW_ALPHA)
    / (ANCHOR_0198_DUAL_EMA_FAST_ALPHA - ANCHOR_0198_DUAL_EMA_SLOW_ALPHA)
)

METHODS = (
    "frame_only",
    "cumulative_mean",
    "ema",
    "dual_ema_matched",
    "dual_ema_short_tail",
    "dual_ema_slow_0022",
    "dual_ema_anchor_0198",
    "dual_ema_norm_p0",
    "dual_ema_norm_p2",
)


def _validate_features(input_features: np.ndarray) -> np.ndarray:
    features = np.asarray(input_features)
    if features.ndim != 2:
        raise ValueError(f"Expected a 2-D feature array, got shape {features.shape}")
    if features.shape[0] == 0:
        raise ValueError("Expected at least one frame")
    if not np.issubdtype(features.dtype, np.floating):
        raise ValueError(f"Expected floating features, got dtype {features.dtype}")
    if not np.isfinite(features).all():
        raise ValueError("Input features contain NaN or Inf")
    return features


def frame_only(input_features: np.ndarray) -> np.ndarray:
    """Return each current-frame embedding without temporal aggregation."""
    features = _validate_features(input_features)
    return features.copy()


def cumulative_mean(input_features: np.ndarray) -> np.ndarray:
    """Reproduce the official causal cumulative-mean implementation."""
    features = _validate_features(input_features)
    frame_count = features.shape[0]
    return np.cumsum(features, axis=0) / np.arange(1, frame_count + 1)[:, None]


def causal_ema(input_features: np.ndarray, alpha: float) -> np.ndarray:
    """Return an online exponential moving average initialized at frame zero."""
    features = _validate_features(input_features)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    # Float64 state prevents avoidable long-sequence recurrence drift. Submission
    # scores are explicitly cast to float32 by the artifact builder.
    output = np.empty(features.shape, dtype=np.float64)
    output[0] = features[0]
    for index in range(1, features.shape[0]):
        output[index] = alpha * features[index] + (1.0 - alpha) * output[index - 1]
    return output


def dual_ema_matched(input_features: np.ndarray) -> np.ndarray:
    """Blend fixed slow/fast EMAs with initial response matched to EMA 0.03."""
    features = _validate_features(input_features)
    slow = causal_ema(features, DUAL_EMA_SLOW_ALPHA)
    fast = causal_ema(features, DUAL_EMA_FAST_ALPHA)
    return (1.0 - DUAL_EMA_FAST_WEIGHT) * slow + DUAL_EMA_FAST_WEIGHT * fast


def dual_ema_short_tail(input_features: np.ndarray) -> np.ndarray:
    """Use a shorter slow tail while retaining an initial response of 0.03."""
    features = _validate_features(input_features)
    slow = causal_ema(features, SHORT_TAIL_DUAL_EMA_SLOW_ALPHA)
    fast = causal_ema(features, SHORT_TAIL_DUAL_EMA_FAST_ALPHA)
    return (
        (1.0 - SHORT_TAIL_DUAL_EMA_FAST_WEIGHT) * slow
        + SHORT_TAIL_DUAL_EMA_FAST_WEIGHT * fast
    )


def dual_ema_slow_0022(input_features: np.ndarray) -> np.ndarray:
    """Fixed memory-axis control with a 0.022 slow update rate."""
    features = _validate_features(input_features)
    slow = causal_ema(features, SLOW_0022_DUAL_EMA_SLOW_ALPHA)
    fast = causal_ema(features, SLOW_0022_DUAL_EMA_FAST_ALPHA)
    return (
        (1.0 - SLOW_0022_DUAL_EMA_FAST_WEIGHT) * slow
        + SLOW_0022_DUAL_EMA_FAST_WEIGHT * fast
    )


def dual_ema_anchor_0198(input_features: np.ndarray) -> np.ndarray:
    """Fixed p=1 memory-family anchor with a 0.0198 slow update rate."""
    features = _validate_features(input_features)
    slow = causal_ema(features, ANCHOR_0198_DUAL_EMA_SLOW_ALPHA)
    fast = causal_ema(features, ANCHOR_0198_DUAL_EMA_FAST_ALPHA)
    return (
        (1.0 - ANCHOR_0198_DUAL_EMA_FAST_WEIGHT) * slow
        + ANCHOR_0198_DUAL_EMA_FAST_WEIGHT * fast
    )


def _dual_ema_slow_0022_norm_power(
    input_features: np.ndarray, norm_power: int
) -> np.ndarray:
    """Apply one fixed norm exponent before the submitted 0.022 dual EMA."""
    features = _validate_features(input_features)
    if norm_power == 1:
        # Route the control directly so its recurrence stays bit-identical to
        # submission 921783 instead of relying on power/multiply identities.
        return dual_ema_slow_0022(features)

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-12)
    if norm_power == 0:
        scaled = features / safe_norms
    elif norm_power == 2:
        scaled = features * safe_norms
    else:
        raise ValueError("norm_power must be one of {0, 1, 2}")
    return dual_ema_slow_0022(scaled)


def dual_ema_norm_p0(input_features: np.ndarray) -> np.ndarray:
    """Remove raw embedding-norm weighting before the fixed dual EMA."""
    return _dual_ema_slow_0022_norm_power(input_features, 0)


def dual_ema_norm_p2(input_features: np.ndarray) -> np.ndarray:
    """Square raw embedding-norm weighting before the fixed dual EMA."""
    return _dual_ema_slow_0022_norm_power(input_features, 2)


def process_embeddings(
    input_features: np.ndarray, method: str, *, ema_alpha: float | None = None
) -> np.ndarray:
    """Dispatch to one causal processor using an explicit configuration."""
    if method == "frame_only":
        if ema_alpha is not None:
            raise ValueError("ema_alpha is only valid for method='ema'")
        return frame_only(input_features)
    if method == "cumulative_mean":
        if ema_alpha is not None:
            raise ValueError("ema_alpha is only valid for method='ema'")
        return cumulative_mean(input_features)
    if method == "ema":
        if ema_alpha is None:
            raise ValueError("method='ema' requires ema_alpha")
        return causal_ema(input_features, ema_alpha)
    if method == "dual_ema_matched":
        if ema_alpha is not None:
            raise ValueError("ema_alpha is only valid for method='ema'")
        return dual_ema_matched(input_features)
    if method == "dual_ema_short_tail":
        if ema_alpha is not None:
            raise ValueError("ema_alpha is only valid for method='ema'")
        return dual_ema_short_tail(input_features)
    if method == "dual_ema_slow_0022":
        if ema_alpha is not None:
            raise ValueError("ema_alpha is only valid for method='ema'")
        return dual_ema_slow_0022(input_features)
    if method == "dual_ema_anchor_0198":
        if ema_alpha is not None:
            raise ValueError("ema_alpha is only valid for method='ema'")
        return dual_ema_anchor_0198(input_features)
    if method == "dual_ema_norm_p0":
        if ema_alpha is not None:
            raise ValueError("ema_alpha is only valid for method='ema'")
        return dual_ema_norm_p0(input_features)
    if method == "dual_ema_norm_p2":
        if ema_alpha is not None:
            raise ValueError("ema_alpha is only valid for method='ema'")
        return dual_ema_norm_p2(input_features)
    raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
