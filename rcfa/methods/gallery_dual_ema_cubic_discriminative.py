"""Fixed local dual-EMA cubic candidate with unchanged gallery margin readout.

This is not registered in the submission builder. No Track or attack label is
used. Local synthetic validation does not authorize a challenge run or ZIP.
"""

import numpy as np

from rcfa.methods.gallery_discriminative_readout import (
    FINAL_ERROR_LIMIT,
    apply_discriminative_readout,
    discriminative_mapping,
)
from scripts.cubic_kernel_reference import prepare
from scripts.dual_ema_cubic_stable import stable_dual_ema_cubic_raw_prepared


def dual_ema_cubic_discriminative_diagnostics(input_features, gallery):
    projected, coordinates, _ = prepare(input_features, gallery)
    transform, bias, mapping = discriminative_mapping(coordinates)
    amplification = float(np.max(np.sum(np.abs(transform), axis=1)))
    raw_error_limit = FINAL_ERROR_LIMIT / (2 * max(1.0, amplification))
    raw, info = stable_dual_ema_cubic_raw_prepared(
        projected, coordinates, raw_error_limit)
    scores = apply_discriminative_readout(raw, transform, bias)
    info.update(raw_error_limit=raw_error_limit,
                map_amplification=amplification,
                map_error=mapping["map_error"],
                projected_shape=list(projected.shape))
    return scores, info


def dual_ema_cubic_discriminative_scores(input_features, gallery):
    return dual_ema_cubic_discriminative_diagnostics(input_features, gallery)[0]
