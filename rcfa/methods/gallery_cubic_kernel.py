"""Fixed degree-three causal matching; shared numerical core with validated reference.

No track, degree, blend or tuning argument. Each call starts fresh state.
Builder consumes these direct float32 scores without further normalization.
"""
from scripts.cubic_kernel_stable import stable_history_scores


def cubic_kernel_scores(input_features, gallery):
    return stable_history_scores(input_features, gallery, 3)[0]


def cubic_kernel_diagnostics(input_features, gallery):
    return stable_history_scores(input_features, gallery, 3)
