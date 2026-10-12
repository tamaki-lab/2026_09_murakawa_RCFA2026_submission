"""Fixed mean-then-cube causal matching candidate.

The degree1 history state is unchanged. Only its uncentered normalized cosine
readout is cubed before the existing float32/cohort-centering/float32 path.
No track, degree, temperature, blend or tuning argument is exposed.
"""
from scripts.cubic_kernel_stable import stable_post_aggregation_cubic_scores


def post_aggregation_cubic_scores(input_features, gallery):
    return stable_post_aggregation_cubic_scores(input_features, gallery)[0]


def post_aggregation_cubic_diagnostics(input_features, gallery):
    return stable_post_aggregation_cubic_scores(input_features, gallery)
