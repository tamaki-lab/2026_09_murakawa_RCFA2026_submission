"""Emission-only diagnostic; original run-length path recursion is unchanged."""
import numpy as np
from rcfa.methods.causal_runlength import RunLengthFilter, logsumexp
from example import normalize
from rcfa.embedding import _validate_features


class ContinueOutputFilter:
    """Condition emission on continuation, never feed that output into path state.

    Matches analyze_runlength_failures.observe's continue_emission exactly.
    No prior paths, zero input, or no finite continuation likelihood falls back
    to the original emission. Hazard/outlier constants cancel in conditional
    weights; the original model still uses them for ALL state updates.
    """
    def __init__(self, dimension):
        self.model = RunLengthFilter(dimension)

    def step(self, observation):
        prior = self.model.paths
        original, trace = self.model.step(observation)
        x = normalize(np.asarray(observation, dtype=float)[None])[0]
        logs = np.array([p.log_weight+p.log_predictive_ratio(x) for p in prior])
        valid = bool(len(logs) and np.any(np.isfinite(logs)) and np.any(x))
        if valid:
            weights = np.exp(logs-logsumexp(logs))
            directions = np.array([normalize((p.total+x)[None])[0] for p in prior])
            output = normalize(np.sum(weights[:, None]*directions, axis=0)[None])[0]
        else:
            output = original.copy()
        if not np.isfinite(output).all():
            raise FloatingPointError('Nonfinite continuation output')
        return output, trace


def process_continue_output(features):
    x = _validate_features(features)
    model = ContinueOutputFilter(x.shape[1])
    outputs, rows = [], []
    for observation in x:
        output, row = model.step(observation)
        outputs.append(output)
        rows.append(row)
    return np.asarray(outputs), {k: np.array([row[k] for row in rows]) for k in rows[0]}
