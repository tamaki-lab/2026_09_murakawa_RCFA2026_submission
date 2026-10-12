"""Shared causal path filter; plug-in spherical likelihood, NOT authenticity.

Power Spherical observation law: De Cao and Aziz (2020). Path recursion and
outlier-output policy are our experiment, not an exact conjugate BOCPD model.
"""
from dataclasses import dataclass
from math import lgamma, log
import numpy as np
from example import normalize
from rcfa.embedding import _validate_features

HAZARD = 1 / 200
OUTLIER_PRIOR = .01
BEAM = 32


def log_power_ratio(cosine, concentration, dimension):
    """Log density ratio against uniform sphere; works on scalar or array c."""
    if dimension < 2 or not np.isfinite(concentration) or concentration < 0:
        raise ValueError('Invalid spherical distribution')
    c = np.asarray(cosine, dtype=np.float64)
    if not np.isfinite(c).all() or np.any(np.abs(c) > 1 + 1e-12):
        raise ValueError('Invalid cosine')
    if concentration == 0:
        return np.zeros_like(c)
    b = (dimension - 1) / 2
    normalizer = (lgamma(b) - lgamma(2*b) - lgamma(b+concentration)
                  + lgamma(2*b+concentration))
    u = (1 + np.clip(c, -1, 1)) / 2
    value = np.full_like(u, -np.inf)
    np.log(u, out=value, where=u > 0)
    return concentration * value + normalizer


def logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values)
    return float(maximum + np.log(np.exp(values-maximum).sum()))


@dataclass
class PathState:
    total: np.ndarray
    count: int
    age: int
    log_weight: float

    def log_predictive_ratio(self, x):
        length = float(np.linalg.norm(self.total))
        if length == 0:
            return 0.
        resultant = min(length / (self.count + 1), 1 - 1e-6)
        kappa = (len(x)-1) * resultant / (1-resultant)
        cosine = float(np.sum((self.total/length)*x))
        return float(log_power_ratio(cosine, kappa, len(x)))


class RunLengthFilter:
    def __init__(self, dimension):
        if not isinstance(dimension, int) or dimension < 2:
            raise ValueError('Expected dimension >= 2')
        self.dimension = dimension
        self.paths = []

    def step(self, observation):
        raw = np.asarray(observation, dtype=np.float64)
        if raw.shape != (self.dimension,) or not np.isfinite(raw).all():
            raise ValueError('Invalid observation')
        x = normalize(raw[None, :])[0]
        if not np.any(x):
            self.paths = [PathState(s.total, s.count, s.age+1, s.log_weight)
                          for s in self.paths]
            return x.copy(), dict(continue_probability=0., new_probability=0.,
                outlier_probability=0., dropped_mass=0., paths=len(self.paths),
                expected_age=self._expected('age'), expected_count=self._expected('count'))
        if not self.paths:
            self.paths = [PathState(x.copy(), 1, 1, 0.)]
            return x.copy(), dict(continue_probability=0., new_probability=1.,
                outlier_probability=0., dropped_mass=0., paths=1,
                expected_age=1., expected_count=1.)

        branches, vectors, kinds = [], [], []
        for state in self.paths:
            predictive = state.log_predictive_ratio(x)  # Strictly prior statistics.
            updated = state.total + x
            branches.append(PathState(updated, state.count+1, state.age+1,
                state.log_weight + log(1-HAZARD) + log(1-OUTLIER_PRIOR) + predictive))
            vectors.append(normalize(updated[None, :])[0]); kinds.append(0)
            branches.append(PathState(state.total, state.count, state.age+1,
                state.log_weight + log(1-HAZARD) + log(OUTLIER_PRIOR)))
            vectors.append(x); kinds.append(2)  # Preserve state, but never emit old identity here.
        branches.append(PathState(x.copy(), 1, 1, log(HAZARD)))
        vectors.append(x); kinds.append(1)
        logs = np.array([s.log_weight for s in branches])
        normalizer = logsumexp(logs)
        weights = np.exp(logs-normalizer)
        mixed = np.sum(weights[:, None]*np.asarray(vectors), axis=0)
        output = normalize(mixed[None, :])[0]

        # Always reserve a new-segment path; never prune by age.
        new = len(branches)-1
        keep = list(np.argsort(-logs[:-1], kind='stable')[:BEAM-1]) + [new]
        retained_normalizer = logsumexp(logs[keep])
        self.paths = [PathState(branches[i].total, branches[i].count,
                               branches[i].age, float(logs[i]-retained_normalizer))
                      for i in keep]
        kinds = np.asarray(kinds)
        trace = dict(continue_probability=float(weights[kinds == 0].sum()),
            new_probability=float(weights[kinds == 1].sum()),
            outlier_probability=float(weights[kinds == 2].sum()),
            dropped_mass=float(max(0., 1-weights[keep].sum())), paths=len(keep),
            expected_age=self._expected('age'), expected_count=self._expected('count'))
        if not np.isfinite(output).all() or not np.isfinite(list(trace.values())).all():
            raise FloatingPointError('Nonfinite filter output')
        return output, trace

    def _expected(self, attribute):
        return float(sum(np.exp(p.log_weight)*getattr(p, attribute) for p in self.paths))


def process_runlength(features):
    x = _validate_features(features)
    model = RunLengthFilter(x.shape[1])
    outputs, rows = [], []
    for observation in x:
        out, row = model.step(observation)
        outputs.append(out); rows.append(row)
    return np.asarray(outputs), {key: np.array([row[key] for row in rows]) for key in rows[0]}
