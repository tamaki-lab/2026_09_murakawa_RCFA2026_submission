"""LOCAL same-formula exact-rational recomputation for unresolved history scores.

No challenge reader or builder registration. Q/N are exact for stored a,u,g;
preprocessing and the fast-path error estimate are NOT interval certified.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
from time import perf_counter

import numpy as np

from rcfa.methods.gallery_cohort_centering import cohort_center_scores
from scripts.cubic_kernel_reference import (
    RAW_ERROR_LIMIT,
    KernelHistory,
    NumericalUnresolved,
    gamma_bound,
    prepare,
)


def exact_dot(a, b):
    return sum((x * y for x, y in zip(a, b) if x and y), Fraction(0))


def group_prefix(units, weights, grouped=None):
    """Collapse exact antipodes only; optional causal cache avoids prefix rescans."""
    grouped = {} if grouped is None else grouped
    for u, a in zip(units, weights):
        if not a:
            continue
        nonzero = np.flatnonzero(u)
        if not len(nonzero):
            raise NumericalUnresolved('Positive weight on zero stored direction')
        sign = 1 if u[nonzero[0]] > 0 else -1
        key = tuple(float(sign * v) for v in u)
        grouped[key] = grouped.get(key, Fraction(0)) + sign * Fraction(float(a))
        if not grouped[key]:
            del grouped[key]
    return grouped


def exact_kernel(units, weights, gallery, degree):
    return evaluate_grouped(group_prefix(units, weights), gallery, degree)


def evaluate_grouped(grouped, gallery, degree, rational_gallery=None):
    if degree not in (1, 3):
        raise ValueError('Only degree1 and degree3')
    support = [(tuple(Fraction(v) for v in key), a)
               for key, a in grouped.items() if a]
    if not support:
        return Fraction(0), [Fraction(0) for _ in gallery], 0
    if rational_gallery is None:
        rational_gallery = [tuple(Fraction(float(v)) for v in g) for g in gallery]
    q = Fraction(0)
    if degree == 1:
        total = [sum((a * u[j] for u, a in support), Fraction(0))
                 for j in range(gallery.shape[1])]
        q = exact_dot(total, total)
        numerator = [exact_dot(total, g) for g in rational_gallery]
    elif degree == 3:
        for i, (u, a) in enumerate(support):
            q += a * a * exact_dot(u, u) ** 3
            for v, b in support[:i]:
                q += 2 * a * b * exact_dot(u, v) ** 3
        numerator = []
        for gf in rational_gallery:
            numerator.append(sum((a * exact_dot(u, gf) ** 3 for u, a in support), Fraction(0)))
    else:
        raise ValueError('Only degree1 and degree3')
    if q < 0 or (not q and any(numerator)):
        raise NumericalUnresolved('Exact kernel norm/numerator inconsistency')
    return q, numerator, len(support)


def rational_readout(q, numerator, frames):
    """Fixed 80/160-digit agreement check, not an interval-proof claim."""
    if q < 0 or frames < 1:
        raise NumericalUnresolved('Invalid exact norm/frame count')
    if not q:
        if any(numerator):
            raise NumericalUnresolved('Zero Q with nonzero numerator')
        return np.zeros(len(numerator), dtype=np.float32)
    outputs = []
    for precision in (80, 160):
        with localcontext() as ctx:
            ctx.prec = precision
            norm = (Decimal(q.numerator) / Decimal(q.denominator)).sqrt()
            denominator = max(norm, Decimal.from_float(frames * 1e-12))
            raw = np.array([float((Decimal(n.numerator) / Decimal(n.denominator)) / denominator)
                            for n in numerator], dtype=np.float32)
            outputs.append(raw)
    if not np.array_equal(outputs[0], outputs[1]):
        raise NumericalUnresolved('Fixed precision cross-check unresolved')
    return cohort_center_scores(outputs[-1][None, :])[0]


def raw_rational_readout(q, numerator, frames):
    """Return the uncentered float64 kernel readout at fixed80/160 precision."""
    if q < 0 or frames < 1:
        raise NumericalUnresolved('Invalid exact norm/frame count')
    if not q:
        if any(numerator):
            raise NumericalUnresolved('Zero Q with nonzero numerator')
        return np.zeros(len(numerator), dtype=np.float64)
    outputs = []
    for precision in (80, 160):
        with localcontext() as ctx:
            ctx.prec = precision
            norm = (Decimal(q.numerator) / Decimal(q.denominator)).sqrt()
            denominator = max(norm, Decimal.from_float(frames * 1e-12))
            outputs.append(np.array([
                float((Decimal(n.numerator) / Decimal(n.denominator)) / denominator)
                for n in numerator
            ], dtype=np.float64))
    if not np.array_equal(outputs[0], outputs[1]):
        raise NumericalUnresolved('Fixed precision raw cross-check unresolved')
    return outputs[-1]


def post_aggregation_rational_readout(q, numerator, frames):
    """Exact stored degree1 state with fixed mean-then-cube readout."""
    if q < 0 or frames < 1:
        raise NumericalUnresolved('Invalid exact norm/frame count')
    if not q:
        if any(numerator):
            raise NumericalUnresolved('Zero Q with nonzero numerator')
        return np.zeros(len(numerator), dtype=np.float32)
    outputs = []
    for precision in (80, 160):
        with localcontext() as ctx:
            ctx.prec = precision
            norm = (Decimal(q.numerator) / Decimal(q.denominator)).sqrt()
            denominator = max(norm, Decimal.from_float(frames * 1e-12))
            attenuation = norm / denominator
            raw = np.array([
                float(((Decimal(n.numerator) / Decimal(n.denominator)) / norm) ** 3
                      * attenuation)
                for n in numerator
            ], dtype=np.float32)
            outputs.append(raw)
    if not np.array_equal(outputs[0], outputs[1]):
        raise NumericalUnresolved('Fixed precision post-cubic cross-check unresolved')
    return cohort_center_scores(outputs[-1][None, :])[0]


def _post_aggregation_fast_readout(state):
    """Screen the approximate degree1 state for the fixed post-cubic readout."""
    if state.mass == 0:
        return np.zeros(len(state.gallery), dtype=np.float32)
    floor = state.frames * 1e-12
    lo, hi = state.q - state.q_error, state.q + state.q_error
    # Unlike the degree1 readout, a floor-bound Q is still needed to define the
    # direction before cubing. Any interval touching zero goes to exact replay.
    if hi < 0:
        raise NumericalUnresolved('Negative Q outside roundoff estimate')
    if lo <= 0:
        raise NumericalUnresolved('Post-cubic direction unresolved near zero Q')

    norm = np.sqrt(state.q)
    cosine = state.numerator / norm
    attenuation = norm / max(norm, floor)
    raw = cosine ** 3 * attenuation

    nlo = state.numerator - state.numerator_error
    nhi = state.numerator + state.numerator_error
    qlo = float(np.nextafter(lo, 0.0))
    qhi = float(np.nextafter(hi, np.inf))
    if qlo <= 0 or not np.isfinite(qhi):
        raise NumericalUnresolved('Post-cubic norm interval unresolved')
    dlo = qlo * max(np.sqrt(qlo), floor)
    dhi = qhi * max(np.sqrt(qhi), floor)
    clo, chi = nlo ** 3, nhi ** 3
    endpoints = np.stack((clo / dlo, clo / dhi, chi / dlo, chi / dhi))
    lower, upper = endpoints.min(axis=0), endpoints.max(axis=0)
    arithmetic = gamma_bound(12) * np.maximum.reduce(
        (np.abs(raw), np.abs(lower), np.abs(upper), np.ones_like(raw)))
    diagnostic = np.maximum(np.abs(raw - lower), np.abs(upper - raw)) + arithmetic
    worst = float(np.max(diagnostic))
    state.max_raw_error = max(state.max_raw_error, worst)
    if worst > RAW_ERROR_LIMIT:
        raise NumericalUnresolved('Estimated post-cubic uncertainty exceeds fixed local limit')
    return cohort_center_scores(raw.astype(np.float32)[None, :])[0]


class StableKernelHistory(KernelHistory):
    """Fast history unchanged; unresolved readout recomputed from stored prefix."""
    def __init__(self, gallery_coordinates, degree, capacity):
        super().__init__(gallery_coordinates, degree, capacity)
        self.recomputations = 0
        self.recompute_seconds = 0.
        self.max_exact_support = 0
        self._grouped = {}
        self._grouped_frames = 0
        self._rational_gallery = None

    def _readout(self):
        try:
            return super()._readout()
        except NumericalUnresolved:
            start = perf_counter()
            group_prefix(self.units[self._grouped_frames:self.frames],
                         self.weights[self._grouped_frames:self.frames], self._grouped)
            self._grouped_frames = self.frames
            if self._grouped and self._rational_gallery is None:
                self._rational_gallery = [tuple(Fraction(float(v)) for v in g) for g in self.gallery]
            q, numerator, count = evaluate_grouped(self._grouped, self.gallery,
                self.degree, self._rational_gallery)
            output = rational_readout(q, numerator, self.frames)
            self.recomputations += 1
            self.recompute_seconds += perf_counter() - start
            self.max_exact_support = max(self.max_exact_support, count)
            # Do not reset approximate recurrence/error counters or alter future history.
            return output


class StablePostAggregationCubicHistory(KernelHistory):
    """Degree1 history with a fixed cube-before-centering readout."""
    def __init__(self, gallery_coordinates, capacity):
        super().__init__(gallery_coordinates, 1, capacity)
        self.recomputations = 0
        self.recompute_seconds = 0.
        self.max_exact_support = 0
        self._grouped = {}
        self._grouped_frames = 0
        self._rational_gallery = None

    def _readout(self):
        try:
            return _post_aggregation_fast_readout(self)
        except NumericalUnresolved:
            start = perf_counter()
            group_prefix(self.units[self._grouped_frames:self.frames],
                         self.weights[self._grouped_frames:self.frames], self._grouped)
            self._grouped_frames = self.frames
            if self._grouped and self._rational_gallery is None:
                self._rational_gallery = [tuple(Fraction(float(v)) for v in g)
                                          for g in self.gallery]
            q, numerator, count = evaluate_grouped(
                self._grouped, self.gallery, 1, self._rational_gallery)
            output = post_aggregation_rational_readout(q, numerator, self.frames)
            self.recomputations += 1
            self.recompute_seconds += perf_counter() - start
            self.max_exact_support = max(self.max_exact_support, count)
            return output


class StableRawKernelHistory(KernelHistory):
    """Stable uncentered history readout for a fixed downstream score map."""
    def __init__(self, gallery_coordinates, degree, capacity, raw_error_limit):
        super().__init__(gallery_coordinates, degree, capacity)
        if not (np.isfinite(raw_error_limit) and raw_error_limit > 0):
            raise ValueError('raw_error_limit must be positive and finite')
        self.raw_error_limit = float(raw_error_limit)
        self.recomputations = 0
        self.recompute_seconds = 0.
        self.max_exact_support = 0
        self._grouped = {}
        self._grouped_frames = 0
        self._rational_gallery = None

    def _fast_raw_readout(self):
        if self.mass == 0:
            return np.zeros(len(self.gallery), dtype=np.float64)
        floor = self.frames * 1e-12
        lo, hi = self.q - self.q_error, self.q + self.q_error
        if hi < 0:
            raise NumericalUnresolved('Negative Q outside roundoff estimate')
        if hi <= floor * floor:
            denominator = lower = upper = floor
        elif lo <= 0:
            raise NumericalUnresolved('Q cancellation unresolved; no zero clamp')
        else:
            denominator = max(np.sqrt(self.q), floor)
            lower, upper = max(np.sqrt(lo), floor), max(np.sqrt(hi), floor)
        raw = self.numerator / denominator
        diagnostic = (self.numerator_error / lower
                      + np.abs(self.numerator) * (1 / lower - 1 / upper)
                      + gamma_bound(2) * np.abs(raw))
        worst = float(np.max(diagnostic))
        self.max_raw_error = max(self.max_raw_error, worst)
        if worst > self.raw_error_limit:
            raise NumericalUnresolved('Estimated raw-score uncertainty exceeds downstream limit')
        return raw

    def _readout(self):
        try:
            return self._fast_raw_readout()
        except NumericalUnresolved:
            start = perf_counter()
            group_prefix(self.units[self._grouped_frames:self.frames],
                         self.weights[self._grouped_frames:self.frames], self._grouped)
            self._grouped_frames = self.frames
            if self._grouped and self._rational_gallery is None:
                self._rational_gallery = [tuple(Fraction(float(v)) for v in g)
                                          for g in self.gallery]
            q, numerator, count = evaluate_grouped(
                self._grouped, self.gallery, self.degree, self._rational_gallery)
            output = raw_rational_readout(q, numerator, self.frames)
            self.recomputations += 1
            self.recompute_seconds += perf_counter() - start
            self.max_exact_support = max(self.max_exact_support, count)
            return output


def stable_history_scores(input_features, gallery, degree):
    z, coords, _ = prepare(input_features, gallery)
    state = StableKernelHistory(coords, degree, len(z))
    scores = np.array([state.step(row) for row in z], dtype=np.float32)
    return scores, dict(recomputations=state.recomputations,
        recompute_seconds=state.recompute_seconds, max_exact_support=state.max_exact_support)


def stable_post_aggregation_cubic_scores(input_features, gallery):
    z, coords, _ = prepare(input_features, gallery)
    state = StablePostAggregationCubicHistory(coords, len(z))
    scores = np.array([state.step(row) for row in z], dtype=np.float32)
    return scores, dict(recomputations=state.recomputations,
        recompute_seconds=state.recompute_seconds, max_exact_support=state.max_exact_support,
        max_raw_error_estimate=state.max_raw_error)


def stable_raw_history_scores(input_features, gallery, degree, raw_error_limit):
    """Return uncentered float64 history scores for one fixed downstream map."""
    z, coords, _ = prepare(input_features, gallery)
    return stable_raw_prepared_scores(z, coords, degree, raw_error_limit)


def stable_raw_prepared_scores(projected, gallery_coordinates, degree, raw_error_limit):
    """Same raw history readout for already frozen prepared coordinates."""
    z = np.asarray(projected, dtype=np.float64)
    coords = np.asarray(gallery_coordinates, dtype=np.float64)
    if (z.ndim != 2 or coords.ndim != 2 or z.shape[1] != coords.shape[1]
            or len(z) < 1 or not np.isfinite(z).all()
            or not np.isfinite(coords).all()):
        raise ValueError('Invalid prepared raw-history inputs')
    state = StableRawKernelHistory(coords, degree, len(z), raw_error_limit)
    scores = np.array([state.step(row) for row in z], dtype=np.float64)
    return scores, dict(
        recomputations=state.recomputations,
        recompute_seconds=state.recompute_seconds,
        max_exact_support=state.max_exact_support,
        max_raw_error_estimate=state.max_raw_error,
    )
