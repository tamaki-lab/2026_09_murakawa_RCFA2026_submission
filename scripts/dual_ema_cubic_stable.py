"""Local finite-memory cubic history with screened raw-score readout.

The kernel recurrence and fixed coefficients match dual_ema_cubic_reference.
The screen is a forward-error estimate for stored prepared coordinates, not
an outward-rounded proof of infinite-precision preprocessing. Unresolved
prefixes are replayed with exact rational arithmetic on stored coordinates
and effective EMA weights. No builder or challenge reader is registered here.
"""

from decimal import Decimal, localcontext
from fractions import Fraction
from time import perf_counter

import numpy as np

from scripts.cubic_kernel_reference import NumericalUnresolved, gamma_bound, power_with_error
from scripts.cubic_kernel_stable import exact_dot
from scripts.dual_ema_cubic_reference import FAST_ALPHA, FAST_WEIGHT, FLOOR, SLOW_ALPHA

EXACT_SLOW_ALPHA = Fraction(11, 500)
EXACT_FAST_ALPHA = Fraction(1, 10)
EXACT_FAST_WEIGHT = Fraction(4, 39)
MAX_EXACT_SECONDS = 10.0


def _coefficient_delta(approximate, exact):
    """Conservatively convert exact binary-vs-rational coefficient distance."""
    distance = abs(Fraction(float(approximate)) - exact)
    return np.nextafter(float(distance), np.inf) if distance else 0.0


def _linear_error(coefficients, exact_coefficients, values, errors, operations):
    """Propagate input, coefficient-representation and arithmetic error."""
    bound = 0.0
    magnitude = 0.0
    for c, exact, value, error in zip(coefficients, exact_coefficients, values, errors):
        coefficient_error = _coefficient_delta(c, exact)
        bound = (bound + (abs(c) + coefficient_error) * error
                 + coefficient_error * np.abs(value))
        magnitude = magnitude + np.abs(c * value)
    return bound + gamma_bound(operations) * magnitude


def _bounded_cubic_grouped(grouped, gallery, rational_gallery, deadline):
    """Exact stored-coordinate cubic replay, failing closed on work budget."""
    support = []
    for key, weight in grouped.items():
        if perf_counter() > deadline:
            raise NumericalUnresolved("Finite-memory exact replay time budget exceeded")
        if weight:
            support.append((tuple(Fraction(value) for value in key), weight))
    if not support:
        return Fraction(0), [Fraction(0) for _ in gallery], 0
    q = Fraction(0)
    for i, (u, a) in enumerate(support):
        if perf_counter() > deadline:
            raise NumericalUnresolved("Finite-memory exact replay time budget exceeded")
        q += a * a * exact_dot(u, u) ** 3
        for v, b in support[:i]:
            if perf_counter() > deadline:
                raise NumericalUnresolved("Finite-memory exact replay time budget exceeded")
            q += 2 * a * b * exact_dot(u, v) ** 3
    numerator = []
    for gf in rational_gallery:
        total = Fraction(0)
        for u, a in support:
            if perf_counter() > deadline:
                raise NumericalUnresolved("Finite-memory exact replay time budget exceeded")
            total += a * exact_dot(u, gf) ** 3
        numerator.append(total)
    if q < 0 or (not q and any(numerator)):
        raise NumericalUnresolved("Exact finite-memory norm/numerator inconsistency")
    return q, numerator, len(support)


class StableDualEMACubicHistory:
    def __init__(self, gallery_coordinates, capacity, raw_error_limit):
        gallery = np.asarray(gallery_coordinates, dtype=np.float64)
        if (gallery.ndim != 2 or gallery.shape[0] < 2 or gallery.shape[1] < 1
                or capacity < 1 or not np.isfinite(gallery).all()
                or not (np.isfinite(raw_error_limit) and raw_error_limit > 0)):
            raise ValueError("Invalid prepared gallery, capacity or error limit")
        self.gallery = gallery.copy()
        self.capacity = capacity
        self.raw_error_limit = float(raw_error_limit)
        self.units = np.zeros((capacity, gallery.shape[1]), dtype=np.float64)
        self.mass = np.zeros(capacity, dtype=np.float64)
        self.frames = 0
        self.ws = np.empty(0, dtype=np.float64)
        self.wf = np.empty(0, dtype=np.float64)
        self.ws_error = np.empty(0, dtype=np.float64)
        self.wf_error = np.empty(0, dtype=np.float64)
        self.ns = np.zeros(len(gallery), dtype=np.float64)
        self.nf = np.zeros(len(gallery), dtype=np.float64)
        self.ns_error = np.zeros(len(gallery), dtype=np.float64)
        self.nf_error = np.zeros(len(gallery), dtype=np.float64)
        self.qss = self.qff = self.qsf = 0.0
        self.qss_error = self.qff_error = self.qsf_error = 0.0
        self.recomputations = 0
        self.recompute_seconds = 0.0
        self.max_exact_support = 0
        self.max_raw_error_estimate = 0.0
        self.screen_zero_count = 0
        self.screen_floor_count = 0
        self.screen_regular_count = 0
        self.minimum_q_lower_bound = float("inf")
        self._rational_gallery = None

    def step(self, row):
        z = np.asarray(row, dtype=np.float64)
        if (z.shape != (self.gallery.shape[1],) or not np.isfinite(z).all()
                or self.frames >= self.capacity):
            raise ValueError("Expected one finite prepared frame within capacity")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
                return self._step(z)
        except NumericalUnresolved:
            raise
        except FloatingPointError as error:
            # In particular, coefficient/term underflow on sequences much
            # longer than the reviewed challenge length must not be hidden.
            raise NumericalUnresolved("Finite-memory arithmetic failed closed") from error

    def _step(self, z):
        t = self.frames
        a = float(np.linalg.norm(z))
        u = z / a if a else np.zeros_like(z)
        dotself = float(np.dot(u, u))
        self_kernel, self_kernel_error = power_with_error(dotself, dotself, 3, len(u))
        diag = a * a * self_kernel
        diag_error = (a * a * self_kernel_error + gamma_bound(3) * abs(diag))
        gd = np.einsum("ij,j->i", self.gallery, u, optimize=False)
        ga = np.einsum("ij,j->i", np.abs(self.gallery), np.abs(u), optimize=False)
        gk, gke = power_with_error(gd, ga, 3, len(u))
        observed = a * gk
        observed_error = a * gke + gamma_bound(2) * np.abs(observed)

        if t == 0:
            self.ns = self.nf = observed.copy()
            self.ns_error = self.nf_error = observed_error.copy()
            self.qss = self.qff = self.qsf = diag
            self.qss_error = self.qff_error = self.qsf_error = diag_error
            self.ws = self.wf = np.ones(1, dtype=np.float64)
            self.ws_error = self.wf_error = np.zeros(1, dtype=np.float64)
        else:
            old_units = self.units[:t]
            dot = np.einsum("ij,j->i", old_units, u, optimize=False)
            absdot = np.einsum("ij,j->i", np.abs(old_units), np.abs(u), optimize=False)
            kernel, kernel_error = power_with_error(dot, absdot, 3, len(u))
            s_terms = self.ws * self.mass[:t] * kernel
            f_terms = self.wf * self.mass[:t] * kernel
            cross_s = a * float(np.sum(s_terms))
            cross_f = a * float(np.sum(f_terms))
            # Bound dot/cube, stored-EMA-weight and reduction roundoff before
            # the two recurrence updates. Preprocessing roundoff is excluded.
            s_error = a * float(np.sum(self.mass[:t] * (
                np.abs(self.ws) * kernel_error
                + self.ws_error * (np.abs(kernel) + kernel_error))))
            f_error = a * float(np.sum(self.mass[:t] * (
                np.abs(self.wf) * kernel_error
                + self.wf_error * (np.abs(kernel) + kernel_error))))
            s_error += a * gamma_bound(max(1, t) + 4) * float(np.sum(np.abs(s_terms)))
            f_error += a * gamma_bound(max(1, t) + 4) * float(np.sum(np.abs(f_terms)))
            bs, bf = 1 - SLOW_ALPHA, 1 - FAST_ALPHA
            ss_terms = (bs * bs * self.qss, 2 * bs * SLOW_ALPHA * cross_s,
                        SLOW_ALPHA**2 * diag)
            ff_terms = (bf * bf * self.qff, 2 * bf * FAST_ALPHA * cross_f,
                        FAST_ALPHA**2 * diag)
            sf_terms = (bs * bf * self.qsf, bs * FAST_ALPHA * cross_s,
                        SLOW_ALPHA * bf * cross_f, SLOW_ALPHA * FAST_ALPHA * diag)
            next_qss = sum(ss_terms)
            next_qff = sum(ff_terms)
            next_qsf = sum(sf_terms)
            es, ef = EXACT_SLOW_ALPHA, EXACT_FAST_ALPHA
            bs_exact, bf_exact = 1 - es, 1 - ef
            self.qss_error = _linear_error(
                (bs * bs, 2 * bs * SLOW_ALPHA, SLOW_ALPHA**2),
                (bs_exact**2, 2 * bs_exact * es, es**2),
                (self.qss, cross_s, diag),
                (self.qss_error, s_error, diag_error), 12)
            self.qff_error = _linear_error(
                (bf * bf, 2 * bf * FAST_ALPHA, FAST_ALPHA**2),
                (bf_exact**2, 2 * bf_exact * ef, ef**2),
                (self.qff, cross_f, diag),
                (self.qff_error, f_error, diag_error), 12)
            self.qsf_error = _linear_error(
                (bs * bf, bs * FAST_ALPHA, SLOW_ALPHA * bf,
                 SLOW_ALPHA * FAST_ALPHA),
                (bs_exact * bf_exact, bs_exact * ef, es * bf_exact, es * ef),
                (self.qsf, cross_s, cross_f, diag),
                (self.qsf_error, s_error, f_error, diag_error), 16)
            self.qss, self.qff, self.qsf = next_qss, next_qff, next_qsf
            old_ns, old_nf = self.ns, self.nf
            self.ns = bs * old_ns + SLOW_ALPHA * observed
            self.nf = bf * old_nf + FAST_ALPHA * observed
            self.ns_error = _linear_error(
                (bs, SLOW_ALPHA), (bs_exact, es),
                (old_ns, observed), (self.ns_error, observed_error), 5)
            self.nf_error = _linear_error(
                (bf, FAST_ALPHA), (bf_exact, ef),
                (old_nf, observed), (self.nf_error, observed_error), 5)
            next_ws = np.append(bs * self.ws, SLOW_ALPHA)
            next_wf = np.append(bf * self.wf, FAST_ALPHA)
            bs_error = _coefficient_delta(bs, bs_exact)
            bf_error = _coefficient_delta(bf, bf_exact)
            self.ws_error = np.append(
                (abs(bs) + bs_error) * self.ws_error
                + bs_error * np.abs(self.ws)
                + gamma_bound(3) * np.abs(next_ws[:-1]),
                _coefficient_delta(SLOW_ALPHA, es))
            self.wf_error = np.append(
                (abs(bf) + bf_error) * self.wf_error
                + bf_error * np.abs(self.wf)
                + gamma_bound(3) * np.abs(next_wf[:-1]),
                _coefficient_delta(FAST_ALPHA, ef))
            self.ws, self.wf = next_ws, next_wf

        self.units[t], self.mass[t] = u, a
        self.frames += 1
        w, v = 1 - FAST_WEIGHT, FAST_WEIGHT
        terms = (w * w * self.qss, 2 * w * v * self.qsf, v * v * self.qff)
        q = sum(terms)
        ew, ev = 1 - EXACT_FAST_WEIGHT, EXACT_FAST_WEIGHT
        q_error = _linear_error(
            (w * w, 2 * w * v, v * v), (ew * ew, 2 * ew * ev, ev * ev),
            (self.qss, self.qsf, self.qff),
            (self.qss_error, self.qsf_error, self.qff_error), 8)
        numerator = w * self.ns + v * self.nf
        numerator_error = _linear_error(
            (w, v), (ew, ev), (self.ns, self.nf),
            (self.ns_error, self.nf_error), 5)
        if (not np.isfinite(q) or not np.isfinite(q_error)
                or not np.isfinite(numerator).all()
                or not np.isfinite(numerator_error).all()):
            raise NumericalUnresolved("Nonfinite finite-memory kernel state")
        try:
            return self._screen_raw(q, q_error, numerator, numerator_error)
        except NumericalUnresolved:
            start = perf_counter()
            exact = self._exact_raw()
            self.recomputations += 1
            self.recompute_seconds += perf_counter() - start
            return exact

    def _screen_raw(self, q, q_error, numerator, numerator_error):
        if not np.any(self.mass[:self.frames]):
            self.screen_zero_count += 1
            return np.zeros(len(self.gallery), dtype=np.float64)
        lo, hi = q - q_error, q + q_error
        self.minimum_q_lower_bound = min(self.minimum_q_lower_bound, lo)
        if hi < 0:
            raise NumericalUnresolved("Negative squared norm outside error estimate")
        if hi <= FLOOR * FLOOR:
            self.screen_floor_count += 1
            denom = lower = upper = FLOOR
        elif lo <= 0:
            raise NumericalUnresolved("Finite-memory kernel cancellation unresolved")
        else:
            self.screen_regular_count += 1
            denom = max(np.sqrt(q), FLOOR)
            lower, upper = max(np.sqrt(lo), FLOOR), max(np.sqrt(hi), FLOOR)
        raw = numerator / denom
        diagnostic = (numerator_error / lower
                      + np.abs(numerator) * (1 / lower - 1 / upper)
                      + gamma_bound(3) * np.abs(raw))
        worst = float(np.max(diagnostic))
        self.max_raw_error_estimate = max(self.max_raw_error_estimate, worst)
        if worst > self.raw_error_limit or not np.isfinite(raw).all():
            raise NumericalUnresolved("Finite-memory raw score uncertainty exceeds limit")
        return raw

    def _exact_raw(self):
        # Replay the fixed decimal/rational EMA coefficients exactly. The
        # observed a,u and gallery coordinates are exact stored binary values;
        # the fast coefficient arrays above are only numerical approximations.
        deadline = perf_counter() + MAX_EXACT_SECONDS
        t = self.frames
        slow_powers = [Fraction(1)]
        fast_powers = [Fraction(1)]
        for _ in range(1, t):
            if perf_counter() > deadline:
                raise NumericalUnresolved("Finite-memory exact replay time budget exceeded")
            slow_powers.append(slow_powers[-1] * (1 - EXACT_SLOW_ALPHA))
            fast_powers.append(fast_powers[-1] * (1 - EXACT_FAST_ALPHA))
        grouped = {}
        for i, (u, a) in enumerate(zip(self.units[:t], self.mass[:t])):
            if perf_counter() > deadline:
                raise NumericalUnresolved("Finite-memory exact replay time budget exceeded")
            if a == 0:
                continue
            slow = slow_powers[t - 1] if i == 0 else EXACT_SLOW_ALPHA * slow_powers[t - 1 - i]
            fast = fast_powers[t - 1] if i == 0 else EXACT_FAST_ALPHA * fast_powers[t - 1 - i]
            coefficient = (1 - EXACT_FAST_WEIGHT) * slow + EXACT_FAST_WEIGHT * fast
            nonzero = np.flatnonzero(u)
            if not len(nonzero):
                raise NumericalUnresolved("Positive mass on zero direction")
            sign = 1 if u[nonzero[0]] > 0 else -1
            key = tuple(float(sign * value) for value in u)
            weight = Fraction(float(a)) * coefficient * sign
            grouped[key] = grouped.get(key, Fraction(0)) + weight
            if not grouped[key]:
                del grouped[key]
        if grouped and self._rational_gallery is None:
            converted = []
            for g in self.gallery:
                if perf_counter() > deadline:
                    raise NumericalUnresolved("Finite-memory exact replay time budget exceeded")
                converted.append(tuple(Fraction(float(v)) for v in g))
            self._rational_gallery = converted
        q, numerator, count = _bounded_cubic_grouped(
            grouped, self.gallery, self._rational_gallery, deadline)
        self.max_exact_support = max(self.max_exact_support, count)
        if q == 0:
            return np.zeros(len(self.gallery), dtype=np.float64)
        outputs = []
        for precision in (80, 160):
            with localcontext() as ctx:
                ctx.prec = precision
                norm = (Decimal(q.numerator) / Decimal(q.denominator)).sqrt()
                denominator = max(norm, Decimal.from_float(FLOOR))
                values = []
                for n in numerator:
                    if perf_counter() > deadline:
                        raise NumericalUnresolved("Finite-memory exact replay time budget exceeded")
                    values.append(float(
                        (Decimal(n.numerator) / Decimal(n.denominator)) / denominator))
                outputs.append(np.array(values, dtype=np.float64))
        if not np.array_equal(outputs[0], outputs[1]) or not np.isfinite(outputs[-1]).all():
            raise NumericalUnresolved("Finite-memory exact readout unresolved")
        return outputs[-1]


def stable_dual_ema_cubic_raw_prepared(projected, gallery_coordinates, raw_error_limit):
    z = np.asarray(projected, dtype=np.float64)
    g = np.asarray(gallery_coordinates, dtype=np.float64)
    if (z.ndim != 2 or g.ndim != 2 or len(z) < 1 or z.shape[1] != g.shape[1]
            or not np.isfinite(z).all() or not np.isfinite(g).all()):
        raise ValueError("Invalid prepared finite-memory inputs")
    state = StableDualEMACubicHistory(g, len(z), raw_error_limit)
    raw = np.array([state.step(row) for row in z], dtype=np.float64)
    return raw, dict(recomputations=state.recomputations,
                     recompute_seconds=state.recompute_seconds,
                     max_exact_support=state.max_exact_support,
                     max_raw_error_estimate=state.max_raw_error_estimate,
                     screen_zero_count=state.screen_zero_count,
                     screen_floor_count=state.screen_floor_count,
                     screen_regular_count=state.screen_regular_count,
                     minimum_q_lower_bound=(state.minimum_q_lower_bound
                                            if np.isfinite(state.minimum_q_lower_bound)
                                            else None))
