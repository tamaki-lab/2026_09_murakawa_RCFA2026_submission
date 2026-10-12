"""Fixed gallery-discriminative readout on the successful cubic prefix state.

The temporal state and degree-three kernel are unchanged from935112.  Gallery-
only convex-hull margins form one fixed affine map, followed by tanh and the
existing float32/cohort-centering/float32 contract.  No Track, temperature,
ridge, top-k, soft-margin or tuning argument is exposed.
"""
from functools import lru_cache

import numpy as np

from rcfa.methods.gallery_cohort_centering import cohort_center_scores
from scripts.cubic_kernel_reference import prepare
from scripts.cubic_kernel_stable import stable_raw_prepared_scores
from scripts.gallery_discriminative_diagnostic import (
    FEASIBILITY_LIMIT,
    KKT_LIMIT,
    MAP_LIMIT,
    build_margin_map,
    cubic_gram,
)

FINAL_ERROR_LIMIT = 1e-7


def _validate_mapping(mapping):
    details = mapping["details"]
    feasibility = max(
        max(row["kkt"]["sum_residual"], row["kkt"]["negative_violation"])
        for row in details
    )
    kkt = max(
        max(row["kkt"]["scaled_residual"], row["reference_kkt"]["scaled_residual"])
        for row in details
    )
    minimum_distance = min(row["distance_squared"] for row in details)
    if feasibility > FEASIBILITY_LIMIT:
        raise FloatingPointError("Gallery margin simplex feasibility unresolved")
    if kkt > KKT_LIMIT:
        raise FloatingPointError("Gallery margin KKT conditions unresolved")
    if mapping["map_error"] > MAP_LIMIT:
        raise FloatingPointError("Gallery margin independent map mismatch")
    if not (minimum_distance > 0 and np.isfinite(minimum_distance)):
        raise FloatingPointError("Gallery margin distance unresolved")


@lru_cache(maxsize=8)
def _cached_mapping(shape, coordinate_bytes):
    coordinates = np.frombuffer(coordinate_bytes, dtype=np.float64).reshape(shape)
    gram, spectrum = cubic_gram(coordinates)
    if spectrum["rank"] != shape[0] or spectrum["smallest_eigenvalue"] <= 0:
        raise FloatingPointError("Cubic gallery Gram is not positive definite")
    transform, bias, mapping = build_margin_map(gram)
    _validate_mapping(mapping)
    transform.setflags(write=False)
    bias.setflags(write=False)
    return transform, bias, mapping


def discriminative_mapping(gallery_coordinates):
    coordinates = np.ascontiguousarray(gallery_coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or not np.isfinite(coordinates).all():
        raise ValueError("Expected finite two-dimensional gallery coordinates")
    return _cached_mapping(coordinates.shape, coordinates.tobytes())


def apply_discriminative_readout(raw_scores, transform, bias):
    raw = np.asarray(raw_scores, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    offset = np.asarray(bias, dtype=np.float64)
    if (raw.ndim != 2 or matrix.shape != (raw.shape[1], raw.shape[1])
            or offset.shape != (raw.shape[1],) or not np.isfinite(raw).all()
            or not np.isfinite(matrix).all() or not np.isfinite(offset).all()):
        raise ValueError("Invalid discriminative readout inputs")
    with np.errstate(over="raise", invalid="raise"):
        margins = raw @ matrix.T + offset
        bounded = np.tanh(margins).astype(np.float32)
    scores = cohort_center_scores(bounded)
    if scores.dtype != np.float32 or not np.isfinite(scores).all():
        raise FloatingPointError("Discriminative score contract failed")
    return scores


def gallery_discriminative_diagnostics(input_features, gallery):
    projected, coordinates, _ = prepare(input_features, gallery)
    transform, bias, mapping = discriminative_mapping(coordinates)
    amplification = float(np.max(np.sum(np.abs(transform), axis=1)))
    # Reserve half the final tolerance for history error propagation; the rest
    # covers fixed-map arithmetic and final readout validation.
    raw_error_limit = FINAL_ERROR_LIMIT / (2 * max(1.0, amplification))
    raw_scores, info = stable_raw_prepared_scores(
        projected, coordinates, 3, raw_error_limit)
    scores = apply_discriminative_readout(raw_scores, transform, bias)
    info.update(
        raw_error_limit=raw_error_limit,
        map_amplification=amplification,
        map_error=mapping["map_error"],
        projected_shape=list(projected.shape),
    )
    return scores, info


def gallery_discriminative_scores(input_features, gallery):
    return gallery_discriminative_diagnostics(input_features, gallery)[0]
