"""Gallery-only numerical diagnostic for the fixed discriminative readout.

This module never reads probe sequences, labels, scores, or submission ZIPs.
It diagnoses whether the one-positive-versus-negative-convex-hull construction
is numerically well defined on the two supplied galleries.  Passing this file's
gates is not evidence of authentication performance and does not register a
submission processor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from example import normalize
from rcfa.methods.gallery_cohort_centering import cohort_center_scores
from rcfa.methods.gallery_subspace_residual import GallerySubspace
from rcfa.embedding import _validate_features


FEASIBILITY_LIMIT = 1e-12
KKT_LIMIT = 1e-12
MAP_LIMIT = 1e-8
SCORE_LIMIT = 1e-7
MAX_UPDATES = 10_000
MAX_SECONDS = 3_600.0
MAX_RSS_BYTES = 1 << 30
COEFFICIENT_CLEAR = 1e-10
COEFFICIENT_AMBIGUOUS = 1e-14
MARGIN_EQUAL = 1e-8
MARGIN_AMBIGUOUS = 1e-6


class DiagnosticUnresolved(FloatingPointError):
    """The fixed gallery-only numerical contract could not be resolved."""


def gallery_coordinates(gallery: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the unchanged cubic row-space gallery coordinates and basis."""
    values = _validate_features(gallery)
    subspace = GallerySubspace(values)
    if subspace.rank != len(values):
        raise DiagnosticUnresolved("Gallery is not full row rank")
    coordinates = np.einsum(
        "kd,rd->kr", normalize(values), subspace.basis, optimize=False
    )
    if not np.isfinite(coordinates).all():
        raise DiagnosticUnresolved("Nonfinite gallery coordinates")
    return coordinates, subspace.basis


def cubic_gram(coordinates: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Construct the stored-coordinate cubic Gram and a long-double reference."""
    gamma = np.asarray(coordinates, dtype=np.float64)
    linear = np.einsum("ir,jr->ij", gamma, gamma, optimize=False)
    gram = linear ** 3
    gamma_ld = gamma.astype(np.longdouble)
    linear_ld = np.einsum("ir,jr->ij", gamma_ld, gamma_ld, optimize=False)
    gram_ld = linear_ld ** 3
    reference_error = float(np.max(np.abs(gram_ld - gram.astype(np.longdouble))))
    symmetry_error = float(np.max(np.abs(gram - gram.T)))
    eigenvalues = np.linalg.eigvalsh((gram + gram.T) / 2)
    largest = float(eigenvalues[-1])
    tolerance = largest * max(gram.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    smallest = float(eigenvalues[0])
    condition = float(largest / smallest) if smallest > 0 else float("inf")
    return gram, {
        "eigenvalues": eigenvalues.tolist(),
        "smallest_eigenvalue": smallest,
        "largest_eigenvalue": largest,
        "condition": condition,
        "rank": rank,
        "rank_tolerance": float(tolerance),
        "symmetry_error": symmetry_error,
        "longdouble_gram_error": reference_error,
    }


def _support_solution(
    matrix: np.ndarray, linear: np.ndarray, support: np.ndarray, method: str
) -> np.ndarray:
    """Solve the equality-constrained QP on one fixed support."""
    block = matrix[np.ix_(support, support)]
    size = len(support)
    kkt = np.empty((size + 1, size + 1), dtype=np.float64)
    kkt[:size, :size] = block
    kkt[:size, size] = 1.0
    kkt[size, :size] = 1.0
    kkt[size, size] = 0.0
    rhs = np.concatenate((linear[support], [1.0]))
    if method == "solve":
        solution = np.linalg.solve(kkt, rhs)
    elif method == "lstsq":
        solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    else:
        raise ValueError("Unknown support solver")
    return solution[:-1]


def kkt_metrics(
    matrix: np.ndarray, linear: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    """Evaluate simplex feasibility and KKT residuals in long double."""
    a = matrix.astype(np.longdouble)
    b = linear.astype(np.longdouble)
    x = weights.astype(np.longdouble)
    gradient = a @ x - b
    support = np.flatnonzero(weights > COEFFICIENT_AMBIGUOUS)
    inactive = np.setdiff1d(np.arange(len(weights)), support, assume_unique=True)
    if not len(support):
        raise DiagnosticUnresolved("Empty simplex support")
    multiplier = np.mean(gradient[support])
    support_residual = float(np.max(np.abs(gradient[support] - multiplier)))
    inactive_violation = (
        float(np.max(np.maximum(multiplier - gradient[inactive], 0)))
        if len(inactive) else 0.0
    )
    scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return {
        "sum_residual": float(abs(np.sum(x) - 1)),
        "negative_violation": float(max(0, -np.min(x))),
        "support_residual": support_residual,
        "inactive_violation": inactive_violation,
        "scaled_residual": max(support_residual, inactive_violation) / scale,
        "multiplier": float(multiplier),
    }


def solve_simplex_qp(
    matrix: np.ndarray, linear: np.ndarray, max_updates: int = MAX_UPDATES
) -> tuple[np.ndarray, dict[str, object]]:
    """Deterministic primal active-set solver, initialized at uniform weights."""
    a = np.asarray(matrix, dtype=np.float64)
    b = np.asarray(linear, dtype=np.float64)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != (len(a),)
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Expected one finite square simplex QP")
    count = len(b)
    weights = np.full(count, 1.0 / count)
    active = list(range(count))
    scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    for updates in range(1, max_updates + 1):
        support = np.array(active, dtype=int)
        candidate = _support_solution(a, b, support, "solve")
        materially_negative = candidate < -FEASIBILITY_LIMIT
        if np.any(materially_negative):
            current = weights[support]
            negative = np.flatnonzero(candidate < 0)
            ratios = current[negative] / (current[negative] - candidate[negative])
            step = float(np.min(ratios))
            if not (0 <= step <= 1 and np.isfinite(step)):
                raise DiagnosticUnresolved("Invalid active-set boundary step")
            weights[support] = current + step * (candidate - current)
            if float(np.min(weights)) < -FEASIBILITY_LIMIT:
                raise DiagnosticUnresolved("Active-set feasibility lost")
            weights[np.abs(weights) <= COEFFICIENT_AMBIGUOUS] = 0.0
            hit = negative[np.flatnonzero(np.abs(ratios - step) <= 8 * np.finfo(float).eps)]
            if not len(hit):
                hit = np.array([negative[int(np.argmin(ratios))]])
            drop = {int(support[index]) for index in hit}
            active = [index for index in active if index not in drop]
            if not active:
                raise DiagnosticUnresolved("Active set became empty")
            continue

        if float(np.min(candidate)) < -FEASIBILITY_LIMIT:
            raise DiagnosticUnresolved("Affine support solution is infeasible")
        candidate[candidate < 0] = 0.0
        total = float(np.sum(candidate))
        if total <= 0 or not np.isfinite(total):
            raise DiagnosticUnresolved("Invalid simplex normalization")
        candidate /= total
        weights.fill(0.0)
        weights[support] = candidate
        gradient = a @ weights - b
        multiplier = float(np.mean(gradient[support]))
        inactive = np.array([index for index in range(count) if index not in active])
        if not len(inactive):
            break
        violations = multiplier - gradient[inactive]
        worst = float(np.max(violations))
        if worst <= KKT_LIMIT * scale:
            break
        add_candidates = inactive[np.flatnonzero(violations == np.max(violations))]
        active.append(int(np.min(add_candidates)))
        active.sort()
    else:
        raise DiagnosticUnresolved("Active-set update limit exceeded")

    metrics = kkt_metrics(a, b, weights)
    metrics.update({
        "updates": updates,
        "active_indices": np.flatnonzero(weights > COEFFICIENT_AMBIGUOUS).tolist(),
    })
    return weights, metrics


def _reference_weights(
    matrix: np.ndarray, linear: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Independent SVD/lstsq solve on the KKT-certified support."""
    support = np.flatnonzero(weights > COEFFICIENT_AMBIGUOUS)
    reference = np.zeros_like(weights)
    reference[support] = _support_solution(matrix, linear, support, "lstsq")
    return reference


def build_margin_map(gram: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build all convex-hull margins and compare with a support reference."""
    h = np.asarray(gram, dtype=np.float64)
    count = len(h)
    transform = np.zeros_like(h)
    bias = np.zeros(count)
    reference_transform = np.zeros_like(h)
    reference_bias = np.zeros(count)
    lambdas = np.zeros_like(h)
    details = []
    maximum_updates = 0
    for identity in range(count):
        negative = np.delete(np.arange(count), identity)
        matrix = h[np.ix_(negative, negative)]
        linear = h[negative, identity]
        weights, metrics = solve_simplex_qp(matrix, linear)
        reference = _reference_weights(matrix, linear, weights)
        reference_metrics = kkt_metrics(matrix, linear, reference)
        quadratic = float(weights @ matrix @ weights)
        distance = float(h[identity, identity] - 2 * linear @ weights + quadratic)
        quadratic_reference = float(reference @ matrix @ reference)
        distance_reference = float(
            h[identity, identity] - 2 * linear @ reference + quadratic_reference)
        if not (distance > 0 and distance_reference > 0):
            raise DiagnosticUnresolved("Nonpositive hull distance")
        transform[identity, identity] = 2 / distance
        transform[identity, negative] = -2 * weights / distance
        bias[identity] = -(h[identity, identity] - quadratic) / distance
        reference_transform[identity, identity] = 2 / distance_reference
        reference_transform[identity, negative] = -2 * reference / distance_reference
        reference_bias[identity] = (
            -(h[identity, identity] - quadratic_reference) / distance_reference
        )
        lambdas[identity, negative] = weights
        candidate_margin = h @ transform[identity] + bias[identity]
        negative_margin = candidate_margin[negative]
        support_clear = int(np.count_nonzero(weights > COEFFICIENT_CLEAR))
        support_ambiguous = int(np.count_nonzero(
            (weights > COEFFICIENT_AMBIGUOUS) & (weights <= COEFFICIENT_CLEAR)))
        equal = np.abs(negative_margin + 1)
        details.append({
            "identity": identity,
            "distance_squared": distance,
            "distance_reference": distance_reference,
            "error_amplification_4_over_d2": 4 / distance,
            "support_count": support_clear,
            "support_ambiguous_count": support_ambiguous,
            "margin_equal_count": int(np.count_nonzero(equal <= MARGIN_EQUAL)),
            "margin_ambiguous_count": int(np.count_nonzero(
                (equal > MARGIN_EQUAL) & (equal <= MARGIN_AMBIGUOUS))),
            "self_margin_error": float(abs(candidate_margin[identity] - 1)),
            "maximum_negative_margin": float(np.max(negative_margin)),
            "kkt": metrics,
            "reference_kkt": reference_metrics,
        })
        maximum_updates = max(maximum_updates, int(metrics["updates"]))

    diagonal_norm = np.sqrt(np.maximum(np.diag(h), 0))
    difference = np.abs(transform - reference_transform)
    map_errors = difference @ diagonal_norm + np.abs(bias - reference_bias)
    gallery_margins = h @ transform.T + bias
    inverse = np.linalg.solve(h, np.eye(count))
    r = np.linalg.solve(h, np.ones(count))
    s = float(np.sum(r))
    inverse_l = inverse - np.outer(r, r) / s
    inverse_transform = 2 * inverse_l
    inverse_bias = 2 * r / s - 1
    all_active = np.array([row["support_count"] == count - 1
                           and row["support_ambiguous_count"] == 0
                           for row in details])
    if np.any(all_active):
        inverse_error = float(np.max(
            np.abs(transform[all_active] - inverse_transform[all_active])
        ))
        inverse_bias_error = float(np.max(
            np.abs(bias[all_active] - inverse_bias[all_active])
        ))
    else:
        inverse_error = inverse_bias_error = None
    return transform, bias, {
        "details": details,
        "lambdas": lambdas,
        "reference_transform": reference_transform,
        "reference_bias": reference_bias,
        "map_error": float(np.max(map_errors)),
        "map_errors": map_errors.tolist(),
        "maximum_updates": maximum_updates,
        "gallery_self_margin_error": float(np.max(np.abs(np.diag(gallery_margins) - 1))),
        "gallery_maximum_negative_margin": float(np.max(
            gallery_margins[~np.eye(count, dtype=bool)])),
        "all_active_count": int(np.count_nonzero(all_active)),
        "inverse_gram_transform_error_on_all_active": inverse_error,
        "inverse_gram_bias_error_on_all_active": inverse_bias_error,
        "inverse_gram_global_map_difference": float(np.max(
            np.abs(transform - inverse_transform))),
    }


def _centered_margin_scores(
    score_profiles: np.ndarray, transform: np.ndarray, bias: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    margins = score_profiles @ transform.T + bias
    if not np.isfinite(margins).all():
        raise DiagnosticUnresolved("Nonfinite fixed-set margin")
    bounded = np.tanh(margins).astype(np.float32)
    scores = cohort_center_scores(bounded)
    return margins, scores


def inspect_fixed_set(
    gram: np.ndarray,
    transform: np.ndarray,
    bias: np.ndarray,
    lambdas: np.ndarray,
    reference_transform: np.ndarray,
    reference_bias: np.ndarray,
) -> dict[str, object]:
    """Evaluate fixed gallery-derived stress inputs; never infer performance."""
    count = len(gram)
    profiles = [np.zeros(count)]
    labels = ["zero"]
    diagonal = np.diag(gram)
    for identity in range(count):
        norm = np.sqrt(diagonal[identity])
        profiles.extend((gram[identity] / norm, -gram[identity] / norm))
        labels.extend(("gallery", "negative_gallery"))
    normalized_hull_target_margin = []
    for identity in range(count):
        weights = lambdas[identity]
        norm_squared = float(weights @ gram @ weights)
        if norm_squared <= 0:
            raise DiagnosticUnresolved("Zero negative-hull norm")
        profile = weights @ gram / np.sqrt(norm_squared)
        profiles.append(profile)
        labels.append("normalized_negative_hull")
        margin = float(profile @ transform[identity] + bias[identity])
        normalized_hull_target_margin.append(margin)
    for left in range(count):
        for right in range(left + 1, count):
            plus_norm = diagonal[left] + diagonal[right] + 2 * gram[left, right]
            minus_norm = diagonal[left] + diagonal[right] - 2 * gram[left, right]
            if plus_norm > 0:
                profiles.append((gram[left] + gram[right]) / np.sqrt(plus_norm))
                labels.append("pair_sum")
            if minus_norm > 0:
                profiles.append((gram[left] - gram[right]) / np.sqrt(minus_norm))
                labels.append("pair_difference")
    matrix = np.asarray(profiles, dtype=np.float64)
    margins, scores = _centered_margin_scores(matrix, transform, bias)
    reference_margins, reference_scores = _centered_margin_scores(
        matrix, reference_transform, reference_bias)
    bounded = np.tanh(margins).astype(np.float32)
    saturation = np.abs(bounded) == 1
    tie_rows = sum(np.unique(row).size != count for row in scores)
    zero_margin, zero_score = margins[0], scores[0]
    return {
        "rows": len(matrix),
        "type_counts": {label: labels.count(label) for label in sorted(set(labels))},
        "margin_min": float(np.min(margins)),
        "margin_max": float(np.max(margins)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "saturated_elements": int(np.count_nonzero(saturation)),
        "saturated_rows": int(np.count_nonzero(np.any(saturation, axis=1))),
        "tie_rows": int(tie_rows),
        "zero_margin_min": float(np.min(zero_margin)),
        "zero_margin_max": float(np.max(zero_margin)),
        "zero_score_min": float(np.min(zero_score)),
        "zero_score_max": float(np.max(zero_score)),
        "zero_score_linf": float(np.max(np.abs(zero_score))),
        "normalized_hull_target_margins": normalized_hull_target_margin,
        "normalized_hull_margin_min": float(np.min(normalized_hull_target_margin)),
        "normalized_hull_margin_max": float(np.max(normalized_hull_target_margin)),
        "reference_margin_error": float(np.max(
            np.abs(margins - reference_margins))),
        "reference_score_error": float(np.max(
            np.abs(scores.astype(np.float64) - reference_scores.astype(np.float64)))),
    }


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def diagnose_gallery(gallery: np.ndarray) -> dict[str, object]:
    """Run the fixed diagnostic on one gallery matrix."""
    started = time.perf_counter()
    values = _validate_features(gallery)
    gamma, basis = gallery_coordinates(values)
    gram, spectrum = cubic_gram(gamma)
    transform, bias, mapping = build_margin_map(gram)
    fixed = inspect_fixed_set(
        gram,
        transform,
        bias,
        mapping["lambdas"],
        mapping["reference_transform"],
        mapping["reference_bias"],
    )

    permutation = np.arange(len(values) - 1, -1, -1)
    permuted_gamma, _ = gallery_coordinates(values[permutation])
    permuted_gram, _ = cubic_gram(permuted_gamma)
    permuted_transform, permuted_bias, _ = build_margin_map(permuted_gram)
    restored_transform = np.empty_like(transform)
    restored_transform[np.ix_(permutation, permutation)] = permuted_transform
    restored_bias = np.empty_like(bias)
    restored_bias[permutation] = permuted_bias
    diagonal_norm = np.sqrt(np.maximum(np.diag(gram), 0))
    permutation_map_error = float(np.max(
        np.abs(restored_transform - transform) @ diagonal_norm
        + np.abs(restored_bias - bias)
    ))

    # Powers of two preserve the stored floating-point values exactly.  This
    # checks positive row-scale invariance without silently changing dtype or
    # introducing a second quantization experiment.
    exponents = (np.arange(len(values)) % 3 - 1)[:, None]
    scaled_gamma, _ = gallery_coordinates(np.ldexp(values, exponents))
    scaled_gram, _ = cubic_gram(scaled_gamma)
    scale_gram_error = float(np.max(np.abs(scaled_gram - gram)))

    details = mapping["details"]
    maximum_feasibility = max(
        max(row["kkt"]["sum_residual"], row["kkt"]["negative_violation"])
        for row in details
    )
    maximum_kkt = max(row["kkt"]["scaled_residual"] for row in details)
    maximum_reference_kkt = max(
        row["reference_kkt"]["scaled_residual"] for row in details
    )
    minimum_distance = min(row["distance_squared"] for row in details)
    distance_reference_error = max(
        abs(row["distance_squared"] - row["distance_reference"])
        for row in details
    )
    elapsed = time.perf_counter() - started
    rss = _rss_bytes()
    gates = {
        "gram": bool(spectrum["rank"] == len(values)
                     and spectrum["smallest_eigenvalue"] > 0
                     and np.isfinite(spectrum["condition"])),
        "simplex_feasibility": bool(maximum_feasibility <= FEASIBILITY_LIMIT),
        "kkt": bool(max(maximum_kkt, maximum_reference_kkt) <= KKT_LIMIT),
        "distance": bool(minimum_distance > 0 and np.isfinite(minimum_distance)),
        "independent_map": bool(mapping["map_error"] <= MAP_LIMIT),
        "final_readout": bool(fixed["reference_score_error"] <= SCORE_LIMIT),
        "permutation": bool(permutation_map_error <= SCORE_LIMIT),
        "positive_scale": bool(scale_gram_error <= SCORE_LIMIT),
        "finite_range": bool(fixed["score_min"] >= -1 and fixed["score_max"] <= 1),
        "update_budget": bool(mapping["maximum_updates"] <= MAX_UPDATES),
        "runtime_budget": bool(elapsed <= MAX_SECONDS),
        "rss_budget": bool(rss <= MAX_RSS_BYTES),
    }
    # Large arrays remain owner-only in the ignored report; console output is summary only.
    return {
        "gallery_shape": list(values.shape),
        "basis_orthogonality_error": float(np.max(
            np.abs(basis @ basis.T - np.eye(len(basis))))),
        "spectrum": spectrum,
        "minimum_distance_squared": minimum_distance,
        "maximum_error_amplification": max(
            row["error_amplification_4_over_d2"] for row in details),
        "distance_reference_error": distance_reference_error,
        "maximum_feasibility_residual": maximum_feasibility,
        "maximum_kkt_residual": maximum_kkt,
        "maximum_reference_kkt_residual": maximum_reference_kkt,
        "support_counts": [row["support_count"] for row in details],
        "support_ambiguous_counts": [row["support_ambiguous_count"] for row in details],
        "margin_equal_counts": [row["margin_equal_count"] for row in details],
        "margin_ambiguous_counts": [row["margin_ambiguous_count"] for row in details],
        "all_active_count": mapping["all_active_count"],
        "map_error": mapping["map_error"],
        "permutation_map_error": permutation_map_error,
        "positive_scale_gram_error": scale_gram_error,
        "gallery_self_margin_error": mapping["gallery_self_margin_error"],
        "gallery_maximum_negative_margin": mapping["gallery_maximum_negative_margin"],
        "inverse_gram_transform_error_on_all_active": mapping[
            "inverse_gram_transform_error_on_all_active"],
        "inverse_gram_bias_error_on_all_active": mapping[
            "inverse_gram_bias_error_on_all_active"],
        "inverse_gram_global_map_difference": mapping[
            "inverse_gram_global_map_difference"],
        "maximum_updates": mapping["maximum_updates"],
        "bias_min": float(np.min(bias)),
        "bias_max": float(np.max(bias)),
        "fixed_inspection": fixed,
        "elapsed_seconds": elapsed,
        "maximum_rss_bytes": rss,
        "state_bytes": int(gram.nbytes + transform.nbytes + bias.nbytes),
        "gates": gates,
        "verdict": "PASS_NUMERICAL_ONLY" if all(gates.values()) else "NO_GO_NUMERICAL",
    }


def run(data_root: Path) -> dict[str, object]:
    # Local import keeps the diagnostic math reusable by the fixed processor
    # without creating a builder -> processor -> diagnostic -> builder cycle.
    from scripts.build_candidate_submission import load_features

    started = time.perf_counter()
    tracks = {}
    for track in ("track1", "track2"):
        gallery_path = data_root / track / "gallery.npz"
        if not gallery_path.is_file():
            raise FileNotFoundError(f"Missing gallery only input: {gallery_path}")
        tracks[track] = diagnose_gallery(load_features(gallery_path))
        summary = tracks[track]
        print(
            f"{track}: verdict={summary['verdict']} condition={summary['spectrum']['condition']:.6g} "
            f"min_d2={summary['minimum_distance_squared']:.6g} "
            f"all_active={summary['all_active_count']}/{summary['gallery_shape'][0]} "
            f"seconds={summary['elapsed_seconds']:.3f}",
            flush=True,
        )
    return {
        "status": (
            "PASS_GALLERY_ONLY_NUMERICAL" if all(
                row["verdict"] == "PASS_NUMERICAL_ONLY" for row in tracks.values()
            ) else "NO_GO_GALLERY_ONLY_NUMERICAL"
        ),
        "contract": {
            "feasibility_limit": FEASIBILITY_LIMIT,
            "kkt_limit": KKT_LIMIT,
            "map_limit": MAP_LIMIT,
            "score_limit": SCORE_LIMIT,
            "max_updates": MAX_UPDATES,
            "max_seconds": MAX_SECONDS,
            "max_rss_bytes": MAX_RSS_BYTES,
            "probe_files_read": 0,
            "score_files_read": 0,
            "labels_read": 0,
        },
        "tracks": tracks,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite diagnostic report")
    report = run(args.data_root)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as target:
        target.write(payload)
    print("status=" + report["status"], flush=True)
    print("report_sha256=" + hashlib.sha256(payload.encode()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
