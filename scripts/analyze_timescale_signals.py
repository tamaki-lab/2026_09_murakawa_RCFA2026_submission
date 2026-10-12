#!/usr/bin/env python3
"""Analyze causal fast/slow state disagreement without private labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from example import normalize  # noqa: E402
from rcfa.embedding import causal_ema  # noqa: E402


QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
DEFAULT_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
SCORE_DROP_THRESHOLDS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
QUALITY_DEVIATION_LIMITS = (0.05, 0.10, 0.15)
DEFAULT_PERSISTENCE = (3, 5, 10)


def load_features(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        features = archive["feature"].astype(np.float32)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError(f"Invalid features in {path}")
    return features


def select_evenly(paths: list[Path], count: int) -> list[Path]:
    if count >= len(paths):
        return paths
    indices = np.linspace(0, len(paths) - 1, count, dtype=int)
    return [paths[int(index)] for index in indices]


def state_signals(
    features: np.ndarray,
    gallery_normalized: np.ndarray,
    slow_alpha: float,
    fast_alpha: float,
) -> dict[str, np.ndarray]:
    slow = normalize(causal_ema(features, slow_alpha))
    fast = normalize(causal_ema(features, fast_alpha))
    current = normalize(features)
    feature_norm = np.linalg.norm(features, axis=1)
    slow_feature_norm = causal_ema(feature_norm[:, None], slow_alpha)[:, 0]
    norm_ratio = feature_norm / np.maximum(slow_feature_norm, 1e-12)
    disagreement = 1.0 - np.clip(np.sum(slow * fast, axis=1), -1.0, 1.0)
    current_slow = 1.0 - np.clip(np.sum(current * slow, axis=1), -1.0, 1.0)
    current_fast = 1.0 - np.clip(np.sum(current * fast, axis=1), -1.0, 1.0)

    slow_scores = slow @ gallery_normalized.T
    fast_scores = fast @ gallery_normalized.T
    slow_top1 = np.argmax(slow_scores, axis=1)
    row_indices = np.arange(features.shape[0])
    slow_top2 = np.partition(slow_scores, -2, axis=1)[:, -2:]
    fast_top2 = np.partition(fast_scores, -2, axis=1)[:, -2:]
    return {
        "fast_slow_disagreement": disagreement,
        "current_slow_disagreement": current_slow,
        "current_fast_disagreement": current_fast,
        "slow_gallery_margin": np.max(slow_top2, axis=1) - np.min(slow_top2, axis=1),
        "fast_gallery_margin": np.max(fast_top2, axis=1) - np.min(fast_top2, axis=1),
        "fast_slow_top1_disagree": (
            np.argmax(fast_scores, axis=1) != slow_top1
        ).astype(np.float64),
        "slow_identity_fast_score_drop": (
            slow_scores[row_indices, slow_top1] - fast_scores[row_indices, slow_top1]
        ),
        "feature_norm": feature_norm,
        "feature_norm_ratio_to_slow": norm_ratio,
        "feature_norm_log_deviation": np.abs(np.log(np.maximum(norm_ratio, 1e-12))),
    }


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, QUANTILES)
    summary = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }
    summary.update(
        {f"q{quantile:g}": float(value) for quantile, value in zip(QUANTILES, quantiles)}
    )
    return summary


def true_runs(mask: np.ndarray) -> list[int]:
    runs: list[int] = []
    current = 0
    for value in mask:
        if bool(value):
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def first_persistent_true(mask: np.ndarray, persistence: int) -> int | None:
    current = 0
    for index, value in enumerate(mask):
        current = current + 1 if bool(value) else 0
        if current >= persistence:
            return index - persistence + 1
    return None


def detection_summary(
    per_sequence: list[np.ndarray],
    thresholds: tuple[float, ...],
    persistence_values: tuple[int, ...],
    additional_masks: list[np.ndarray] | None = None,
) -> dict[str, dict[str, dict[str, float | None]]]:
    result: dict[str, dict[str, dict[str, float | None]]] = {}
    for threshold in thresholds:
        threshold_result: dict[str, dict[str, float | None]] = {}
        for persistence in persistence_values:
            delays = []
            for index, values in enumerate(per_sequence):
                mask = values >= threshold
                if additional_masks is not None:
                    mask = mask & additional_masks[index]
                delays.append(first_persistent_true(mask, persistence))
            observed = [float(delay) for delay in delays if delay is not None]
            threshold_result[str(persistence)] = {
                "detected_fraction": len(observed) / len(delays),
                "delay_mean": float(np.mean(observed)) if observed else None,
                "delay_max": float(max(observed)) if observed else None,
            }
        result[f"{threshold:g}"] = threshold_result
    return result


def activation_summary(
    per_sequence: list[np.ndarray],
    thresholds: tuple[float, ...],
    persistence_values: tuple[int, ...],
    additional_masks: list[np.ndarray] | None = None,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    total_frames = sum(values.size for values in per_sequence)
    for threshold in thresholds:
        run_lengths: list[int] = []
        active_frames = 0
        sequence_runs: list[list[int]] = []
        for index, values in enumerate(per_sequence):
            mask = values >= threshold
            if additional_masks is not None:
                mask = mask & additional_masks[index]
            active_frames += int(np.sum(mask))
            runs = true_runs(mask)
            run_lengths.extend(runs)
            sequence_runs.append(runs)
        result[f"{threshold:g}"] = {
            "active_frame_fraction": active_frames / total_frames,
            "activation_runs_per_1000_frames": len(run_lengths) * 1000.0 / total_frames,
            "run_length_mean": float(np.mean(run_lengths)) if run_lengths else 0.0,
            "run_length_max": float(max(run_lengths, default=0)),
            "persistent": {
                str(persistence): {
                    "runs_per_1000_frames": (
                        sum(run >= persistence for run in run_lengths) * 1000.0 / total_frames
                    ),
                    "sequence_fraction": (
                        sum(any(run >= persistence for run in runs) for runs in sequence_runs)
                        / len(sequence_runs)
                    ),
                }
                for persistence in persistence_values
            },
        }
    return result


def choose_transition_pairs(
    sequences: list[np.ndarray], pair_count: int, anchor_frames: int
) -> list[tuple[int, int]]:
    anchors = np.stack(
        [normalize(sequence[:anchor_frames].mean(axis=0, keepdims=True))[0] for sequence in sequences]
    )
    similarity = anchors @ anchors.T
    candidates: list[tuple[float, int, int]] = []
    for left in range(len(sequences)):
        for right in range(left + 1, len(sequences)):
            candidates.append((float(similarity[left, right]), left, right))
    candidates.sort()
    return [(left, right) for _, left, right in candidates[:pair_count]]


def transition_detection_summary(
    sequences: list[np.ndarray],
    pairs: list[tuple[int, int]],
    slow_alpha: float,
    fast_alpha: float,
    gallery_normalized: np.ndarray,
    pre_frames: int,
    post_frames: int,
    thresholds: tuple[float, ...],
    persistence_values: tuple[int, ...],
) -> dict[str, object]:
    transition_disagreements: list[np.ndarray] = []
    transition_top1_mismatch: list[np.ndarray] = []
    transition_score_drops: list[np.ndarray] = []
    transition_norm_deviations: list[np.ndarray] = []
    pre_disagreements: list[np.ndarray] = []
    for left, right in pairs:
        synthetic = np.concatenate(
            [sequences[left][-pre_frames:], sequences[right][:post_frames]], axis=0
        )
        slow = normalize(causal_ema(synthetic, slow_alpha))
        fast = normalize(causal_ema(synthetic, fast_alpha))
        disagreement = 1.0 - np.clip(np.sum(slow * fast, axis=1), -1.0, 1.0)
        slow_scores = slow @ gallery_normalized.T
        fast_scores = fast @ gallery_normalized.T
        slow_top1 = np.argmax(slow_scores, axis=1)
        fast_top1 = np.argmax(fast_scores, axis=1)
        row_indices = np.arange(synthetic.shape[0])
        score_drop = slow_scores[row_indices, slow_top1] - fast_scores[row_indices, slow_top1]
        feature_norm = np.linalg.norm(synthetic, axis=1)
        slow_feature_norm = causal_ema(feature_norm[:, None], slow_alpha)[:, 0]
        norm_deviation = np.abs(
            np.log(feature_norm / np.maximum(slow_feature_norm, 1e-12))
        )
        pre_disagreements.append(disagreement[:pre_frames])
        transition_disagreements.append(disagreement[pre_frames:])
        transition_top1_mismatch.append((slow_top1 != fast_top1)[pre_frames:])
        transition_score_drops.append(score_drop[pre_frames:])
        transition_norm_deviations.append(norm_deviation[pre_frames:])

    detection: dict[str, dict[str, dict[str, float | None]]] = {}
    detection_with_top1: dict[str, dict[str, dict[str, float | None]]] = {}
    for threshold in thresholds:
        threshold_result: dict[str, dict[str, float | None]] = {}
        combined_result: dict[str, dict[str, float | None]] = {}
        for persistence in persistence_values:
            delays = [
                first_persistent_true(values >= threshold, persistence)
                for values in transition_disagreements
            ]
            combined_delays = [
                first_persistent_true(
                    (values >= threshold) & mismatch, persistence
                )
                for values, mismatch in zip(
                    transition_disagreements, transition_top1_mismatch
                )
            ]
            observed = [float(delay) for delay in delays if delay is not None]
            combined_observed = [
                float(delay) for delay in combined_delays if delay is not None
            ]
            threshold_result[str(persistence)] = {
                "detected_fraction": len(observed) / len(delays),
                "delay_mean": float(np.mean(observed)) if observed else None,
                "delay_max": float(max(observed)) if observed else None,
            }
            combined_result[str(persistence)] = {
                "detected_fraction": len(combined_observed) / len(combined_delays),
                "delay_mean": (
                    float(np.mean(combined_observed)) if combined_observed else None
                ),
                "delay_max": float(max(combined_observed)) if combined_observed else None,
            }
        detection[f"{threshold:g}"] = threshold_result
        detection_with_top1[f"{threshold:g}"] = combined_result

    return {
        "pre_transition_quantiles": quantile_summary(np.concatenate(pre_disagreements)),
        "post_frames_0_9_quantiles": quantile_summary(
            np.concatenate([values[:10] for values in transition_disagreements])
        ),
        "post_frames_10_49_quantiles": quantile_summary(
            np.concatenate([values[10:50] for values in transition_disagreements])
        ),
        "post_frames_50_99_quantiles": quantile_summary(
            np.concatenate([values[50:100] for values in transition_disagreements])
        ),
        "detection": detection,
        "detection_with_top1_disagreement": detection_with_top1,
        "score_drop_quantiles": quantile_summary(
            np.concatenate(transition_score_drops)
        ),
        "score_drop_detection": detection_summary(
            transition_score_drops, SCORE_DROP_THRESHOLDS, persistence_values
        ),
        "quality_gated_score_drop_detection": {
            f"{limit:g}": detection_summary(
                transition_score_drops,
                SCORE_DROP_THRESHOLDS,
                persistence_values,
                [values <= limit for values in transition_norm_deviations],
            )
            for limit in QUALITY_DEVIATION_LIMITS
        },
    }


def analyze_track(
    data_root: Path,
    track: str,
    max_sequences: int,
    transition_pairs: int,
    pre_frames: int,
    post_frames: int,
    slow_alpha: float,
    fast_alpha: float,
    thresholds: tuple[float, ...],
    persistence_values: tuple[int, ...],
) -> dict[str, object]:
    track_root = data_root / track
    all_paths = sorted((track_root / "probe_seq_embeddings").glob("*.npz"))
    paths = select_evenly(all_paths, max_sequences)
    if len(paths) < 2:
        raise ValueError(f"Need at least two sequences under {track_root}")
    sequences = [load_features(path) for path in paths]
    if any(sequence.shape[0] < max(pre_frames, post_frames) for sequence in sequences):
        raise ValueError("A sequence is shorter than the requested synthetic segments")
    gallery = normalize(load_features(track_root / "gallery.npz"))

    per_sequence_signals = [
        state_signals(sequence, gallery, slow_alpha, fast_alpha) for sequence in sequences
    ]
    signal_names = tuple(per_sequence_signals[0])
    signals = {
        name: np.concatenate([record[name] for record in per_sequence_signals])
        for name in signal_names
    }
    disagreements = [record["fast_slow_disagreement"] for record in per_sequence_signals]
    top1_mismatches = [
        record["fast_slow_top1_disagree"].astype(bool)
        for record in per_sequence_signals
    ]
    score_drops = [
        record["slow_identity_fast_score_drop"] for record in per_sequence_signals
    ]
    norm_deviations = [
        record["feature_norm_log_deviation"] for record in per_sequence_signals
    ]
    pair_count = min(transition_pairs, math.comb(len(sequences), 2))
    pairs = choose_transition_pairs(sequences, pair_count, min(100, pre_frames))
    return {
        "sequence_count": len(sequences),
        "frame_count": int(sum(sequence.shape[0] for sequence in sequences)),
        "signal_quantiles": {name: quantile_summary(values) for name, values in signals.items()},
        "fast_slow_activations": activation_summary(
            disagreements, thresholds, persistence_values
        ),
        "fast_slow_top1_gated_activations": activation_summary(
            disagreements, thresholds, persistence_values, top1_mismatches
        ),
        "slow_identity_score_drop_activations": activation_summary(
            score_drops, SCORE_DROP_THRESHOLDS, persistence_values
        ),
        "quality_gated_score_drop_activations": {
            f"{limit:g}": activation_summary(
                score_drops,
                SCORE_DROP_THRESHOLDS,
                persistence_values,
                [values <= limit for values in norm_deviations],
            )
            for limit in QUALITY_DEVIATION_LIMITS
        },
        "synthetic_transitions": transition_detection_summary(
            sequences,
            pairs,
            slow_alpha,
            fast_alpha,
            gallery,
            pre_frames,
            post_frames,
            thresholds,
            persistence_values,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("challenge_data"))
    parser.add_argument("--max-sequences", type=int, default=40)
    parser.add_argument("--transition-pairs", type=int, default=20)
    parser.add_argument("--pre-frames", type=int, default=200)
    parser.add_argument("--post-frames", type=int, default=400)
    parser.add_argument("--slow-alpha", type=float, default=0.01)
    parser.add_argument("--fast-alpha", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=Path("results/timescale_signal_analysis.json"))
    args = parser.parse_args()

    if not 0.0 < args.slow_alpha < args.fast_alpha <= 1.0:
        raise ValueError("Expected 0 < slow_alpha < fast_alpha <= 1")
    if args.max_sequences < 2 or args.transition_pairs < 1:
        raise ValueError("Need at least two sequences and one transition pair")

    report = {
        "protocol": {
            "label_independent": True,
            "offline_synthetic_transition_only": True,
            "selection": "evenly spaced paths; names not logged",
            "slow_alpha": args.slow_alpha,
            "fast_alpha": args.fast_alpha,
            "thresholds": list(DEFAULT_THRESHOLDS),
            "score_drop_thresholds": list(SCORE_DROP_THRESHOLDS),
            "quality_deviation_limits": list(QUALITY_DEVIATION_LIMITS),
            "persistence_values": list(DEFAULT_PERSISTENCE),
        },
        "tracks": {
            track: analyze_track(
                args.data_root,
                track,
                args.max_sequences,
                args.transition_pairs,
                args.pre_frames,
                args.post_frames,
                args.slow_alpha,
                args.fast_alpha,
                DEFAULT_THRESHOLDS,
                DEFAULT_PERSISTENCE,
            )
            for track in ("track1", "track2")
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Signal analysis: {args.output}")
    for track, track_report in report["tracks"].items():
        disagreement = track_report["signal_quantiles"]["fast_slow_disagreement"]
        top1 = track_report["signal_quantiles"]["fast_slow_top1_disagree"]
        print(
            f"{track}: sequences={track_report['sequence_count']} "
            f"frames={track_report['frame_count']} "
            f"disagreement_q95={disagreement['q0.95']:.6f} "
            f"q99={disagreement['q0.99']:.6f} "
            f"top1_disagree_mean={top1['mean']:.6f} "
            f"top1_disagree_q99={top1['q0.99']:.6f}"
        )


if __name__ == "__main__":
    main()
