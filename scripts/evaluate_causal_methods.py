#!/usr/bin/env python3
"""Run label-independent diagnostics for bounded causal RCFA processors."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from example import normalize, your_algorithm  # noqa: E402
from rcfa.embedding import process_embeddings  # noqa: E402


DEFAULT_EMA_ALPHAS = (0.01, 0.03, 0.10)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    method: str
    ema_alpha: float | None = None


@dataclass
class SequenceMetrics:
    consecutive_state_cosine_mean: float
    current_frame_cosine_mean: float
    top1_switch_rate: float
    elapsed_seconds: float


def method_specs(alphas: tuple[float, ...]) -> list[MethodSpec]:
    specs = [
        MethodSpec("frame_only", "frame_only"),
        MethodSpec("cumulative_mean", "cumulative_mean"),
    ]
    specs.extend(MethodSpec(f"ema_{alpha:.2f}", "ema", alpha) for alpha in alphas)
    specs.append(MethodSpec("dual_ema_matched", "dual_ema_matched"))
    specs.append(MethodSpec("dual_ema_short_tail", "dual_ema_short_tail"))
    return specs


def load_features(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        features = archive["feature"].astype(np.float32)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError(f"Invalid features in {path}")
    return features


def process(spec: MethodSpec, features: np.ndarray) -> np.ndarray:
    return process_embeddings(features, spec.method, ema_alpha=spec.ema_alpha)


def first_reach(values: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(values >= threshold)
    return int(indices[0]) if indices.size else None


def prefix_invariance_error(spec: MethodSpec, features: np.ndarray) -> float:
    full = process(spec, features)
    cut_points = sorted({1, max(1, features.shape[0] // 3), features.shape[0] // 2, features.shape[0]})
    maximum = 0.0
    for end in cut_points:
        prefix = process(spec, features[:end])
        maximum = max(maximum, float(np.max(np.abs(prefix - full[:end]))))
    return maximum


def sequence_metrics(
    spec: MethodSpec, features: np.ndarray, gallery_normalized: np.ndarray
) -> SequenceMetrics:
    started = time.perf_counter()
    states = normalize(process(spec, features))
    elapsed = time.perf_counter() - started
    current = normalize(features)
    consecutive = np.sum(states[1:] * states[:-1], axis=1)
    current_similarity = np.sum(states * current, axis=1)
    top1 = np.argmax(states @ gallery_normalized.T, axis=1)
    switch_rate = float(np.mean(top1[1:] != top1[:-1])) if top1.size > 1 else 0.0
    return SequenceMetrics(
        consecutive_state_cosine_mean=float(np.mean(consecutive)),
        current_frame_cosine_mean=float(np.mean(current_similarity)),
        top1_switch_rate=switch_rate,
        elapsed_seconds=elapsed,
    )


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


def transition_metrics(
    spec: MethodSpec,
    first: np.ndarray,
    second: np.ndarray,
    pre_frames: int,
    post_frames: int,
) -> dict[str, float | int | None]:
    first_part = first[-pre_frames:]
    second_part = second[:post_frames]
    synthetic = np.concatenate([first_part, second_part], axis=0)
    states = normalize(process(spec, synthetic))[pre_frames:]

    # The complete post-transition mean is an offline diagnostic anchor only. It
    # is never passed to the causal processor or used to generate submissions.
    old_anchor = normalize(first_part.mean(axis=0, keepdims=True))[0]
    new_anchor = normalize(second_part.mean(axis=0, keepdims=True))[0]
    similarity_new = states @ new_anchor
    similarity_old = states @ old_anchor
    final_new = float(np.mean(similarity_new[-min(25, post_frames) :]))
    initial_new = float(similarity_new[0])
    target50 = initial_new + 0.5 * (final_new - initial_new)
    target90 = initial_new + 0.9 * (final_new - initial_new)

    initial_old = float(similarity_old[0])
    final_old = float(np.mean(similarity_old[-min(25, post_frames) :]))
    old_drop = initial_old - similarity_old
    final_drop = initial_old - final_old
    forget50 = first_reach(old_drop, 0.5 * final_drop) if final_drop > 0 else None

    return {
        "old_new_anchor_cosine": float(old_anchor @ new_anchor),
        "new_anchor_similarity_at_switch": initial_new,
        "new_anchor_similarity_final": final_new,
        "adaptation_frames_50pct": first_reach(similarity_new, target50),
        "adaptation_frames_90pct": first_reach(similarity_new, target90),
        "old_anchor_similarity_at_switch": initial_old,
        "old_anchor_similarity_final": final_old,
        "forgetting_frames_50pct": forget50,
    }


def aggregate(records: list[dict[str, float | int | None]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in records[0]:
        values = [float(record[key]) for record in records if record[key] is not None]
        result[f"{key}_mean"] = float(np.mean(values)) if values else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("challenge_data"))
    parser.add_argument("--track", choices=("track1", "track2"), default="track2")
    parser.add_argument("--max-sequences", type=int, default=12)
    parser.add_argument("--transition-pairs", type=int, default=6)
    parser.add_argument("--pre-frames", type=int, default=200)
    parser.add_argument("--post-frames", type=int, default=400)
    parser.add_argument("--ema-alpha", type=float, action="append")
    parser.add_argument("--output", type=Path, default=Path("results/causal_method_diagnostics.json"))
    args = parser.parse_args()

    if args.max_sequences < 2 or args.transition_pairs < 1:
        raise ValueError("Need at least two sequences and one transition pair")
    alphas = tuple(args.ema_alpha) if args.ema_alpha else DEFAULT_EMA_ALPHAS
    for alpha in alphas:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")

    track_root = args.data_root / args.track
    probe_paths = sorted((track_root / "probe_seq_embeddings").glob("*.npz"))[: args.max_sequences]
    if len(probe_paths) < 2:
        raise ValueError(f"Need at least two probe sequences under {track_root}")
    sequences = [load_features(path) for path in probe_paths]
    if any(sequence.shape[0] < max(args.pre_frames, args.post_frames) for sequence in sequences):
        raise ValueError("At least one sequence is shorter than the requested transition segments")
    gallery = normalize(load_features(track_root / "gallery.npz"))

    official = your_algorithm(sequences[0], method=1)
    shared = process_embeddings(sequences[0], "cumulative_mean")
    baseline_max_abs_error = float(np.max(np.abs(official - shared)))
    baseline_exact = bool(np.array_equal(official, shared))
    if not baseline_exact:
        raise ValueError(f"Shared cumulative mean differs from official baseline: {baseline_max_abs_error}")

    pair_count = min(args.transition_pairs, math.comb(len(sequences), 2))
    pairs = choose_transition_pairs(sequences, pair_count, min(100, args.pre_frames))
    report: dict[str, object] = {
        "protocol": {
            "label_independent": True,
            "offline_future_anchor_used_only_for_diagnostics": True,
            "track": args.track,
            "sequence_count": len(sequences),
            "transition_pair_count": len(pairs),
            "pre_frames": args.pre_frames,
            "post_frames": args.post_frames,
            "ema_alphas_predeclared": list(alphas),
        },
        "baseline_equivalence": {
            "array_equal": baseline_exact,
            "max_abs_error": baseline_max_abs_error,
        },
        "methods": {},
    }

    for spec in method_specs(alphas):
        prefix_errors = [prefix_invariance_error(spec, sequence) for sequence in sequences]
        if max(prefix_errors) != 0.0:
            raise ValueError(f"Prefix invariance failed for {spec.name}: {max(prefix_errors)}")
        per_sequence = [asdict(sequence_metrics(spec, sequence, gallery)) for sequence in sequences]
        transitions = [
            transition_metrics(spec, sequences[left], sequences[right], args.pre_frames, args.post_frames)
            for left, right in pairs
        ]
        report["methods"][spec.name] = {
            "configuration": asdict(spec),
            "causal_prefix_max_abs_error": max(prefix_errors),
            "sequence_metrics": aggregate(per_sequence),
            "transition_metrics": aggregate(transitions),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Baseline exact equivalence: PASS (max_abs_error={baseline_max_abs_error:.3g})")
    print(f"Causal prefix invariance: PASS ({len(report['methods'])} methods)")
    print(f"Diagnostic report: {args.output}")
    for name, method_report in report["methods"].items():
        transition = method_report["transition_metrics"]
        sequence = method_report["sequence_metrics"]
        print(
            f"{name}: adapt50={transition['adaptation_frames_50pct_mean']} "
            f"adapt90={transition['adaptation_frames_90pct_mean']} "
            f"forget50={transition['forgetting_frames_50pct_mean']} "
            f"state_cos={sequence['consecutive_state_cosine_mean_mean']:.6f} "
            f"top1_switch_rate={sequence['top1_switch_rate_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
