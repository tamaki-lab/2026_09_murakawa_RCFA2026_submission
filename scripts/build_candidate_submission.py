#!/usr/bin/env python3
"""Build one deterministic, validated-input RCFA candidate artifact."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from example import normalize  # noqa: E402
from rcfa.embedding import METHODS as EMBEDDING_METHODS, process_embeddings  # noqa: E402
from rcfa.methods.adaptive_score_memory import process_adaptive  # noqa: E402
from rcfa.methods.bounded_output import process_bounded, raw_scores  # noqa: E402
from rcfa.methods.predictive_output import process_predictive  # noqa: E402
from rcfa.methods.predictive_state_feedback import process_predictive_state_feedback  # noqa: E402
from rcfa.methods.persistence_output import process_persistence  # noqa: E402
from rcfa.methods.causal_runlength import process_runlength  # noqa: E402
from rcfa.methods.runlength_continue_output import process_continue_output  # noqa: E402
from rcfa.methods.gallery_common_deflation import process_gallery_common_deflation  # noqa: E402
from rcfa.methods.gallery_symmetric_orthogonalization import (  # noqa: E402
    process_gallery_symmetric_orthogonalization,
)
from rcfa.methods.gallery_row_space_projection import (  # noqa: E402
    process_gallery_row_space_projection,
)
from rcfa.methods.gallery_cumulative_projection import process_gallery_cumulative_projection  # noqa: E402
from rcfa.methods.gallery_cumulative_norm_p2 import process_gallery_cumulative_norm_p2  # noqa: E402
from rcfa.methods.pairwise_evidence import pairwise_evidence_scores  # noqa: E402
from rcfa.methods.gallery_cohort_centering import cumulative_cohort_centered_scores  # noqa: E402
from rcfa.methods.gallery_distribution_shape import distribution_shape_scores  # noqa: E402
from rcfa.methods.gallery_background_centering import cumulative_gallery_background_scores  # noqa: E402
from rcfa.methods.gallery_cubic_kernel import cubic_kernel_scores  # noqa: E402
from rcfa.methods.gallery_degree1_kernel import degree1_kernel_scores  # noqa: E402
from rcfa.methods.gallery_post_aggregation_cubic import post_aggregation_cubic_scores  # noqa: E402
from rcfa.methods.gallery_discriminative_readout import gallery_discriminative_scores  # noqa: E402
from rcfa.methods.gallery_discriminative_row_l2 import (  # noqa: E402
    gallery_discriminative_row_l2_scores,
)
from rcfa.methods.gallery_dual_ema_cubic_discriminative import (  # noqa: E402
    dual_ema_cubic_discriminative_scores,
)
from rcfa.methods.gallery_frame_cubic_discriminative import (  # noqa: E402
    frame_cubic_discriminative_row_l2_scores,
)
from rcfa.methods.gallery_continuity_reset import gallery_continuity_reset_scores  # noqa: E402
from rcfa.methods.gallery_frame_prefix_rms import frame_prefix_rms_scores  # noqa: E402
from rcfa.methods.gallery_frame_two_sided_rms import frame_two_sided_rms_scores  # noqa: E402
from rcfa.methods.gallery_frame_max_prefix_rms import frame_max_prefix_rms_scores  # noqa: E402


TRACKS = ("track1", "track2")
METHODS = (
    *EMBEDDING_METHODS,
    "adaptive_score_memory",
    "bounded_relative_output",
    "predictive_output",
    "predictive_state_feedback",
    "persistence_output",
    "causal_runlength",
    "runlength_continue_output",
    "gallery_common_deflation",
    "gallery_symmetric_orthogonalization",
    "gallery_row_space_projection",
    "gallery_cumulative_projection",
    "pairwise_evidence",
    "gallery_cumulative_norm_p2",
    "gallery_cohort_centering",
    "gallery_distribution_shape",
    "gallery_background_centering",
    "gallery_cubic_kernel",
    "gallery_degree1_kernel",
    "gallery_post_aggregation_cubic",
    "gallery_discriminative_readout",
    "gallery_discriminative_row_l2",
    "gallery_dual_ema_cubic_discriminative",
    "gallery_frame_cubic_discriminative_row_l2",
    "gallery_continuity_reset",
    "gallery_frame_prefix_rms",
    "gallery_frame_two_sided_rms",
    "gallery_frame_max_prefix_rms",
)
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def fixed_zip_info(name: str, *, compressed: bool) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    info.external_attr = 0o600 << 16
    return info


def serialize_score(score: np.ndarray) -> bytes:
    """Create a deterministic NPZ containing exactly one `score` array."""
    npy_buffer = io.BytesIO()
    np.lib.format.write_array(npy_buffer, score, allow_pickle=False)
    npz_buffer = io.BytesIO()
    with zipfile.ZipFile(npz_buffer, "w") as archive:
        archive.writestr(
            fixed_zip_info("score.npy", compressed=True),
            npy_buffer.getvalue(),
            compresslevel=6,
        )
    return npz_buffer.getvalue()


def load_features(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["feature"]:
            raise ValueError(f"{path}: expected only key 'feature', got {archive.files}")
        features = archive["feature"].astype(np.float32)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError(f"{path}: expected a finite 2-D floating feature array")
    return features


def build_track_archive(
    data_root: Path,
    temporary_root: Path,
    track: str,
    method: str,
    ema_alpha: float | None,
) -> tuple[Path, int, float, float]:
    track_root = data_root / track
    gallery_features = load_features(track_root / "gallery.npz")
    gallery = normalize(gallery_features)
    probe_paths = sorted((track_root / "probe_seq_embeddings").glob("*.npz"))
    if not probe_paths:
        raise ValueError(f"No probe sequences under {track_root}")

    track_zip = temporary_root / f"{track}.zip"
    global_min = float("inf")
    global_max = float("-inf")
    with zipfile.ZipFile(track_zip, "w") as archive:
        for probe_path in probe_paths:
            probe = load_features(probe_path)
            # Track selects data paths only; every sequence uses the same method,
            # constants and fresh initialization. No track label reaches a processor.
            if method == "bounded_relative_output":
                embeddings, _, _, _ = process_bounded(probe, gallery)
                score = raw_scores(embeddings, gallery)
            elif method == "pairwise_evidence":
                # Direct causal matching scores: no embedding normalization or
                # second gallery multiplication after column-specific fusion.
                score = pairwise_evidence_scores(probe, gallery_features)
            elif method == "gallery_cohort_centering":
                # Direct scores after the frozen parent's float32 rounding;
                # do not normalize again or multiply by gallery a second time.
                score = cumulative_cohort_centered_scores(probe, gallery_features)
            elif method == "gallery_background_centering":
                # Frozen parent score plus fixed gallery background offsets.
                # Direct scores: no extra normalization or gallery multiplication.
                score = cumulative_gallery_background_scores(probe, gallery_features)
            elif method == "gallery_cubic_kernel":
                # Fixed degree3 direct scores, including cohort centering.
                score = cubic_kernel_scores(probe, gallery_features)
            elif method == "gallery_degree1_kernel":
                # Precision/readout-matched degree1 control for the cubic path.
                # Direct scores: no extra normalization or gallery product.
                score = degree1_kernel_scores(probe, gallery_features)
            elif method == "gallery_post_aggregation_cubic":
                # Fixed degree1 history followed by cube-before-centering.
                # Direct scores: no extra normalization or gallery product.
                score = post_aggregation_cubic_scores(probe, gallery_features)
            elif method == "gallery_discriminative_readout":
                # Fixed cubic prefix plus gallery-only convex-hull margin map.
                # Direct scores: no extra normalization or gallery product.
                score = gallery_discriminative_scores(probe, gallery_features)
            elif method == "gallery_discriminative_row_l2":
                # Normalize only the final float32 parent score; no extra
                # cohort centering, embedding normalization or gallery product.
                score = gallery_discriminative_row_l2_scores(probe, gallery_features)
            elif method == "gallery_dual_ema_cubic_discriminative":
                # Fixed dual-EMA cubic state; unchanged 941755 margin readout.
                # Direct scores: no extra normalization or gallery product.
                score = dual_ema_cubic_discriminative_scores(probe, gallery_features)
            elif method == "gallery_frame_cubic_discriminative_row_l2":
                # The frozen singleton path, independently for every frame.
                # Already includes cohort centering and row L2. No postprocess.
                score = frame_cubic_discriminative_row_l2_scores(probe, gallery_features)
            elif method == "gallery_frame_prefix_rms":
                # Singleton centered score plus causal prefix RMS, fresh per
                # sequence. Direct scores: no extra normalization or product.
                score = frame_prefix_rms_scores(probe, gallery_features)
            elif method == "gallery_frame_two_sided_rms":
                # Same singleton centered input; fixed two-sided RMS readout.
                # Fresh full-sequence state and no postprocessing of the scores.
                score = frame_two_sided_rms_scores(probe, gallery_features)
                if score.dtype != np.float32:
                    raise ValueError("Two-sided RMS must return float32 scores")
            elif method == "gallery_frame_max_prefix_rms":
                # Frozen singleton centered input plus historical peak prefix RMS.
                # Fresh per sequence; save direct scores without postprocessing.
                score = frame_max_prefix_rms_scores(probe, gallery_features)
                if score.dtype != np.float32:
                    raise ValueError("Max-prefix RMS must return float32 scores")
            elif method == "gallery_distribution_shape":
                # Fixed metric matching scores; preserve raw gallery for SVD.
                # No subsequent embedding normalization or gallery product.
                score = distribution_shape_scores(probe, gallery_features)
            elif method == "gallery_continuity_reset":
                # Fixed detector/reset state and parent's complete row-L2 readout.
                # Raw gallery, fresh sequence, no second normalization/product.
                score = gallery_continuity_reset_scores(probe, gallery_features)
            else:
                if method == "runlength_continue_output":
                    embeddings, _ = process_continue_output(probe)
                elif method == "causal_runlength":
                    embeddings, _ = process_runlength(probe)
                elif method == "persistence_output":
                    embeddings, _ = process_persistence(probe, gallery)
                elif method == "predictive_output":
                    embeddings, _ = process_predictive(probe, gallery)
                elif method == "predictive_state_feedback":
                    embeddings, _ = process_predictive_state_feedback(probe, gallery)
                elif method == "adaptive_score_memory":
                    embeddings, _ = process_adaptive(probe, gallery)
                elif method == "gallery_common_deflation":
                    embeddings, _ = process_gallery_common_deflation(probe, gallery)
                elif method == "gallery_symmetric_orthogonalization":
                    embeddings, _ = process_gallery_symmetric_orthogonalization(
                        probe, gallery
                    )
                elif method == "gallery_row_space_projection":
                    embeddings, _ = process_gallery_row_space_projection(
                        probe, gallery
                    )
                elif method == "gallery_cumulative_projection":
                    # Match the frozen audit's raw-gallery SVD input exactly;
                    # float32 gallery normalization is still used for scoring.
                    embeddings, _ = process_gallery_cumulative_projection(
                        probe, gallery_features
                    )
                elif method == "gallery_cumulative_norm_p2":
                    embeddings, _ = process_gallery_cumulative_norm_p2(
                        probe, gallery_features
                    )
                else:
                    embeddings = process_embeddings(probe, method, ema_alpha=ema_alpha)
                processed = normalize(embeddings)
                score = (processed @ gallery.T).astype(np.float32)
            if score.shape != (probe.shape[0], gallery.shape[0]):
                raise ValueError(f"{track}/{probe_path.name}: invalid score shape {score.shape}")
            if not np.isfinite(score).all():
                raise ValueError(f"{track}/{probe_path.name}: score contains NaN or Inf")
            score_min = float(score.min())
            score_max = float(score.max())
            if score_min < -1.0 or score_max > 1.0:
                raise ValueError(
                    f"{track}/{probe_path.name}: score range [{score_min}, {score_max}]"
                )
            global_min = min(global_min, score_min)
            global_max = max(global_max, score_max)
            archive.writestr(
                fixed_zip_info(f"score_{probe_path.name}", compressed=False),
                serialize_score(score),
            )
    os.chmod(track_zip, 0o600)
    return track_zip, len(probe_paths), global_min, global_max


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_submission(
    data_root: Path,
    output_root: Path,
    method: str,
    ema_alpha: float | None = None,
) -> tuple[Path, str, float]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
    if method == "ema" and ema_alpha is None:
        raise ValueError("method='ema' requires --ema-alpha")
    if method == "ema" and not (np.isfinite(ema_alpha) and 0.0 < ema_alpha <= 1.0):
        raise ValueError("--ema-alpha must be finite and in (0, 1]")
    if method != "ema" and ema_alpha is not None:
        raise ValueError("--ema-alpha is only valid with --method ema")

    output_root.mkdir(parents=True, exist_ok=True)
    os.chmod(output_root, 0o700)
    artifact = output_root / "submission.zip"
    if artifact.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {artifact}")

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".build-", dir=output_root) as temp_name:
        temporary_root = Path(temp_name)
        os.chmod(temporary_root, 0o700)
        track_archives = []
        for track in TRACKS:
            track_zip, count, score_min, score_max = build_track_archive(
                data_root, temporary_root, track, method, ema_alpha
            )
            track_archives.append(track_zip)
            print(
                f"{track}: files={count} finite=yes "
                f"range=[{score_min:.8f}, {score_max:.8f}]"
            )

        temporary_artifact = temporary_root / "submission.zip"
        with zipfile.ZipFile(temporary_artifact, "w") as outer:
            for track_zip in track_archives:
                with track_zip.open("rb") as source, outer.open(
                    fixed_zip_info(track_zip.name, compressed=False), "w", force_zip64=True
                ) as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        os.replace(temporary_artifact, artifact)

    os.chmod(artifact, 0o600)
    elapsed = time.perf_counter() - started
    digest = sha256_file(artifact)
    return artifact, digest, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One shared causal method and parameter set for both tracks.",
        allow_abbrev=False,
    )
    parser.add_argument("--data-root", type=Path, default=Path("challenge_data"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--ema-alpha", type=float)
    args = parser.parse_args()

    artifact, digest, elapsed = build_submission(
        args.data_root, args.output_root, args.method, args.ema_alpha
    )
    print(f"artifact: {artifact}")
    print(f"size_bytes: {artifact.stat().st_size}")
    print(f"sha256: {digest}")
    print(f"elapsed_seconds: {elapsed:.3f}")


if __name__ == "__main__":
    main()
