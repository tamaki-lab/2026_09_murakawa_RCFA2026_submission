#!/usr/bin/env python3
"""Validate the nested RCFA submission archive against released probe data."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np


TRACKS = ("track1", "track2")


def safe_flat_names(names: list[str], archive_label: str) -> None:
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
            raise ValueError(f"{archive_label}: unsafe or nested member {name!r}")


def validate_track(
    track: str, inner_bytes: bytes, data_root: Path
) -> tuple[int, float, float, set[str]]:
    track_root = data_root / track
    probe_paths = sorted((track_root / "probe_seq_embeddings").glob("*.npz"))
    with np.load(track_root / "gallery.npz", allow_pickle=False) as archive:
        gallery_count = archive["feature"].shape[0]

    expected = {f"score_{path.name}" for path in probe_paths}
    observed_dtypes: set[str] = set()
    global_min = float("inf")
    global_max = float("-inf")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        members = [info.filename for info in inner.infolist() if not info.is_dir()]
        safe_flat_names(members, f"{track}.zip")
        if len(members) != len(set(members)):
            raise ValueError(f"{track}.zip: duplicate member names")
        if set(members) != expected:
            missing = sorted(expected - set(members))
            extra = sorted(set(members) - expected)
            raise ValueError(
                f"{track}.zip: sequence mismatch; missing={missing[:5]} extra={extra[:5]}"
            )

        probes_by_output = {f"score_{path.name}": path for path in probe_paths}
        for member in members:
            with np.load(io.BytesIO(inner.read(member)), allow_pickle=False) as archive:
                if archive.files != ["score"]:
                    raise ValueError(f"{track}.zip/{member}: keys={archive.files}, expected ['score']")
                score = archive["score"]
            with np.load(probes_by_output[member], allow_pickle=False) as probe_archive:
                frame_count = probe_archive["feature"].shape[0]
            expected_shape = (frame_count, gallery_count)
            if score.shape != expected_shape:
                raise ValueError(
                    f"{track}.zip/{member}: shape={score.shape}, expected={expected_shape}"
                )
            if not np.issubdtype(score.dtype, np.floating):
                raise ValueError(f"{track}.zip/{member}: non-float dtype={score.dtype}")
            if not np.isfinite(score).all():
                raise ValueError(f"{track}.zip/{member}: contains NaN/Inf")
            score_min = float(score.min())
            score_max = float(score.max())
            if score_min < -1.0 or score_max > 1.0:
                raise ValueError(
                    f"{track}.zip/{member}: range [{score_min}, {score_max}] outside [-1, 1]"
                )
            observed_dtypes.add(str(score.dtype))
            global_min = min(global_min, score_min)
            global_max = max(global_max, score_max)

    return len(members), global_min, global_max, observed_dtypes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path, nargs="?", default=Path("results/submission.zip"))
    parser.add_argument("--data-root", type=Path, default=Path("challenge_data"))
    args = parser.parse_args()

    with zipfile.ZipFile(args.submission) as outer:
        members = [info.filename for info in outer.infolist() if not info.is_dir()]
        safe_flat_names(members, args.submission.name)
        expected_outer = {f"{track}.zip" for track in TRACKS}
        if len(members) != len(set(members)) or set(members) != expected_outer:
            raise ValueError(f"Outer archive members={members}, expected={sorted(expected_outer)}")
        inner_archives = {name: outer.read(name) for name in members}

    print(f"outer archive: {args.submission}")
    print(f"outer members: {sorted(inner_archives)}")
    for track in TRACKS:
        count, score_min, score_max, dtypes = validate_track(
            track, inner_archives[f"{track}.zip"], args.data_root
        )
        print(
            f"{track}: files={count} one_to_one=yes key=score dtypes={sorted(dtypes)} "
            f"finite=yes range=[{score_min:.8f}, {score_max:.8f}]"
        )
    print("Submission validation: PASS")


if __name__ == "__main__":
    main()
