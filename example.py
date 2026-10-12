import os
import shutil
from pathlib import Path

import numpy as np


def your_algorithm(input_features, method=1):
    """Return one processed 512-D embedding per input frame."""
    if method == 0:
        return input_features
    if method == 1:
        n = input_features.shape[0]
        return np.cumsum(input_features, axis=0) / np.arange(1, n + 1)[:, None]
    raise NotImplementedError("Implement your method here")


def normalize(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def build_track(data_root, results_root, track):
    gallery = np.load(data_root / track / "gallery.npz", allow_pickle=False)["feature"]
    gallery = normalize(gallery.astype(np.float32))
    output_dir = results_root / track
    output_dir.mkdir(parents=True, exist_ok=True)
    for probe_path in sorted((data_root / track / "probe_seq_embeddings").glob("*.npz")):
        probe = np.load(probe_path, allow_pickle=False)["feature"].astype(np.float32)
        processed = normalize(your_algorithm(probe, method=1))
        score = processed @ gallery.T
        np.savez_compressed(output_dir / f"score_{probe_path.name}", score=score.astype(np.float32))
    shutil.make_archive(str(results_root / track), "zip", output_dir)


if __name__ == "__main__":
    data_root = Path(os.environ.get("RCFA_DATA", "challenge_data"))
    results_root = Path(os.environ.get("RCFA_RESULTS", "results"))
    for track in ("track1", "track2"):
        build_track(data_root, results_root, track)
    package = results_root / "submission"
    package.mkdir(parents=True, exist_ok=True)
    shutil.copy2(results_root / "track1.zip", package / "track1.zip")
    shutil.copy2(results_root / "track2.zip", package / "track2.zip")
    shutil.make_archive(str(results_root / "submission"), "zip", package)
    print(f"Upload: {results_root / 'submission.zip'}")
