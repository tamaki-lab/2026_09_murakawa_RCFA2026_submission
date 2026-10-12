"""Parallel version of build_candidate.py: same configs, checks, ZIP layout and receipt, sequences scored in workers.

Usage: build_candidate_par.py CONFIG_NAME OUTPUT_DIR [workers<=6]
Each sequence is scored in one worker process (fresh state per call, as in the serial builder). The first three
files of each track get the causal prefix spot check and the bit-identity check against the research module, both
inside the worker that scored them. Members are written in sorted order, exactly as build_candidate.py writes them.
"""
import json
import multiprocessing as mp
import os
import sys
import time
import zipfile

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import build_candidate as BC  # noqa: E402
from scripts.build_candidate_submission import fixed_zip_info, load_features, serialize_score  # noqa: E402

W = {}


def _score(task):
    i, p = task
    fn, g, name = W["fn"], W["g"], W["name"]
    x = load_features(p)
    s = fn(x, g)
    if s.shape != (x.shape[0], g.shape[0]) or s.dtype != np.float32 or not np.isfinite(s).all():
        raise ValueError(f"{p.name}: bad score")
    if s.min() < -1 or s.max() > 1:
        raise ValueError(f"{p.name}: out of range")
    perr, ref_ok = 0.0, 0
    if i < 3:
        h = max(1, x.shape[0] // 2)
        perr = float(np.abs(fn(x[:h], g) - s[:h]).max())
        if name in BC.REFERENCE:
            mod, key = BC.REFERENCE[name]
            ref = __import__(mod).outputs_fn(x, g)[key]
            if not np.array_equal(ref, s):
                raise ValueError(f"{p.name}: differs from {mod}.{key}")
            ref_ok = 1
    return i, p.name, serialize_score(s), float(s.min()), float(s.max()), x.shape[0], perr, ref_ok


def main(name, out, workers=6):
    import candidates
    fn = eval(BC.CONFIGS[name], vars(candidates))
    data = Path(os.environ.get("RCFA_DATA", ROOT / "challenge_data"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    art = out / "submission.zip"
    if art.exists():
        raise FileExistsError(art)
    nw = max(1, min(int(workers), 6))
    t0 = time.time()
    receipt = {"config": name, "expr": BC.CONFIGS[name], "builder": "build_candidate_par", "workers": nw, "tracks": {}}
    inner = {}
    for track in BC.TRACKS:
        g = load_features(data / track / "gallery.npz")
        paths = sorted((data / track / "probe_seq_embeddings").glob("*.npz"))
        W.update(fn=fn, g=g, name=name)
        tz = out / f".{track}.zip.part"
        lo, hi, frames, prefix_err, ref_checked = 1.0, -1.0, 0, 0.0, 0
        with zipfile.ZipFile(tz, "w") as z, \
                ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("fork")) as ex:
            for i, pname, blob, smin, smax, T, perr, rok in ex.map(_score, list(enumerate(paths)), chunksize=1):
                lo, hi, frames = min(lo, smin), max(hi, smax), frames + T
                prefix_err, ref_checked = max(prefix_err, perr), ref_checked + rok
                z.writestr(fixed_zip_info(f"score_{pname}", compressed=False), blob)
        inner[track] = tz
        receipt["tracks"][track] = {"files": len(paths), "frames": frames, "min": lo, "max": hi,
                                    "prefix_max_abs_diff": prefix_err, "reference_identical_files": ref_checked}
        print(track, receipt["tracks"][track], flush=True)
    part = out / ".submission.zip.part"
    with zipfile.ZipFile(part, "w", allowZip64=False) as z:
        for track in BC.TRACKS:
            z.write(inner[track], f"{track}.zip", compress_type=zipfile.ZIP_STORED)
    os.link(part, art)
    part.unlink()
    for t in inner.values():
        t.unlink()
    os.chmod(art, 0o600)
    receipt.update(size=art.stat().st_size, sha256=BC.sha256(art), seconds=round(time.time() - t0, 1))
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
