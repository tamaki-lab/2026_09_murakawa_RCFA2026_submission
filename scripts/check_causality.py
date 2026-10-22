"""Prefix check of the submission 980314 scorer: removing future frames must not change any earlier score.

Usage: python scripts/check_causality.py DATA_DIR [--sequences N]

For the first N probe sequences of each track (sorted file order), the exact build function
(build_candidate.CONFIGS["bal_ete_b8d"], the one scripts/core/build_candidate_par.py uses) scores the whole sequence
once and then each prefix x[:c]; the rows 0..c-1 must be bit-identical to the full run. The cuts cover the opening
(t < 10), the jump search window (t < 60) and later parts of the sequence. Exit code 0 means every cut matched.
"""
import argparse
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "core"))
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import build_candidate as BC  # noqa: E402
import candidates  # noqa: E402
from scripts.build_candidate_submission import load_features  # noqa: E402

FIXED_CUTS = (1, 2, 5, 9, 10, 11, 15, 30, 59, 60, 61, 100, 300)
FRACTIONS = (0.25, 0.5, 0.75)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--sequences", type=int, default=2)
    args = ap.parse_args()
    fn = eval(BC.CONFIGS["bal_ete_b8d"], vars(candidates))
    ok = True
    for track in BC.TRACKS:
        g = load_features(args.data_dir / track / "gallery.npz")
        for p in sorted((args.data_dir / track / "probe_seq_embeddings").glob("*.npz"))[:args.sequences]:
            x = load_features(p)
            T = x.shape[0]
            t0 = time.time()
            full = fn(x, g)
            cuts = sorted({c for c in FIXED_CUTS if c < T} | {int(T * f) for f in FRACTIONS})
            bad = [c for c in cuts if not np.array_equal(fn(x[:c], g), full[:c])]
            ok &= not bad
            print(f"{track}/{p.name}: {T} frames, {len(cuts)} cuts {cuts}, "
                  f"{'all bit-identical' if not bad else 'DIFFER at ' + str(bad)} ({time.time() - t0:.0f} s)",
                  flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
