"""T1-specialist candidate b8v: bank8 with the teammate's P0 veto ordering (no high band).

out = L except where L == -1 (bank8's vetoes), which become -1 + .025 (1 + b) (inside [-1, -.95], ordered by b);
L = bank8 on the fixed 4096-row block (`t1_bank8p.bank8p`), b = 943955 (gallery-discriminative row-L2). The constants
are those of the teammate's stage-3 P0 (`analyze_t1_stage3_bank.VETO_LO / VETO_W`, 4e2a29f), as in 976810. Every entry
bank8 does not veto is unchanged, and every changed entry stays at or below -.95, far below the Track-1 EER threshold
(~ -.0998) and the FAR = 1e-5 threshold (~ .43), so the EER, FRR@FAR1e-5 and the sequence FRRs are bank8's; only the
order of the bottom of the genuine distribution (FAR@FRR1e-5, 1.0 for bank8) changes. One code path and the same
constants on both tracks; fresh state per sequence; every component reads frames <= t only.
`b8v(x, g)` is the build path; `outputs_fn(x, g)` is the builder's reference with L from `t1_bank8.bank8_ref`.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ogata"))
sys.path.insert(0, str(HERE.parents[1]))
import numpy as np  # noqa: E402

import analyze_t1_stage3_bank as S3  # noqa: E402
import t1_bank8 as B8  # noqa: E402
import t1_bank8p as B8P  # noqa: E402
from rcfa.methods.gallery_discriminative_row_l2 import gallery_discriminative_row_l2_scores  # noqa: E402


def b_scores(x, g):
    return np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)


def compose(L32, b32):
    """bank8's vetoes ordered by b in [-1, -.95] (float64 arithmetic, float32 output); other entries unchanged."""
    L32 = np.asarray(L32, np.float32)
    v = L32.astype(np.float64)
    m = L32 == np.float32(-1.0)
    v[m] = S3.VETO_LO + S3.VETO_W * (1.0 + np.asarray(b32, np.float64)[m])
    return v.astype(np.float32)


def b8v(x, g):
    return compose(B8P.bank8p(x, g), b_scores(x, g))


def outputs_fn(x, g):
    return {"b8v": compose(B8.bank8_ref(x, g), b_scores(x, g))}
