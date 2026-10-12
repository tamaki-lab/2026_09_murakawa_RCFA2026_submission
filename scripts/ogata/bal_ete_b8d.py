"""Balanced candidate ete_b8d: the teammate's or75v_b8d composite with 980179's high band (stage S ete) for H.

out = veto_band(H, L, b) (`analyze_t1_stage3_bank.veto_band`, unchanged): H where H >= .75, otherwise min(L, .5), except
L's vetoes (L == -1, H < .75), which become -1 + .025 (1 + b).
  H = ete (`analyze_t2_stage_s.variants_h(x, g)["ete"]`, float32): 980179's high band (official BE .08577983)
  L = bank8 + DBS10 (`t1_b8d.b8d_core`, the teammate's 979990 low band before P0; fixed 4096-row block)
  b = 943955 (gallery_discriminative_row_l2_scores), as in 976810 / 979937 / or75v_b8d
Every entry with H >= .75 is ete's and every other entry is <= .5, so the Track-1 FAR = 1e-5 operating threshold and
every Track-2 decision are 980179's; L moves the Track-1 EER (and FAR@FRR1e-5) only. One code path and the same
constants on both tracks; fresh state per sequence; every component reads frames <= t only.
`bal(x, g)` is the build path; `outputs_fn(x, g)` is the builder's reference (L = `t1_b8d.b8d_ref_core`, unpadded).
Plan and frozen rules: docs/experiments/20261012_ogata_bal_etx_b8d.md (ete addendum).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "final_stretch"))
import numpy as np  # noqa: E402

import analyze_t1_stage3_bank as S3  # noqa: E402
import analyze_t2_stage_s as SS  # noqa: E402
import t1_b8d as BD  # noqa: E402
from rcfa.methods.gallery_discriminative_row_l2 import gallery_discriminative_row_l2_scores  # noqa: E402


def high_and_b(x, g):
    H32 = SS.variants_h(x, g)["ete"].astype(np.float32)
    b32 = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    return H32, b32


def bal(x, g):
    H32, b32 = high_and_b(x, g)
    return S3.veto_band(H32, np.asarray(BD.b8d_core(x, g), np.float32), b32)


def outputs_fn(x, g):
    H32, b32 = high_and_b(x, g)
    return {"bal_ete_b8d": S3.veto_band(H32, np.asarray(BD.b8d_ref_core(x, g), np.float32), b32)}
