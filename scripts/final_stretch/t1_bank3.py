"""T1-specialist bank3 candidate: bank2 plus the opening map continued into its owner-less frames (same on both tracks).

Evaluated in docs/experiments/20261009_final_stretch_t1_direction.md (`analyze_t1_mid2.py`, `analyze_t1_mid3.py`:
MID_UNTIL 70, TAU_M 1, KAPPA_M .03 on the 971136 scores; analyze_t1_mid4.py: one shared kappa is worse).

  bank2 : t1_bank2.bank2, unchanged (971136's processor, the opening override, own2 ownership, the ownership extension)
  mid   : in a frame 10 <= t < MID_UNTIL that has no owner after bank2 (own2 leaves it owner-less and the extension does
          not own it), every column becomes clip(TH0 + KAPPA_M (zP - TAU_M), -1, 1), zP the HCP memory z cast to
          float32: the opening map of t1_bank with the memory z in place of the cumulative-mean z.
The owner-less test reads the bank output (before the extension) exactly as the extension does, so the step of one
frame never changes another frame. All state is per sequence and uses frames <= t only. `bank3` is the vectorised
processor; `bank3_ref` is an independent per-frame loop (own2, the extension and the mid step re-implemented frame by
frame) used as the builder's bit-identity reference.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ogata"))
import numpy as np  # noqa: E402

import analyze_hvc_stage0 as HV  # noqa: E402
import t1_bank as B  # noqa: E402
import t1_bank2 as B2  # noqa: E402
import x_fr_hcp  # noqa: E402

MID_UNTIL = 70
TAU_M = 1.0
KAPPA_M = 0.03


def mid_map(zp):
    """The opening-type map on memory z values (float32 in, float32 out)."""
    zp = np.asarray(zp, np.float32)
    return np.clip(B.TH0 + np.float32(KAPPA_M) * (zp - np.float32(TAU_M)), -1.0, 1.0).astype(np.float32)


def mid(s2, b, x, g):
    """Apply the mid step to bank2's output s2 (T x K float32), b being the bank output it was extended from."""
    s = np.asarray(s2, np.float32).copy()
    T = s.shape[0]
    own = B.owners(x, g)
    zP = x_fr_hcp._fe(x, g)["zP"]
    a, run = B2.leader_runs(b)
    t = np.arange(T)
    ext = (zP[t, a] >= B2.ZLEAD_OWN) | (run >= B2.RUN_OWN)
    free = (own < 0) & (t >= B2.EXT_FROM) & (t < MID_UNTIL) & ~ext
    idx = np.nonzero(free)[0]
    if idx.size:
        s[idx] = mid_map(zP[idx].astype(np.float32))
    return s


def bank3(x, g):
    b = B.bank(x, g)
    return mid(B2.extend(b, x, g), b, x, g)


def bank3_ref(x, g):
    """Per-frame loop reference: the opening override, own2, the extension and the mid step are applied frame by frame
    with independent bookkeeping; the numeric primitives (base scores, row norms, the block matrix product, the LOO z
    and zP) are the same calls as in `bank3`, so the result must be bit-identical."""
    s = np.asarray(B._BASE(x, g), np.float32).copy()
    xb = np.asarray(x, np.float64)
    T, K = s.shape
    nrm = np.linalg.norm(xb, axis=1)
    zall = HV.verify(x, g)[1]
    zP = x_fr_hcp._fe(x, g)["zP"]
    e = min(B.EARLY, T)
    u = B._unit(xb[:e])
    C = np.zeros_like(u)
    acc = np.zeros(xb.shape[1])
    for t in range(e):
        acc = acc + u[t]
        C[t] = acc
    early = B._loo_z(B._unit(C) @ B._unit(g).T).astype(np.float32)
    cnt = np.zeros(K, np.int64)
    prev, run = -1, 0
    for t in range(T):
        if t < e:
            s[t] = np.clip(B.TH0 + B.KAPPA * (early[t] - B.TAU), -1.0, 1.0)
        if nrm[t] >= B.CLEAN_NORM and nrm[t] > 0:
            cnt = cnt + (zall[t] >= HV.Z0)
        order = np.argsort(-cnt, kind="stable")
        owner = -1
        if cnt[order[0]] >= 1 and cnt[order[0]] > cnt[order[1]]:
            owner = int(order[0])
        else:
            zo = np.argsort(-zP[t], kind="stable")
            if zP[t, zo[0]] >= B.ZP_OWN and zP[t, zo[0]] > zP[t, zo[1]] and zP[t, zo[1]] <= B.ZP_RUNNER:
                owner = int(zo[0])
        if owner >= 0:
            keep = s[t, owner]
            s[t] = -1.0
            s[t, owner] = keep
        row = s[t].astype(np.float64)
        lead = int(np.flatnonzero(row == row.max())[0])
        run = run + 1 if lead == prev else 1
        prev = lead
        extended = False
        if owner < 0 and t >= B2.EXT_FROM and (zP[t, lead] >= B2.ZLEAD_OWN or run >= B2.RUN_OWN):
            keep = s[t, lead]
            s[t] = -1.0
            s[t, lead] = keep
            extended = True
        if owner < 0 and not extended and B2.EXT_FROM <= t < MID_UNTIL:
            for j in range(K):
                s[t, j] = np.clip(B.TH0 + np.float32(KAPPA_M) * (np.float32(zP[t, j]) - np.float32(TAU_M)), -1.0, 1.0)
    return s


def outputs_fn(x, g):
    return {"bank3": bank3_ref(x, g)}
