"""T1-specialist bank5 candidate: bank4 plus the opening-type map on never-owned late rows (same on both tracks).

Evaluated in docs/experiments/20261009_final_stretch_t1_direction.md (`analyze_t1_never.py`, `analyze_t1_bank5d.py`,
`analyze_t1_bank5e.py`: T1 proxy .002503355468825 on the 971136 scores, +46.8 gen-eq against bank4, every one of 12
subsets positive).

  bank4 : t1_bank4.post4, unchanged (the HCP memory z restarted at the regime jump J in bank3's steps)
  late  : a frame t >= MID_UNTIL (70) that bank4 leaves owner-less and does not extend, in a sequence where no frame
          0 .. t-1 has been owned (own2) or extended, and whose mean frame norm over frames max(0, t-49) .. t is at
          least NORM_GATE (14), gets clip(TH0 + KAPPA (float32(zSr[t, j]) - TAU_L), -1, 1) in every column; zSr is HCP's
          short EMA memory z (x_fr_hcp._fe's zS) computed afresh from frame J (from frame 0 when there is no jump);
          KAPPA = t1_bank.KAPPA (.05), TAU_L = 1.5
The owner-less and extension tests read bank4's own masks (the extension's leader and run from the post-ownership
bank output), so this step never changes another frame. Before the first ownership the linear map on a memory z
ranks columns against the global threshold better than the base scorer's posterior map (bank3's mid finding, here
past t 70 with the short memory); the norm gate keeps low-quality stretches on the base scorer. All state is per
sequence and uses frames <= t only. `bank5` is the vectorised processor (`post5` on given base scores); `bank5_ref`
is an independent per-frame loop used as the builder's bit-identity reference.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "core"))
import numpy as np  # noqa: E402

import analyze_hvc_stage0 as HV  # noqa: E402
import t1_bank as B  # noqa: E402
import t1_bank2 as B2  # noqa: E402
import t1_bank3 as B3  # noqa: E402
import t1_bank4 as B4  # noqa: E402
import x_fr_hcp  # noqa: E402

NORM_GATE = 14.0
NORM_WINDOW = 50
TAU_L = 1.5


def late_map(z):
    """The opening-type map on short-memory z values (float32 in, float32 out)."""
    z = np.asarray(z, np.float32)
    return np.clip(B.TH0 + np.float32(B.KAPPA) * (z - np.float32(TAU_L)), -1.0, 1.0).astype(np.float32)


def recent_norm(x):
    """Mean frame norm over frames max(0, t - NORM_WINDOW + 1) .. t (float64)."""
    nrm = np.linalg.norm(np.asarray(x, np.float64), axis=1)
    cs = np.concatenate([[0.0], np.cumsum(nrm)])
    t = np.arange(nrm.shape[0])
    lo = np.maximum(t - (NORM_WINDOW - 1), 0)
    return (cs[t + 1] - cs[lo]) / (t + 1 - lo)


def post5(base, x, g):
    """bank5's steps on a (T x K) float32 base score matrix."""
    J = B4.jump_index(x)
    j0 = max(J, 0)
    fe = x_fr_hcp._fe(x[j0:], g)
    zP = x_fr_hcp._fe(x, g)["zP"]
    if J > 0:
        zP = zP.copy()
        zP[J:] = fe["zP"]
    s = np.asarray(base, np.float32).copy()
    cmz = B.early_cmz(x, g)
    e = cmz.shape[0]
    s[:e] = np.clip(B.TH0 + B.KAPPA * (cmz - B.TAU), -1.0, 1.0)
    own = B4.owners_r(x, g, zP)
    has = np.nonzero(own >= 0)[0]
    if has.size:
        keep = s[has, own[has]].copy()
        s[has] = -1.0
        s[has, own[has]] = keep
    a, run = B2.leader_runs(s)
    T = s.shape[0]
    t = np.arange(T)
    lead_ok = (zP[t, a] >= B2.ZLEAD_OWN) | (run >= B2.RUN_OWN)
    out = s.copy()
    ext = (own < 0) & (t >= B2.EXT_FROM) & lead_ok
    idx = np.nonzero(ext)[0]
    if idx.size:
        keep = out[idx, a[idx]].copy()
        out[idx] = -1.0
        out[idx, a[idx]] = keep
    idx = np.nonzero((own < 0) & (t >= B2.EXT_FROM) & (t < B3.MID_UNTIL) & ~lead_ok)[0]
    if idx.size:
        out[idx] = B3.mid_map(zP[idx].astype(np.float32))
    held = (own >= 0) | ext
    before = np.concatenate([[False], np.cumsum(held)[:-1] > 0])
    late = (own < 0) & ~ext & (t >= B3.MID_UNTIL) & ~before & (recent_norm(x) >= NORM_GATE)
    idx = np.nonzero(late)[0]
    if idx.size:
        out[idx] = late_map(fe["zS"][idx - j0])
    return out


def bank5(x, g):
    return post5(B._BASE(x, g), x, g)


def bank5_ref(x, g):
    """Per-frame loop reference: the jump detector, the opening override, own2, the extension, the mid step and the
    late step are applied frame by frame with independent bookkeeping (jump accumulator, clean count, leader run,
    the ever-held flag, a scalar sequential prefix of frame norms); the numeric primitives (base scores, unit frames, row norms,
    the block matrix product, the LOO z and the HCP memories of a sequence or of its suffix) are the same calls as in
    `bank5`, so the result must be bit-identical."""
    s = np.asarray(B._BASE(x, g), np.float32).copy()
    xb = np.asarray(x, np.float64)
    T, K = s.shape
    nrm = np.linalg.norm(xb, axis=1)
    zall = HV.verify(x, g)[1]
    zfull = x_fr_hcp._fe(x, g)["zP"]
    uj = B._unit(xb[:B4.JUMP_UNTIL])
    jacc = np.zeros(xb.shape[1])
    J = -1
    for t in range(1, min(T, B4.JUMP_UNTIL)):
        if t <= B.EARLY:
            jacc = jacc + uj[t - 1]
        nj = float(np.linalg.norm(jacc))
        if nj > 0 and float(np.dot(uj[t], jacc)) / nj < B4.JUMP_CUT:
            J = t
            break
    fe = x_fr_hcp._fe(x[max(J, 0):], g)
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
    ever = False
    pre = [0.0]
    for t in range(T):
        pre.append(pre[-1] + float(nrm[t]))
    for t in range(T):
        zt = fe["zP"][t - J] if J >= 0 and t >= J else zfull[t]
        if t < e:
            s[t] = np.clip(B.TH0 + B.KAPPA * (early[t] - B.TAU), -1.0, 1.0)
        if nrm[t] >= B.CLEAN_NORM and nrm[t] > 0:
            cnt = cnt + (zall[t] >= HV.Z0)
        order = np.argsort(-cnt, kind="stable")
        owner = -1
        if cnt[order[0]] >= 1 and cnt[order[0]] > cnt[order[1]]:
            owner = int(order[0])
        else:
            zo = np.argsort(-zt, kind="stable")
            if zt[zo[0]] >= B.ZP_OWN and zt[zo[0]] > zt[zo[1]] and zt[zo[1]] <= B.ZP_RUNNER:
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
        if owner < 0 and t >= B2.EXT_FROM and (zt[lead] >= B2.ZLEAD_OWN or run >= B2.RUN_OWN):
            keep = s[t, lead]
            s[t] = -1.0
            s[t, lead] = keep
            extended = True
        if owner < 0 and not extended and B2.EXT_FROM <= t < B3.MID_UNTIL:
            for j in range(K):
                s[t, j] = np.clip(B.TH0 + np.float32(B3.KAPPA_M) * (np.float32(zt[j]) - np.float32(B3.TAU_M)),
                                  -1.0, 1.0)
        if owner < 0 and not extended and t >= B3.MID_UNTIL and not ever:
            lo = max(0, t - (NORM_WINDOW - 1))
            if (pre[t + 1] - pre[lo]) / (t + 1 - lo) >= NORM_GATE:
                zs = fe["zS"][t - max(J, 0)]
                for j in range(K):
                    s[t, j] = np.clip(B.TH0 + np.float32(B.KAPPA) * (np.float32(zs[j]) - np.float32(TAU_L)), -1.0, 1.0)
        ever = ever or owner >= 0 or extended
    return s


def outputs_fn(x, g):
    return {"bank5": bank5_ref(x, g)}
