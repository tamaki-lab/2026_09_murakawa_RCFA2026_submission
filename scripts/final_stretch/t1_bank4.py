"""T1-specialist bank4 candidate: bank3 with the HCP memory restarted at the regime jump (same on both tracks).

Evaluated in docs/experiments/20261009_final_stretch_t1_direction.md (`analyze_t1_jump.py`, `analyze_t1_restart.py`
variant V2: T1 proxy .002551194636869801 on the 971136 scores, +74.5 gen-eq against bank3, every subset positive;
`analyze_t1_restart2.py`: the jump cut, the detector's reference and bank3's other constants are at their optimum).

  jump : J = the first frame t, 1 <= t < JUMP_UNTIL, whose unit frame has cosine < JUMP_CUT to the sum of the unit
         frames 0 .. min(t, EARLY) - 1 (decided at frame J from frames <= J); J = -1 when there is none (no restart).
         This sum-based test is the specification; V2's mean-based statement gives the same J on all 400 released
         sequences (closest examined cosine .00049 from the cut; check_t1_bank4_v2_equiv.py)
  zPr  : the HCP memory z of x_fr_hcp; rows t < J from the whole sequence, rows t >= J computed afresh from frame J on,
         as if the sequence started at frame J
  bank3: t1_bank3's steps — the opening override, own2 ownership, the ownership extension and the mid step — with zPr
         wherever bank3 reads the HCP memory z (own2's zP rule, the extension's leader test, the mid map). The base
         scores, own2's clean-frame count and the opening override are unchanged.
Rows t < J are bank3's rows. All state is per sequence and uses frames <= t only. `bank4` is the vectorised processor
(`post4` applies its steps to given base scores); `bank4_ref` is an independent per-frame loop (the jump detector, own2,
the extension and the mid step re-implemented frame by frame) used as the builder's bit-identity reference.
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
import t1_bank3 as B3  # noqa: E402
import x_fr_hcp  # noqa: E402

JUMP_CUT = 0.5
JUMP_UNTIL = 60


def jump_index(x):
    """The regime jump J (int, -1 when none)."""
    u = B._unit(np.asarray(x, np.float64)[:JUMP_UNTIL])
    T = u.shape[0]
    S = np.cumsum(u[:min(T, B.EARLY)], axis=0)
    for t in range(1, T):
        s = S[min(t, B.EARLY) - 1]
        ns = float(np.linalg.norm(s))
        if ns > 0 and float(np.dot(u[t], s)) / ns < JUMP_CUT:
            return t
    return -1


def zp_restart(x, g, J):
    """HCP memory z (float64, T x K) with rows >= J computed afresh from frame J."""
    zP = x_fr_hcp._fe(x, g)["zP"]
    if J > 0:
        zP = zP.copy()
        zP[J:] = x_fr_hcp._fe(x[J:], g)["zP"]
    return zP


def owners_r(x, g, zP):
    """own2 per frame (int64, -1 = no owner) with the given memory z in the zP rule."""
    xb = np.asarray(x, np.float64)
    nrm = np.linalg.norm(xb, axis=1)
    v, _, _ = HV.verify(x, g)
    cnt = np.cumsum(v & (nrm >= B.CLEAN_NORM)[:, None], axis=0)
    arg, top, second = B._top2(cnt)
    own = np.where((top >= 1) & (top > second), arg, -1)
    a2, t2, s2 = B._top2(zP)
    alt = np.where((t2 >= B.ZP_OWN) & (t2 > s2) & (s2 <= B.ZP_RUNNER), a2, -1)
    return np.where(own >= 0, own, alt).astype(np.int64)


def post4(base, x, g):
    """bank4's steps on a (T x K) float32 base score matrix."""
    zP = zp_restart(x, g, jump_index(x))
    s = np.asarray(base, np.float32).copy()
    cmz = B.early_cmz(x, g)
    e = cmz.shape[0]
    s[:e] = np.clip(B.TH0 + B.KAPPA * (cmz - B.TAU), -1.0, 1.0)
    own = owners_r(x, g, zP)
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
    idx = np.nonzero((own < 0) & (t >= B2.EXT_FROM) & lead_ok)[0]
    if idx.size:
        keep = out[idx, a[idx]].copy()
        out[idx] = -1.0
        out[idx, a[idx]] = keep
    idx = np.nonzero((own < 0) & (t >= B2.EXT_FROM) & (t < B3.MID_UNTIL) & ~lead_ok)[0]
    if idx.size:
        out[idx] = B3.mid_map(zP[idx].astype(np.float32))
    return out


def bank4(x, g):
    return post4(B._BASE(x, g), x, g)


def bank4_ref(x, g):
    """Per-frame loop reference: the jump detector, the opening override, own2, the extension and the mid step are
    applied frame by frame with independent bookkeeping; the numeric primitives (base scores, unit frames, row norms,
    the block matrix product, the LOO z and the HCP memory z of a sequence or of its suffix) are the same calls as in
    `bank4`, so the result must be bit-identical."""
    s = np.asarray(B._BASE(x, g), np.float32).copy()
    xb = np.asarray(x, np.float64)
    T, K = s.shape
    nrm = np.linalg.norm(xb, axis=1)
    zall = HV.verify(x, g)[1]
    zfull = x_fr_hcp._fe(x, g)["zP"]
    uj = B._unit(xb[:JUMP_UNTIL])
    jacc = np.zeros(xb.shape[1])
    J = -1
    zsuf = None
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
        if J < 0 and 1 <= t < min(T, JUMP_UNTIL):
            nj = float(np.linalg.norm(jacc))
            if nj > 0 and float(np.dot(uj[t], jacc)) / nj < JUMP_CUT:
                J = t
                zsuf = x_fr_hcp._fe(x[J:], g)["zP"]
        if t < min(T, JUMP_UNTIL, B.EARLY):
            jacc = jacc + uj[t]
        zt = zsuf[t - J] if J >= 0 else zfull[t]
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
    return s


def outputs_fn(x, g):
    return {"bank4": bank4_ref(x, g)}
