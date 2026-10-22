"""T1-specialist bank6 candidate: bank5 plus a fallback for uninformative base rows (same on both tracks).

Evaluated in docs/experiments/20261009_final_stretch_t1_direction.md (`analyze_t1_lowq.py`, `analyze_t1_lowq2.py`,
`analyze_t1_lowq3.py`: T1 proxy +153.0 gen-eq against bank5; the rule fires in one Track 1 sequence and in no
Track 2 sequence).

  bank5 : t1_bank5.post5, unchanged
  flat  : a row that bank5 leaves on the base scorer (owner-less, not extended, t >= MID_UNTIL (70) and not in bank5's
          late step) in which more than half of the base scorer's columns are >= TH0 gets
          clip(TH0 + KAPPA (float32(cumJ[t, j]) - TAU_U), -1, 1) in every column; cumJ is the LOO z of the mean of the
          unit frames J .. t (0 .. t when there is no jump) against the unit gallery, KAPPA = t1_bank.KAPPA (.05),
          TAU_U = .25
A base row that puts most of the gallery at or above the operating level carries no ranking it can act on (in the
low-quality stretch where this happens it accepts ~93 % of the impostors); the cumulative mean since the jump still
ranks the target above the lower part of the gallery there, so the opening-type map with a low cut replaces it. The
flat test reads the base row itself, so the step never changes another frame or a row that any earlier step set. All
state is per sequence and uses frames <= t only. `bank6` is the vectorised processor (`post6` on given base scores);
`bank6_ref` is an independent per-frame loop used as the builder's bit-identity reference.
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
import t1_bank5 as B5  # noqa: E402
import x_fr_hcp  # noqa: E402

TAU_U = 0.25


def flat_map(z):
    """The opening-type map with the low cut (float32 in, float32 out)."""
    z = np.asarray(z, np.float32)
    return np.clip(B.TH0 + np.float32(B.KAPPA) * (z - np.float32(TAU_U)), -1.0, 1.0).astype(np.float32)


def cum_z(x, g, j0):
    """LOO z of the mean of unit frames j0 .. t for every t >= j0 (rows t - j0; one block product)."""
    u = B._unit(np.asarray(x, np.float64)[j0:])
    return B._loo_z(B._unit(np.cumsum(u, axis=0)) @ B._unit(g).T)


def is_flat(row):
    """More than half of the columns at or above TH0 (counted, no rounding)."""
    return 2 * int(np.count_nonzero(np.asarray(row, np.float32) >= B.TH0)) > row.shape[-1]


def post6(base, x, g):
    """bank6's steps on a (T x K) float32 base score matrix."""
    base = np.asarray(base, np.float32)
    J = B4.jump_index(x)
    j0 = max(J, 0)
    fe = x_fr_hcp._fe(x[j0:], g)
    zP = x_fr_hcp._fe(x, g)["zP"]
    if J > 0:
        zP = zP.copy()
        zP[J:] = fe["zP"]
    s = base.copy()
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
    late = (own < 0) & ~ext & (t >= B3.MID_UNTIL) & ~before & (B5.recent_norm(x) >= B5.NORM_GATE)
    idx = np.nonzero(late)[0]
    if idx.size:
        out[idx] = B5.late_map(fe["zS"][idx - j0])
    base_rows = (own < 0) & ~ext & (t >= B3.MID_UNTIL) & ~late
    flat = base_rows & (2 * np.count_nonzero(base >= B.TH0, axis=1) > base.shape[1])
    idx = np.nonzero(flat)[0]
    if idx.size:
        out[idx] = flat_map(cum_z(x, g, j0)[idx - j0])
    return out


def bank6(x, g):
    return post6(B._BASE(x, g), x, g)


def bank6_ref(x, g):
    """Per-frame loop reference: bank5's steps frame by frame (as in t1_bank5.bank5_ref) and then the flat test on the
    row the base scorer gave, with independent bookkeeping; the numeric primitives (base scores, unit frames, row
    norms, the block matrix products, the LOO z and the HCP memories of a sequence or of its suffix) are the same calls
    as in `bank6`, so the result must be bit-identical."""
    s = np.asarray(B._BASE(x, g), np.float32).copy()
    b0 = s.copy()
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
    j0 = max(J, 0)
    fe = x_fr_hcp._fe(x[j0:], g)
    cz = cum_z(x, g, j0)
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
    th0 = np.float32(B.TH0)
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
        late = False
        if owner < 0 and not extended and t >= B3.MID_UNTIL and not ever:
            lo = max(0, t - (B5.NORM_WINDOW - 1))
            if (pre[t + 1] - pre[lo]) / (t + 1 - lo) >= B5.NORM_GATE:
                late = True
                zs = fe["zS"][t - j0]
                for j in range(K):
                    s[t, j] = np.clip(B.TH0 + np.float32(B.KAPPA) * (np.float32(zs[j]) - np.float32(B5.TAU_L)),
                                      -1.0, 1.0)
        if owner < 0 and not extended and t >= B3.MID_UNTIL and not late:
            above = 0
            for j in range(K):
                if b0[t, j] >= th0:
                    above += 1
            if above + above > K:
                zc = cz[t - j0]
                for j in range(K):
                    s[t, j] = np.clip(B.TH0 + np.float32(B.KAPPA) * (np.float32(zc[j]) - np.float32(TAU_U)), -1.0, 1.0)
        ever = ever or owner >= 0 or extended
    return s


def outputs_fn(x, g):
    return {"bank6": bank6_ref(x, g)}
