"""T1-specialist bank7 candidate: bank6 with its cuts re-placed after bank6 moved the pooled threshold (same on both
tracks).

Evaluated in docs/experiments/20261009_final_stretch_t1_direction.md (`analyze_t1_retune6.py`, `analyze_t1_rival.py`,
`analyze_t1_joint7.py`, `analyze_t1_held7.py`: T1 proxy about +64 gen-eq against bank6). bank6's flat step moved the
pooled EER threshold from -.0909 to -.0952, so every opening-type map's effective cut moved with it; the cells want
different cuts (accepting a column pays when its posterior exceeds 1/200, i.e. at about half the target's expected z,
which differs between the cells). Changes against bank6, nothing else:

  early : for t < 10 every column's output becomes clip(TH0 + KAPPA (w - TAU_O), -1, 1),
          w = float32(cmz - OPEN_A max_k cmz) (float64 inside), OPEN_A .2, TAU_O .5 (bank6: OPEN_A 0, TAU 1); an owned
          opening row keeps that value at the owner and -1 elsewhere. A row with a strong leader needs a higher z from
          every column. The new map sets output values only: leaders, runs, extension and every other row class are
          computed from bank6's opening map exactly as in bank6 (the shift is constant within a row, but float32
          rounding can merge near-equal top values and move a first-index leader; Codex review 2026-10-10).
  mid   : TAU_M_PRE .9 in mid rows while no jump has been detected (t < J, or no jump in frames <= t) and
          TAU_M_POST 1.25 from the detected jump on (t >= J > 0) (bank6 1 in both). The jump is detected at frame J
          itself, so the class of row t reads frames <= t only (every Track 1 sequence has a jump at some J >= 1).
  late  : TAU_L7 1.4 (bank6 1.5)
The flat step (TAU_U .25) and every row class are bank6's, so bank7 differs from bank6 only in the values of the
opening, mid and late-step rows. All state is per sequence and uses frames <= t only.
`bank7` is the vectorised processor (`post7` on given base scores); `bank7_ref` is an independent per-frame loop used as
the builder's bit-identity reference.
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
import t1_bank6 as B6  # noqa: E402
import x_fr_hcp  # noqa: E402

OPEN_A = 0.2
TAU_O = 0.5
TAU_M_PRE = 0.9
TAU_M_POST = 1.25
TAU_L7 = 1.4


def lin(z, kappa, tau):
    """clip(TH0 + kappa (z - tau)) in float32 (float32 in, float32 out)."""
    z = np.asarray(z, np.float32)
    return np.clip(B.TH0 + np.float32(kappa) * (z - np.float32(tau)), -1.0, 1.0).astype(np.float32)


def early_map(cmz):
    """The re-placed opening map on a (rows x K) float32 cmz block."""
    c = np.asarray(cmz, np.float32).astype(np.float64)
    w = (c - OPEN_A * c.max(axis=1, keepdims=True)).astype(np.float32)
    return lin(w, B.KAPPA, TAU_O)


def post7(base, x, g):
    """bank7's steps on a (T x K) float32 base score matrix."""
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
    s[:e] = np.clip(B.TH0 + B.KAPPA * (cmz - B.TAU), -1.0, 1.0)  # bank6's opening map: decisions only
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
    out[:e] = early_map(cmz)  # bank7's opening values, with the same ownership
    he = np.nonzero(own[:e] >= 0)[0]
    if he.size:
        keep = out[he, own[he]].copy()
        out[he] = -1.0
        out[he, own[he]] = keep
    ext = (own < 0) & (t >= B2.EXT_FROM) & lead_ok
    idx = np.nonzero(ext)[0]
    if idx.size:
        keep = out[idx, a[idx]].copy()
        out[idx] = -1.0
        out[idx, a[idx]] = keep
    midr = (own < 0) & (t >= B2.EXT_FROM) & (t < B3.MID_UNTIL) & ~lead_ok
    pre = (t < J) if J > 0 else np.ones(T, bool)  # no jump detected at or before t
    for m, tau in ((midr & pre, TAU_M_PRE), (midr & ~pre, TAU_M_POST)):
        idx = np.nonzero(m)[0]
        if idx.size:
            out[idx] = lin(zP[idx].astype(np.float32), B3.KAPPA_M, tau)
    held = (own >= 0) | ext
    before = np.concatenate([[False], np.cumsum(held)[:-1] > 0])
    late = (own < 0) & ~ext & (t >= B3.MID_UNTIL) & ~before & (B5.recent_norm(x) >= B5.NORM_GATE)
    idx = np.nonzero(late)[0]
    if idx.size:
        out[idx] = lin(fe["zS"][idx - j0], B.KAPPA, TAU_L7)
    base_rows = (own < 0) & ~ext & (t >= B3.MID_UNTIL) & ~late
    flat = base_rows & (2 * np.count_nonzero(base >= B.TH0, axis=1) > base.shape[1])
    idx = np.nonzero(flat)[0]
    if idx.size:
        out[idx] = B6.flat_map(B6.cum_z(x, g, j0)[idx - j0])
    return out


def bank7(x, g):
    return post7(B._BASE(x, g), x, g)


def bank7_ref(x, g):
    """Per-frame loop reference: bank6's steps frame by frame (as in t1_bank6.bank6_ref) with bank7's maps, with
    independent bookkeeping; the numeric primitives (base scores, unit frames, row norms, the block matrix products,
    the LOO z and the HCP memories of a sequence or of its suffix) are the same calls as in `bank7`, so the result must
    be bit-identical."""
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
    cz = B6.cum_z(x, g, j0)
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
    k32, km32 = np.float32(B.KAPPA), np.float32(B3.KAPPA_M)
    for t in range(T):
        zt = fe["zP"][t - J] if J >= 0 and t >= J else zfull[t]
        if t < e:
            s[t] = np.clip(B.TH0 + B.KAPPA * (early[t] - B.TAU), -1.0, 1.0)  # bank6's map: decisions only
            top = float(np.max(early[t].astype(np.float64)))
            nr = np.empty(K, np.float32)
            for j in range(K):
                w = np.float32(float(early[t, j]) - OPEN_A * top)
                nr[j] = np.clip(th0 + k32 * (w - np.float32(TAU_O)), -1.0, 1.0)
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
        if t < e:  # output: bank7's opening values with the same owner
            if owner >= 0:
                kv = nr[owner]
                nr[:] = -1.0
                nr[owner] = kv
            s[t] = nr
        extended = False
        if owner < 0 and t >= B2.EXT_FROM and (zt[lead] >= B2.ZLEAD_OWN or run >= B2.RUN_OWN):
            keep = s[t, lead]
            s[t] = -1.0
            s[t, lead] = keep
            extended = True
        if owner < 0 and not extended and B2.EXT_FROM <= t < B3.MID_UNTIL:
            tau_m = np.float32(TAU_M_POST if (J > 0 and t >= J) else TAU_M_PRE)
            for j in range(K):
                s[t, j] = np.clip(th0 + km32 * (np.float32(zt[j]) - tau_m), -1.0, 1.0)
        late = False
        if owner < 0 and not extended and t >= B3.MID_UNTIL and not ever:
            lo = max(0, t - (B5.NORM_WINDOW - 1))
            if (pre[t + 1] - pre[lo]) / (t + 1 - lo) >= B5.NORM_GATE:
                late = True
                zs = fe["zS"][t - j0]
                for j in range(K):
                    s[t, j] = np.clip(th0 + k32 * (np.float32(zs[j]) - np.float32(TAU_L7)), -1.0, 1.0)
        if owner < 0 and not extended and t >= B3.MID_UNTIL and not late:
            above = 0
            for j in range(K):
                if b0[t, j] >= th0:
                    above += 1
            if above + above > K:
                zc = cz[t - j0]
                for j in range(K):
                    s[t, j] = np.clip(th0 + k32 * (np.float32(zc[j]) - np.float32(B6.TAU_U)), -1.0, 1.0)
        ever = ever or owner >= 0 or extended
    return s


def outputs_fn(x, g):
    return {"bank7": bank7_ref(x, g)}
