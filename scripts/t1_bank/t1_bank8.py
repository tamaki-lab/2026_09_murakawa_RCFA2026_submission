"""T1-specialist bank8 candidate: bank7 with a norm-direction cross-moment statistic in the flat rows (same on both
tracks).

Evaluated in docs/experiments/20261009_final_stretch_t1_direction.md (fifth consultation; `analyze_t1_v5a.py`,
`analyze_t1_e2flat.py`: T1 proxy +105.9 gen-eq against bank7). bank6's flat rows (owner-less, not extended, t >= 70, not
the late step, base row with more than half of the gallery at or above TH0) are rows where the base scorer has no
usable ranking; on the released data they occur in one Track 1 sequence (a long low-norm stretch) and in no Track 2
row. The change against bank7, nothing else:

  flat  : every column becomes clip(TH0 + KAPPA (zE - TAU_E), -1, 1), KAPPA .05 and TAU_E 1.0 (the bank's original
          opening constants, t1_bank.KAPPA / t1_bank.TAU; bank7: the cumulative unit mean since the jump, TAU_U .25).
          zE is the LOO z over the gallery of h_t . (g_j - mean_k g_k), where, over the frames s <= t of the current
          segment ([0, J) before the jump, [J, T) from it), with unit frames u_s, y_s = u_s - mean_k g_k,
          q_s = |x_s| / 20 and weights w_s = 1 at the segment start, else (1 - rho_s) / (1 + rho_s) with
          rho_s = clip(u_s . u_{s-1}, 0, 1): h_t = V - R U / S, S = sum w, R = sum w q, U = sum w y, V = sum w q y,
          i.e. the weighted covariance of frame norm and frame direction. A row whose covariance is zero by
          construction keeps bank7's value: the first frame of a segment, and any row whose segment prefix has all
          frame norms equal or all unit frames equal (exact comparisons; floating-point residue of an exactly zero
          covariance would otherwise be amplified by the LOO z; Codex review 2026-10-11), as well as a computed
          h_t of zero.
All other rows are bank7's bitwise. All state is per sequence and uses frames <= t only.
`bank8` is the vectorised processor (`post8` on given base scores); `bank8_ref` is an independent per-frame loop used as
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
import t1_bank7 as B7  # noqa: E402
import x_fr_hcp  # noqa: E402

TAU_E = B.TAU  # 1.0


def cross_z(x, g, j0):
    """zE for every frame (T x K, float64) and the mask of frames whose covariance is not zero by construction (one
    block per segment; a prefix with all frame norms equal or all unit frames equal has an exactly zero covariance)."""
    xb = np.asarray(x, np.float64)
    u = B._unit(xb)
    q = np.linalg.norm(xb, axis=1) / 20.0
    gu = B._unit(g)
    mu = gu.mean(0)
    V = gu - mu
    y = u - mu
    T = u.shape[0]
    H = np.zeros_like(u)
    ok = np.zeros(T, bool)
    for a, b in ((0, j0), (j0, T)) if j0 > 0 else ((0, T),):
        if b <= a:
            continue
        rho = np.clip((u[a + 1: b] * u[a: b - 1]).sum(1), 0.0, 1.0)
        w = np.r_[1.0, (1 - rho) / (1 + rho)]
        S = np.cumsum(w)
        R = np.cumsum(w * q[a:b])
        U = np.cumsum(w[:, None] * y[a:b], axis=0)
        Vv = np.cumsum((w * q[a:b])[:, None] * y[a:b], axis=0)
        H[a:b] = Vv - (R / S)[:, None] * U
        qs = q[a:b]
        varied_q = np.maximum.accumulate(qs) > np.minimum.accumulate(qs)
        varied_u = np.logical_or.accumulate((u[a:b] != u[a]).any(1))
        ok[a:b] = varied_q & varied_u
        ok[a] = False
    ok &= np.linalg.norm(H, axis=1) > 0
    return B._loo_z(H @ V.T), ok


def post8(base, x, g):
    """bank8's steps on a (T x K) float32 base score matrix (bank7's, flat rows on zE)."""
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
    out[:e] = B7.early_map(cmz)  # bank7's opening values, with the same ownership
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
    for m, tau in ((midr & pre, B7.TAU_M_PRE), (midr & ~pre, B7.TAU_M_POST)):
        idx = np.nonzero(m)[0]
        if idx.size:
            out[idx] = B7.lin(zP[idx].astype(np.float32), B3.KAPPA_M, tau)
    held = (own >= 0) | ext
    before = np.concatenate([[False], np.cumsum(held)[:-1] > 0])
    late = (own < 0) & ~ext & (t >= B3.MID_UNTIL) & ~before & (B5.recent_norm(x) >= B5.NORM_GATE)
    idx = np.nonzero(late)[0]
    if idx.size:
        out[idx] = B7.lin(fe["zS"][idx - j0], B.KAPPA, B7.TAU_L7)
    base_rows = (own < 0) & ~ext & (t >= B3.MID_UNTIL) & ~late
    flat = base_rows & (2 * np.count_nonzero(base >= B.TH0, axis=1) > base.shape[1])
    idx = np.nonzero(flat)[0]
    if idx.size:
        out[idx] = B6.flat_map(B6.cum_z(x, g, j0)[idx - j0])  # bank7's value, kept where h_t is zero
        zE, okE = cross_z(x, g, j0)
        ie = idx[okE[idx]]
        if ie.size:
            out[ie] = B7.lin(zE[ie], B.KAPPA, TAU_E)
    return out


def bank8(x, g):
    return post8(B._BASE(x, g), x, g)


def bank8_ref(x, g):
    """Per-frame loop reference: bank7_ref's steps frame by frame with the flat rows on zE, with independent
    bookkeeping; the numeric primitives (base scores, unit frames, row norms, the block matrix products, the LOO z,
    the HCP memories of a sequence or of its suffix and `cross_z`) are the same calls as in `bank8`, so the result
    must be bit-identical."""
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
    ze, oke = cross_z(x, g, j0)
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
                w = np.float32(float(early[t, j]) - B7.OPEN_A * top)
                nr[j] = np.clip(th0 + k32 * (w - np.float32(B7.TAU_O)), -1.0, 1.0)
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
            tau_m = np.float32(B7.TAU_M_POST if (J > 0 and t >= J) else B7.TAU_M_PRE)
            for j in range(K):
                s[t, j] = np.clip(th0 + km32 * (np.float32(zt[j]) - tau_m), -1.0, 1.0)
        late = False
        if owner < 0 and not extended and t >= B3.MID_UNTIL and not ever:
            lo = max(0, t - (B5.NORM_WINDOW - 1))
            if (pre[t + 1] - pre[lo]) / (t + 1 - lo) >= B5.NORM_GATE:
                late = True
                zs = fe["zS"][t - j0]
                for j in range(K):
                    s[t, j] = np.clip(th0 + k32 * (np.float32(zs[j]) - np.float32(B7.TAU_L7)), -1.0, 1.0)
        if owner < 0 and not extended and t >= B3.MID_UNTIL and not late:
            above = 0
            for j in range(K):
                if b0[t, j] >= th0:
                    above += 1
            if above + above > K:
                if oke[t]:
                    zc, tau_f = ze[t], np.float32(TAU_E)
                else:
                    zc, tau_f = cz[t - j0], np.float32(B6.TAU_U)
                for j in range(K):
                    s[t, j] = np.clip(th0 + k32 * (np.float32(zc[j]) - tau_f), -1.0, 1.0)
        ever = ever or owner >= 0 or extended
    return s


def outputs_fn(x, g):
    return {"bank8": bank8_ref(x, g)}
