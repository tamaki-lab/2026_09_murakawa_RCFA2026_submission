"""T1 premise checks of the fifth consultation, on top of bank7 (aggregates only).

Five causal statistics proposed in the fifth consultation (Codex gpt-6-astra, effort high; receipts in the direction
doc), each substituted into bank7's existing map of the cells it targets. Decisions, ownership, the extension and every
other row stay bank7's. Cells (bank7's causal row states, `t1_eval7.state7`): open = free rows t < 10 before the jump;
pre10 = free rows 10 <= t < J; mid = free rows J <= t < 70; late = the late step; flat = bank6's flat rows. Maps:
open bank7's opening map (z - .2 max z, tau .5, kappa .05), pre10 / mid kappa .03 with tau .9 / 1.25, late kappa .05
tau 1.4, flat kappa .05 tau .25. "b7" substitutes bank7's own statistic (cmz, zP, restarted zP, restarted zS, cumJ)
and must reproduce bank7's rows bitwise (self-check).
  A1 (open, pre10): nullspace-amplitude regression intercept on the raw prefix x_0 .. x_t (n >= 4): Q = projection off
     the gallery span, v = unit(Q mean x), q_i = x_i . v, h = sum (q_i - qbar)(x_i - xbar) / sum (q_i - qbar)^2,
     r = xbar - qbar h, statistic LOO z of cos(r, g_j). Diagnostics: amplitude CV, variance inflation 1 + qbar^2 /
     var q, forward prediction of the current frame's gallery-span part (fit on frames < t) against the prefix mean.
  A2 (open, pre10): identity-dependent covariance likelihood. R_i = x_i - xbar, s2 = sum |Q R_i|^2 / ((n - 1)(512 -
     rank G)), u_j = sum (R_i . g_j)^2 / ((n - 1) s2), L_j = (n - 1)/2 (u_j - 1 - log u_j) for u_j > 1 else 0;
     statistic LOO z(z_old + LOO z(L)), z_old = cmz (open) or zP (pre10). Diagnostic: split-half agreement of the
     target's LOO z(L).
  E2 (all cells): signed norm-direction cross-moment. y_t = unit(x_t) - mean gallery, q_t = |x_t| / 20, weights w = 1
     at a segment start, else (1 - rho) / (1 + rho), rho = clip(cos(x_t, x_{t-1}), 0, 1); segments [0, J) and [J, T);
     h_t = V - R U / S with S = sum w, R = sum w q, U = sum w y, V = sum w q y; statistic LOO z of h_t . (g_j - mean g).
  D1 (all cells): contrast against the gallery neighbourhood, v_j = unit(g_j - mean of its 5 nearest other gallery
     vectors); statistic LOO z of cos(cumulative unit mean, v_j), cumulative from 0 before the jump, from J after it.
  D2 (mid, late): retrieval of matching views. Among unit frames J .. t-1 with cosine >= .5 to the current frame, the
     five most similar (ties at the fifth included); statistic LOO z of cos(unit(u_t + their sum), g_j); none -> bank7.
Rows where a statistic is undefined keep bank7. Per (statistic, cell): rows, target mean z, median rank, top-1 / 5 / 20
shares, the single-threshold floor of the mapped scores against bank7's on the same rows, and the cost at bank7's
threshold. Exact pooled T1 proxy gains against bank7 per (statistic, cell) and per statistic over its cells; 12 subsets
and per-sequence deltas for the best five. Pseudo targets are evaluator-only.
Usage (server): python -B scripts/t1_bank/analyze_t1_v5a.py PARENT_ZIP > log
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "core"))
sys.path.insert(0, str(HERE.parents[1]))
import x_r5  # noqa: E402,F401  (pins BLAS threads)
import numpy as np  # noqa: E402

import t1_bank as B  # noqa: E402
import t1_bank3 as B3  # noqa: E402
import t1_bank6 as B6  # noqa: E402
import t1_bank7 as B7  # noqa: E402
import t1_eval7 as E7  # noqa: E402
import x_fr_hcp  # noqa: E402

CELLS = ("open", "pre10", "mid", "late", "flat")
APPLY = {"b7": CELLS, "A1": ("open", "pre10"), "A2": ("open", "pre10"), "E2": CELLS, "D1": CELLS,
         "D2": ("mid", "late")}
STATS = tuple(APPLY)
D = E7.D


def loo(r):
    return B._loo_z(np.atleast_2d(np.asarray(r, np.float64)))


def cell_map(cell, z):
    z = np.asarray(z, np.float32)
    if cell == "open":
        return B7.early_map(z)
    if cell == "pre10":
        return B7.lin(z, B3.KAPPA_M, B7.TAU_M_PRE)
    if cell == "mid":
        return B7.lin(z, B3.KAPPA_M, B7.TAU_M_POST)
    if cell == "late":
        return B7.lin(z, B.KAPPA, B7.TAU_L7)
    return B7.lin(z, B.KAPPA, B6.TAU_U)


def a1_rows(xr, Qg, rows):
    """Intercept directions for rows (prefix from frame 0), diagnostics."""
    out, cv, vif, fwd = {}, [], [], []
    prev = None
    for t in range(3, (rows.max() + 1) if rows.size else 0):
        X = xr[: t + 1]
        xbar = X.mean(0)
        qx = xbar - Qg @ (Qg.T @ xbar)
        nq = np.linalg.norm(qx)
        fit = None
        if nq > 0:
            v = qx / nq
            q = X @ v
            qbar = q.mean()
            dq = q - qbar
            Sq = float(dq @ dq)
            if Sq > 0:
                h = (dq @ (X - xbar)) / Sq
                r = xbar - qbar * h
                if np.any(r):
                    fit = (v, h, r, xbar)
                    if t in rows:
                        out[t] = r
                        cv.append(np.sqrt(Sq / (t + 1)) / abs(qbar) if qbar != 0 else np.inf)
                        vif.append(1.0 + qbar * qbar / (Sq / (t + 1)))
        if prev is not None and t in rows:
            pv, ph, pr, pxbar = prev
            xt = xr[t]
            pred = pr + (xt @ pv) * ph
            e1 = Qg.T @ (xt - pred)
            e0 = Qg.T @ (xt - pxbar)
            fwd.append((float(e1 @ e1), float(e0 @ e0)))
        prev = fit
    return out, cv, vif, fwd


def a2_rows(xr, Qg, rG, gu, rows, zold):
    out, split = {}, []
    d = xr.shape[1]

    def zl(X):
        n = X.shape[0]
        R = X - X.mean(0)
        QR = R - (R @ Qg) @ Qg.T
        s2 = float((QR * QR).sum()) / ((n - 1) * (d - rG))
        if s2 <= 0:
            return None
        p = R @ gu.T
        u = (p * p).sum(0) / ((n - 1) * s2)
        L = np.where(u > 1, (n - 1) / 2.0 * (u - 1 - np.log(np.maximum(u, 1e-300))), 0.0)
        if np.ptp(L) == 0:
            return None
        return loo(L)[0]

    for k, t in enumerate(rows):
        if t < 3:
            continue
        z = zl(xr[: t + 1])
        if z is None:
            continue
        out[t] = loo(np.asarray(zold[k], np.float64) + z)[0]
        n = t + 1
        if n >= 8:
            h1, h2 = zl(xr[: n // 2]), zl(xr[n // 2: n])
            if h1 is not None and h2 is not None:
                split.append((h1, h2))
    return out, split


def e2_all(u, nrm, gu, j0):
    T = u.shape[0]
    mu = gu.mean(0)
    V = gu - mu
    y = u - mu
    q = nrm / 20.0
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
        ok[a + 1: b] = True
    ok &= np.linalg.norm(H, axis=1) > 0
    return H @ V.T, ok


def d1_all(u, vnn, j0):
    T = u.shape[0]
    S = np.zeros_like(u)
    for a, b in ((0, j0), (j0, T)) if j0 > 0 else ((0, T),):
        if b > a:
            S[a:b] = np.cumsum(u[a:b], axis=0)
    return B._unit(S) @ vnn.T


def d2_rows(u, gu, j0, rows):
    out, share_seg = {}, []
    for t in rows:
        if t <= j0:
            continue
        prev = u[j0:t]
        cs = prev @ u[t]
        sel = np.nonzero(cs >= 0.5)[0]
        if sel.size == 0:
            continue
        if sel.size > 5:
            kth = np.sort(cs[sel])[-5]
            sel = sel[cs[sel] >= kth]
        out[t] = loo(B._unit(u[t] + prev[sel].sum(0)) @ gu.T)[0]
        share_seg.append(float((t - (j0 + sel)).mean()))
    return out, share_seg


def _job(i):
    x, g, c = D["s1"][i], D["g1"], int(D["c1"][i])
    base0 = np.asarray(D["base"][i], np.float32)
    st = E7.state7(base0, x, g)
    s7, own, state, J, j0, zP = st["s7"], st["own"], st["state"], st["J"], st["j0"], st["zP"]
    T = s7.shape[0]
    t = np.arange(T)
    res = E7.entry(s7, c)
    pre = t < j0
    rows = {"open": np.nonzero((t < B.EARLY) & pre & (own < 0))[0], "pre10": np.nonzero(state == "pre_free_t10")[0],
            "mid": np.nonzero(state == "mid")[0], "late": np.nonzero(state == "late_new")[0],
            "flat": np.nonzero(state == "flat")[0]}
    xr = np.asarray(x, np.float64)
    nrm = np.linalg.norm(xr, axis=1)
    u = B._unit(xr)
    gu = B._unit(g)
    Qg, rG = D["Qg"], D["rG"]
    Z = {s: {} for s in STATS}  # stat -> cell -> (rows, z)
    cmz = B.early_cmz(x, g)
    fe = x_fr_hcp._fe(x[j0:], g)
    cumJ = B6.cum_z(x, g, j0)
    own7 = {"open": cmz, "pre10": zP, "mid": zP}
    for cell in CELLS:
        r = rows[cell]
        if not r.size:
            continue
        if cell in own7:
            Z["b7"][cell] = (r, np.asarray(own7[cell][r], np.float64))
        elif cell == "late":
            Z["b7"][cell] = (r, np.asarray(fe["zS"][r - j0], np.float64))
        else:
            Z["b7"][cell] = (r, np.asarray(cumJ[r - j0], np.float64))
    diag = {"a1_cv": [], "a1_vif": [], "a1_fwd": [], "a2_split": [], "d2_dist": []}
    for cell in ("open", "pre10"):
        r = rows[cell]
        if not r.size:
            continue
        rr = r[r >= 3]
        a1, cv, vif, fwd = a1_rows(xr, Qg, rr)
        diag["a1_cv"] += cv
        diag["a1_vif"] += vif
        diag["a1_fwd"] += fwd
        if a1:
            ks = np.array(sorted(a1))
            Z["A1"][cell] = (ks, loo(np.stack([B._unit(a1[k]) for k in ks]) @ gu.T))
        zold = Z["b7"][cell][1][np.searchsorted(r, rr)] if rr.size else np.zeros((0, 200))
        a2, split = a2_rows(xr, Qg, rG, gu, rr, zold)
        diag["a2_split"] += [(float(h1[c]), float(h2[c])) for h1, h2 in split]
        if a2:
            ks = np.array(sorted(a2))
            Z["A2"][cell] = (ks, np.stack([a2[k] for k in ks]))
    E2raw, e2ok = e2_all(u, nrm, gu, j0)
    D1raw = d1_all(u, D["vnn"], j0)
    for cell in CELLS:
        r = rows[cell]
        if not r.size:
            continue
        re = r[e2ok[r]]
        if re.size:
            Z["E2"][cell] = (re, loo(E2raw[re]))
        Z["D1"][cell] = (r, loo(D1raw[r]))
    for cell in ("mid", "late"):
        r = rows[cell]
        if not r.size:
            continue
        d2, dist = d2_rows(u, gu, j0, r)
        diag["d2_dist"] += dist
        if d2:
            ks = np.array(sorted(d2))
            Z["D2"][cell] = (ks, np.stack([d2[k] for k in ks]))
    keep = np.ones(g.shape[0], bool)
    keep[c] = False
    cellres = {}
    self_ok = True
    for s in STATS:
        allr, alln = [], []
        for cell, (r, z) in Z[s].items():
            new = cell_map(cell, z)
            if s == "b7":
                self_ok &= bool(np.array_equal(new.view(np.uint32), s7[r].view(np.uint32)))
            res["chg"][(s, cell)] = E7.swap(s7, r, new, c)
            allr.append(r)
            alln.append(new)
            zt = z[:, c]
            cellres[(s, cell)] = {"z_t": zt, "rank": (z > zt[:, None]).sum(1), "g_new": new[:, c].astype(np.float64),
                                  "i_new": new[:, keep].ravel(), "g_old": s7[r, c].astype(np.float64),
                                  "i_old": s7[r][:, keep].ravel()}
        if allr and s != "b7":
            r = np.concatenate(allr)
            o = np.argsort(r)
            res["chg"][(s, "all")] = E7.swap(s7, r[o], np.concatenate(alln)[o], c)
    res["cell"] = cellres
    res["diag"] = diag
    res["self_ok"] = self_ok
    res["J"] = J
    return i, res


def floor_geq(gen, imp):
    if gen.size == 0:
        return 0.0
    gs = np.sort(gen)
    ims = np.sort(imp.ravel())
    cand = np.unique(np.r_[gs, ims, np.inf])
    miss = np.searchsorted(gs, cand, side="left")
    acc = ims.size - np.searchsorted(ims, cand, side="left")
    return float(np.min(miss + acc / 199.0))


def main(zpath, workers=24):
    t0 = time.time()
    E7.load(zpath)
    g = B._unit(D["g1"])
    Qg, Rg = np.linalg.qr(g.T)
    rG = int(np.sum(np.abs(np.diag(Rg)) > 1e-10 * np.abs(Rg).max()))
    C = g @ g.T
    np.fill_diagonal(C, -np.inf)
    nn = np.argsort(-C, axis=1)[:, :5]
    v = g - g[nn].mean(1)
    nv = np.linalg.norm(v, axis=1, keepdims=True)
    D.update(Qg=Qg[:, :rG], rG=rG, vnn=np.where(nv > 0, v / np.where(nv > 0, nv, 1.0), g))
    R = E7.run(_job, workers)
    keys = sorted({k for o in R for k in o["chg"] if k[0] != "b7"})
    res = {"rank_G": rG, "self_check_b7_equals_bank7": all(o["self_ok"] for o in R),
           "b7_keys_change_nothing": all(not o["chg"][k][0].size for o in R for k in o["chg"] if k[0] == "b7")}
    rep = E7.report(R, keys, name=lambda k: f"{k[0]}_{k[1]}", top=6)
    thr = rep["thr"]
    th32 = np.float32(thr)
    cells = {}
    for k in sorted({k for o in R for k in o["cell"]}):
        parts = [o["cell"][k] for o in R if k in o["cell"]]
        zt = np.concatenate([p["z_t"] for p in parts])
        rk = np.concatenate([p["rank"] for p in parts])
        gn, inew = np.concatenate([p["g_new"] for p in parts]), np.concatenate([p["i_new"] for p in parts])
        go, iold = np.concatenate([p["g_old"] for p in parts]), np.concatenate([p["i_old"] for p in parts])
        cells[f"{k[0]}_{k[1]}"] = {
            "rows": int(zt.size), "seqs": len(parts), "target_mean_z": round(float(zt.mean()), 3),
            "target_median_rank": float(np.median(rk)), "top1": round(float((rk == 0).mean()), 3),
            "top5": round(float((rk < 5).mean()), 3), "top20": round(float((rk < 20).mean()), 3),
            "floor_new": round(floor_geq(gn, inew), 1), "floor_bank7_same_rows": round(floor_geq(go, iold), 1),
            "cost_new_at_thr": round(float((gn < thr).sum() + (inew >= th32).sum() / 199.0), 1),
            "cost_bank7_at_thr": round(float((go < thr).sum() + (iold >= th32).sum() / 199.0), 1)}
    res["cells"] = cells
    dg = {k: [x for o in R for x in o["diag"][k]] for k in ("a1_cv", "a1_vif", "a1_fwd", "a2_split", "d2_dist")}
    cv, vif = np.array(dg["a1_cv"]), np.array(dg["a1_vif"])
    fwd = np.array(dg["a1_fwd"]) if dg["a1_fwd"] else np.zeros((0, 2))
    sp = np.array(dg["a2_split"]) if dg["a2_split"] else np.zeros((0, 2))
    res["diag"] = {
        "a1_rows": int(cv.size),
        "a1_amplitude_cv_q10_50_90": [round(float(v), 4) for v in np.quantile(cv, [.1, .5, .9])] if cv.size else [],
        "a1_variance_inflation_q10_50_90": [round(float(v), 1) for v in np.quantile(vif, [.1, .5, .9])] if vif.size
        else [],
        "a1_forward_error_ratio_mean": round(float(fwd[:, 0].mean() / fwd[:, 1].mean()), 4) if fwd.size else None,
        "a1_forward_better_share": round(float((fwd[:, 0] < fwd[:, 1]).mean()), 3) if fwd.size else None,
        "a2_split_rows": int(sp.shape[0]),
        "a2_split_target_zL_means": [round(float(v), 3) for v in sp.mean(0)] if sp.size else [],
        "a2_split_target_zL_corr": round(float(np.corrcoef(sp[:, 0], sp[:, 1])[0, 1]), 3) if sp.shape[0] > 2 else None,
        "d2_rows": len(dg["d2_dist"]),
        "d2_mean_lag_q10_50_90": [round(float(v), 1) for v in np.quantile(dg["d2_dist"], [.1, .5, .9])]
        if dg["d2_dist"] else []}
    res.update(rep)
    res["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
