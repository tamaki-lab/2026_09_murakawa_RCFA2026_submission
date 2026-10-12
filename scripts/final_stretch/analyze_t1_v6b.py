"""T1 premise checks of the sixth consultation's second round, on top of bank8 (aggregates only).

Three fixed proposals (Codex gpt-6-astra high, round 6; receipts in the direction doc), each against bank8
(t1_bank8.post8; decisions, ownership and the extension unchanged; only output values of free rows change; undefined
statistics keep bank8). Maps: lin(z, kappa, tau) = clip(TH0 + kappa (z - tau)).
  NO (D): norm-order association in bank8's flat rows. Within the current segment (from J), H_t = H_{t-1} +
     sum_{s < t in the segment} sign(q_t - q_s) (u_t - u_s), u = unit frame, q = |x| / 20; z = row LOO z of H_t . g_j
     (unit gallery); map lin(z, .05, 1.0); H = 0 or zero spread keeps bank8.
  DEL (C): history-deletion sensitivity in bank8's mid rows after the jump with n = t - J + 1 >= 3. The frames J .. t
     are cut into three consecutive blocks at floor(b n / 3), b = 0 .. 3; for each block, the HCP cumulative memory z of
     the remaining frames in time order (x_fr_hcp._fe(...)["zP"], last row); d_j = population sd over the three;
     z' = zPr_t - d (zPr = bank8's restarted memory z); map lin(z', .03, 1.25) (bank8's mid map after the jump).
  CUR (A): quality curvature after removing linear quality, gated. Post-jump rows t >= 10 that bank8 leaves free (mid,
     late step, late base-scored, flat) with no owned or extended row before t, and either flat or D(zP) <= .5 and
     D(zS) <= .5 (D(z) = max_j (z_j^+)^2 / sum_j (z_j^+)^2, 0 when the sum is 0; zP / zS the restarted HCP memories).
     c_tj = (u_t - mu) . (g_j - mu) (mu = mean unit gallery row); E2's weights from J; weighted covariances C, V;
     R_j = [C(q^2, c_j) - C(q^2, q) C(q, c_j) / V(q)] / sqrt([V(q^2) - C(q^2, q)^2 / V(q)] [V(c_j) - C(q, c_j)^2 /
     V(q)]); z = row LOO z of R (positive sign only). Stability gate: the same R on two streams (alternating 10-frame
     blocks since J); both streams' effective counts (sum w)^2 / sum w^2 >= 10 and the cosine of their LOO-z vectors
     > .5. Maps: mid lin(z, .03, 1.25), late step lin(z, .05, 1.4), flat and late base-scored lin(z, .05, 1.0).
Revised before its first run after the proposers' fidelity reviews (round 7; the first version at 34c7968 never ran):
  - undefined statistics keep bank8 exactly: NO accumulates the literal pair differences (u_t - u_s), so identical unit
    frames give H = 0 exactly; CUR uses centred two-pass moments and falls back when the positive-weight prefix has at
    most two distinct quality values (q^2 is then affine in q), identical unit frames or a column exactly affine in q,
    and the stability cosine has no epsilon; LOO z is computed from centred arithmetic and falls back when any column's
    LOO variance is exactly zero or not positive (strict_loo); CUR's conditional-information gate fails closed when an
    applied row has no defined E2c z;
  - NO also runs without a detected jump (segment from frame 0) and its floors cover every flat row (fallback rows keep
    bank8); DEL's floor is the row-wise oracle floor sum_rows #{j != c: s_j >= s_c} / 199 (ties accepted), with the
    single-threshold floor reported too; CUR reports the eligible cost by baseline sequence rank and state, the largest
    fixed-threshold loss outside the two costliest baseline sequences and the target-vs-impostor AUC of its z within
    quintiles of the E2c z (the quality-correlation statistic on the same support), pooled over the applied rows;
  - every gain, floor and cost is reported unrounded, with the kill gates evaluated before any rounding.
Per variant: coverage counts, target mean z / median rank on the applied rows, floors, exact pooled T1 proxy gain against
bank8, subsets (even, odd, minus the costliest one / three) and per-sequence deltas at bank8's threshold. Kill rules
(fixed by the proposers): NO pooled < 20, fixed-threshold <= 0 or floor gain <= 0; DEL row-oracle floor gain < 30,
pooled < 20, the even or odd half negative or minus the costliest three negative; CUR eligible cost < 50, floor gain
< 50, pooled < 30, any sequence outside the two costliest losing > 10 at the fixed threshold, or no conditional
information (within-quintile AUC <= .5). Pseudo targets are evaluator-only.
Usage (server): python -B scripts/final_stretch/analyze_t1_v6b.py PARENT_ZIP [workers] > log
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ogata"))
sys.path.insert(0, str(HERE.parents[1]))
import x_r5  # noqa: E402,F401  (pins BLAS threads)
import numpy as np  # noqa: E402

import t1_bank as B  # noqa: E402
import t1_bank2 as B2  # noqa: E402
import t1_bank3 as B3  # noqa: E402
import t1_bank7 as B7  # noqa: E402
import t1_eval7 as E7  # noqa: E402
import t1_eval8 as E8  # noqa: E402
import x_fr_hcp  # noqa: E402
from analyze_t1_v5a import floor_geq  # noqa: E402

D = E8.D
KEYS = ("NO", "DEL", "CUR")


def e2_weights(u, a, b):
    rho = np.clip((u[a + 1: b] * u[a: b - 1]).sum(1), 0.0, 1.0)
    return np.r_[1.0, (1 - rho) / (1 + rho)]


def strict_loo(s):
    """Row LOO z of a score vector (each column against the population mean / sd of the other K - 1), from centred
    arithmetic: with d = s - mean(s) and D2 = sum d^2, column j's LOO variance is (D2 - d_j^2 K / (K - 1)) / (K - 1) and
    its numerator K d_j / (K - 1). None when some column's LOO variance is exactly zero (all entries equal, or all but one
    equal), when any computed variance is not positive, or when the result is not finite."""
    s = np.asarray(s, np.float64)
    K = s.size
    vals, counts = np.unique(s, return_counts=True)
    if vals.size == 1 or (vals.size == 2 and counts.min() == 1):
        return None
    d = s - s.mean()
    var = ((d * d).sum() - d * d * (K / (K - 1))) / (K - 1)
    if not (np.isfinite(var).all() and (var > 0).all()):
        return None
    z = (K / (K - 1)) * d / np.sqrt(var)
    return z if np.isfinite(z).all() else None


def norm_order(u, q, gu, a, rows):
    """NO: z of H_t . g_j for the given rows (all >= a); H accumulated from a with the literal pair differences, so
    identical unit frames give H = 0 exactly. Returns ({row: z}, fallback counts)."""
    out, why = {}, {"zero_h": 0, "zero_spread": 0, "loo_undefined": 0}
    if not rows.size:
        return out, why
    H = np.zeros(u.shape[1])
    want = set(rows.tolist())
    for t in range(a, int(rows.max()) + 1):
        if t > a:
            sg = np.sign(q[t] - q[a:t])
            H = H + (sg[:, None] * (u[t] - u[a:t])).sum(0)
        if t not in want:
            continue
        if not H.any():
            why["zero_h"] += 1
            continue
        sc = gu @ H
        if (sc == sc[0]).all():
            why["zero_spread"] += 1
            continue
        z = strict_loo(sc)
        if z is None:
            why["loo_undefined"] += 1
            continue
        out[t] = z
    return out, why


def deletion_z(x, g, J, rows, zPr):
    """DEL: z' = zPr_t - sd over the three one-block deletions of frames J .. t. Returns ({row: z'}, fallback counts)."""
    out, why = {}, {"short": 0, "nonfinite": 0}
    for t in rows:
        n = t - J + 1
        if n < 3:
            why["short"] += 1
            continue
        e = [(b * n) // 3 for b in range(4)]
        zs = []
        for b in range(3):
            keep = np.r_[np.arange(J, J + e[b]), np.arange(J + e[b + 1], t + 1)]
            zs.append(x_fr_hcp._fe(x[keep], g)["zP"][-1])
        Z = np.stack(zs)
        d = np.sqrt(((Z - Z.mean(0)) ** 2).mean(0))
        zz = zPr[t] - d
        if np.isfinite(zz).all():
            out[t] = zz
        else:
            why["nonfinite"] += 1
    return out, why


def cmoments(w, q, c):
    """CUR's curvature correlation R and the quality correlation (E2c) on the positive-weight frames, from centred
    two-pass weighted moments (float64). (None, reason) when undefined: no positive weight, at most two distinct quality
    values (q^2 is then affine in q), a column of c constant over the frames (identical unit frames make every column
    constant) or exactly affine in q, or a non-positive / non-finite residual variance."""
    pos = w > 0
    if not pos.any():
        return None, "zero_weight"
    w, q, c = w[pos], q[pos], c[pos]
    if np.unique(q).size <= 2:
        return None, "q_levels"
    if (c == c[0]).all(0).any():
        return None, "const_col"
    i0, i1 = int(np.argmin(q)), int(np.argmax(q))
    if (((c - c[i0]) * (q[i1] - q[i0])) == ((c[i1] - c[i0]) * (q - q[i0])[:, None])).all(0).any():
        return None, "affine_col"  # a column exactly affine in q (exact cross-product test against the extreme-q pair)
    S0 = w.sum()
    dq = q - (w @ q) / S0
    q2 = q * q
    dq2 = q2 - (w @ q2) / S0
    dc = c - (w @ c) / S0
    Vq = (w @ (dq * dq)) / S0
    Vq2 = (w @ (dq2 * dq2)) / S0
    Cq2q = (w @ (dq2 * dq)) / S0
    Vc = (w @ (dc * dc)) / S0
    Cqc = ((w * dq) @ dc) / S0
    Cq2c = ((w * dq2) @ dc) / S0
    a = Vq2 - Cq2q * Cq2q / Vq
    bb = Vc - Cqc * Cqc / Vq
    if not (np.isfinite(a) and a > 0 and np.isfinite(bb).all() and (bb > 0).all()):
        return None, "residual_var"
    R = (Cq2c - Cq2q * Cqc / Vq) / np.sqrt(a * bb)
    E2c = Cqc / np.sqrt(Vq * Vc)
    if not (np.isfinite(R).all() and np.isfinite(E2c).all()):
        return None, "nonfinite"
    return (R, E2c), None


def curvature(u, q, gu, a, rows):
    """CUR: z per row (rows > a) with the two-stream stability gate (alternating 10-frame blocks since a). Returns
    ({row: (z, E2c z or None)}, fallback counts)."""
    out = {}
    why = {"full_undefined": 0, "full_loo": 0, "stream_count": 0, "stream_undefined": 0, "stream_cos": 0,
           "unstable": 0}
    if not rows.size:
        return out, why
    mu = gu.mean(0)
    last = int(rows.max()) + 1
    C = (u[a:last] - mu) @ (gu - mu).T
    w = e2_weights(u, a, last)
    blk = (np.arange(a, last) - a) // 10 % 2
    for t in rows:
        k = t - a + 1
        ww, qq, cc = w[:k], q[a:t + 1], C[:k]
        full, reason = cmoments(ww, qq, cc)
        if full is None:
            why["full_undefined"] += 1
            continue
        zf = strict_loo(full[0])
        if zf is None:
            why["full_loo"] += 1
            continue
        zs, fail = [], None
        for st in (0, 1):
            msk = blk[:k] == st
            ws = ww[msk]
            sw = ws.sum()
            if not sw > 0 or sw * sw / (ws * ws).sum() < 10:
                fail = "stream_count"
                break
            mres, _ = cmoments(ws, qq[msk], cc[msk])
            zst = None if mres is None else strict_loo(mres[0])
            if zst is None:
                fail = "stream_undefined"
                break
            zs.append(zst)
        if fail:
            why[fail] += 1
            continue
        n0, n1 = np.linalg.norm(zs[0]), np.linalg.norm(zs[1])
        if not (np.isfinite(n0) and np.isfinite(n1) and n0 > 0 and n1 > 0):
            why["stream_cos"] += 1
            continue
        if float(zs[0] @ zs[1]) / (n0 * n1) > 0.5:
            out[int(t)] = (zf, strict_loo(full[1]))
        else:
            why["unstable"] += 1
    return out, why


def conc(z):
    zp = np.maximum(z, 0.0)
    s = (zp * zp).sum(1)
    return np.where(s > 0, (zp.max(1) ** 2) / np.where(s > 0, s, 1.0), 0.0)


def row_ranks(S, c):
    """#{j != c: S[i, j] >= S[i, c]} per row (ties accepted): the row-wise oracle floor in impostor counts."""
    return (S >= S[:, c:c + 1]).sum(1) - 1


def _job(i):
    x, g, c = D["s1"][i], D["g1"], int(D["c1"][i])
    base0 = np.asarray(D["base"][i], np.float32)
    st = E8.state8(base0, x, g)
    s8, state, J, j0, zP, own, lead_ok = st["s8"], st["state"], st["J"], st["j0"], st["zP"], st["own"], st["lead_ok"]
    T = s8.shape[0]
    t = np.arange(T)
    res = E8.entry(s8, c)
    keep = np.ones(g.shape[0], bool)
    keep[c] = False
    thr = D["thr"]
    th32 = np.float32(thr)
    xr = np.asarray(x, np.float64)
    u = B._unit(xr)
    q = np.linalg.norm(xr, axis=1) / 20.0
    gu = B._unit(g)
    info = {}

    def put(k, elig, zz, maps, why, extra=None):
        """Variant k on eligible rows elig: applied rows get maps(row, z); fallback rows keep bank8."""
        new = s8[elig].copy()
        app = np.array(sorted(zz), dtype=int)
        pos = {r: n for n, r in enumerate(elig.tolist())}
        for r in app:
            new[pos[r]] = maps(r, zz[r])
        res["chg"][k] = E8.swap(s8, elig, new, c)
        a_idx = np.array([pos[r] for r in app], dtype=int)
        zt = np.array([zz[r][c] for r in app]) if app.size else np.zeros(0)
        rk = np.array([(zz[r] > zz[r][c]).sum() for r in app]) if app.size else np.zeros(0, int)
        info[k] = {"eligible": int(elig.size), "applied": int(app.size), "why": why, "z_t": zt.tolist(),
                   "rank": rk.tolist(),
                   # all eligible rows (fallback rows keep bank8)
                   "g_old_e": s8[elig, c].astype(np.float64), "i_old_e": s8[elig][:, keep].ravel(),
                   "g_new_e": new[:, c].astype(np.float64), "i_new_e": new[:, keep].ravel(),
                   "r_old_e": int(row_ranks(s8[elig], c).sum()), "r_new_e": int(row_ranks(new, c).sum()),
                   # applied rows only
                   "g_old_a": s8[app, c].astype(np.float64), "i_old_a": s8[app][:, keep].ravel(),
                   "g_new_a": new[a_idx, c].astype(np.float64), "i_new_a": new[a_idx][:, keep].ravel(),
                   "cost_a": float((s8[app, c] < thr).sum() + (s8[app][:, keep] >= th32).sum() / 199.0),
                   "cost_a_by_state": {str(sname): float((s8[app[state[app] == sname], c] < thr).sum()
                                                         + (s8[app[state[app] == sname]][:, keep] >= th32).sum()
                                                         / 199.0) for sname in sorted(set(state[app].tolist()))}}
        if extra:
            info[k].update(extra(app))

    # NO: every flat row; segment from j0 (frame 0 when no jump is detected)
    fr = np.nonzero(state == "flat")[0]
    if fr.size:
        zz, why = norm_order(u, q, gu, j0, fr)
        put("NO", fr, zz, lambda r, z: B7.lin(z, B.KAPPA, 1.0), why)
    # DEL: mid rows after the jump
    mr = np.nonzero((state == "mid") & (t >= j0) & (J > 0))[0]
    if mr.size:
        zz, why = deletion_z(x, g, J, mr, zP)
        put("DEL", mr, zz, lambda r, z: B7.lin(z, B3.KAPPA_M, B7.TAU_M_POST), why)
    # CUR: gated post-jump free rows
    if J > 0:
        ext = (own < 0) & (t >= B2.EXT_FROM) & lead_ok
        held = (own >= 0) | ext
        before = np.concatenate([[False], np.cumsum(held)[:-1] > 0])
        fe = x_fr_hcp._fe(x[j0:], g)
        zS = np.zeros_like(zP)
        zS[j0:] = fe["zS"]
        cand = np.isin(state, ("mid", "late_new", "late_gated", "late_after_held", "flat")) & (t >= 10) & (t > j0)
        cand &= ~before
        diffuse = (conc(zP) <= 0.5) & (conc(zS) <= 0.5)
        rows = np.nonzero(cand & ((state == "flat") | diffuse))[0]
        if rows.size:
            zz2, why = curvature(u, q, gu, j0, rows)
            zz = {r: v[0] for r, v in zz2.items()}

            def cur_map(r, z):
                s_ = state[r]
                if s_ == "mid":
                    return B7.lin(z, B3.KAPPA_M, B7.TAU_M_POST)
                if s_ == "late_new":
                    return B7.lin(z, B.KAPPA, B7.TAU_L7)
                return B7.lin(z, B.KAPPA, 1.0)

            def extra(app):
                e2 = [(zz2[r][0], zz2[r][1]) for r in app if zz2[r][1] is not None]
                return {"cur_z": np.stack([a for a, _ in e2]) if e2 else np.zeros((0, g.shape[0])),
                        "e2c_z": np.stack([b for _, b in e2]) if e2 else np.zeros((0, g.shape[0])),
                        "e2c_missing": int(len(app) - len(e2))}

            put("CUR", rows, zz, cur_map, why, extra)
    res["info"] = info
    res["c"] = c
    return i, res


def auc(pos, neg):
    """Mann-Whitney AUC of pos over neg with average ranks for ties."""
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    v = allv[order]
    ranks = np.empty(v.size)
    _, start, cnt = np.unique(v, return_index=True, return_counts=True)
    for s_, n_ in zip(start, cnt):
        ranks[s_: s_ + n_] = s_ + (n_ + 1) / 2.0
    r = np.empty(v.size)
    r[order] = ranks
    rp = r[: pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def exact(R, keys):
    """Unrounded pooled / subset gains (gen-eq) and per-sequence deltas at bank8's threshold."""
    n = len(R)
    ids = list(range(n))
    imp = np.concatenate([o["imp"] for o in R])
    imp.sort()
    e8, thr = E7._ev(R, None, ids, imp)
    out = {"bank8": e8, "thr": thr, "keys": {k: {"pooled": (e8 - E7._ev(R, k, ids, imp)[0]) * E7.GEQ} for k in keys}}
    del imp
    th32 = np.float32(thr)
    cost = np.array([(o["gen"] < thr).sum() + (o["imp"] >= th32).sum() / 199.0 for o in R])
    order = np.argsort(-cost)
    out["top_cost"] = [[int(j), float(cost[j])] for j in order[:3]]
    sub = {"even": list(range(0, n, 2)), "odd": list(range(1, n, 2)),
           "minus_top1": sorted(set(ids) - set(order[:1].tolist())),
           "minus_top3": sorted(set(ids) - set(order[:3].tolist()))}
    for sname, sids in sub.items():
        simp = np.concatenate([R[j]["imp"] for j in sids])
        simp.sort()
        b = E7._ev(R, None, sids, simp)[0]
        for k in keys:
            out["keys"][k].setdefault("subsets", {})[sname] = (b - E7._ev(R, k, sids, simp)[0]) * E7.GEQ
        del simp
    top2 = set(order[:2].tolist())
    for k in keys:
        d = np.zeros(n)
        for j, o in enumerate(R):
            if k in o["chg"] and o["chg"][k][0].size:
                rows, gnew, iold, inew = o["chg"][k]
                d[j] = ((o["gen"][rows] < thr).sum() + (iold >= th32).sum() / 199.0
                        - (gnew < thr).sum() - (inew >= th32).sum() / 199.0)
        rest = np.array([d[j] for j in range(n) if j not in top2])
        out["keys"][k].update({"fixed_net": float(d.sum()), "changed_seqs": int((d != 0).sum()),
                               "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
                               "largest": sorted(d.tolist(), reverse=True)[:3], "smallest": sorted(d.tolist())[:3],
                               "max_loss_outside_top2": float(max(0.0, -rest.min())) if rest.size else 0.0,
                               "top2_deltas": [float(d[j]) for j in order[:2]]})
    return out


def main(zpath, workers=24):
    t0 = time.time()
    E8.load(zpath)
    D["thr"] = -0.09978974610567093  # bank8's proxy EER threshold
    R = E8.run(_job, workers)
    keys = [k for k in KEYS if any(k in o["info"] for o in R)]
    ex = exact(R, keys)
    top2 = {j for j, _ in ex["top_cost"][:2]}
    res = {"bank8": ex["bank8"], "thr": ex["thr"], "top_cost": ex["top_cost"], "variants": {}}
    for k in keys:
        parts = [(j, o["info"][k]) for j, o in enumerate(R) if k in o["info"]]
        P = [p for _, p in parts]
        cat = lambda f: np.concatenate([p[f] for p in P])  # noqa: E731
        rk = np.concatenate([np.asarray(p["rank"], float) for p in P])
        zt = np.concatenate([np.asarray(p["z_t"], float) for p in P])
        why = {}
        for p in P:
            for a, b in p["why"].items():
                why[a] = why.get(a, 0) + b
        fo, fn = floor_geq(cat("g_old_e"), cat("i_old_e")), floor_geq(cat("g_new_e"), cat("i_new_e"))
        fao, fan = floor_geq(cat("g_old_a"), cat("i_old_a")), floor_geq(cat("g_new_a"), cat("i_new_a"))
        ro, rn = sum(p["r_old_e"] for p in P), sum(p["r_new_e"] for p in P)
        cost_a = sum(p["cost_a"] for p in P)
        by_rank = {"seq1": 0.0, "seq2": 0.0, "rest": 0.0}
        top_order = [j for j, _ in ex["top_cost"][:2]]
        by_state = {}
        for j, p in parts:
            key = "seq1" if j == top_order[0] else "seq2" if j == top_order[1] else "rest"
            by_rank[key] += p["cost_a"]
            for a, b in p["cost_a_by_state"].items():
                by_state[a] = by_state.get(a, 0.0) + b
        v = {"eligible_rows": int(sum(p["eligible"] for p in P)), "applied_rows": int(rk.size), "seqs": len(P),
             "fallback": why, "target_mean_z": float(zt.mean()) if zt.size else None,
             "target_median_rank": float(np.median(rk)) if rk.size else None,
             "floor_single_eligible": [fo, fn, fo - fn], "floor_single_applied": [fao, fan, fao - fan],
             "floor_row_oracle_eligible": [ro / 199.0, rn / 199.0, (ro - rn) / 199.0],
             "applied_cost_bank8": cost_a, "applied_cost_by_seq_rank": by_rank, "applied_cost_by_state": by_state}
        v.update(ex["keys"][k])
        if k == "CUR":
            cz = np.concatenate([p["cur_z"] for p in P if p["cur_z"].size]) if any(p["cur_z"].size for p in P) else None
            if cz is not None:
                ez = np.concatenate([p["e2c_z"] for p in P if p["e2c_z"].size])
                tg = np.concatenate([np.eye(cz.shape[1], dtype=bool)[R[j]["c"]][None, :].repeat(p["cur_z"].shape[0], 0)
                                     for j, p in parts if p["cur_z"].size])
                edges = np.quantile(ez.ravel(), [.2, .4, .6, .8])
                bins = np.digitize(ez.ravel(), edges)
                cr, tr = cz.ravel(), tg.ravel()
                aucs, wts = [], []
                for b in range(5):
                    m = bins == b
                    if (m & tr).any() and (m & ~tr).any():
                        aucs.append(auc(cr[m & tr], cr[m & ~tr]))
                        wts.append(int((m & tr).sum()))
                v["e2c_quintile_auc"] = aucs
                v["e2c_quintile_targets"] = wts
                v["e2c_weighted_auc"] = float(np.average(aucs, weights=wts)) if wts else None
            # the conditional gate is certified only when every applied row has a defined E2c z
            v["e2c_missing_rows"] = int(sum(p["e2c_missing"] for p in P))
            v["conditional_gate_certified"] = v["e2c_missing_rows"] == 0
        res["variants"][k] = v
    # kill gates, evaluated on unrounded values
    gates = {}
    if "NO" in res["variants"]:
        v = res["variants"]["NO"]
        gates["NO"] = {"pooled_lt_20": v["pooled"] < 20, "fixed_le_0": v["fixed_net"] <= 0,
                       "floor_gain_le_0": v["floor_single_eligible"][2] <= 0}
    if "DEL" in res["variants"]:
        v = res["variants"]["DEL"]
        gates["DEL"] = {"row_floor_gain_lt_30": v["floor_row_oracle_eligible"][2] < 30, "pooled_lt_20": v["pooled"] < 20,
                        "even_lt_0": v["subsets"]["even"] < 0, "odd_lt_0": v["subsets"]["odd"] < 0,
                        "minus_top3_lt_0": v["subsets"]["minus_top3"] < 0}
    if "CUR" in res["variants"]:
        v = res["variants"]["CUR"]
        gates["CUR"] = {"applied_cost_lt_50": v["applied_cost_bank8"] < 50,
                        "floor_gain_lt_50": v["floor_single_applied"][2] < 50, "pooled_lt_30": v["pooled"] < 30,
                        "loss_outside_top2_gt_10": v["max_loss_outside_top2"] > 10,
                        "no_conditional_info": not (v.get("e2c_weighted_auc") is not None
                                                    and v["e2c_weighted_auc"] > 0.5
                                                    and v["conditional_gate_certified"])}
    for k, gk in gates.items():
        gk = {a: bool(b) for a, b in gk.items()}
        res["variants"][k]["gates"] = gk
        res["variants"][k]["verdict"] = "KILL" if any(gk.values()) else "SURVIVES"
    res["top2"] = sorted(top2)
    res["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(res, default=float), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 24)
