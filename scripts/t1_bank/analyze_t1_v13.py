"""T1 premise check of round J (three `gpt-6-astra` high sessions): non-negative gallery decomposition (NNLS) as a new
ranking signal, output by rank transplant on top of bank8.

Aggregates only; pseudo targets are evaluator-only. Common pieces (fixed by the proposers before this check):
  G = unit gallery rows, C = G G^T, L = largest eigenvalue of C. For a vector c = G v, the statistic is the output of a
  FIXED number n of projected-gradient steps from zero, a <- max(0, a + (c - C a) / L) (not a converged NNLS solution).
  Rank transplant: a changed row keeps bank8's 200 values and re-assigns them in descending order to the columns in
  descending order of the new signal r (ties of r broken by bank8's value, then by column index), so every row keeps
  its accept count at every threshold. Non-finite r or a constant r keeps the bank8 row. Decisions, ownership, runs,
  the extension and the row states are bank8's.
Variants (each with the kill rule its proposer fixed; all pooled gains unrounded, threshold -0.09978974610567093):
  NNm (J1)  rows t < 60, jump not yet detected at t (t < J), the row neither owned nor extended; r = n-step NNLS (n 64) of the
            cumulative mean m_t = (1 / (t + 1)) sum_{s <= t} u_s. KILL: pooled < 100, fixed <= 0, summed fixed losses
            outside #170 / #99 > 10, fixed gain on rows where bank8's leader is wrong <= 0, or net fixed gain on rows
            where it is right < 0.
  NNf (J2)  rows t >= J (a jump detected), t >= 10, bank8 states mid / late_new / flat / late_gated / late_after_held;
            r = mean over s = J .. t of the per-frame n-step NNLS (n 32) of u_s. KILL: pooled < 100, fixed <= 0, summed
            fixed losses outside #170 / #99 > 10.
  NNw (J3)  rows owner-less and not extended with t < 70, plus bank8's late-step rows (late_new); r = weighted mean over
            the current segment (from 0; from J once the jump is detected) of the per-frame n-step NNLS (n 64) of u_s,
            weights w_b = 1 at the segment's first frame, w_s = (1 - rho_s) / (1 + rho_s), rho_s = clip(u_s . u_{s-1},
            0, 1) (bank8's flat-row weights). KILL: pooled < 100, fixed <= 0, summed fixed losses outside #170 / #99
            > 10, or the target's rank sum over rows where bank8's leader is wrong does not improve.
Usage (server): python -B scripts/t1_bank/analyze_t1_v13.py PARENT_ZIP [workers] > log
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

import analyze_t1_v6b as V6  # noqa: E402
import t1_bank as B  # noqa: E402
import t1_bank2 as B2  # noqa: E402
import t1_eval8 as E8  # noqa: E402

D = E8.D
KEYS = ("NNm", "NNf", "NNw")
HARD2 = (170, 99)
THR = -0.09978974610567093


def nnls_steps(Cv, C, L, n):
    """n projected-gradient steps from zero for every row of Cv (rows = vectors G v): (m x K) float64."""
    A = np.zeros_like(Cv)
    for _ in range(n):
        A = np.maximum(0.0, A + (Cv - A @ C) / L)
    return A


def transplant(s_row, r):
    """bank8's row values re-assigned by r's order (ties: bank8 value, then column); None to keep the row."""
    if not np.isfinite(r).all() or np.all(r == r[0]):
        return None
    K = r.size
    order = np.lexsort((np.arange(K), -s_row.astype(np.float64), -r))  # columns by r desc, s desc, index asc
    vals = np.sort(s_row)[::-1]
    out = np.empty_like(s_row)
    out[order] = vals
    return out


def _rank(row, c):
    return int((row > row[c]).sum())


def _job(i):
    x, g, c = D["s1"][i], D["g1"], int(D["c1"][i])
    base0 = np.asarray(D["base"][i], np.float32)
    st = E8.state8(base0, x, g)
    s8, state, J, own, lead_ok = st["s8"], st["state"], st["J"], st["own"], st["lead_ok"]
    T = s8.shape[0]
    t = np.arange(T)
    res = E8.entry(s8, c)
    ext = (own < 0) & (t >= B2.EXT_FROM) & lead_ok
    held = (own >= 0) | ext
    u = B._unit(np.asarray(x, np.float64))
    G = B._unit(np.asarray(g, np.float64))
    C = G @ G.T
    L = float(np.linalg.eigvalsh(C)[-1])
    jd = J if J > 0 else T  # first row at which the jump is detected (T: never)
    lead_right = np.array([s8[r, c] > np.delete(s8[r], c).max() for r in range(T)])
    rows = {"NNm": np.nonzero((t < 60) & (t < jd) & ~held)[0],
            "NNf": np.nonzero((t >= jd) & (t >= 10) & np.isin(state, ("mid", "late_new", "flat", "late_gated",
                                                                         "late_after_held")))[0],
            "NNw": np.nonzero((~held & (t < 70)) | (state == "late_new"))[0]}
    info = {}
    for k in KEYS:
        rk = rows[k]
        if not rk.size:
            continue
        top = int(rk.max()) + 1
        if k == "NNm":
            m = np.cumsum(u[:top], axis=0) / (np.arange(top)[:, None] + 1.0)
            R = nnls_steps(m[rk] @ G.T, C, L, 64)
        else:
            n = 32 if k == "NNf" else 64
            Af = nnls_steps(u[:top] @ G.T, C, L, n)
            if k == "NNf":
                S = np.cumsum(Af[jd:top], axis=0) if jd < top else None
                R = np.stack([S[r - jd] / (r - jd + 1.0) for r in rk]) if S is not None else None
            else:
                rho = np.clip(np.einsum("ij,ij->i", u[1:top], u[:top - 1]), 0.0, 1.0)
                w = np.concatenate([[1.0], (1.0 - rho) / (1.0 + rho)])
                R = np.empty((rk.size, G.shape[0]))
                for a, seg_rows in ((0, rk[rk < jd]), (jd, rk[rk >= jd])):
                    if not seg_rows.size:
                        continue
                    ww = w[a:top].copy()
                    ww[0] = 1.0  # the segment's first frame
                    Sw = np.cumsum(Af[a:top] * ww[:, None], axis=0)
                    Wc = np.cumsum(ww)
                    for r in seg_rows:
                        R[np.searchsorted(rk, r)] = Sw[r - a] / Wc[r - a]
        new = s8[rk].copy()
        kept = 0
        for n_, r in enumerate(rk):
            o = transplant(s8[r], R[n_]) if R is not None else None
            if o is None:
                kept += 1
            else:
                new[n_] = o
        res["chg"][k] = E8.swap(s8, rk, new, c)
        wrong = ~lead_right[rk]
        g_old = s8[rk, c] >= THR
        g_new = new[:, c] >= THR
        info[k] = {"rows": int(rk.size), "kept": kept, "wrong_rows": int(wrong.sum()),
                   "rank_old_wrong": sum(_rank(s8[r], c) for r in rk[wrong]),
                   "rank_new_wrong": sum(_rank(new[j], c) for j in np.nonzero(wrong)[0]),
                   "rank_old_right": sum(_rank(s8[r], c) for r in rk[~wrong]),
                   "rank_new_right": sum(_rank(new[j], c) for j in np.nonzero(~wrong)[0]),
                   "A_wrong": int((~g_old & g_new & wrong).sum()), "B_wrong": int((g_old & ~g_new & wrong).sum()),
                   "A_right": int((~g_old & g_new & ~wrong).sum()), "B_right": int((g_old & ~g_new & ~wrong).sum())}
    res["info"] = info
    return i, res


def main(zpath, workers=24):
    t0 = time.time()
    E8.load(zpath)
    D["thr"] = THR
    R = E8.run(_job, workers)
    keys = [k for k in KEYS if any(k in o["info"] for o in R)]
    ex = V6.exact(R, keys)
    top2 = [j for j, _ in ex["top_cost"][:2]]
    res = {"bank8": ex["bank8"], "thr": ex["thr"], "thr_equals_fixed": bool(ex["thr"] == THR),
           "top_cost": ex["top_cost"], "hard2_matches": bool(sorted(top2) == sorted(HARD2)), "variants": {}}
    for k in keys:
        P = [o["info"][k] for o in R if k in o["info"]]
        s = {f: sum(p[f] for p in P) for f in P[0]}
        d = np.zeros(len(R))
        for j, o in enumerate(R):
            if k in o["chg"] and o["chg"][k][0].size:
                rows, gnew, iold, inew = o["chg"][k]
                d[j] = ((o["gen"][rows] < THR).sum() + (iold >= np.float32(THR)).sum() / 199.0
                        - (gnew < THR).sum() - (inew >= np.float32(THR)).sum() / 199.0)
        outside = np.array([d[j] for j in range(len(R)) if j not in HARD2])
        v = dict(s)
        v.update(ex["keys"][k])
        v["summed_loss_outside_top2"] = float(np.maximum(0.0, -outside).sum())
        v["fixed_wrong"] = 200.0 / 199.0 * (s["A_wrong"] - s["B_wrong"])
        v["fixed_right"] = 200.0 / 199.0 * (s["A_right"] - s["B_right"])
        gates = {"pooled_lt_100": v["pooled"] < 100, "fixed_le_0": v["fixed_net"] <= 0,
                 "summed_loss_outside_top2_gt_10": v["summed_loss_outside_top2"] > 10}
        if k == "NNm":
            gates.update(fixed_wrong_le_0=v["fixed_wrong"] <= 0, fixed_right_lt_0=v["fixed_right"] < 0)
        if k == "NNw":
            gates["wrong_rank_sum_not_improved"] = s["rank_new_wrong"] >= s["rank_old_wrong"]
        v["gates"] = {a: bool(b) for a, b in gates.items()}
        v["verdict"] = "KILL" if any(v["gates"].values()) else "SURVIVES"
        res["variants"][k] = v
    res["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(res, default=float), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 24)
