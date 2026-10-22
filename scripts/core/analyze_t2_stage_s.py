"""T2 high-band stage S: an early, frozen residual template beside rtpl's cumulative one (analysis only).

Plan, provenance and frozen rules: docs/experiments/20261012_ogata_t2_stage_s.md.

H0 = rtpl_pez = out(oE oV oEn fac(ratio_prefix(q))) as in analyze_t2_stage_r (q = cos(quality-EMA residual, the
column's cumulative template of verified-clean unit residuals), analyze_round2.rtpl_q). The early template of column j is
the sum of its first N_EARLY verified-clean unit residuals, then frozen; q_e is the same cosine against it (defined
exactly where q is, since both templates start at the column's first verified-clean frame). Cells:
  etx   q' = max(q, q_e)     release where the frame matches the early template better than the cumulative one
  ete   q' = q_e             the early template alone
  etn   q' = min(q, q_e)     reject where the frame mismatches the early template although it matches the cumulative one
Everything else (ratio against the prefix mean over defined frames, oen, act_of, the PEZ base) is H0's. One constant,
N_EARLY = 100 verified-clean frames, fixed before any run. N_EARLY = infinity gives H0 for every cell (synth check).
Every cell is also composed as the 976810 recipe, veto_band(H', L = bank8, b = 943955).
Pseudo targets on the evaluator side only (T1 targets, T2 claimed columns). Prints JSON aggregates only.
Usage (server): python scripts/core/analyze_t2_stage_s.py [workers<=6] > results/ogata/t2_stage_s.log
                python scripts/core/analyze_t2_stage_s.py --synth
"""
import json
import multiprocessing as mp
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "t1_bank"))
import x_r5  # noqa: E402,F401  (imports common_check first: pins BLAS threads)
import numpy as np  # noqa: E402

import analyze_round2 as AR2  # noqa: E402
import analyze_t1_stage3_bank as S3  # noqa: E402
import analyze_t2_family as T2F  # noqa: E402
import analyze_t2_stage_h as SH  # noqa: E402
import t1_bank8 as B8  # noqa: E402
import x_r4 as R4  # noqa: E402
import x_r5 as R5  # noqa: E402
from rcfa.methods.gallery_discriminative_row_l2 import gallery_discriminative_row_l2_scores  # noqa: E402
from rcfa.methods.gallery_subspace_residual import GallerySubspace  # noqa: E402

N_EARLY = 100
CELLS = ["etx", "ete", "etn"]
HS = ["h0"] + CELLS
CMP = {k: "cmp_" + k for k in HS}
HI = SH.HI
H0_REC = SH.H0_REC
CMP_REC = SH.CMP_REC
GEN_EQ = SH.GEN_EQ
LATE = 200
D = {}


def rtpl_q_early(P, g, n_early=N_EARLY):
    """analyze_round2.rtpl_q's q (same operations, same order) and q_e against the early frozen template."""
    al, nrm = P["al"], P["nrm"]
    xb = P["xb"]
    u = xb / np.maximum(nrm, 1e-12)[:, None]
    B = np.asarray(GallerySubspace(np.asarray(g, np.float32)).basis, np.float64)
    rr = u - (u @ B.T) @ B
    rn_ = np.linalg.norm(rr, axis=1)
    rh = np.where(rn_[:, None] > 0, rr / np.maximum(rn_, 1e-300)[:, None], 0.0)
    R = T2F.q_ema(rh, al)
    T, K = P["v"].shape
    q = np.zeros((T, K))
    qe = np.zeros((T, K))
    dfn = np.zeros((T, K), bool)
    tmpl, tmpe, cnt = {}, {}, {}
    vc = P["v"] & P["clean"][:, None]
    for t in range(T):
        Rn = np.linalg.norm(R[t])
        for j, s in tmpl.items():
            sn = np.linalg.norm(s)
            if Rn > 0 and sn > 0:
                q[t, j] = max(0.0, float(R[t] @ s) / (Rn * sn))
                dfn[t, j] = True
                se = tmpe[j]
                sen = np.linalg.norm(se)
                qe[t, j] = max(0.0, float(R[t] @ se) / (Rn * sen)) if sen > 0 else q[t, j]
        for j in np.nonzero(vc[t])[0]:
            j = int(j)
            tmpl[j] = tmpl.get(j, 0.0) + rh[t]
            if cnt.get(j, 0) < n_early:
                tmpe[j] = tmpe.get(j, 0.0) + rh[t]
                cnt[j] = cnt.get(j, 0) + 1
    return q, qe, dfn


def variants_h(x, g, P=None, n_early=N_EARLY):
    """float64 H0 and the cells for one sequence."""
    P = T2F.parts(x, g) if P is None else P
    K, zP, zS, LP, LS, oE, oV = P["K"], P["zP"], P["f"]["zS"], P["LP"], P["LS"], P["oE"], P["oV"]

    def out(om):
        return R4.fco_core(R5._numN(LP, LS, om), P["den"], P["C"], P["M"], K)[0]

    def fac(rt):
        return T2F.oen(rt, zS, K, T2F.act_of(rt, LS, zP, oV))

    q, qe, dfn = rtpl_q_early(P, g, n_early)
    base = oE * oV * P["oEn"]
    return {"h0": out(base * fac(T2F.ratio_prefix(q, defined=dfn))),
            "etx": out(base * fac(T2F.ratio_prefix(np.maximum(q, qe), defined=dfn))),
            "ete": out(base * fac(T2F.ratio_prefix(qe, defined=dfn))),
            "etn": out(base * fac(T2F.ratio_prefix(np.minimum(q, qe), defined=dfn)))}


def outputs_fn(x, g, with_low=True):
    out = {k: v.astype(np.float32) for k, v in variants_h(x, g).items()}
    if with_low:
        b32 = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
        L = np.asarray(B8.bank8(x, g), np.float32)
        for k in HS:
            out[CMP[k]] = S3.veto_band(out[k], L, b32)
    return out


def composite_h(x, g, cell):
    """The build path of one cell: veto_band(H', bank8p, b), as the teammate's bal_or75v_b8.bal with H' for H."""
    import t1_bank8p as B8P
    H = variants_h(x, g)[cell].astype(np.float32)
    b32 = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    return S3.veto_band(H, np.asarray(B8P.bank8p(x, g), np.float32), b32)


def _job(task):
    tr, i = task
    x, g = D[f"s{tr}"][i], D[f"g{tr}"]
    o = outputs_fn(x, g, with_low=(tr == 1))
    zchk = (0, 0)
    Z = D.get(f"Z{tr}")
    if Z is not None:
        ref = Z[i]
        h = o["h0"]
        m = (h >= np.float32(HI)) | (ref >= np.float32(HI))
        zchk = (int(m.sum()), int((h[m] != ref[m]).sum()))
    inv = {}
    if tr == 1:
        for k in HS:
            H, C = o[k], o[CMP[k]]
            hi = H.astype(np.float64) >= HI
            inv[k] = (int((C[hi] != H[hi]).sum()), float(C[~hi].max()) if (~hi).any() else -1.0)
        return tr, i, o, zchk, inv
    c = int(D["c2"][i])
    return tr, i, {k: v[:, c].copy() for k, v in o.items()}, zchk, inv


def main(workers=6):
    import analyze_hvc_stage0 as HV
    import analyze_t1_residual as ar
    import common_check as cc
    from proxy_lib import load_track, pseudo_targets
    t0 = time.time()
    g1, s1, n1 = load_track("track1")
    g2, s2, n2 = load_track("track2")
    c1, _ = pseudo_targets(g1, s1)
    c2, _ = pseudo_targets(g2, s2)
    D.update(s1=s1, g1=g1, s2=s2, g2=g2, c1=c1, c2=c2)
    S = HV._read_zip(str(HERE.parents[1] / SH.ZIP_973452))
    D["Z1"] = [S["track1"][f"score_{n}.npz"] for n in n1]
    D["Z2"] = [S["track2"][f"score_{n}.npz"] for n in n2]
    del S
    tasks = [(1, i) for i in range(len(s1))] + [(2, i) for i in range(len(s2))]
    R = {1: {}, 2: {}}
    zc = [0, 0]
    inv = {k: [0, -1.0] for k in HS}
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 6)), mp_context=mp.get_context("fork")) as ex:
        for tr, i, o, z, iv in ex.map(_job, tasks, chunksize=1):
            R[tr][i] = o
            zc[0] += z[0]
            zc[1] += z[1]
            for k, (m_, mx) in iv.items():
                inv[k][0] += m_
                inv[k][1] = max(inv[k][1], mx)
    D.pop("Z1")
    D.pop("Z2")
    n1s, n2s = len(s1), len(s2)
    rng_ok = all(bool(np.isfinite(v).all() and (np.abs(v) <= 1).all()) for tr in (1, 2) for o in R[tr].values()
                 for v in o.values())
    print(json.dumps({"stage": "outputs", "seconds": round(time.time() - t0, 1), "finite_in_range": rng_ok,
                      "h0_vs_973452_zip_hi_entries": zc[0], "h0_vs_973452_zip_mismatch": zc[1],
                      "composite_hi_changed": {k: v[0] for k, v in inv.items()},
                      "composite_low_max": {k: v[1] for k, v in inv.items()}}), flush=True)
    t1 = {}
    for k in HS + [CMP[k] for k in HS]:
        gen, imp = ar._gen_imp([R[1][i][k] for i in range(n1s)], c1)
        m = ar._metrics(gen, imp)
        t1[k] = {"eer_full": cc._eer_full(gen, imp), "far1e5_thr": m["far1e5_thr"], "frr_far1e5": m["frr_at_far1e5"],
                 "far_at_frr1e5": m["far_at_frr1e5"]}
        del gen, imp
        print(json.dumps({"setting": k, **t1[k]}), flush=True)
    col = {1: {k: [R[1][i][k][:, int(c1[i])] for i in range(n1s)] for k in HS},
           2: {k: [R[2][i][k] for i in range(n2s)] for k in HS}}
    acc = {tr: {k: [v >= np.float32(t1[k]["far1e5_thr"]) for v in col[tr][k]] for k in HS} for tr in (1, 2)}
    rep = {}
    for k in CELLS:
        A0 = np.concatenate(acc[2]["h0"])
        Ak = np.concatenate(acc[2][k])
        d2 = float(A0.mean() - Ak.mean())
        g1 = t1[k]["frr_far1e5"] - t1["h0"]["frr_far1e5"]
        r = {"d2_drop": d2, "g1": g1, "thr_shift": t1[k]["far1e5_thr"] - t1["h0"]["far1e5_thr"],
             "cmp_eer_vs_control_gen_eq": (t1[CMP[k]]["eer_full"] - t1[CMP["h0"]]["eer_full"]) / GEN_EQ}
        for tr in (1, 2):
            dn = up = dn_late = up_late = seq_ch = 0
            for a, b in zip(acc[tr]["h0"], acc[tr][k]):
                late = np.arange(a.shape[0]) >= LATE
                dn += int((a & ~b).sum())
                up += int((~a & b).sum())
                dn_late += int((a & ~b & late).sum())
                up_late += int((~a & b & late).sum())
                seq_ch += int((a != b).any())
            r[f"t{tr}"] = {"down": dn, "up": up, "down_late": dn_late, "up_late": up_late, "seq_changed": seq_ch}
        rep[k] = r
        print(json.dumps({"cell": k, **r}), flush=True)
    stop = {"finite_in_range": rng_ok, "h0_matches_973452_zip": zc[1] == 0 and zc[0] > 0,
            "h0_thr_recorded": t1["h0"]["far1e5_thr"] == H0_REC["far1e5_thr"],
            "h0_frr_recorded": abs(t1["h0"]["frr_far1e5"] - H0_REC["frr_far1e5"]) <= 1e-12,
            "control_cmp_reproduces_976810_proxy": abs(t1[CMP["h0"]]["eer_full"] - CMP_REC) <= 1e-12,
            "composite_hi_unchanged": all(v[0] == 0 for v in inv.values()),
            "composite_low_below_075": all(v[1] < HI for v in inv.values())}
    gates = {}
    for k in CELLS:
        m = rep[k]
        gates[k] = {"thr": abs(m["thr_shift"]) <= 1e-4, "frr_cap": t1[k]["frr_far1e5"] <= 0.333,
                    "cmp": m["cmp_eer_vs_control_gen_eq"] <= 5.0,
                    "effect": m["t2"]["down"] + m["t2"]["up"] >= 500}
    verdict = {k: ("ELIGIBLE" if all(gates[k].values()) else "KILL") for k in CELLS}
    if not all(stop.values()):
        verdict = "STOP"
    print(json.dumps({"stop": stop, "gates": gates, "verdict": verdict, "seconds": round(time.time() - t0, 1)}),
          flush=True)
    return 0 if all(stop.values()) else 2


def synth():
    ok = {}
    for seed, kind in ((3, "switch"), (5, "mng"), (7, "switch")):
        x, g = R4._synth_data(seed, 240, 60, kind, norm_hi=26.0)
        P = T2F.parts(x, g)
        V = variants_h(x, g, P)
        ref = AR2.variants2(x, g, P, only=["rtpl_pez"])["rtpl_pez"]
        ok[f"h0_equals_rtpl_pez_{seed}"] = bool(np.array_equal(V["h0"], ref))
        Vinf = variants_h(x, g, P, n_early=10 ** 9)
        ok[f"n_inf_equals_h0_{seed}"] = all(bool(np.array_equal(Vinf[k], V["h0"])) for k in CELLS)
        q, qe, dfn = rtpl_q_early(P, g)
        q0, _, d0 = AR2.rtpl_q(P, g)
        ok[f"q_equals_rtpl_q_{seed}"] = bool(np.array_equal(q, q0) and np.array_equal(dfn, d0))
        ok[f"finite_in_range_{seed}"] = all(bool(np.isfinite(v).all() and (np.abs(v) <= 1).all()) for v in V.values())
        Vs = variants_h(x, g, P, n_early=10)
        ok[f"small_n_changes_{seed}"] = bool(any((Vs[k] != Vs["h0"]).any() for k in CELLS))
        o = outputs_fn(x, g)
        for k in HS:
            H, C = o[k], o[CMP[k]]
            hi = H.astype(np.float64) >= HI
            ok[f"{k}_cmp_hi_{seed}"] = bool(np.array_equal(C[hi], H[hi]) and ((~hi).sum() == 0 or C[~hi].max() < HI))
        for h in (30, 75, 150):
            oh = outputs_fn(x[:h], g)
            ok[f"prefix_exact_{seed}_{h}"] = all(bool(np.array_equal(oh[k], o[k][:h])) for k in o)
        for cell in CELLS:
            ok[f"{cell}_build_path_{seed}"] = bool(np.array_equal(composite_h(x, g, cell), o[CMP[cell]]))
        print(json.dumps({"seed": seed, "verified_clean_max_per_col": int((P["v"] & P["clean"][:, None]).sum(0).max()),
                          "changed_entries_n100": {k: int((V[k] != V["h0"]).sum()) for k in CELLS},
                          "changed_entries_n10": {k: int((Vs[k] != Vs["h0"]).sum()) for k in CELLS}}), flush=True)
    print(json.dumps({"synth": "PASS" if all(ok.values()) else "FAIL", "failed": [k for k, v in ok.items() if not v]}))
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    if "--synth" in sys.argv:
        sys.exit(synth())
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 6))
