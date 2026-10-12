"""T2 high-band stage H: three H candidates on rtpl_pez with label-free gates, composed over bank8 (analysis only).

Plan, provenance and frozen rules: docs/experiments/20261011_ogata_t2_stage_h.md.

H0 = rtpl_pez (973452's / 976810's high band): out(oE oV oEn fr), fr = oen(rt, zS, K, act_of(rt, LS, zP, oV)),
rt = rtpl's prefix ratio (analyze_round2.rtpl_q). Cells (no new constant):
  mat   MAT (10/10 critique): rtpl's discount acts only once column j has at least ceil(1/al_t) verified clean frames
        strictly before t: out(oE oV oEn oen(rt, zS, K, act_of(rt, LS, zP, oV) & (cnt >= ceil(1/al_t))))
  s4    S4 (10/10 critique): rtpl_pez computed from the two-sided Cauchy P (analyze_t1_cau2_chain.two_sided)
  nst   NST (Opus lens B, 10/10): a negative residual template N_j collects verified clean frames inside runs that
        rtpl discounts (fr < 1 and the memoryless ratio rm < 1) once the run weight W_j (sum of al over the run) is at
        least 1; the cue c = a / b when the frame matches N_j better than rtpl's template (b > a), else 1;
        out(oE oV oEn fr oen(c, zS, K, act_of(c, LS, zP, oV)))
Every cell is also composed as the 976810 recipe, veto_band(H', L = bank8, b = 943955) (control: 976810's proxy).
Pseudo targets on the evaluator side only (T1 targets, T2 claimed columns). Prints JSON aggregates only.
Usage (server): python scripts/ogata/analyze_t2_stage_h.py [workers<=6] > results/ogata/t2_stage_h.log
                python scripts/ogata/analyze_t2_stage_h.py --synth
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
sys.path.insert(0, str(HERE.parent / "final_stretch"))
import x_r5  # noqa: E402,F401  (imports common_check first: pins BLAS threads)
import numpy as np  # noqa: E402

import analyze_round2 as AR2  # noqa: E402
import analyze_t1_cau2_chain as C2  # noqa: E402
import analyze_t1_stage3_bank as S3  # noqa: E402
import analyze_t2_family as T2F  # noqa: E402
import t1_bank8 as B8  # noqa: E402
import x_r4 as R4  # noqa: E402
import x_r5 as R5  # noqa: E402
from rcfa.methods.gallery_discriminative_row_l2 import gallery_discriminative_row_l2_scores  # noqa: E402
from rcfa.methods.gallery_subspace_residual import GallerySubspace  # noqa: E402

CELLS = ["mat", "s4", "nst"]
HS = ["h0"] + CELLS
CMP = {k: "cmp_" + k for k in HS}
HI = 0.75
H0_REC = {"far1e5_thr": 0.8431214690208435, "frr_far1e5": 0.28247691534194724}   # rtpl_pez, local convention
CMP_REC = 0.0021734726698571204                                                   # 976810's proxy (or75v_b8)
ZIP_973452 = "results/ogata/artifacts/tfr2_rtpl_pez/submission.zip"
GEN_EQ = 1.0 / (2 * 489286)
POS = [(0, 100), (100, None)]
D = {}


def nst_cue(P, g, dfn, fr, rm):
    """NST's cue c (T x K) and, per column, whether N_j ever formed. Same residual arithmetic as rtpl_q."""
    al, nrm, xb = P["al"], P["nrm"], P["xb"]
    u = xb / np.maximum(nrm, 1e-12)[:, None]
    B = np.asarray(GallerySubspace(np.asarray(g, np.float32)).basis, np.float64)
    rr = u - (u @ B.T) @ B
    rn_ = np.linalg.norm(rr, axis=1)
    rh = np.where(rn_[:, None] > 0, rr / np.maximum(rn_, 1e-300)[:, None], 0.0)
    T, K = P["v"].shape
    vc = P["v"] & P["clean"][:, None]
    tmpl, neg = {}, {}
    W = np.zeros(K)
    c = np.ones((T, K))
    for t in range(T):
        r = rh[t]
        rnorm = np.linalg.norm(r)
        if rnorm > 0:
            for j, N in neg.items():
                s = tmpl.get(j)
                if s is None:
                    continue
                sn, Nn = np.linalg.norm(s), np.linalg.norm(N)
                if sn > 0 and Nn > 0:
                    a = max(0.0, float(r @ s)) / sn
                    b = max(0.0, float(r @ N)) / Nn
                    if b > a:
                        c[t, j] = a / b
        m = dfn[t] & (fr[t] < 1.0) & (rm[t] < 1.0)
        W = np.where(m, W + al[t], 0.0)
        for j in np.nonzero(vc[t] & m & (W >= 1.0))[0]:
            neg[int(j)] = neg.get(int(j), 0.0) + r
        for j in np.nonzero(vc[t])[0]:
            tmpl[int(j)] = tmpl.get(int(j), 0.0) + r
    formed = np.zeros(K, bool)
    formed[list(neg)] = True
    return c, formed


def variants_h(x, g, P=None):
    """float64 H0 and the three cells for one sequence, plus NST's formation flags."""
    P = T2F.parts(x, g) if P is None else P
    K, zP, zS, LP, LS, oE, oV = P["K"], P["zP"], P["f"]["zS"], P["LP"], P["LS"], P["oE"], P["oV"]

    def out(om):
        return R4.fco_core(R5._numN(LP, LS, om), P["den"], P["C"], P["M"], K)[0]

    q, qm, dfn = AR2.rtpl_q(P, g)
    rt = T2F.ratio_prefix(q, defined=dfn)
    rm = T2F.ratio_prefix(qm, defined=dfn)
    fr = T2F.oen(rt, zS, K, T2F.act_of(rt, LS, zP, oV))
    base = oE * oV * P["oEn"]
    vc = P["v"] & P["clean"][:, None]
    cnt = np.cumsum(vc, axis=0) - vc
    need = np.ceil(1.0 / P["al"])[:, None]
    c, formed = nst_cue(P, g, dfn, fr, rm)
    res = {"h0": out(base * fr),
           "mat": out(base * T2F.oen(rt, zS, K, T2F.act_of(rt, LS, zP, oV) & (cnt >= need))),
           "nst": out(base * fr * T2F.oen(c, zS, K, T2F.act_of(c, LS, zP, oV))),
           "s4": AR2.variants2(x, g, C2.two_sided(P), only=["rtpl_pez"])["rtpl_pez"]}
    return res, formed


def outputs_fn(x, g, with_low=True):
    """float32 H's and (with_low) their 976810-recipe composites over bank8; key "_formed" holds NST's flags."""
    V, formed = variants_h(x, g)
    out = {k: v.astype(np.float32) for k, v in V.items()}
    out["_formed"] = formed
    if with_low:
        b32 = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
        L = np.asarray(B8.bank8(x, g), np.float32)
        for k in HS:
            out[CMP[k]] = S3.veto_band(out[k], L, b32)
    return out


def composite_h(x, g, cell):
    """The build path of one cell: veto_band(H', bank8p, b), as the teammate's bal_or75v_b8.bal with H' for H."""
    import t1_bank8p as B8P
    H = variants_h(x, g)[0][cell].astype(np.float32)
    b32 = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    return S3.veto_band(H, np.asarray(B8P.bank8p(x, g), np.float32), b32)


def _job(task):
    tr, i = task
    x, g = D[f"s{tr}"][i], D[f"g{tr}"]
    o = outputs_fn(x, g, with_low=(tr == 1))
    formed = o.pop("_formed")
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
        t = int(D["c1"][i])
        return tr, i, o, bool(formed[t]), zchk, inv
    c = int(D["c2"][i])
    return tr, i, {k: v[:, c].copy() for k, v in o.items()}, bool(formed[c]), zchk, inv


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
    S = HV._read_zip(str(HERE.parents[1] / ZIP_973452))
    D["Z1"] = [S["track1"][f"score_{n}.npz"] for n in n1]
    D["Z2"] = [S["track2"][f"score_{n}.npz"] for n in n2]
    del S
    tasks = [(1, i) for i in range(len(s1))] + [(2, i) for i in range(len(s2))]
    R = {1: {}, 2: {}}
    formed = {1: [], 2: []}
    zc = [0, 0]
    inv = {k: [0, -1.0] for k in HS}
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 6)), mp_context=mp.get_context("fork")) as ex:
        for tr, i, o, fm, z, iv in ex.map(_job, tasks, chunksize=1):
            R[tr][i] = o
            formed[tr].append(fm)
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
        t1[k] = {"eer_full": cc._eer_full(gen, imp), "eer_thr": m["eer_thr"], "far1e5_thr": m["far1e5_thr"],
                 "frr_far1e5": m["frr_at_far1e5"], "far_at_frr1e5": m["far_at_frr1e5"]}
        del gen, imp
        print(json.dumps({"setting": k, **t1[k]}), flush=True)
    acc = {k: [R[2][i][k] >= np.float32(t1[k]["far1e5_thr"]) for i in range(n2s)] for k in HS}
    A = {k: np.concatenate(v) for k, v in acc.items()}
    pos = np.concatenate([np.arange(v.shape[0]) for v in acc["h0"]])
    t2 = {}
    for k in CELLS:
        d2 = float(A["h0"].mean() - A[k].mean())          # drop convention: positive = k rejects more
        g1 = t1[k]["frr_far1e5"] - t1["h0"]["frr_far1e5"]
        down, up = A["h0"] & ~A[k], A[k] & ~A["h0"]
        ch = down | up
        early = (pos < 100)
        t2[k] = {"d2_drop": d2, "g1": g1, "down": int(down.sum()), "up": int(up.sum()),
                 "changed_share": float(ch.mean()), "changed_share_t_lt_100": float((ch & early).sum() / max(ch.sum(), 1)),
                 "dBE_screen": (-d2 + g1) / 0.6,
                 "thr_shift": t1[k]["far1e5_thr"] - t1["h0"]["far1e5_thr"],
                 "cmp_eer_vs_control_gen_eq": (t1[CMP[k]]["eer_full"] - t1[CMP["h0"]]["eer_full"]) / GEN_EQ}
        print(json.dumps({"t2": k, **t2[k]}), flush=True)
    form = {"t1_target_formed_share": float(np.mean(formed[1])), "t2_claimed_formed_share": float(np.mean(formed[2]))}
    stop = {"finite_in_range": rng_ok, "h0_matches_973452_zip": zc[1] == 0 and zc[0] > 0,
            "h0_thr_recorded": t1["h0"]["far1e5_thr"] == H0_REC["far1e5_thr"],
            "h0_frr_recorded": abs(t1["h0"]["frr_far1e5"] - H0_REC["frr_far1e5"]) <= 1e-12,
            "control_cmp_reproduces_976810_proxy": abs(t1[CMP["h0"]]["eer_full"] - CMP_REC) <= 1e-12,
            "composite_hi_unchanged": all(v[0] == 0 for v in inv.values()),
            "composite_low_below_075": all(v[1] < HI for v in inv.values())}
    common = {k: {"thr_min": t1[k]["far1e5_thr"] >= 0.8381215, "frr_max": t1[k]["frr_far1e5"] <= 0.30,
                  "cmp_t1_within_5_gen_eq": t2[k]["cmp_eer_vs_control_gen_eq"] <= 5.0} for k in CELLS}
    m, s, n = t2["mat"], t2["s4"], t2["nst"]
    verdict = {}
    if m["d2_drop"] > -0.001:
        verdict["mat"] = "INERT"
    else:
        ok = (m["d2_drop"] <= -0.001 and m["changed_share_t_lt_100"] >= 0.6 and m["g1"] < 0
              and abs(m["g1"]) / abs(m["d2_drop"]) >= 1.263 and all(common["mat"].values()))
        verdict["mat"] = "BUILD" if ok else "KILL"
    if s["changed_share"] < 0.005:
        verdict["s4"] = "INERT"
    else:
        dirn = ((s["d2_drop"] > 0 and s["g1"] > 0 and s["d2_drop"] / s["g1"] >= 0.7917)
                or (s["d2_drop"] < 0 and s["g1"] < 0 and abs(s["g1"]) / abs(s["d2_drop"]) >= 1.27))
        verdict["s4"] = "BUILD" if dirn and all(common["s4"].values()) else "KILL"
    ok = (abs(n["thr_shift"]) <= 1e-4 and n["g1"] <= 0.005 and n["d2_drop"] >= 0.005
          and (n["g1"] <= 0 or n["d2_drop"] / n["g1"] >= 0.79)
          and form["t1_target_formed_share"] <= form["t2_claimed_formed_share"] / 3.0 and all(common["nst"].values()))
    verdict["nst"] = "BUILD" if ok else "KILL"
    if not all(stop.values()):
        verdict = "STOP"
    print(json.dumps({"stop": stop, "formation": form, "common": common, "verdict": verdict,
                      "seconds": round(time.time() - t0, 1)}), flush=True)
    return 0 if all(stop.values()) else 2


def _nst_ref(P, g, dfn, fr, rm):
    """Independent restatement of nst_cue: templates from cumulative sums, N per column by its own loop."""
    al, nrm, xb = P["al"], P["nrm"], P["xb"]
    u = xb / np.maximum(nrm, 1e-12)[:, None]
    B = np.asarray(GallerySubspace(np.asarray(g, np.float32)).basis, np.float64)
    rr = u - (u @ B.T) @ B
    rn_ = np.linalg.norm(rr, axis=1)
    rh = np.where(rn_[:, None] > 0, rr / np.maximum(rn_, 1e-300)[:, None], 0.0)
    T, K = P["v"].shape
    vc = P["v"] & P["clean"][:, None]
    c = np.ones((T, K))
    formed = np.zeros(K, bool)
    for j in range(K):
        s_ = np.zeros(rh.shape[1])
        N = np.zeros(rh.shape[1])
        has_s = has_n = False
        w = 0.0
        for t in range(T):
            if has_s and has_n and np.linalg.norm(rh[t]) > 0:
                sn, Nn = np.linalg.norm(s_), np.linalg.norm(N)
                if sn > 0 and Nn > 0:
                    a = max(0.0, float(rh[t] @ s_)) / sn
                    b = max(0.0, float(rh[t] @ N)) / Nn
                    if b > a:
                        c[t, j] = a / b
            m = bool(dfn[t, j] and fr[t, j] < 1.0 and rm[t, j] < 1.0)
            w = w + al[t] if m else 0.0
            if vc[t, j] and m and w >= 1.0:
                N = N + rh[t]
                has_n = True
                formed[j] = True
            if vc[t, j]:
                s_ = s_ + rh[t]
                has_s = True
    return c, formed


def _nst_case(seed, K=60, T=240):
    """A verified identity whose gallery-orthogonal residual follows style A (t < 100), B (100-159), A (160-199) and
    B (200-239) at frame norm 26 (al ~ .14): rtpl discounts the first B run, N forms from it, and the second B run
    matches N better than rtpl's template, so NST's cue must fall below 1 there."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(K, 512)).astype(np.float32)
    Q = np.linalg.qr(g.astype(np.float64).T)[0]

    def orth(v):
        v = v - Q @ (Q.T @ v)
        return v / np.linalg.norm(v)
    a_, b_ = orth(rng.normal(size=512)), orth(rng.normal(size=512))
    style = np.r_[np.repeat(a_[None], 100, 0), np.repeat(b_[None], 60, 0), np.repeat(a_[None], 40, 0),
                  np.repeat(b_[None], T - 200, 0)]
    x = g[5][None].astype(np.float64) + 15.0 * style + 0.3 * rng.normal(size=(T, 512))
    x = x / np.linalg.norm(x, axis=1, keepdims=True) * 26.0
    return x.astype(np.float32), g


def _formed_upto(P, g, dfn, fr, rm, h):
    """Formation flags of nst_cue using frames < h only (restated on the first h rows of the full-run inputs)."""
    P_h = dict(P)
    for key in ("al", "nrm", "xb", "v", "clean"):
        P_h[key] = P[key][:h]
    return nst_cue(P_h, g, dfn[:h], fr[:h], rm[:h])[1]


def synth_nst():
    """NST on the designed case: the cue fires (below 1) only from the second B run on, equals the restatement,
    keeps nst <= h0 and is prefix-exact."""
    ok = {}
    for seed in (2, 4):
        x, g = _nst_case(seed)
        P = T2F.parts(x, g)
        q, qm, dfn = AR2.rtpl_q(P, g)
        K, zP, zS, LS, oV = P["K"], P["zP"], P["f"]["zS"], P["LS"], P["oV"]
        rt = T2F.ratio_prefix(q, defined=dfn)
        rm = T2F.ratio_prefix(qm, defined=dfn)
        fr = T2F.oen(rt, zS, K, T2F.act_of(rt, LS, zP, oV))
        c1_, f1_ = nst_cue(P, g, dfn, fr, rm)
        c2_, f2_ = _nst_ref(P, g, dfn, fr, rm)
        rows = np.nonzero((c1_ < 1).any(1))[0]
        ok[f"nst_case_equals_ref_{seed}"] = bool(np.array_equal(c1_, c2_) and np.array_equal(f1_, f2_))
        ok[f"nst_case_formed_{seed}"] = bool(f1_[5])
        ok[f"nst_case_fires_{seed}"] = bool(rows.size > 0 and (c1_[:, 5] < 1).any())
        ok[f"nst_case_fires_only_after_first_b_{seed}"] = bool(rows.size > 0 and rows.min() >= 107)
        V, _ = variants_h(x, g, P)
        ok[f"nst_case_le_h0_{seed}"] = bool((V["nst"] <= V["h0"] + 1e-15).all())
        ok[f"nst_case_changes_output_{seed}"] = bool((V["nst"] < V["h0"]).any())
        f64 = 0.0
        for h in (120, 205):
            Vh, _ = variants_h(x[:h], g)
            # the submitted contract is float32; H's float64 values inherit the cumulative memory's known
            # length-dependent rounding (zP), so float64 is reported, not gated
            ok[f"nst_case_prefix_f32_{seed}_{h}"] = all(
                bool(np.array_equal(Vh[k].astype(np.float32), V[k][:h].astype(np.float32))) for k in HS)
            f64 = max(f64, max(float(np.abs(Vh[k] - V[k][:h]).max()) for k in HS))
            Ph = T2F.parts(x[:h], g)
            qh, qmh, dfnh = AR2.rtpl_q(Ph, g)
            rth = T2F.ratio_prefix(qh, defined=dfnh)
            rmh = T2F.ratio_prefix(qmh, defined=dfnh)
            frh = T2F.oen(rth, Ph["f"]["zS"], K, T2F.act_of(rth, Ph["LS"], Ph["zP"], Ph["oV"]))
            ch, fh = nst_cue(Ph, g, dfnh, frh, rmh)
            formed_by_h = np.zeros(K, bool)
            formed_by_h[np.nonzero(_formed_upto(P, g, dfn, fr, rm, h))[0]] = True
            ok[f"nst_case_cue_prefix_{seed}_{h}"] = bool(np.array_equal(ch, c1_[:h]) and np.array_equal(fh, formed_by_h))
        print(json.dumps({"nst_case": seed, "rows_with_cue_below_one": int(rows.size),
                          "first_row": int(rows.min()) if rows.size else -1,
                          "min_cue_col5": float(c1_[:, 5].min()), "h_float64_prefix_max_diff": f64}), flush=True)
    return ok


def synth():
    ok = synth_nst()
    fired = 0
    for seed, kind in ((3, "switch"), (5, "mng"), (7, "switch")):
        x, g = R4._synth_data(seed, 240, 60, kind, norm_hi=26.0)
        P = T2F.parts(x, g)
        V, formed = variants_h(x, g, P)
        ref = AR2.variants2(x, g, P, only=["rtpl_pez"])["rtpl_pez"]
        ok[f"h0_equals_rtpl_pez_{seed}"] = bool(np.array_equal(V["h0"], ref))
        ok[f"mat_ge_h0_{seed}"] = bool((V["mat"] >= V["h0"] - 1e-15).all())
        ok[f"nst_le_h0_{seed}"] = bool((V["nst"] <= V["h0"] + 1e-15).all())
        ok[f"finite_in_range_{seed}"] = all(bool(np.isfinite(v).all() and (np.abs(v) <= 1).all()) for v in V.values())
        o = outputs_fn(x, g)
        o.pop("_formed")
        for k in HS:
            H, C = o[k], o[CMP[k]]
            hi = H.astype(np.float64) >= HI
            ok[f"{k}_cmp_hi_{seed}"] = bool(np.array_equal(C[hi], H[hi]) and ((~hi).sum() == 0 or C[~hi].max() < HI))
        for h in (30, 75, 150):
            oh = outputs_fn(x[:h], g)
            oh.pop("_formed")
            ok[f"prefix_exact_H_{seed}_{h}"] = all(bool(np.array_equal(oh[k], o[k][:h])) for k in HS)
            ok[f"prefix_exact_cmp_{seed}_{h}"] = all(bool(np.array_equal(oh[CMP[k]], o[CMP[k]][:h])) for k in HS)
        for cell in CELLS:
            ok[f"{cell}_build_path_{seed}"] = bool(np.array_equal(composite_h(x, g, cell), o[CMP[cell]]))
        q, qm, dfn = AR2.rtpl_q(P, g)
        K, zP, zS, LS, oV = P["K"], P["zP"], P["f"]["zS"], P["LS"], P["oV"]
        rt = T2F.ratio_prefix(q, defined=dfn)
        rm = T2F.ratio_prefix(qm, defined=dfn)
        fr = T2F.oen(rt, zS, K, T2F.act_of(rt, LS, zP, oV))
        c1_, f1_ = nst_cue(P, g, dfn, fr, rm)
        c2_, f2_ = _nst_ref(P, g, dfn, fr, rm)
        ok[f"nst_cue_equals_ref_{seed}"] = bool(np.array_equal(c1_, c2_) and np.array_equal(f1_, f2_))
        fired += int((c1_ < 1).sum())
    good = all(ok.values())
    print(json.dumps({"synth": ok, "nst_cue_below_one_entries": fired, "all_ok": good}), flush=True)
    return good


if __name__ == "__main__":
    if "--synth-nst" in sys.argv:
        r = synth_nst()
        print(json.dumps({"synth_nst": r, "all_ok": all(r.values())}), flush=True)
        sys.exit(0 if all(r.values()) else 1)
    if "--synth" in sys.argv:
        sys.exit(0 if synth() else 1)
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 6))
