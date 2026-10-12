"""T1 premise search stage 1d: low-band cells under 973452's high band (analysis only).

Plan, provenance and frozen rules: docs/experiments/20261009_ogata_t1_premise_search.md (stage 1d).
Designs: docs/experiments/20261010_ogata_next_improvements.md (Claude lenses and GPT round 10).

Per T1 sequence (float64 internals, fresh state, the same constants on both tracks). H = rtpl_pez (one-sided P),
b = 943955, every composite is H where H >= .5 and a low-band value below .75 elsewhere:
  controls  cmp_pez    = cmp(H | hold_nh)                      (973452's own function)
            cmp_pez_c2 = cmp(H | hold_nh_c2)                   (the pending T1 candidate, the parent of every cell)
  L cells   p2s1       LP two-sided, LS one-sided               (look split; Claude L1 / S3)
            s2p1       LP one-sided, LS two-sided
            psg        LP two-sided, LS two-sided only where zP < 0
            c2num      both two-sided in the numerator and the discounts, den one-sided (S3)
            lid        GPT-T1-1: L = (sqrt R - 1)/(sqrt R + 1), R_j = A_j / mean_{k!=j} A_k, A = C(zP) C(zS)
  band      pure       (min(fco_c2, .5) + b)/2 below H = .5, fco_c2 = two-sided HCP + FCO, no discount (PURE-L)
  partner   geo        sqrt((1 + a)(1 + b)) - 1 with a = min(hold_nh_c2, H) (GPT-T1-2)
            bs         (a + (b + bS)/2)/2 with bS = row-L2 of S (Claude L4)
Pseudo targets on the evaluator side only. Prints JSON aggregates only.
Usage (server): python scripts/ogata/analyze_t1_stage1d.py [workers<=6] > results/ogata/t1_stage1d.log
                python scripts/ogata/analyze_t1_stage1d.py --synth
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
import x_r5  # noqa: E402,F401  (imports common_check first: pins BLAS threads)
import numpy as np  # noqa: E402

import analyze_fusion_family as FF  # noqa: E402
import analyze_round2 as AR2  # noqa: E402
import analyze_rtpl_cmp as RC  # noqa: E402
import analyze_t1_cau2_chain as C2  # noqa: E402
import analyze_t2_family as T2F  # noqa: E402
import x_r4 as R4  # noqa: E402
from rcfa.methods.gallery_discriminative_row_l2 import gallery_discriminative_row_l2_scores, row_l2_scores  # noqa: E402

PEZ_CMP = 0.002944727767301255
PARENT = 0.002876373627558647
BAR = 2e-5
HI = 0.75
CELLS = ["p2s1", "s2p1", "psg", "c2num", "lid", "pure", "geo", "bs"]
SETTINGS = ["cmp_pez", "cmp_pez_c2"] + CELLS
D = {}


def _with(P, LP, LS, den=None):
    """P with the likelihood ratios replaced; den from the same LRs unless given (x_fr_hcp.hcp_parts' operations)."""
    num, den2 = C2.hcp_parts2(LP, LS, P["K"])
    P2 = dict(P)
    P2.update(LP=LP, LS=LS, num=num, den=den2 if den is None else den)
    return P2


def lid(zP, zS, K):
    """GPT-T1-1: R_j = A_j / mean_{k!=j} A_k (leader-exact), A = C(zP) C(zS); out = (sqrt R - 1)/(sqrt R + 1)."""
    A = C2.lr2(zP) * C2.lr2(zS)
    m = (A.sum(1, keepdims=True) - A) / (K - 1)
    r = np.arange(A.shape[0])
    i1 = A.argmax(1)
    Ao = A.copy()
    Ao[r, i1] = 0.0
    m[r, i1] = Ao.sum(1) / (K - 1)                    # leader-exact
    R = A / m
    rr = np.sqrt(R)
    return (rr - 1.0) / (rr + 1.0)


def _band(H32, low64):
    H64 = H32.astype(np.float64)
    return np.where(H64 >= FF.C0, H64, low64).astype(np.float32)


def outputs_fn(x, g):
    """Every setting (float32) for one sequence plus H (key "_H"); the reference for build_candidate."""
    P = T2F.parts(x, g)
    K, zP, zS = P["K"], P["zP"], P["f"]["zS"]
    V = AR2.variants2(x, g, P, only=["rtpl_pez", "hold_nh"])
    H32 = V["rtpl_pez"].astype(np.float32)
    b = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    P2 = C2.two_sided(P)

    def hold(Px):
        return T2F.variants(x, g, Px, only=["hold_nh"])["hold_nh"].astype(np.float32)

    Lc2 = hold(P2)
    LP1, LS1, LP2, LS2 = P["LP"], P["LS"], C2.lr2(zP), C2.lr2(zS)
    out = {"_H": H32,
           "cmp_pez": RC.cmp_(H32, V["hold_nh"].astype(np.float32), b),
           "cmp_pez_c2": RC.cmp_(H32, Lc2, b),
           "p2s1": RC.cmp_(H32, hold(_with(P, LP2, LS1)), b),
           "s2p1": RC.cmp_(H32, hold(_with(P, LP1, LS2)), b),
           "psg": RC.cmp_(H32, hold(_with(P, LP2, np.where(zP < 0, LS2, LS1))), b),
           "c2num": RC.cmp_(H32, hold(_with(P, LP2, LS2, den=P["den"])), b),
           "lid": RC.cmp_(H32, lid(zP, zS, K).astype(np.float32), b)}
    fco_c2 = R4.fco_core(P2["num"], P2["den"], P["C"], P["M"], K)[0].astype(np.float32)
    out["pure"] = _band(H32, FF._mean(np.minimum(fco_c2, np.float32(FF.C0)), b))
    a = np.minimum(Lc2, H32).astype(np.float64)
    out["geo"] = _band(H32, np.sqrt((1.0 + a) * (1.0 + b.astype(np.float64))) - 1.0)
    bS = row_l2_scores(np.asarray(P["f"]["S"], np.float32))
    partner = (b + bS) / np.float32(2)
    out["bs"] = _band(H32, FF._mean(np.minimum(Lc2, H32), partner))
    return out


def composite_geo(x, g):
    """The build path for the stage-1d BUILD cell geo: H = rtpl_pez, a = min(hold_nh_c2, H),
    low band sqrt((1 + a)(1 + b)) - 1 below H = .5 (only the parents and the partner are computed)."""
    P = T2F.parts(x, g)
    H32 = AR2.variants2(x, g, P, only=["rtpl_pez"])["rtpl_pez"].astype(np.float32)
    Lc2 = T2F.variants(x, g, C2.two_sided(P), only=["hold_nh"])["hold_nh"].astype(np.float32)
    b = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    a = np.minimum(Lc2, H32).astype(np.float64)
    return _band(H32, np.sqrt((1.0 + a) * (1.0 + b.astype(np.float64))) - 1.0)


def _job(i):
    o = outputs_fn(D["s"][i], D["g"])
    H = o.pop("_H")
    hi = H.astype(np.float64) >= FF.C0
    chk = {k: (int((o[k][hi] != H[hi]).sum()), float(o[k][~hi].max()) if (~hi).any() else -1.0) for k in SETTINGS}
    return i, o, H, chk


def main(workers=6):
    import analyze_t1_residual as ar
    import common_check as cc
    from analyze_t1_hcpg import NAMES, _counts
    from proxy_lib import load_track, pseudo_targets
    t0 = time.time()
    g, s, _ = load_track("track1")
    tg, _ = pseudo_targets(g, s)
    D.update(s=s, g=g)
    n = len(s)
    R, Hs, hi_mis, lo_max = {}, {}, {k: 0 for k in SETTINGS}, {k: -1.0 for k in SETTINGS}
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 6)), mp_context=mp.get_context("fork")) as ex:
        for i, o, H, chk in ex.map(_job, range(n), chunksize=1):
            R[i], Hs[i] = o, H
            for k, (m_, mx) in chk.items():
                hi_mis[k] += m_
                lo_max[k] = max(lo_max[k], mx)
    rng_ok = all(bool(np.isfinite(v).all() and (np.abs(v) <= 1).all()) for i in range(n) for v in R[i].values())
    print(json.dumps({"stage": "outputs", "sequences": n, "finite_in_range": rng_ok, "hi_band_mismatch": hi_mis,
                      "low_band_max": lo_max, "seconds": round(time.time() - t0, 1)}), flush=True)
    even, odd = np.arange(0, n, 2), np.arange(1, n, 2)
    rest = np.array([i for i in range(n) if i not in C2.HARD])
    t1, per = {}, {}
    for k in SETTINGS:
        gen, imp = ar._gen_imp([R[i][k] for i in range(n)], tg)
        m = ar._metrics(gen, imp)
        del gen, imp
        gm, ia, per[k] = _counts(R, tg, k, np.float32(m["eer_thr"]))
        sub = {}
        for nm, idx in (("even", even), ("odd", odd), ("no_hard", rest)):
            gg, ii = ar._gen_imp([R[i][k] for i in idx], tg[idx])
            sub[nm] = cc._eer_full(gg, ii)
        t1[k] = {"eer_full": m["eer_full"], "eer_thr": m["eer_thr"], "far1e5_thr": m["far1e5_thr"], **sub,
                 "gen_eq_by_bin": {nm: round(float(gm[j] + ia[j] / 199.0), 1) for j, nm in enumerate(NAMES)}}
        print(json.dumps({"setting": k, **t1[k]}), flush=True)
    # report-only lock accounting at the parent's own cut: impostor accepts with H >= .5 cannot be moved by any L
    cut = np.float32(t1["cmp_pez_c2"]["eer_thr"])
    lock_imp = lock_gen_ok = 0
    for i, t in enumerate(tg):
        o, H = R[i]["cmp_pez_c2"], Hs[i]
        mask = np.ones(o.shape[1], bool)
        mask[t] = False
        hi = H.astype(np.float64) >= FF.C0
        lock_imp += int((hi[:, mask] & (o[:, mask] >= cut)).sum())
        lock_gen_ok += int(hi[:, t].sum())
    conc = {}
    for k in CELLS:
        d = np.array([(per[k][i][0] + per[k][i][1] / 199.0) - (per["cmp_pez_c2"][i][0] + per["cmp_pez_c2"][i][1] / 199.0)
                      for i in range(n)])
        top = np.argsort(d)[:5]
        conc[k] = {"d_eer": t1[k]["eer_full"] - PARENT, "d_even": t1[k]["even"] - t1["cmp_pez_c2"]["even"],
                   "d_odd": t1[k]["odd"] - t1["cmp_pez_c2"]["odd"],
                   "d_no_hard": t1[k]["no_hard"] - t1["cmp_pez_c2"]["no_hard"],
                   "top5": [int(i) for i in top], "top5_gen_eq": round(float(d[top].sum()), 1),
                   "total_gen_eq": round(float(d.sum()), 1), "better": int((d < -0.5).sum()), "worse": int((d > 0.5).sum())}
    stop = {"cmp_pez_repro": abs(t1["cmp_pez"]["eer_full"] - PEZ_CMP) <= 1e-9,
            "parent_repro": abs(t1["cmp_pez_c2"]["eer_full"] - PARENT) <= 1e-9,
            "finite_in_range": rng_ok, "hi_band_unchanged": all(v == 0 for v in hi_mis.values()),
            "low_band_below_075": all(v < HI for v in lo_max.values())}
    passing = sorted((k for k in CELLS if t1[k]["eer_full"] <= PARENT - BAR), key=lambda k: t1[k]["eer_full"])
    verdict = "STOP" if not all(stop.values()) else ("BUILD:" + passing[0] if passing else "KILL")
    print(json.dumps({"stop": stop, "lock": {"impostor_accepts_with_H_ge_.5": lock_imp,
                                             "as_gen_eq": round(lock_imp / 199.0, 1),
                                             "genuine_entries_with_H_ge_.5": lock_gen_ok},
                      "cells": conc, "passing": passing, "verdict": verdict,
                      "seconds": round(time.time() - t0, 1)}), flush=True)


def synth():
    ok = {}
    K = 60
    rng = np.random.default_rng(1)
    zP, zS = 3.0 * rng.normal(size=(30, K)), 3.0 * rng.normal(size=(30, K))
    A = C2.lr2(zP) * C2.lr2(zS)
    ref = np.empty_like(A)
    for j in range(K):
        R = A[:, j] / (np.delete(A, j, axis=1).sum(1) / (K - 1))
        ref[:, j] = (np.sqrt(R) - 1) / (np.sqrt(R) + 1)
    ok["lid_brute"] = bool(np.abs(lid(zP, zS, K) - ref).max() <= 1e-12)
    for seed, kind in ((3, "switch"), (5, "mng")):
        x, g = R4._synth_data(seed, 240, 60, kind, norm_hi=26.0)
        o = outputs_fn(x, g)
        H = o.pop("_H")
        ref = AR2.composite(x, g, "rtpl_pez")
        import analyze_t1_c2_pez as CP
        ok[f"cmp_pez_equals_973452_fn_{seed}"] = bool(np.array_equal(o["cmp_pez"], ref))
        ok[f"parent_equals_stage1c_{seed}"] = bool(np.array_equal(o["cmp_pez_c2"], CP.composite_pez_c2(x, g)))
        P = T2F.parts(x, g)
        both = _with(P, C2.lr2(P["zP"]), C2.lr2(P["f"]["zS"]))
        P2 = C2.two_sided(P)
        ok[f"with_equals_two_sided_{seed}"] = bool(np.array_equal(both["num"], P2["num"]) and np.array_equal(both["den"], P2["den"]))
        one = _with(P, P["LP"], P["LS"])
        ok[f"with_one_sided_identity_{seed}"] = bool(np.array_equal(one["num"], P["num"]) and np.array_equal(one["den"], P["den"]))
        hi = H.astype(np.float64) >= 0.5
        ok[f"hi_band_{seed}"] = all(bool(np.array_equal(o[k][hi], H[hi])) for k in SETTINGS)
        ok[f"low_below_075_{seed}"] = all(bool((o[k][~hi] < HI).all()) for k in SETTINGS)
        ok[f"cells_change_{seed}"] = all(bool(np.abs(o[k] - o["cmp_pez_c2"]).max() > 0) for k in CELLS if k != "psg")
        ok[f"finite_in_range_{seed}"] = all(bool(np.isfinite(v).all() and (np.abs(v) <= 1).all()) for v in o.values())
        ok[f"geo_build_path_{seed}"] = bool(np.array_equal(composite_geo(x, g), o["geo"]))
        h = 150
        oh = outputs_fn(x[:h], g)
        ok[f"prefix_exact_{seed}"] = all(bool(np.array_equal(oh[k], o[k][:h])) for k in SETTINGS)
    good = all(ok.values())
    print(json.dumps({"synth": ok, "all_ok": good}), flush=True)
    return good


if __name__ == "__main__":
    if "--synth" in sys.argv:
        sys.exit(0 if synth() else 1)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
