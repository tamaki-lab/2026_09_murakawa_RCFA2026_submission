"""T1 premise search stage 1b: the two-sided Cauchy likelihood ported into the current T1 chain (analysis only).

Plan, provenance and frozen rules: docs/experiments/20261009_ogata_t1_premise_search.md (stage 1b).

Per T1 sequence, float64 internals, fresh state per call, the same constants on both tracks:
  P  = analyze_t2_family.parts(x, g), unchanged (one-sided LR = 1 + z+^2).
  P2 = P with LP, LS, num, den replaced: LR(z) = 1 + z^2 for z >= 0 (bit-identical), 1 / (1 + z^2) for z < 0;
       num = 4 LP LS, den = LP (2 + mS) + 1 + Q (x_fr_hcp.hcp_parts' operations, leader-exact Q).
       Every omega and the FCO state are P's own.
  settings: nh, hold_nh, bl_hold_nh, cmp = cmp(rtpl_nh | hold_nh)        controls (from P)
            fco, fco_c2                                                 HCP + FCO, no discount (P, P2)
            nh_c2, hold_nh_c2, bl_hold_nh_c2                            the chain from P2
            cmp_c2 = cmp(rtpl_nh | hold_nh_c2)                          PRIMARY (rtpl_nh from P)
Pseudo targets on the evaluator side only. Prints JSON aggregates only.
Usage (server): python scripts/ogata/analyze_t1_cau2_chain.py [workers<=6] > results/ogata/t1_cau2_chain.log
                python scripts/ogata/analyze_t1_cau2_chain.py --synth
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
import x_r5 as R5  # noqa: E402  (imports common_check first: pins BLAS threads)
import numpy as np  # noqa: E402

import analyze_fusion_family as FF  # noqa: E402
import analyze_rtpl_cmp as RC  # noqa: E402
import analyze_t2_family as T2F  # noqa: E402
import x_fr_hcp as H  # noqa: E402
import x_r4 as R4  # noqa: E402
from analyze_t1_hcpg import NAMES, _counts  # noqa: E402
from rcfa.methods.gallery_discriminative_row_l2 import gallery_discriminative_row_l2_scores  # noqa: E402

NH_PROXY = 0.002989023673650839
CMP_PROXY = 0.002944727767301255
BAR = 2e-5
HI = 0.75
HARD = [11, 61, 99, 109, 130, 170, 185]               # stage 1's hard sequences of hcp (report-only)
SETTINGS = ["nh", "hold_nh", "bl_hold_nh", "cmp", "fco", "fco_c2", "nh_c2", "hold_nh_c2", "bl_hold_nh_c2", "cmp_c2"]
PAIRS = {"fco_c2": "fco", "nh_c2": "nh", "hold_nh_c2": "hold_nh", "bl_hold_nh_c2": "bl_hold_nh", "cmp_c2": "cmp"}
D = {}


def lr2(z):
    """Two-sided Cauchy LR; equals x_fr_hcp's one-sided 1 + z+^2 bit for bit where z >= 0."""
    pos = 1.0 + np.maximum(z, 0.0) ** 2
    return np.where(z >= 0, pos, 1.0 / (1.0 + z ** 2))


def hcp_parts2(LP, LS, K):
    """x_fr_hcp.hcp_parts with the likelihood ratios given (same operations and order)."""
    wS = LS - 1.0
    mS = (wS.sum(1, keepdims=True) - wS) / (K - 1)
    Pr = LP * LS
    Q = (Pr.sum(1, keepdims=True) - Pr) / (K - 1)
    r = np.arange(Pr.shape[0])
    i1 = Pr.argmax(1)
    Po = Pr.copy()
    Po[r, i1] = 0.0
    Q[r, i1] = Po.sum(1) / (K - 1)                   # leader-exact
    num = 4.0 * LP * LS
    den = LP * (2.0 + mS) + 1.0 + Q
    return num, den


def two_sided(P):
    K, zP, zS = P["K"], P["zP"], P["f"]["zS"]
    LP, LS = lr2(zP), lr2(zS)
    num, den = hcp_parts2(LP, LS, K)
    P2 = dict(P)
    P2.update(LP=LP, LS=LS, num=num, den=den)
    return P2


def outputs(x, g, P=None):
    """Every setting (float32) for one sequence, plus the identity check of the replaced algebra."""
    P = T2F.parts(x, g) if P is None else P
    K = P["K"]
    n1, d1 = hcp_parts2(1.0 + np.maximum(P["zP"], 0.0) ** 2, 1.0 + np.maximum(P["f"]["zS"], 0.0) ** 2, K)
    ident = bool(np.array_equal(n1, P["num"]) and np.array_equal(d1, P["den"]))
    P2 = two_sided(P)
    V = T2F.variants(x, g, P, only=["nh", "hold_nh", "rtpl_nh"])
    V2 = T2F.variants(x, g, P2, only=["nh", "hold_nh"])
    b = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    f32 = {k: v.astype(np.float32) for k, v in V.items()}
    c2 = {k: v.astype(np.float32) for k, v in V2.items()}
    out = {"nh": f32["nh"], "hold_nh": f32["hold_nh"],
           "bl_hold_nh": FF._bl(f32["hold_nh"], b).astype(np.float32),
           "cmp": RC.cmp_(f32["rtpl_nh"], f32["hold_nh"], b),
           "fco": R4.fco_core(P["num"], P["den"], P["C"], P["M"], K)[0].astype(np.float32),
           "fco_c2": R4.fco_core(P2["num"], P2["den"], P["C"], P["M"], K)[0].astype(np.float32),
           "nh_c2": c2["nh"], "hold_nh_c2": c2["hold_nh"],
           "bl_hold_nh_c2": FF._bl(c2["hold_nh"], b).astype(np.float32),
           "cmp_c2": RC.cmp_(f32["rtpl_nh"], c2["hold_nh"], b)}
    hi = f32["rtpl_nh"].astype(np.float64) >= HI
    inv = int((out["cmp_c2"][hi] != out["cmp"][hi]).sum())
    return out, ident, inv


def outputs_fn(x, g):
    """Reference for build_candidate's bit-identity check (keys = SETTINGS)."""
    return outputs(x, g)[0]


def composite_c2(x, g):
    """The build path for cmp(rtpl_nh | hold_nh_c2): only the two parents and the partner are computed."""
    P = T2F.parts(x, g)
    rt = T2F.variants(x, g, P, only=["rtpl_nh"])["rtpl_nh"].astype(np.float32)
    hc = T2F.variants(x, g, two_sided(P), only=["hold_nh"])["hold_nh"].astype(np.float32)
    b = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    return RC.cmp_(rt, hc, b)


def _job(i):
    out, ident, inv = outputs(D["s"][i], D["g"])
    return i, out, ident, inv


def main(workers=6):
    import analyze_t1_residual as ar
    import common_check as cc
    from proxy_lib import load_track, pseudo_targets
    t0 = time.time()
    g, s, _ = load_track("track1")
    tg, _ = pseudo_targets(g, s)
    D.update(s=s, g=g)
    n = len(s)
    R, ident, inv = {}, True, 0
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 6)), mp_context=mp.get_context("fork")) as ex:
        for i, out, idn, iv in ex.map(_job, range(n), chunksize=1):
            R[i] = out
            ident = ident and idn
            inv += iv
    rng_ok = all(bool(np.isfinite(v).all() and (np.abs(v) <= 1).all()) for i in range(n) for v in R[i].values())
    print(json.dumps({"stage": "outputs", "sequences": n, "identity_bitwise": ident, "hi_band_mismatch": inv,
                      "finite_in_range": rng_ok, "seconds": round(time.time() - t0, 1)}), flush=True)

    def eer(idx, k):
        gen, imp = ar._gen_imp([R[i][k] for i in idx], tg[idx])
        return cc._eer_full(gen, imp)

    even, odd = np.arange(0, n, 2), np.arange(1, n, 2)
    rest = np.array([i for i in range(n) if i not in HARD])
    t1, per = {}, {}
    for k in SETTINGS:
        gen, imp = ar._gen_imp([R[i][k] for i in range(n)], tg)
        m = ar._metrics(gen, imp)
        del gen, imp
        gm, ia, per[k] = _counts(R, tg, k, np.float32(m["eer_thr"]))
        t1[k] = {"eer_full": m["eer_full"], "eer_thr": m["eer_thr"], "far1e5_thr": m["far1e5_thr"],
                 "frr_far1e5": m["frr_at_far1e5"], "even": eer(even, k), "odd": eer(odd, k), "no_hard": eer(rest, k),
                 "gen_eq_by_bin": {nm: round(float(gm[j] + ia[j] / 199.0), 1) for j, nm in enumerate(NAMES)},
                 "gen_misses": int(gm.sum()), "imp_accepts": int(ia.sum())}
        print(json.dumps({"setting": k, **t1[k]}), flush=True)
    conc = {}
    for k, base in PAIRS.items():
        d = np.array([(per[k][i][0] + per[k][i][1] / 199.0) - (per[base][i][0] + per[base][i][1] / 199.0)
                      for i in range(n)])
        top = np.argsort(d)[:5]
        conc[k] = {"vs": base, "d_eer": t1[k]["eer_full"] - t1[base]["eer_full"],
                   "d_even": t1[k]["even"] - t1[base]["even"], "d_odd": t1[k]["odd"] - t1[base]["odd"],
                   "d_no_hard": t1[k]["no_hard"] - t1[base]["no_hard"], "d_gen_eq_total": round(float(d.sum()), 1),
                   "top5_gain_seqs": [int(i) for i in top], "top5_gain_gen_eq": round(float(d[top].sum()), 1),
                   "seqs_better": int((d < -0.5).sum()), "seqs_worse": int((d > 0.5).sum())}
    stop = {"nh_repro": abs(t1["nh"]["eer_full"] - NH_PROXY) <= 1e-9,
            "bl_hold_nh_repro": abs(t1["bl_hold_nh"]["eer_full"] - CMP_PROXY) <= 1e-9,
            "cmp_repro": abs(t1["cmp"]["eer_full"] - CMP_PROXY) <= 1e-9,
            "identity_bitwise": ident, "finite_in_range": rng_ok}
    build_cmp = t1["cmp_c2"]["eer_full"] <= CMP_PROXY - BAR and inv == 0
    build_bl = (not build_cmp) and t1["bl_hold_nh_c2"]["eer_full"] <= CMP_PROXY - BAR
    verdict = "STOP" if not all(stop.values()) else \
        ("BUILD:cmp_c2" if build_cmp else ("BUILD:bl_hold_nh_c2" if build_bl else "KILL"))
    print(json.dumps({"stop": stop, "hi_band_mismatch": inv, "pairs": conc, "verdict": verdict,
                      "seconds": round(time.time() - t0, 1)}), flush=True)


def synth():
    ok = {}
    rng = np.random.default_rng(0)
    K = 200
    for nm, (zh, zs) in {"extreme": H._extreme_rows(K), "random": (3.0 * rng.normal(size=(40, K)),
                                                                   3.0 * rng.normal(size=(40, K)))}.items():
        n0, d0, _ = H.hcp_parts(zh, zs, K)
        n1, d1 = hcp_parts2(1.0 + np.maximum(zh, 0.0) ** 2, 1.0 + np.maximum(zs, 0.0) ** 2, K)
        ok[f"identity_{nm}"] = bool(np.array_equal(n0, n1) and np.array_equal(d0, d1))
        pos = zh >= 0
        ok[f"lr2_pos_bitwise_{nm}"] = bool(np.array_equal(lr2(zh)[pos], (1.0 + np.maximum(zh, 0.0) ** 2)[pos]))
        ok[f"lr2_neg_below1_{nm}"] = bool((lr2(zh)[~pos] < 1).all())
        n2, d2 = hcp_parts2(lr2(zh), lr2(zs), K)
        ok[f"finite_pos_{nm}"] = bool(np.isfinite(n2).all() and np.isfinite(d2).all() and (d2 > 0).all())
    for seed, kind in ((3, "switch"), (5, "mng")):
        x, g = R4._synth_data(seed, 240, 60, kind, norm_hi=26.0)
        out, ident, inv = outputs(x, g)
        ref = RC.outputs_fn(x, g)
        ok[f"identity_seq_{seed}"] = ident
        for k, rk in (("nh", "nh"), ("hold_nh", "hold_nh"), ("bl_hold_nh", "bl_hold_nh"), ("cmp", "cmp_rtpl_nh__hold_nh")):
            ok[f"{k}_equals_parent_{seed}"] = bool(np.array_equal(out[k], ref[rk]))
        ok[f"fco_equals_r4_{seed}"] = bool(np.array_equal(out["fco"], R4.fco(x, g).astype(np.float32)))
        ok[f"hi_band_invariant_{seed}"] = inv == 0
        rt = ref["rtpl_nh"].astype(np.float64) >= 0.5
        ok[f"cmp_c2_keeps_rtpl_{seed}"] = bool(np.array_equal(out["cmp_c2"][rt], ref["rtpl_nh"][rt]))
        ok[f"c2_changes_{seed}"] = bool(np.abs(out["nh_c2"] - out["nh"]).max() > 0)
        ok[f"build_path_equals_{seed}"] = bool(np.array_equal(composite_c2(x, g), out["cmp_c2"]))
        ok[f"shapes_{seed}"] = sorted(out) == sorted(SETTINGS) and all(
            v.shape == (240, 60) and v.dtype == np.float32 for v in out.values())
        ok[f"finite_in_range_{seed}"] = all(bool(np.isfinite(v).all() and (np.abs(v) <= 1).all()) for v in out.values())
        h = 150
        oh, _, _ = outputs(x[:h], g)
        ok[f"prefix_invariant_{seed}"] = all(bool(np.allclose(oh[k], out[k][:h], rtol=0, atol=1e-6)) for k in out)
        ok[f"prefix_exact_{seed}"] = all(bool(np.array_equal(oh[k], out[k][:h])) for k in out)
    good = all(v for k, v in ok.items() if not k.startswith("prefix_exact"))
    print(json.dumps({"synth": ok, "all_ok": good}), flush=True)
    return good


if __name__ == "__main__":
    if "--synth" in sys.argv:
        sys.exit(0 if synth() else 1)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
