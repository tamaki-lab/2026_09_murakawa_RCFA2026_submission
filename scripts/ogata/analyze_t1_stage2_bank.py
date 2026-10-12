"""T1 stage 2: the teammate's bank processors as the low band under 973452's high band (analysis only).

Plan, provenance and frozen rules: docs/experiments/20261009_ogata_t1_premise_search.md (stage 2).

L = the teammate's T1 specialists on 971136's base (candidates.t2_family_bl('dframe_nh')): bank5 (974146, official
T1-EER .002503648) and bank6 (not submitted; proxy .002346974384189832). Their processors are taken verbatim from
origin/research/final-stretch-20261009 at 4311e24 (scripts/final_stretch/t1_bank*.py). H = rtpl_pez, b = 943955.
Per T1 sequence (fresh state, the same constants on both tracks):
  controls  bank5, bank6   the bank outputs alone (the teammate's pinned proxies)
            geo            stage 1d BUILD, the parent of this stage (.0028510572795058323)
  cells     or50_bX        H where H >= .5,  min(L, .5) elsewhere          (accept = H >= .5 or L >= cut)
            or75_bX        H where H >= .75, min(L, .5) elsewhere          (H's .5-.75 entries follow the bank)
            cap50_bX       H where H >= .5,  (min(L, .5) + b)/2 elsewhere
            min50_bX       H where H >= .5,  (min(L, H) + b)/2 elsewhere   (974523's band rule)
Every cell keeps every entry with H >= .75 equal to H and every other entry below .75, so the T1 FAR=1e-5 threshold
and every Track 2 decision are H's. Pseudo targets on the evaluator side only. Prints JSON aggregates only.
Usage (server): python scripts/ogata/analyze_t1_stage2_bank.py [workers<=6] > results/ogata/t1_stage2_bank.log
                python scripts/ogata/analyze_t1_stage2_bank.py --synth
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

import analyze_fusion_family as FF  # noqa: E402
import analyze_round2 as AR2  # noqa: E402
import analyze_rtpl_cmp as RC  # noqa: E402
import analyze_t1_stage1d as S1D  # noqa: E402
import analyze_t2_family as T2F  # noqa: E402
import t1_bank as BK  # noqa: E402
import t1_bank5 as B5  # noqa: E402
import t1_bank6 as B6  # noqa: E402
from rcfa.methods.gallery_discriminative_row_l2 import gallery_discriminative_row_l2_scores  # noqa: E402

BANK5 = 0.00250335546882519      # teammate's analyze_t1_bank5_expected.py at 2471d9d (= check_t1_bank5_zip.py pin)
BANK6 = 0.002346974384189832     # teammate's analyze_t1_bank6_expected.py at 90e53ce (= check_t1_bank6_zip.py pin)
GEO = 0.0028510572795058323      # stage 1d (results/ogata/t1_stage1d.log)
TOL = 1e-9
BAR = 2e-5
HI = 0.75
RULES = {"or50": 0.5, "or75": 0.75, "cap50": 0.5, "min50": 0.5}
LOWS = ("b5", "b6")
CELLS = [f"{r}_{l_}" for l_ in LOWS for r in RULES]
CONTROLS = ["bank5", "bank6", "geo"]
SETTINGS = CONTROLS + CELLS
D = {}


def lows(x, g):
    """bank5 and bank6 (float32) on one shared 971136 base."""
    base = np.asarray(BK._BASE(x, g), np.float32)
    return {"b5": B5.post5(base, x, g).astype(np.float32), "b6": B6.post6(base, x, g).astype(np.float32)}


def high(x, g):
    """H = rtpl_pez (float32), 973452's high band."""
    P = T2F.parts(x, g)
    return AR2.variants2(x, g, P, only=["rtpl_pez"])["rtpl_pez"].astype(np.float32)


def rule(name, H32, L32, b32):
    """One band rule on float32 parents; float64 arithmetic, float32 output."""
    H64 = H32.astype(np.float64)
    if name in ("or50", "or75"):
        return np.where(H64 >= RULES[name], H64, np.minimum(L32.astype(np.float64), 0.5)).astype(np.float32)
    if name == "cap50":
        return np.where(H64 >= FF.C0, H64, FF._mean(np.minimum(L32, np.float32(0.5)), b32)).astype(np.float32)
    if name == "min50":
        return RC.cmp_(H32, L32, b32)
    raise KeyError(name)


def outputs_fn(x, g):
    """Every setting (float32) for one sequence plus H (key "_H"); the reference for build_candidate."""
    H32 = high(x, g)
    b32 = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    L = lows(x, g)
    out = {"_H": H32, "bank5": L["b5"], "bank6": L["b6"], "geo": S1D.composite_geo(x, g)}
    for c in CELLS:
        r, l_ = c.split("_")
        out[c] = rule(r, H32, L[l_], b32)
    return out


def composite_bank(x, g, cell):
    """The build path for one stage-2 cell (only the parents and the partner are computed)."""
    r, l_ = cell.split("_")
    H32 = high(x, g)
    b32 = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    base = np.asarray(BK._BASE(x, g), np.float32)
    L32 = (B5.post5(base, x, g) if l_ == "b5" else B6.post6(base, x, g)).astype(np.float32)
    return rule(r, H32, L32, b32)


def _inv(o, H):
    """Entries changed where H >= .75, and the largest value where H < .75 (-1 when none)."""
    hi = H.astype(np.float64) >= HI
    return int((o[hi] != H[hi]).sum()), float(o[~hi].max()) if (~hi).any() else -1.0


def _job(i):
    o = outputs_fn(D["s"][i], D["g"])
    H = o.pop("_H")
    chk = {k: _inv(o[k], H) for k in CELLS + ["geo"]}
    return i, o, H, chk


def main(workers=6):
    import analyze_t1_residual as ar
    import analyze_t1_cau2_chain as C2
    import common_check as cc
    from analyze_t1_hcpg import NAMES, _counts
    from proxy_lib import load_track, pseudo_targets
    t0 = time.time()
    g, s, _ = load_track("track1")
    tg, _ = pseudo_targets(g, s)
    D.update(s=s, g=g)
    n = len(s)
    R, Hs = {}, {}
    hi_mis, lo_max = {k: 0 for k in CELLS + ["geo"]}, {k: -1.0 for k in CELLS + ["geo"]}
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
    # lock accounting at each cell's own cut (report-only): entries decided by H alone, against the bank decision
    lock = {}
    for k in CELLS:
        r, l_ = k.split("_")
        c = RULES[r]
        cut = np.float32(t1[k]["eer_thr"])
        bank = "bank5" if l_ == "b5" else "bank6"
        imp_h = imp_extra = gen_h = gen_rescued = 0
        for i, t in enumerate(tg):
            H = Hs[i].astype(np.float64)
            o, Lb = R[i][k], R[i][bank]
            if r in ("or50", "or75"):
                cf = np.minimum(Lb, np.float32(0.5))
            else:
                cf = o  # the counterfactual of the b rules needs b; report the H-locked counts only
            mask = np.ones(o.shape[1], bool)
            mask[t] = False
            hi = H >= c
            imp_h += int(hi[:, mask].sum())
            imp_extra += int((hi[:, mask] & (cf[:, mask] < cut)).sum())
            gen_h += int(hi[:, t].sum())
            gen_rescued += int((hi[:, t] & (cf[:, t] < cut)).sum())
        lock[k] = {"impostor_H_ge_c": imp_h, "genuine_H_ge_c": gen_h}
        if r in ("or50", "or75"):
            lock[k].update(impostor_accepted_only_by_H=imp_extra, genuine_accepted_only_by_H=gen_rescued,
                           net_gen_eq_of_H=round(imp_extra / 199.0 - gen_rescued, 1))
    conc = {}
    for k in CELLS:
        bank = "bank5" if k.endswith("b5") else "bank6"
        d = np.array([(per[k][i][0] + per[k][i][1] / 199.0) - (per["geo"][i][0] + per["geo"][i][1] / 199.0)
                      for i in range(n)])
        dbank = np.array([(per[k][i][0] + per[k][i][1] / 199.0) - (per[bank][i][0] + per[bank][i][1] / 199.0)
                          for i in range(n)])
        top = np.argsort(d)[:5]
        conc[k] = {"d_eer_vs_geo": t1[k]["eer_full"] - GEO, "d_eer_vs_bank": t1[k]["eer_full"] - t1[bank]["eer_full"],
                   "d_even": t1[k]["even"] - t1["geo"]["even"], "d_odd": t1[k]["odd"] - t1["geo"]["odd"],
                   "d_no_hard": t1[k]["no_hard"] - t1["geo"]["no_hard"],
                   "top5_vs_geo": [int(i) for i in top], "top5_gen_eq": round(float(d[top].sum()), 1),
                   "better_vs_geo": int((d < -0.5).sum()), "worse_vs_geo": int((d > 0.5).sum()),
                   "better_vs_bank": int((dbank < -0.5).sum()), "worse_vs_bank": int((dbank > 0.5).sum())}
    repro = {k: t1[k]["eer_full"] - v for k, v in (("bank5", BANK5), ("bank6", BANK6), ("geo", GEO))}
    stop = {"bank5_repro": abs(repro["bank5"]) <= TOL, "bank6_repro": abs(repro["bank6"]) <= TOL,
            "geo_repro": abs(repro["geo"]) <= TOL, "finite_in_range": rng_ok,
            "hi_band_unchanged": all(v == 0 for v in hi_mis.values()),
            "low_band_below_075": all(v < HI for v in lo_max.values())}
    build = {}
    for l_ in LOWS:
        cells = sorted((k for k in CELLS if k.endswith(l_)), key=lambda k: t1[k]["eer_full"])
        best = cells[0]
        build[l_] = ("BUILD:" + best) if t1[best]["eer_full"] <= GEO - BAR else "KILL"
    verdict = "STOP" if not all(stop.values()) else build
    print(json.dumps({"stop": stop, "repro_delta": repro, "lock": lock, "cells": conc, "verdict": verdict,
                      "seconds": round(time.time() - t0, 1)}), flush=True)


def synth():
    import x_r4 as R4
    ok = {}
    for seed, kind in ((3, "switch"), (5, "mng"), (7, "switch")):
        x, g = R4._synth_data(seed, 240, 60, kind, norm_hi=26.0)
        o = outputs_fn(x, g)
        H = o.pop("_H")
        ok[f"bank5_equals_ref_{seed}"] = bool(np.array_equal(o["bank5"], B5.bank5_ref(x, g)))
        ok[f"bank6_equals_ref_{seed}"] = bool(np.array_equal(o["bank6"], B6.bank6_ref(x, g)))
        ok[f"geo_equals_stage1d_{seed}"] = bool(np.array_equal(o["geo"], S1D.outputs_fn(x, g)["geo"]))
        ok[f"min50_equals_cmp_{seed}"] = bool(np.array_equal(o["min50_b5"], RC.cmp_(H, o["bank5"], np.asarray(
            gallery_discriminative_row_l2_scores(x, g), np.float32))))
        H64 = H.astype(np.float64)
        for k in CELLS:
            c = RULES[k.split("_")[0]]
            m_, mx = _inv(o[k], H)
            ok[f"{k}_hi_{seed}"] = m_ == 0 and mx < HI
            ok[f"{k}_keeps_H_ge_c_{seed}"] = bool(np.array_equal(o[k][H64 >= c], H[H64 >= c]))
            ok[f"{k}_build_path_{seed}"] = bool(np.array_equal(composite_bank(x, g, k), o[k]))
        ok[f"or_rule_low_band_{seed}"] = bool(np.array_equal(o["or50_b5"][H64 < 0.5],
                                                            np.minimum(o["bank5"], np.float32(0.5))[H64 < 0.5]))
        ok[f"finite_in_range_{seed}"] = all(bool(np.isfinite(v).all() and (np.abs(v) <= 1).all()) for v in o.values())
        for h in (30, 75, 150):
            oh = outputs_fn(x[:h], g)
            ok[f"prefix_exact_{seed}_{h}"] = all(bool(np.array_equal(oh[k], o[k][:h])) for k in SETTINGS)
    good = all(ok.values())
    print(json.dumps({"synth": ok, "all_ok": good}), flush=True)
    return good


if __name__ == "__main__":
    if "--synth" in sys.argv:
        sys.exit(0 if synth() else 1)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
