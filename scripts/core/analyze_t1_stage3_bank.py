"""T1 stage 3: three low-band changes on top of the stage-2 bank composites (analysis only).

Plan, provenance and frozen rules: docs/experiments/20261009_ogata_t1_premise_search.md (stage 3).
Designs: the 10/10 evening consults (docs/experiments/20261010_ogata_final_consults.md).

Parents: or75_bX = H where H >= .75, min(L, .5) elsewhere (stage 2), L = the teammate's bank5 / bank6, H = rtpl_pez,
b = 943955. Every cell keeps every entry with H >= .75 equal to H and every other entry below .75.
  controls  or75_b5, or75_b6       stage 2 (.002502990872331916, .002343785448664354)
            base1                  bl(dframe_nh) from the one-sided P must equal the bank's base (971136) bit for bit
  P0        or75v_bX   or75 with the bank's vetoes (L == -1, H < .75) moved to -1 + .025 (1 + b), ordered by b,
                       instead of one tie block at -1 (Opus lens A; FAR@FRR1e-5 from 1.0, ranked metrics unchanged)
  CONS      cons2v_bX  or75v, and in the bank's free rows (more than one entry > -1) at t >= 10 the column named by
                       at least two of argmax b, argmax hold_nh_c2 and argmax of the undiscounted HCP + FCO output
                       becomes max(L, .5) (GPT lens A; at most one added accept per row)
  C2BASE    c2v_b5     or75v over bank5 on the two-sided (c2) base bl(dframe_nh from two_sided(P)) (Opus lens A P1)
            c2v_b6     the same over bank6 (report only, never built)
Pseudo targets on the evaluator side only. Prints JSON aggregates only.
Usage (server): python scripts/core/analyze_t1_stage3_bank.py [workers<=6] > results/ogata/t1_stage3_bank.log
                python scripts/core/analyze_t1_stage3_bank.py --synth
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

import analyze_fusion_family as FF  # noqa: E402
import analyze_round2 as AR2  # noqa: E402
import analyze_t1_cau2_chain as C2  # noqa: E402
import analyze_t1_stage2_bank as S2  # noqa: E402
import analyze_t2_family as T2F  # noqa: E402
import t1_bank as BK  # noqa: E402
import t1_bank5 as B5  # noqa: E402
import t1_bank6 as B6  # noqa: E402
import x_r4 as R4  # noqa: E402
from rcfa.methods.gallery_discriminative_row_l2 import gallery_discriminative_row_l2_scores  # noqa: E402

OR75 = {"b5": 0.002502990872331916, "b6": 0.002343785448664354}
TOL = 1e-9
BAR = 30 * 489286 ** -1 * 0.5  # 30 gen-eq: 30 / (2 * 489286) = 3.0657e-5
HI = 0.75
VETO_LO, VETO_W = -1.0, 0.025
FREE_FROM = 10
C2_HARD = (170, 185)  # stage 1b: the two sequences carrying most of the two-sided gain
CELLS = ["or75v_b5", "or75v_b6", "cons2v_b5", "cons2v_b6", "c2v_b5", "c2v_b6"]
CONTROLS = ["or75_b5", "or75_b6"]
SETTINGS = CONTROLS + CELLS
REPORT_ONLY = ["c2v_b6"]
D = {}


def veto_band(H32, L32, b32):
    """P0: or75 with the bank's vetoes ordered by b in [-1, -.95] (float64 arithmetic, float32 output)."""
    H64 = H32.astype(np.float64)
    v = np.minimum(L32.astype(np.float64), 0.5)
    m = (L32 == np.float32(-1.0)) & (H64 < HI)
    v[m] = VETO_LO + VETO_W * (1.0 + b32.astype(np.float64)[m])
    return np.where(H64 >= HI, H64, v).astype(np.float32)


def _vote(r):
    """argmax per row; -1 where the row maximum is tied."""
    a = r.argmax(1)
    mx = r[np.arange(r.shape[0]), a]
    tied = (r == mx[:, None]).sum(1) > 1
    return np.where(tied, -1, a)


def consensus(L32, votes):
    """CONS: in free rows at t >= FREE_FROM, the column with at least two of the three votes becomes max(L, .5)."""
    jb, jc, ja = votes
    T = L32.shape[0]
    j = np.where((jb >= 0) & ((jb == jc) | (jb == ja)), jb, np.where((jc >= 0) & (jc == ja), jc, -1))
    free = (L32 > np.float32(-1.0)).sum(1) >= 2
    fire = free & (np.arange(T) >= FREE_FROM) & (j >= 0)
    out = L32.copy()
    idx = np.nonzero(fire)[0]
    if idx.size:
        out[idx, j[idx]] = np.maximum(out[idx, j[idx]], np.float32(0.5))
    return out, fire, j


def parts_all(x, g):
    """Every input of the cells for one sequence (float32 where the bank and the composites use float32)."""
    P = T2F.parts(x, g)
    P2 = C2.two_sided(P)
    H32 = AR2.variants2(x, g, P, only=["rtpl_pez"])["rtpl_pez"].astype(np.float32)
    b32 = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    base = np.asarray(BK._BASE(x, g), np.float32)
    base1 = FF._bl(T2F.variants(x, g, P, only=["dframe_nh"])["dframe_nh"].astype(np.float32), b32).astype(np.float32)
    base2 = FF._bl(T2F.variants(x, g, P2, only=["dframe_nh"])["dframe_nh"].astype(np.float32), b32).astype(np.float32)
    c2 = T2F.variants(x, g, P2, only=["hold_nh"])["hold_nh"]
    a = R4.fco_core(P["num"], P["den"], P["C"], P["M"], P["K"])[0]
    return dict(H=H32, b=b32, base=base, base1=base1, base2=base2,
                votes=(_vote(np.asarray(b32, np.float64)), _vote(np.asarray(c2, np.float64)), _vote(np.asarray(a, np.float64))))


def outputs_fn(x, g):
    """Every setting (float32) for one sequence plus H (key "_H") and diagnostics (keys starting with "_")."""
    Q = parts_all(x, g)
    H32, b32 = Q["H"], Q["b"]
    L = {"b5": B5.post5(Q["base"], x, g).astype(np.float32), "b6": B6.post6(Q["base"], x, g).astype(np.float32)}
    Lc2 = {"b5": B5.post5(Q["base2"], x, g).astype(np.float32), "b6": B6.post6(Q["base2"], x, g).astype(np.float32)}
    out = {"_H": H32, "_base1_equal": np.array([np.array_equal(Q["base1"], Q["base"])])}
    for k in ("b5", "b6"):
        out[f"or75_{k}"] = S2.rule("or75", H32, L[k], b32)
        out[f"or75v_{k}"] = veto_band(H32, L[k], b32)
        Lc, fire, j = consensus(L[k], Q["votes"])
        out[f"cons2v_{k}"] = veto_band(H32, Lc, b32)
        out[f"_fire_{k}"] = fire
        out[f"_j_{k}"] = j
        out[f"_L_{k}"] = L[k]
        out[f"c2v_{k}"] = veto_band(H32, Lc2[k], b32)
    out["_b6_ne_b5"] = np.array([int((L["b6"] != L["b5"]).any(1).sum())])
    return out


def composite_stage3(x, g, cell):
    """The build path for one stage-3 cell (only what the cell needs is computed)."""
    kind, k = cell.split("_")
    P = T2F.parts(x, g)
    H32 = AR2.variants2(x, g, P, only=["rtpl_pez"])["rtpl_pez"].astype(np.float32)
    b32 = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    post = B5.post5 if k == "b5" else B6.post6
    if kind == "c2v":
        P2 = C2.two_sided(P)
        base2 = FF._bl(T2F.variants(x, g, P2, only=["dframe_nh"])["dframe_nh"].astype(np.float32), b32).astype(np.float32)
        return veto_band(H32, post(base2, x, g).astype(np.float32), b32)
    L32 = post(np.asarray(BK._BASE(x, g), np.float32), x, g).astype(np.float32)
    if kind == "or75v":
        return veto_band(H32, L32, b32)
    if kind == "cons2v":
        P2 = C2.two_sided(P)
        c2 = T2F.variants(x, g, P2, only=["hold_nh"])["hold_nh"]
        a = R4.fco_core(P["num"], P["den"], P["C"], P["M"], P["K"])[0]
        votes = (_vote(np.asarray(b32, np.float64)), _vote(np.asarray(c2, np.float64)), _vote(np.asarray(a, np.float64)))
        return veto_band(H32, consensus(L32, votes)[0], b32)
    raise KeyError(cell)


def _inv(o, H):
    hi = H.astype(np.float64) >= HI
    return int((o[hi] != H[hi]).sum()), float(o[~hi].max()) if (~hi).any() else -1.0


def _job(i):
    o = outputs_fn(D["s"][i], D["g"])
    H = o.pop("_H")
    diag = {k: o.pop(k) for k in [k for k in o if k.startswith("_")]}
    chk = {k: _inv(o[k], H) for k in SETTINGS}
    return i, o, H, diag, chk


def decide(t1, stop, flat_seqs):
    """Apply the frozen gates without data access; P0 must be adopted before a P0-bearing build."""
    verdict = {}
    for k in ("b5", "b6"):
        par, p0 = t1[f"or75_{k}"], t1[f"or75v_{k}"]
        same = all(p0[field] == par[field] for field in ("eer_full", "eer_thr", "far1e5_thr"))
        verdict[f"P0_{k}"] = ("ADOPT" if p0["far_at_frr1e5"] < 0.5 else "KEEP_PLAIN") if same else "STOP"
        c = t1[f"cons2v_{k}"]
        passes = c["eer_full"] <= par["eer_full"] - BAR and c["even"] <= par["even"] and c["odd"] <= par["odd"]
        verdict[f"CONS_{k}"] = ("BUILD" if passes else "KILL") if verdict[f"P0_{k}"] == "ADOPT" else "HOLD_P0_NOT_ADOPTED"
    premise = bool(flat_seqs) and all(i in C2_HARD for i in flat_seqs)
    c, par = t1["c2v_b5"], t1["or75_b5"]
    passes = c["eer_full"] <= par["eer_full"] - BAR and c["even"] <= par["even"] and c["odd"] <= par["odd"]
    verdict["C2BASE_b5"] = (("BUILD" if passes else "KILL") if premise else "KILL_PREMISE") \
        if verdict["P0_b5"] == "ADOPT" else "HOLD_P0_NOT_ADOPTED"
    failed = not all(stop.values()) or any(verdict[f"P0_{k}"] == "STOP" for k in ("b5", "b6"))
    if not all(stop.values()):
        verdict = "STOP"
    return {"verdict": verdict, "premise_c2_flat_in_hard": premise, "exit_code": 2 if failed else 0}


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
    R, Hs, DG = {}, {}, {}
    hi_mis, lo_max = {k: 0 for k in SETTINGS}, {k: -1.0 for k in SETTINGS}
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 6)), mp_context=mp.get_context("fork")) as ex:
        for i, o, H, diag, chk in ex.map(_job, range(n), chunksize=1):
            R[i], Hs[i], DG[i] = o, H, diag
            for k, (m_, mx) in chk.items():
                hi_mis[k] += m_
                lo_max[k] = max(lo_max[k], mx)
    rng_ok = all(bool(np.isfinite(v).all() and (np.abs(v) <= 1).all()) for i in range(n) for v in R[i].values())
    base1_ok = all(bool(DG[i]["_base1_equal"][0]) for i in range(n))
    flat_seqs = [i for i in range(n) if int(DG[i]["_b6_ne_b5"][0]) > 0]
    print(json.dumps({"stage": "outputs", "sequences": n, "finite_in_range": rng_ok, "base1_equal_bank_base": base1_ok,
                      "hi_band_mismatch": hi_mis, "low_band_max": lo_max,
                      "b6_differs_from_b5": {str(i): int(DG[i]["_b6_ne_b5"][0]) for i in flat_seqs},
                      "seconds": round(time.time() - t0, 1)}), flush=True)
    even, odd = np.arange(0, n, 2), np.arange(1, n, 2)
    t1, per = {}, {}
    for k in SETTINGS:
        gen, imp = ar._gen_imp([R[i][k] for i in range(n)], tg)
        m = ar._metrics(gen, imp)
        del gen, imp
        gm, ia, per[k] = _counts(R, tg, k, np.float32(m["eer_thr"]))
        sub = {}
        for nm, idx in (("even", even), ("odd", odd)):
            gg, ii = ar._gen_imp([R[i][k] for i in idx], tg[idx])
            sub[nm] = cc._eer_full(gg, ii)
        t1[k] = {"eer_full": m["eer_full"], "eer_thr": m["eer_thr"], "far1e5_thr": m["far1e5_thr"],
                 "frr_at_far1e5": m["frr_at_far1e5"], "far_at_frr1e5": m["far_at_frr1e5"], **sub,
                 "gen_eq_by_bin": {nm: round(float(gm[j] + ia[j] / 199.0), 1) for j, nm in enumerate(NAMES)}}
        print(json.dumps({"setting": k, **t1[k]}), flush=True)
    # consensus accounting at the parent's cut (report only): fired rows, added accepts, rescued genuine entries
    cons = {}
    for k in ("b5", "b6"):
        cut = np.float32(t1[f"or75_{k}"]["eer_thr"])
        fired = added = rescued = hit = 0
        for i, t in enumerate(tg):
            f, j, L32 = DG[i][f"_fire_{k}"], DG[i][f"_j_{k}"], DG[i][f"_L_{k}"]
            idx = np.nonzero(f)[0]
            if not idx.size:
                continue
            Hrow = Hs[i][idx, j[idx]].astype(np.float64)
            was = (L32[idx, j[idx]] >= cut) | (Hrow >= HI)
            fired += idx.size
            added += int((~was).sum())
            hit += int((j[idx] == t).sum())
            rescued += int(((~was) & (j[idx] == t)).sum())
        cons[k] = {"fired_rows": fired, "added_accepts": added, "target_named": hit, "genuine_rescued": rescued,
                   "fixed_cut_gain_gen_eq": round(rescued - (added - rescued) / 199.0, 1)}
    cells = {}
    for k in CELLS:
        par = "or75_" + k.split("_")[1]
        d = np.array([(per[k][i][0] + per[k][i][1] / 199.0) - (per[par][i][0] + per[par][i][1] / 199.0) for i in range(n)])
        top = np.argsort(d)[:5]
        cells[k] = {"d_eer": t1[k]["eer_full"] - t1[par]["eer_full"], "d_even": t1[k]["even"] - t1[par]["even"],
                    "d_odd": t1[k]["odd"] - t1[par]["odd"], "top5": [int(i) for i in top],
                    "top5_gen_eq": round(float(d[top].sum()), 1), "better": int((d < -0.5).sum()), "worse": int((d > 0.5).sum())}
    repro = {k: t1[f"or75_{k}"]["eer_full"] - v for k, v in OR75.items()}
    stop = {"or75_b5_repro": abs(repro["b5"]) <= TOL, "or75_b6_repro": abs(repro["b6"]) <= TOL, "base1_equal": base1_ok,
            "finite_in_range": rng_ok, "hi_band_unchanged": all(v == 0 for v in hi_mis.values()),
            "low_band_below_075": all(v < HI for v in lo_max.values())}
    decision = decide(t1, stop, flat_seqs)
    print(json.dumps({"stop": stop, "repro_delta": repro, "consensus": cons, "cells": cells, **decision,
                      "seconds": round(time.time() - t0, 1)}), flush=True)
    return decision["exit_code"]


def synth():
    ok = {}
    rng = np.random.default_rng(11)
    # veto_band on a hand-made matrix
    H = np.array([[0.9, 0.2, 0.6], [0.1, 0.8, -0.3]], np.float32)
    L = np.array([[-1.0, -1.0, 0.7], [-1.0, 0.3, -0.2]], np.float32)
    b = np.array([[0.5, -0.2, 0.1], [1.0, 0.0, 0.3]], np.float32)
    v = veto_band(H, L, b)
    exp = np.array([[0.9, -1 + 0.025 * 0.8, 0.5], [-1 + 0.025 * 2.0, 0.8, -0.2]], np.float32)
    ok["veto_band_hand"] = bool(np.array_equal(v, exp))
    # consensus votes and firing on a hand-made matrix
    Lc = np.full((12, 4), -1.0, np.float32)
    Lc[10] = [0.1, -0.3, 0.2, -0.5]
    Lc[11, 2] = 0.4
    vb = np.array([0] * 10 + [1, 1])
    vc = np.array([0] * 10 + [1, 2])
    va = np.array([0] * 10 + [3, 2])
    o, fire, j = consensus(Lc, (vb, vc, va))
    ok["consensus_hand"] = bool(fire.tolist() == [False] * 10 + [True, False] and o[10, 1] == 0.5
                                and np.array_equal(np.delete(o, 1, axis=1)[10], np.delete(Lc, 1, axis=1)[10])
                                and np.array_equal(o[11], Lc[11]))
    ok["vote_tie_abstains"] = bool(_vote(np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 1.0]])).tolist() == [-1, 1])
    for seed, kind in ((3, "switch"), (5, "mng"), (7, "switch")):
        x, g = R4._synth_data(seed, 240, 60, kind, norm_hi=26.0)
        o = outputs_fn(x, g)
        H = o.pop("_H")
        ok[f"base1_equal_{seed}"] = bool(o["_base1_equal"][0])
        diag = {k: o.pop(k) for k in [k for k in o if k.startswith("_")]}
        ref2 = S2.outputs_fn(x, g)
        ok[f"or75_equal_stage2_{seed}"] = all(bool(np.array_equal(o[f"or75_{k}"], ref2[f"or75_{k}"])) for k in ("b5", "b6"))
        H64 = H.astype(np.float64)
        for k in SETTINGS:
            m_, mx = _inv(o[k], H)
            ok[f"{k}_hi_{seed}"] = m_ == 0 and mx < HI
        for k in ("b5", "b6"):
            p, v = o[f"or75_{k}"], o[f"or75v_{k}"]
            moved = p != v
            ok[f"P0_moves_only_vetoes_{k}_{seed}"] = bool((p[moved] == -1).all() and (v[moved] >= -1).all()
                                                          and (v[moved] <= -0.95).all())
            c = o[f"cons2v_{k}"]
            ch = (c != v).any(1)
            ok[f"cons_changes_fired_rows_only_{k}_{seed}"] = bool(np.array_equal(ch, ch & diag[f"_fire_{k}"]))
            ok[f"cons_at_most_one_per_row_{k}_{seed}"] = bool(((c != v).sum(1) <= 1).all())
        for cell in CELLS:
            ok[f"{cell}_build_path_{seed}"] = bool(np.array_equal(composite_stage3(x, g, cell), o[cell]))
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
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 6))
