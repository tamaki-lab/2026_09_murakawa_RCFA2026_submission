"""Band-split composite cmp(H | L) and hold_rtpl_nh: stage 0 (analysis only).

Plan and frozen gates: docs/experiments/20261009_ogata_rtpl_cmp.md.

cmp(H | L) = H if H >= .5 else (min(L, H) + b) / 2, with b = 943955's row-L2 readout (the fusion family's C0 and
partner, in the float32/float64 arithmetic of analyze_fusion_family._bl). Entries with H >= .5 are H's own and every
changed entry is below .75, under H's FAR=1e-5 threshold, so the operating threshold, every T2 accept decision and BE
are H's; T1-EER comes from the low band. cmp(H | H) is bl(H) bit for bit.
hold_rtpl_nh: rtpl_nh with the hold admission on the residual template. A verified clean frame joins column j's
template if the template is empty, if its cue value q_tj is undefined, if no defined value has been admitted yet, or if
q_tj >= the mean of the previously admitted defined values of j. The cue, the ratio (prefix mean over defined frames)
and the PEZ-form discount stacked on PEZ-NH are rtpl_nh's unchanged; admitting every frame gives rtpl_nh bit for bit.
Pseudo targets on the evaluator side only. Prints JSON aggregates only.
Usage (server): python scripts/core/analyze_rtpl_cmp.py [workers<=6] > results/ogata/rtpl_cmp.log
                python scripts/core/analyze_rtpl_cmp.py --synth
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
import analyze_t2_family as T2F  # noqa: E402  (imports x_r5 first: pins BLAS threads)
import numpy as np  # noqa: E402

import analyze_fusion_family as FF  # noqa: E402
import x_r4 as R4  # noqa: E402
import x_r5 as R5  # noqa: E402
from rcfa.methods.gallery_subspace_residual import GallerySubspace  # noqa: E402
from rcfa.methods.gallery_discriminative_row_l2 import gallery_discriminative_row_l2_scores  # noqa: E402

RS, DC, HV = T2F.RS, T2F.DC, T2F.HV
ZIPS = {"bl_rtpl_nh": (RS.ART / "tfbl_rtpl_nh" / "submission.zip",
                       "0079bb9515eff1549324408f321a439f3101924ba8bc65ede7f69fcceaffe41d"),
        "bl_hold_nh": (RS.ART / "tfbl_hold_nh" / "submission.zip",
                       "4849292fea5b4ebdbcfc723b2af87d8b838cb3a86c7737cbe5d961a43647521e")}
BL_PROXY = {"bl_rtpl_nh": 0.0029567543164168006, "bl_hold_nh": 0.002944727767301255}
CMP_EER_MAX = BL_PROXY["bl_rtpl_nh"] - 1e-5        # the fusion family's 1e-5 build bar under bl_rtpl_nh
HI_MAX = 0.75
KEYS = ["nh", "hold_nh", "rtpl_nh", "hold_rtpl_nh", "bl_rtpl_nh", "bl_hold_nh",
        "cmp_rtpl_nh__hold_nh", "cmp_hold_rtpl_nh__hold_nh"]
CMPS = {"cmp_rtpl_nh__hold_nh": ("rtpl_nh", "hold_nh"), "cmp_hold_rtpl_nh__hold_nh": ("hold_rtpl_nh", "hold_nh")}
D = {}


def rtpl_s2(P, g, hold):
    """analyze_t2_family.cue_s2 'rtpl' (same operations in the same order) with an optional hold admission."""
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
    dfn = np.zeros((T, K), bool)
    tmpl, sq, nq = {}, {}, {}
    vc = P["v"] & P["clean"][:, None]
    for t in range(T):
        Rn = np.linalg.norm(R[t])
        for j, s in tmpl.items():
            sn = np.linalg.norm(s)
            if Rn > 0 and sn > 0:
                q[t, j] = max(0.0, float(R[t] @ s) / (Rn * sn))
                dfn[t, j] = True
        for j in np.nonzero(vc[t])[0]:
            j = int(j)
            if hold and j in tmpl and dfn[t, j]:
                if nq.get(j, 0) and q[t, j] < sq[j] / nq[j]:
                    continue
                sq[j] = sq.get(j, 0.0) + q[t, j]
                nq[j] = nq.get(j, 0) + 1
            tmpl[j] = tmpl.get(j, 0.0) + rh[t]
    return T2F.ratio_prefix(q, defined=dfn)


def stacked(P, s2c):
    """analyze_t2_family._table's stacked(): PEZ-NH times the PEZ-form discount of s2c."""
    K, zP, zS, LP, LS, oE, oV = P["K"], P["zP"], P["f"]["zS"], P["LP"], P["LS"], P["oE"], P["oV"]
    w_nh = oE * oV * P["oEn_nh"]
    om = w_nh * T2F.oen(s2c, zS, K, T2F.act_of(s2c, LS, zP, oV))
    return R4.fco_core(R5._numN(LP, LS, om), P["den"], P["C"], P["M"], K)[0]


def base_variants(x, g, P, names):
    """float64 outputs for names in {nh, hold_nh, rtpl_nh, hold_rtpl_nh}; the T2 family table where it has them."""
    out = {}
    fam = [n for n in names if n != "hold_rtpl_nh"]
    if fam:
        out.update(T2F.variants(x, g, P, only=fam))
    if "hold_rtpl_nh" in names:
        out["hold_rtpl_nh"] = stacked(P, rtpl_s2(P, g, hold=True))
    return out


def cmp_(Hs, Ls, b):
    """Band-split composite in analyze_fusion_family._bl's arithmetic; cmp_(H, H, b) == _bl(H, b)."""
    H32 = np.asarray(Hs, np.float32)
    lo = np.minimum(np.asarray(Ls, np.float32), H32)
    H64 = H32.astype(np.float64)
    return np.where(H64 >= FF.C0, H64, FF._mean(lo, np.asarray(b, np.float32))).astype(np.float32)


def outputs_fn(x, g):
    """Reference for build_candidate's bit-identity check: every key of KEYS as float32."""
    P = T2F.parts(x, g)
    V = {k: v.astype(np.float32) for k, v in base_variants(x, g, P, ["nh", "hold_nh", "rtpl_nh", "hold_rtpl_nh"]).items()}
    b = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    V["bl_rtpl_nh"] = FF._bl(V["rtpl_nh"], b).astype(np.float32)
    V["bl_hold_nh"] = FF._bl(V["hold_nh"], b).astype(np.float32)
    for k, (h, l_) in CMPS.items():
        V[k] = cmp_(V[h], V[l_], b)
    return V


def composite(x, g, h, l_):
    """The build path for cmp(h | l_): only the two parents and the partner are computed."""
    P = T2F.parts(x, g)
    V = base_variants(x, g, P, sorted({h, l_}))
    b = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    return cmp_(V[h].astype(np.float32), V[l_].astype(np.float32), b)


# ---------------------------------------------------------------- run
def _job(task):
    tr, i = task
    x, g = D[f"s{tr}"][i], D[f"g{tr}"]
    out = outputs_fn(x, g)
    rep = {}
    for k, z in (("nh", "ZA"), ("bl_rtpl_nh", "ZR"), ("bl_hold_nh", "ZH")):
        ref = D[f"{z}{tr}"][i]
        rep[k] = (int((out[k] != ref).sum()), float(np.abs(out[k].astype(np.float64) - ref.astype(np.float64)).max()))
    inv = {}
    for k, (h, l_) in CMPS.items():
        hi = out[h] >= np.float32(FF.C0)
        lo_vals = out[k][~hi]
        inv[k] = {"hi_changed": int((out[k][hi] != out[h][hi]).sum()),
                  "lo_max": float(lo_vals.max()) if lo_vals.size else -1.0,
                  "min_binds": int(((out[l_] > out[h]) & ~hi).sum()), "lo_entries": int((~hi).sum())}
    if tr == 1:
        return tr, i, out, rep, inv
    c = int(D["c2"][i])
    return tr, i, {k: v[:, c].copy() for k, v in out.items()}, rep, inv


def main(workers=4):
    import analyze_t1_residual as ar
    import common_check as cc
    from proxy_lib import load_track, pseudo_targets
    t0 = time.time()
    stop = {"zip_sha_nh": RS._sha(DC.ZIP_NH) == DC.SHA_NH}
    stop.update({f"zip_sha_{k}": RS._sha(p) == s for k, (p, s) in ZIPS.items()})
    if not all(stop.values()):
        print(json.dumps({"stop": stop, "verdict": "STOP"}), flush=True)
        return
    g1, s1, n1 = load_track("track1")
    g2, s2_, n2 = load_track("track2")
    c1, _ = pseudo_targets(g1, s1)
    c2, _ = pseudo_targets(g2, s2_)
    D.update(s1=s1, g1=g1, s2=s2_, g2=g2, c2=c2)
    for z, p in (("ZA", DC.ZIP_NH), ("ZR", ZIPS["bl_rtpl_nh"][0]), ("ZH", ZIPS["bl_hold_nh"][0])):
        S = HV._read_zip(p)
        D[f"{z}1"] = [S["track1"][f"score_{n}.npz"] for n in n1]
        D[f"{z}2"] = [S["track2"][f"score_{n}.npz"] for n in n2]
        del S
    tasks = [(1, i) for i in range(len(s1))] + [(2, i) for i in range(len(s2_))]
    R = {1: {}, 2: {}}
    rep = {k: [0, 0.0] for k in ("nh", "bl_rtpl_nh", "bl_hold_nh")}
    inv = {k: {"hi_changed": 0, "lo_max": -1.0, "min_binds": 0, "lo_entries": 0} for k in CMPS}
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 6)), mp_context=mp.get_context("fork")) as ex:
        for tr, i, res, rp, iv in ex.map(_job, tasks, chunksize=1):
            R[tr][i] = res
            for k, (nd, mx) in rp.items():
                rep[k][0] += nd
                rep[k][1] = max(rep[k][1], mx)
            for k, d in iv.items():
                inv[k]["hi_changed"] += d["hi_changed"]
                inv[k]["lo_max"] = max(inv[k]["lo_max"], d["lo_max"])
                inv[k]["min_binds"] += d["min_binds"]
                inv[k]["lo_entries"] += d["lo_entries"]
    for z in ("ZA", "ZR", "ZH"):
        D.pop(f"{z}1")
        D.pop(f"{z}2")
    stop["repro_nh"] = rep["nh"][0] <= T2F.REPRO_NDIFF and rep["nh"][1] <= T2F.REPRO_MAX
    stop["submitted_bl_rtpl_nh_exact"] = rep["bl_rtpl_nh"][0] == 0
    stop["submitted_bl_hold_nh_exact"] = rep["bl_hold_nh"][0] == 0
    n1s, n2s = len(s1), len(s2_)
    t1 = {}
    for k in KEYS:
        gen, imp = ar._gen_imp([R[1][i][k] for i in range(n1s)], c1)
        met = ar._metrics(gen, imp)
        t1[k] = {"eer_full": cc._eer_full(gen, imp), "eer_thr": met["eer_thr"], "far1e5_thr": met["far1e5_thr"],
                 "frr_far1e5": met["frr_at_far1e5"], "far_at_frr1e5": met["far_at_frr1e5"]}
        del gen, imp
    stop["nh_eer"] = abs(t1["nh"]["eer_full"] - T2F.NH["eer_full"]) <= 1e-9
    for k, v in BL_PROXY.items():
        stop[f"{k}_eer"] = abs(t1[k]["eer_full"] - v) <= 1e-12
    acc = {}
    for k in KEYS:
        thr = np.float32(t1[k]["far1e5_thr"])
        acc[k] = np.concatenate([R[2][i][k] >= thr for i in range(n2s)])
    t2 = {k: {"acc_share": float(acc[k].mean())} for k in KEYS}

    def pair(k, b):
        d2 = float(acc[b].mean() - acc[k].mean())
        g1_ = t1[k]["frr_far1e5"] - t1[b]["frr_far1e5"]
        return {"base": b, "d2": d2, "g1": g1_, "ratio": d2 / g1_ if g1_ > 0 else None,
                "down": int((acc[b] & ~acc[k]).sum()), "up": int((acc[k] & ~acc[b]).sum())}
    t2["hold_rtpl_nh"]["vs_rtpl_nh"] = pair("hold_rtpl_nh", "rtpl_nh")
    t2["hold_rtpl_nh"]["vs_nh"] = pair("hold_rtpl_nh", "nh")
    t2["rtpl_nh"]["vs_nh"] = pair("rtpl_nh", "nh")
    parent_ratio = t2["rtpl_nh"]["vs_nh"]["ratio"]
    gate = {}
    for k, (h, _) in CMPS.items():
        gate[k] = {"thr_equal": t1[k]["far1e5_thr"] == t1[h]["far1e5_thr"],
                   "t2_accept_equal": bool(np.array_equal(acc[k], acc[h])),
                   "hi_unchanged": inv[k]["hi_changed"] == 0, "lo_below": inv[k]["lo_max"] < HI_MAX,
                   "eer": t1[k]["eer_full"] <= CMP_EER_MAX}
    m = t2["hold_rtpl_nh"]["vs_rtpl_nh"]
    gate["hold_rtpl_nh"] = {"eer": t1["hold_rtpl_nh"]["eer_full"] <= t1["rtpl_nh"]["eer_full"] + T2F.GATES["eer_max_vs_base"],
                            "thr": t1["hold_rtpl_nh"]["far1e5_thr"] >= T2F.GATES["thr_min"],
                            "frr": t1["hold_rtpl_nh"]["frr_far1e5"] <= T2F.GATES["frr_max"],
                            "adds_rejections": m["d2"] > 0,
                            "rate": m["g1"] <= 0 or (parent_ratio is not None and m["d2"] / m["g1"] >= parent_ratio)}
    gate["cmp_hold_rtpl_nh__hold_nh"]["parent_pass"] = all(gate["hold_rtpl_nh"].values())
    build = [k for k in CMPS if all(gate[k].values())]
    verdict = "STOP" if not all(stop.values()) else ("BUILD:" + ",".join(build) if build else "NONE")
    print(json.dumps({"stop": stop, "repro": rep, "t1": t1, "t2": t2, "invariance": inv, "gate": gate,
                      "parent_ratio_rtpl_vs_nh": parent_ratio, "cmp_eer_max": CMP_EER_MAX}), flush=True)
    print(json.dumps({"verdict": verdict, "seconds": round(time.time() - t0, 1)}), flush=True)


def synth():
    ok = {}
    for seed, kind in ((3, "switch"), (5, "mng")):
        x, g = R4._synth_data(seed, 240, 60, kind, norm_hi=26.0)
        P = T2F.parts(x, g)
        V = T2F.variants(x, g, P, only=["nh", "hold_nh", "rtpl_nh"])
        ok[f"rtpl_admit_all_eq_{seed}"] = bool(np.array_equal(stacked(P, rtpl_s2(P, g, hold=False)), V["rtpl_nh"]))
        hr = stacked(P, rtpl_s2(P, g, hold=True))
        ok[f"hold_rtpl_le_nh_{seed}"] = bool((hr <= V["nh"] + 1e-12).all())
        O = outputs_fn(x, g)
        b = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
        for h in ("rtpl_nh", "hold_nh", "hold_rtpl_nh"):
            ok[f"cmp_self_eq_bl_{h}_{seed}"] = bool(np.array_equal(cmp_(O[h], O[h], b), FF._bl(O[h], b).astype(np.float32)))
        ok[f"bl_eq_t2_family_{seed}"] = bool(np.array_equal(O["bl_rtpl_nh"], T2F.bl_variant(x, g, "rtpl_nh"))
                                             and np.array_equal(O["bl_hold_nh"], T2F.bl_variant(x, g, "hold_nh")))
        for k, (h, l_) in CMPS.items():
            hi = O[h] >= np.float32(0.5)
            ok[f"{k}_keeps_hi_{seed}"] = bool(np.array_equal(O[k][hi], O[h][hi]) and (O[k][~hi] < np.float32(HI_MAX)).all())
            ok[f"{k}_lean_eq_full_{seed}"] = bool(np.array_equal(composite(x, g, h, l_), O[k]))
        ok[f"finite_in_range_{seed}"] = all(bool(np.isfinite(v).all() and (np.abs(v) <= 1).all()) for v in O.values())
        cut = 150
        Oh = outputs_fn(x[:cut], g)
        bad = [k for k in O if not np.array_equal(Oh[k], O[k][:cut])]
        ok[f"prefix_exact_{seed}"] = not bad
        if bad:
            ok[f"prefix_bad_{seed}"] = bad
        ok[f"hold_admission_active_{seed}"] = bool(np.abs(hr - V["rtpl_nh"]).max() > 0)
    allok = all(v is True for k, v in ok.items() if not k.startswith("hold_admission_active"))
    print(json.dumps({"synth": ok, "all_ok": allok}), flush=True)
    return allok


if __name__ == "__main__":
    if sys.argv[1:] == ["--synth"]:
        sys.exit(0 if synth() else 1)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
