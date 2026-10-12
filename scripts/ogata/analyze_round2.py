"""Round-2 screen on rtpl_nh: five fixed variants, label-free stage 0, upload form cmp(H | hold_nh) (analysis only).

Plan, review and frozen gates: docs/experiments/20261009_ogata_round2_rtpl_screen.md.

rt is rtpl's ratio (analyze_t2_family.cue_s2 'rtpl'); fac(s) = oen(s, zS, K, act_of(s, LS, zP, oV)) is the PEZ form;
w_nh = oE oV oEn_nh is PEZ-NH's weight and oE oV oEn PEZ's.
  rtpl_pez      out(oE oV oEn fac(rt))           rtpl on the hist-gated PEZ base
  rtpl_prof_nh  out(w_nh fac(rt) fac(prof))       stack with the driver-trace profile cue
  rtpl_span_nh  out(w_nh fac(rt) fac(span))       stack with the in-span energy fraction cue
  rtpl_mem      out(w_nh fac(max(rt, rm)))        rm: the same ratio of the memoryless cue max(0, rh_t . s / |s|),
                                                  s the strictly past template, on rtpl's defined entries
  rtpl_min      out(oE oV min(oEn_nh, fac(rt)))   the deeper of the energy and rtpl factors instead of their product
  rtpl_prof_and out(w_nh fac(max(rt, prof)))      report only, never built
No new constant. A selected variant is uploaded as cmp(H | hold_nh) (analyze_rtpl_cmp.cmp_).
Pseudo targets on the evaluator side only. Prints JSON aggregates only.
Usage (server): python scripts/ogata/analyze_round2.py [workers<=6] > results/ogata/round2.log
                python scripts/ogata/analyze_round2.py --synth
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
import analyze_rtpl_cmp as RC  # noqa: E402
import x_r4 as R4  # noqa: E402
import x_r5 as R5  # noqa: E402
from rcfa.methods.gallery_subspace_residual import GallerySubspace  # noqa: E402
from rcfa.methods.gallery_discriminative_row_l2 import gallery_discriminative_row_l2_scores  # noqa: E402

RS, DC, HV = T2F.RS, T2F.DC, T2F.HV
VARS = ["rtpl_pez", "rtpl_prof_nh", "rtpl_span_nh", "rtpl_mem", "rtpl_min"]
REPORT = ["rtpl_prof_and"]
KIND = {"rtpl_pez": "pez", "rtpl_prof_nh": "stack", "rtpl_span_nh": "stack", "rtpl_mem": "release",
        "rtpl_min": "release", "rtpl_prof_and": "release"}
BASES = ["nh", "pez", "hold_nh", "rtpl_nh"]
CMPS = {f"cmp_{h}__hold_nh": h for h in VARS}
ZIPS = {"nh": (DC.ZIP_NH, DC.SHA_NH),
        "pez": (RS.ART / "pez_blvs_n_fco" / "submission.zip",
                "f28ce5876ad7e18c0598c07e0a637059958f8d62b52140e6578d7080fc1445b0"),
        "bl_rtpl_nh": RC.ZIPS["bl_rtpl_nh"]}
RTPL_REC = {"eer_full": 0.002989023673650839, "far1e5_thr": 0.8430233597755432, "frr_far1e5": 0.28788070780688596}
CMP_EER_MAX = RC.CMP_EER_MAX
POS = [(0, 50), (50, 200), (200, None)]
D = {}


def rtpl_q(P, g):
    """analyze_t2_family.cue_s2 'rtpl' loop (same operations in the same order), also returning the memoryless cue."""
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
    qm = np.zeros((T, K))
    dfn = np.zeros((T, K), bool)
    tmpl = {}
    vc = P["v"] & P["clean"][:, None]
    for t in range(T):
        Rn = np.linalg.norm(R[t])
        for j, s in tmpl.items():
            sn = np.linalg.norm(s)
            if Rn > 0 and sn > 0:
                q[t, j] = max(0.0, float(R[t] @ s) / (Rn * sn))
                qm[t, j] = max(0.0, float(rh[t] @ s) / sn)
                dfn[t, j] = True
        for j in np.nonzero(vc[t])[0]:
            tmpl[int(j)] = tmpl.get(int(j), 0.0) + rh[t]
    return q, qm, dfn


def variants2(x, g, P=None, only=None):
    """float64 outputs of the bases, the five variants and the report-only one (or only the named ones)."""
    P = T2F.parts(x, g) if P is None else P
    K, zP, zS, LP, LS, oE, oV = P["K"], P["zP"], P["f"]["zS"], P["LP"], P["LS"], P["oE"], P["oV"]
    w_nh = oE * oV * P["oEn_nh"]
    memo = {}

    def lazy(key, fn):
        if key not in memo:
            memo[key] = fn()
        return memo[key]

    def out(om):
        return R4.fco_core(R5._numN(LP, LS, om), P["den"], P["C"], P["M"], K)[0]

    def fac(s):
        return T2F.oen(s, zS, K, T2F.act_of(s, LS, zP, oV))

    def rq():
        return lazy("rq", lambda: rtpl_q(P, g))

    def rt():
        return lazy("rt", lambda: T2F.ratio_prefix(rq()[0], defined=rq()[2]))

    def rm():
        return lazy("rm", lambda: T2F.ratio_prefix(rq()[1], defined=rq()[2]))

    def fr():
        return lazy("fr", lambda: fac(rt()))

    def pr():
        return lazy("pr", lambda: T2F.cue_s2(P, x, g, "prof"))

    def sp():
        return lazy("sp", lambda: T2F.cue_s2(P, x, g, "span"))

    T = {"rtpl_pez": lambda: out(oE * oV * P["oEn"] * fr()),
         "rtpl_prof_nh": lambda: out(w_nh * fr() * fac(pr())),
         "rtpl_span_nh": lambda: out(w_nh * fr() * fac(sp())),
         "rtpl_mem": lambda: out(w_nh * fac(np.maximum(rt(), rm()))),
         "rtpl_min": lambda: out(oE * oV * np.minimum(P["oEn_nh"], fr())),
         "rtpl_prof_and": lambda: out(w_nh * fac(np.maximum(rt(), pr())))}
    names = BASES + VARS + REPORT if only is None else list(only)
    base = [n for n in names if n in BASES]
    res = T2F.variants(x, g, P, only=base) if base else {}
    res.update({n: T[n]() for n in names if n not in BASES})
    return res


def outputs_fn(x, g):
    """Reference for build_candidate's bit-identity check: bases, variants, bl_rtpl_nh and every composite (float32)."""
    V = {k: v.astype(np.float32) for k, v in variants2(x, g).items()}
    b = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    V["bl_rtpl_nh"] = FF._bl(V["rtpl_nh"], b).astype(np.float32)
    for k, h in CMPS.items():
        V[k] = RC.cmp_(V[h], V["hold_nh"], b)
    return V


def composite(x, g, h):
    """The build path for cmp(h | hold_nh): only h, hold_nh and the partner are computed."""
    P = T2F.parts(x, g)
    V = variants2(x, g, P, only=["hold_nh", h])
    b = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    return RC.cmp_(V[h].astype(np.float32), V["hold_nh"].astype(np.float32), b)


# ---------------------------------------------------------------- run
def _job(task):
    tr, i = task
    x, g = D[f"s{tr}"][i], D[f"g{tr}"]
    out = outputs_fn(x, g)
    rep = {}
    for k, z in (("nh", "Znh"), ("pez", "Zpez"), ("bl_rtpl_nh", "Zbl_rtpl_nh")):
        ref = D[f"{z}{tr}"][i]
        rep[k] = (int((out[k] != ref).sum()), float(np.abs(out[k].astype(np.float64) - ref.astype(np.float64)).max()))
    inv = {}
    for k, h in CMPS.items():
        hi = out[h] >= np.float32(FF.C0)
        lo_vals = out[k][~hi]
        inv[k] = {"hi_changed": int((out[k][hi] != out[h][hi]).sum()),
                  "lo_max": float(lo_vals.max()) if lo_vals.size else -1.0}
    if tr == 1:
        return tr, i, out, rep, inv
    c = int(D["c2"][i])
    return tr, i, {k: v[:, c].copy() for k, v in out.items()}, rep, inv


def main(workers=6):
    import analyze_t1_residual as ar
    import common_check as cc
    from proxy_lib import load_track, pseudo_targets
    t0 = time.time()
    stop = {f"zip_sha_{k}": RS._sha(p) == s for k, (p, s) in ZIPS.items()}
    if not all(stop.values()):
        print(json.dumps({"stop": stop, "verdict": "STOP"}), flush=True)
        return
    g1, s1, n1 = load_track("track1")
    g2, s2_, n2 = load_track("track2")
    c1, _ = pseudo_targets(g1, s1)
    c2, _ = pseudo_targets(g2, s2_)
    D.update(s1=s1, g1=g1, s2=s2_, g2=g2, c2=c2)
    for k, (p, _) in ZIPS.items():
        S = HV._read_zip(p)
        D[f"Z{k}1"] = [S["track1"][f"score_{n}.npz"] for n in n1]
        D[f"Z{k}2"] = [S["track2"][f"score_{n}.npz"] for n in n2]
        del S
    tasks = [(1, i) for i in range(len(s1))] + [(2, i) for i in range(len(s2_))]
    R = {1: {}, 2: {}}
    rep = {k: [0, 0.0] for k in ZIPS}
    inv = {k: {"hi_changed": 0, "lo_max": -1.0} for k in CMPS}
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 6)), mp_context=mp.get_context("fork")) as ex:
        for tr, i, res, rp, iv in ex.map(_job, tasks, chunksize=1):
            R[tr][i] = res
            for k, (nd, mx) in rp.items():
                rep[k][0] += nd
                rep[k][1] = max(rep[k][1], mx)
            for k, d in iv.items():
                inv[k]["hi_changed"] += d["hi_changed"]
                inv[k]["lo_max"] = max(inv[k]["lo_max"], d["lo_max"])
    for k in ZIPS:
        D.pop(f"Z{k}1")
        D.pop(f"Z{k}2")
    for k in ZIPS:
        stop[f"repro_{k}"] = rep[k][0] <= T2F.REPRO_NDIFF and rep[k][1] <= T2F.REPRO_MAX
    n1s, n2s = len(s1), len(s2_)
    keys = BASES + VARS + REPORT + ["bl_rtpl_nh"] + list(CMPS)
    t1 = {}
    for k in keys:
        gen, imp = ar._gen_imp([R[1][i][k] for i in range(n1s)], c1)
        met = ar._metrics(gen, imp)
        t1[k] = {"eer_full": cc._eer_full(gen, imp), "eer_thr": met["eer_thr"], "far1e5_thr": met["far1e5_thr"],
                 "frr_far1e5": met["frr_at_far1e5"], "far_at_frr1e5": met["far_at_frr1e5"]}
        del gen, imp
    stop["nh_eer"] = abs(t1["nh"]["eer_full"] - T2F.NH["eer_full"]) <= 1e-9
    stop["pez_eer"] = abs(t1["pez"]["eer_full"] - T2F.PEZ_PROXY["eer_full"]) <= 1e-9
    stop["rtpl_nh_recorded"] = all(abs(t1["rtpl_nh"][k] - v) <= 1e-9 for k, v in RTPL_REC.items())
    stop["bl_rtpl_nh_eer"] = abs(t1["bl_rtpl_nh"]["eer_full"] - RC.BL_PROXY["bl_rtpl_nh"]) <= 1e-12
    acc = {}
    for k in keys:
        thr = np.float32(t1[k]["far1e5_thr"])
        acc[k] = [R[2][i][k] >= thr for i in range(n2s)]
    A = {k: np.concatenate(v) for k, v in acc.items()}
    pos = np.concatenate([np.arange(v.shape[0]) for v in acc["nh"]])

    def pair(k, b):
        d2 = float(A[b].mean() - A[k].mean())
        g1_ = t1[k]["frr_far1e5"] - t1[b]["frr_far1e5"]
        down, up = A[b] & ~A[k], A[k] & ~A[b]
        res = {"d2": d2, "g1": g1_, "down": int(down.sum()), "up": int(up.sum())}
        for lo, hi in POS:
            m = (pos >= lo) & (pos < (hi if hi is not None else pos.max() + 1))
            res[f"down_t{lo}"] = int((down & m).sum())
            res[f"up_t{lo}"] = int((up & m).sum())
        return res
    base_ratio = pair("rtpl_nh", "nh")
    bar = base_ratio["d2"] / base_ratio["g1"]
    rej_rtpl = A["nh"] & ~A["rtpl_nh"]
    t2 = {"rtpl_nh_vs_nh": base_ratio, "bar": bar, "acc_share": {k: float(A[k].mean()) for k in keys}}
    gate, score = {}, {}
    for k in VARS + REPORT:
        m = pair(k, "rtpl_nh")
        rej_k = A["nh"] & ~A[k]
        m["jaccard_rejections_vs_rtpl"] = float((rej_k & rej_rtpl).sum() / max((rej_k | rej_rtpl).sum(), 1))
        t2[k] = m
        if k in REPORT:
            continue
        c = f"cmp_{k}__hold_nh"
        g_ = {"thr": t1[k]["far1e5_thr"] >= T2F.GATES["thr_min"], "frr": t1[k]["frr_far1e5"] <= T2F.GATES["frr_max"],
              "cmp_eer": t1[c]["eer_full"] <= CMP_EER_MAX, "cmp_thr_equal": t1[c]["far1e5_thr"] == t1[k]["far1e5_thr"],
              "cmp_t2_equal": bool(np.array_equal(A[c], A[k])), "cmp_hi_unchanged": inv[c]["hi_changed"] == 0,
              "cmp_lo_below": inv[c]["lo_max"] < RC.HI_MAX}
        if KIND[k] == "pez":
            g_["direction"] = m["g1"] < 0 and m["d2"] < 0
            score[k] = None
        elif KIND[k] == "stack":
            g_["direction"] = m["d2"] > 0 and (m["g1"] <= 0 or m["d2"] / m["g1"] >= bar)
            score[k] = float("inf") if m["g1"] <= 0 else (m["d2"] / m["g1"]) / bar
        else:
            g_["direction"] = m["d2"] < 0 and m["g1"] < 0 and abs(m["g1"]) / abs(m["d2"]) >= 1.0 / bar
            score[k] = (abs(m["g1"]) / abs(m["d2"])) * bar if m["d2"] != 0 else None
        gate[k] = g_
        t2[k]["h_eer_vs_rtpl_report"] = t1[k]["eer_full"] - t1["rtpl_nh"]["eer_full"]
    passing = [k for k in VARS if all(gate[k].values())]
    stacks = [k for k in passing if KIND[k] == "stack"]
    if len(stacks) == 2:                                   # larger marginal d2/g1 survives; ties go to span
        a, b_ = stacks
        keep = "rtpl_span_nh" if score[a] == score[b_] else max(stacks, key=lambda k: score[k])
        passing = [k for k in passing if KIND[k] != "stack" or k == keep]
    others = sorted([k for k in passing if KIND[k] != "pez"], key=lambda k: -score[k])
    sel = (["rtpl_pez"] if "rtpl_pez" in passing else []) + others
    sel = sel[:2]
    verdict = "STOP" if not all(stop.values()) else ("BUILD:" + ",".join(f"cmp_{k}__hold_nh" for k in sel) if sel else "NONE")
    print(json.dumps({"stop": stop, "repro": rep, "t1": t1, "t2": t2, "invariance": inv, "gate": gate,
                      "score": {k: v for k, v in score.items()}, "passing": passing, "selected": sel,
                      "cmp_eer_max": CMP_EER_MAX}), flush=True)
    print(json.dumps({"verdict": verdict, "seconds": round(time.time() - t0, 1)}), flush=True)


def synth():
    ok = {}
    for seed, kind in ((3, "switch"), (5, "mng")):
        x, g = R4._synth_data(seed, 240, 60, kind, norm_hi=26.0)
        P = T2F.parts(x, g)
        V = variants2(x, g, P)
        q, qm, dfn = rtpl_q(P, g)
        ok[f"rtpl_ratio_eq_{seed}"] = bool(np.array_equal(T2F.ratio_prefix(q, defined=dfn), T2F.cue_s2(P, x, g, "rtpl")))
        ok[f"rtpl_stacked_eq_{seed}"] = bool(np.array_equal(RC.stacked(P, T2F.ratio_prefix(q, defined=dfn)), V["rtpl_nh"]))
        e = 1e-12
        ok[f"order_pez_{seed}"] = bool((V["rtpl_nh"] <= V["rtpl_pez"] + e).all() and (V["rtpl_pez"] <= V["pez"] + e).all())
        for k in ("rtpl_mem", "rtpl_min", "rtpl_prof_and"):
            ok[f"order_{k}_{seed}"] = bool((V["rtpl_nh"] <= V[k] + e).all() and (V[k] <= V["nh"] + e).all())
        for k in ("rtpl_prof_nh", "rtpl_span_nh"):
            ok[f"order_{k}_{seed}"] = bool((V[k] <= V["rtpl_nh"] + e).all())
        O = outputs_fn(x, g)
        ok[f"bl_eq_t2_family_{seed}"] = bool(np.array_equal(O["bl_rtpl_nh"], T2F.bl_variant(x, g, "rtpl_nh")))
        for k, h in CMPS.items():
            hi = O[h] >= np.float32(0.5)
            ok[f"{k}_keeps_hi_{seed}"] = bool(np.array_equal(O[k][hi], O[h][hi]) and (O[k][~hi] < np.float32(RC.HI_MAX)).all())
            ok[f"{k}_lean_eq_full_{seed}"] = bool(np.array_equal(composite(x, g, h), O[k]))
        ok[f"finite_in_range_{seed}"] = all(bool(np.isfinite(v).all() and (np.abs(v) <= 1).all()) for v in O.values())
        cut = 150
        Oh = outputs_fn(x[:cut], g)
        bad = [k for k in O if not np.array_equal(Oh[k], O[k][:cut])]
        ok[f"prefix_exact_{seed}"] = not bad
        xf = x.copy()
        xf[cut:] = xf[cut:][::-1] * np.float32(0.9)
        Of = outputs_fn(xf, g)
        badf = [k for k in O if not np.array_equal(Of[k][:cut], O[k][:cut])]
        ok[f"future_perturbation_{seed}"] = not badf
        if bad or badf:
            ok[f"bad_{seed}"] = sorted(set(bad) | set(badf))
        ok[f"variants_differ_from_rtpl_{seed}"] = {k: bool(np.abs(V[k] - V["rtpl_nh"]).max() > 0) for k in VARS + REPORT}
    allok = all(v is True for k, v in ok.items() if not k.startswith("variants_differ"))
    print(json.dumps({"synth": ok, "all_ok": allok}), flush=True)
    return allok


if __name__ == "__main__":
    if sys.argv[1:] == ["--synth"]:
        sys.exit(0 if synth() else 1)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
