"""Fusion family stage 0: attribution of the F-mean gain and fixed fusion rules (analysis only).

Plan, provenance and frozen rules: docs/experiments/20261009_ogata_fusion_family.md.

Per sequence, everything is recomputed with the frozen research code (float64 internals):
  a64 = PEZ-NH, p64 = PEZ, a0 = the same HCP + FCO core with no discount (omega = 1), b = 943955
  (rcfa gallery_discriminative_row_l2_scores, float32), o = OSP-C (x_fr_hcp.OSPC, float32).
  a = float32(a64), p = float32(p64). The submitted PEZ-NH, PEZ and OSP-C ZIPs are a STOP check only.
Rules (per frame, strictly causal, same constants on both tracks; C0 = .5 is the one fixed constant):
  mean     (a + b) / 2                                  F-mean (reproduces 20261009_ogata_fusion_943955)
  atanh    tanh((atanh a + atanh b) / 2)                log-odds mean (rank-equivalent to the product of experts)
  bl_nh    a if a >= C0 else (a + b) / 2                band-limited, PEZ-NH base
  bl_pez   p if p >= C0 else (p + b) / 2                band-limited, PEZ base
  flat_nh  row-wise: (a + b) / 2 if max_j a_tj < C0 else a
  def_nh   (a + tanh(atanh b + d_NH)) / 2,  d_NH = atanh a64 - atanh a0 <= 0 (PEZ-NH's own discount moved to b)
  def_pez  (p + tanh(atanh b + d_PEZ)) / 2, d_PEZ = atanh p64 - atanh a0
  tri_o    (a + b + o) / 3
  preq     w a + (1 - w) b, w = n_a / (n_a + n_b), n_m = 1 + #{s < t : argmax_j m_s == argmax_j mean_s}
  tcf0     Phi(numN + F, 4 Q + G) on the PEZ-NH core (TCF-0 of 20261009_ogata_t1_new_angles)
  report-only: duo_o (a + o) / 2, min(a, b), max(a, b), tcf_x0 (den 1 + Q), tcf_b0 (den LP (2 + mS) + Q).
atanh uses a clip to +-(1 - 2^-24) (numerical guard only). Pseudo targets on the evaluator side only.
Prints JSON aggregates only.
Usage (server): python scripts/ogata/analyze_fusion_family.py [workers<=6] > results/ogata/fusion_family.log
                python scripts/ogata/analyze_fusion_family.py --synth
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

import x_fr_hcp as H  # noqa: E402
import x_r4 as R4  # noqa: E402
import analyze_hvc_stage0 as HV  # noqa: E402
import analyze_pez_residual as RS  # noqa: E402
import analyze_t1_drp_cla as DC  # noqa: E402
from rcfa.methods.gallery_discriminative_row_l2 import gallery_discriminative_row_l2_scores  # noqa: E402

C0 = 0.5
EPS = 2.0 ** -24
NH = DC.NH
PEZ_PROXY = {"eer_full": 0.0030089891830279987, "far1e5_thr": 0.8431214690208435}
FMEAN_PROXY_EER = 0.0029567543164168006
ZIPS = {"a": (DC.ZIP_NH, DC.SHA_NH),
        "p": (RS.ART / "pez_blvs_n_fco" / "submission.zip",
              "f28ce5876ad7e18c0598c07e0a637059958f8d62b52140e6578d7080fc1445b0"),
        "o": (RS.ART / "ospc_vtn" / "submission.zip",
              "ccd18a849a586e3eef488fbd0d3e4873533f89b6051b52d69e8dba1de86abec2")}
REPRO_NDIFF, REPRO_MAX = 10, 1e-6
K_GAIN = 1e-5
PRIMARY = ["mean", "atanh", "bl_nh", "bl_pez", "flat_nh", "def_nh", "def_pez", "tri_o", "preq", "tcf0"]
REPORT = ["duo_o", "min", "max", "tcf_x0", "tcf_b0"]
BASE = {"bl_pez": "p", "def_pez": "p"}            # every other rule is compared with PEZ-NH (a)
BL_PARENT = {"bl_nh": "a", "bl_pez": "p", "flat_nh": "a"}
STD_EDGES = [0, 10, 50, 200, 1000, np.inf]
STD_NAMES = ["t<10", "10-50", "50-200", "200-1000", ">=1000"]
D = {}


def _at(v):
    return np.arctanh(np.clip(np.asarray(v, np.float64), -1.0 + EPS, 1.0 - EPS))


def core_parts(x, g):
    """float64 internals shared by every rule (one FCO state, one front end)."""
    f, oE, oV, oEn, alt, _, _ = RS.states(x, g)
    K, zP, LP, LS, num, den = R5._base(x, g, f)
    w, r, _ = R4.fco_weights(x, g)
    v, _ = R4.fco_votes(x, g, r)
    C, M = R4.fco_state(w, v)
    a64 = R4.fco_core(R5._numN(LP, LS, oE * oV * alt["hist"]), den, C, M, K)[0]
    p64 = R4.fco_core(R5._numN(LP, LS, oE * oV * oEn), den, C, M, K)[0]
    a0 = R4.fco_core(num, den, C, M, K)[0]
    _, _, aux = H.hcp_parts(f["zP"], f["zS"], K)
    numN = R5._numN(LP, LS, oE * oV * alt["hist"])
    tcf = {"tcf0": 4.0 * aux["Q"], "tcf_x0": 1.0 + aux["Q"], "tcf_b0": LP * (2.0 + aux["mS"]) + aux["Q"]}
    tcf = {k: R4.fco_core(numN, dd, C, M, K)[0] for k, dd in tcf.items()}
    return a64, p64, a0, tcf


def _mean(u, v):
    return (np.asarray(u, np.float64) + np.asarray(v, np.float64)) / 2


def _bl(base, b):
    base64 = np.asarray(base, np.float64)
    return np.where(base64 >= C0, base64, _mean(base, b))


def _flat(base, b):
    base64 = np.asarray(base, np.float64)
    flat = base64.max(1) < C0
    return np.where(flat[:, None], _mean(base, b), base64)


def _def(base, b, d):
    return _mean(base, np.tanh(_at(b) + d))


def _preq(a, b):
    m = _mean(a, b)
    la, lb, lm = a.argmax(1), b.argmax(1), m.argmax(1)
    T = a.shape[0]
    na = 1.0 + np.concatenate([[0.0], np.cumsum(la == lm)[:-1]]) if T else np.zeros(0)
    nb = 1.0 + np.concatenate([[0.0], np.cumsum(lb == lm)[:-1]]) if T else np.zeros(0)
    w = (na / (na + nb))[:, None]
    return w * np.asarray(a, np.float64) + (1.0 - w) * np.asarray(b, np.float64)


def rules(x, g, parts=None):
    """All rule outputs (float32) for one sequence."""
    a64, p64, a0, tcf = core_parts(x, g) if parts is None else parts
    a, p = a64.astype(np.float32), p64.astype(np.float32)
    b = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    o = np.asarray(H.OSPC(x, g), np.float32)
    out = {"a": a, "p": p, "b": b, "o": o,
           "mean": _mean(a, b), "atanh": np.tanh((_at(a) + _at(b)) / 2), "bl_nh": _bl(a, b), "bl_pez": _bl(p, b),
           "flat_nh": _flat(a, b), "def_nh": _def(a, b, _at(a64) - _at(a0)), "def_pez": _def(p, b, _at(p64) - _at(a0)),
           "tri_o": (np.asarray(a, np.float64) + np.asarray(b, np.float64) + np.asarray(o, np.float64)) / 3,
           "preq": _preq(a, b), "duo_o": _mean(a, o), "min": np.minimum(a, b), "max": np.maximum(a, b), **tcf}
    return {k: np.asarray(v).astype(np.float32) for k, v in out.items()}


def outputs_fn(x, g):
    """Reference for build_candidate's bit-identity check (keys = rule names)."""
    return rules(x, g)


def bl_pez_943955(x, g):
    return rules(x, g)["bl_pez"]


def bl_nh_943955(x, g):
    return rules(x, g)["bl_nh"]


def def_nh_943955(x, g):
    return rules(x, g)["def_nh"]


def def_pez_943955(x, g):
    return rules(x, g)["def_pez"]


def _job(task):
    tr, i = task
    x, g = D[f"s{tr}"][i], D[f"g{tr}"]
    R = rules(x, g)
    rep = {k: (int((R[k] != D[f"Z{k}{tr}"][i]).sum()), float(np.abs(R[k].astype(np.float64)
                                                                     - D[f"Z{k}{tr}"][i].astype(np.float64)).max()))
           for k in ("a", "p", "o")}
    if tr == 1:
        return tr, i, R, rep, None
    c = int(D["c2"][i])
    thr = {"a": np.float32(NH["far1e5_thr"]), "p": np.float32(PEZ_PROXY["far1e5_thr"])}
    inv = {k: int(((R[k] >= thr[pa]) != (R[pa] >= thr[pa])).sum()) for k, pa in BL_PARENT.items()}
    return tr, i, {k: v[:, c].copy() for k, v in R.items()}, rep, inv


def _runs(b):
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    return np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0]


def _attrib(R1, tg, k, base, cut_r, cut_b, n1):
    """Errors fixed / broken by rule k against its base at each own EER cut, by position, region and row flatness."""
    nb = len(STD_NAMES)
    z = {n: np.zeros((nb, 2, 2)) for n in ("gen_fixed", "gen_broken", "imp_fixed", "imp_broken")}
    for i in range(n1):
        o, q, t = R1[i][k], R1[i][base], int(tg[i])
        T = o.shape[0]
        pos = np.digitize(np.arange(T), STD_EDGES) - 1
        hi = (q >= np.float32(C0)).astype(int)
        fl = (q.max(1) < np.float32(C0)).astype(int)
        eq = q[:, t] < cut_b
        er = o[:, t] < cut_r
        for nm, m in (("gen_fixed", eq & ~er), ("gen_broken", ~eq & er)):
            np.add.at(z[nm], (pos[m], hi[m, t], fl[m]), 1)
        mask = np.ones(o.shape[1], bool)
        mask[t] = False
        eqi = q[:, mask] >= cut_b
        eri = o[:, mask] >= cut_r
        for nm, m in (("imp_fixed", eqi & ~eri), ("imp_broken", ~eqi & eri)):
            rr, cc_ = np.nonzero(m)
            np.add.at(z[nm], (pos[rr], hi[:, mask][rr, cc_], fl[rr]), 1)
    tot = {n: int(v.sum()) for n, v in z.items()}
    net = (tot["gen_fixed"] - tot["gen_broken"]) + (tot["imp_fixed"] - tot["imp_broken"]) / 199.0
    by_pos = {STD_NAMES[j]: {n: int(z[n][j].sum()) for n in z} for j in range(nb)}
    by_region = {f"{'hi' if h else 'lo'}_{'flat' if fl_ else 'peak'}": {n: int(z[n][:, h, fl_].sum()) for n in z}
                 for h in (0, 1) for fl_ in (0, 1)}
    return {"totals": tot, "net_gen_eq": net, "by_pos": by_pos, "by_region_of_base": by_region}


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
    rep = {k: [0, 0, 0.0] for k in ("a", "p", "o")}       # files differing, entries differing, max |d|
    inv = {k: 0 for k in BL_PARENT}
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 6)), mp_context=mp.get_context("fork")) as ex:
        for tr, i, res, rp, iv in ex.map(_job, tasks, chunksize=1):
            R[tr][i] = res
            for k, (nd, mx) in rp.items():
                rep[k][0] += int(nd > 0)
                rep[k][1] += nd
                rep[k][2] = max(rep[k][2], mx)
            if iv is not None:
                for k, v in iv.items():
                    inv[k] += v
    for k in ZIPS:
        D.pop(f"Z{k}1")
        D.pop(f"Z{k}2")
    stop.update({f"repro_{k}": rep[k][1] <= REPRO_NDIFF and rep[k][2] <= REPRO_MAX for k in rep})
    n1s, n2s = len(s1), len(s2_)
    t1 = {}
    for k in ["a", "p", "b", "o"] + PRIMARY + REPORT:
        gen, imp = ar._gen_imp([R[1][i][k] for i in range(n1s)], c1)
        met = ar._metrics(gen, imp)
        t1[k] = {"eer_full": cc._eer_full(gen, imp), "eer_thr": met["eer_thr"], "far1e5_thr": met["far1e5_thr"],
                 "frr_far1e5": met["frr_at_far1e5"], "far_at_frr1e5": met["far_at_frr1e5"]}
        del gen, imp
    stop["parent_a_eer"] = abs(t1["a"]["eer_full"] - NH["eer_full"]) <= 1e-9
    stop["parent_p_eer"] = abs(t1["p"]["eer_full"] - PEZ_PROXY["eer_full"]) <= 1e-9
    stop["fmean_repro"] = abs(t1["mean"]["eer_full"] - FMEAN_PROXY_EER) <= 2e-9
    # Track 2: claimed-column accepted share at each own FAR=1e-5 threshold, and changes against the base
    t2 = {}
    for k in ["a", "p", "b", "o"] + PRIMARY + REPORT:
        thr = np.float32(t1[k]["far1e5_thr"])
        acc = np.stack([R[2][i][k] >= thr for i in range(n2s)])
        t2[k] = {"acc_share": float(acc.mean())}
        if k in ("a", "p", "b", "o"):
            continue
        base = BASE.get(k, "a")
        accb = np.stack([R[2][i][base] >= np.float32(t1[base]["far1e5_thr"]) for i in range(n2s)])
        down, up = accb & ~acc, acc & ~accb
        t2[k].update({"base": base, "d2_vs_base": float(accb.mean() - acc.mean()), "down": int(down.sum()),
                      "up": int(up.sum()), "down_in_runs_ge20": int(sum(_runs(down[i])[_runs(down[i]) >= 20].sum()
                                                                        for i in range(n2s))),
                      "up_in_runs_ge20": int(sum(_runs(up[i])[_runs(up[i]) >= 20].sum() for i in range(n2s)))})
    bl_check = {k: {"t2_all_column_accept_changes": inv[k],
                    "t1_thr_equal": t1[k]["far1e5_thr"] == t1[BL_PARENT[k]]["far1e5_thr"]} for k in BL_PARENT}
    attrib = {}
    for k in ("mean", "bl_nh", "flat_nh", "def_nh", "tri_o", "atanh"):
        attrib[k] = _attrib(R[1], c1, k, "a", np.float32(t1[k]["eer_thr"]), np.float32(t1["a"]["eer_thr"]), n1s)
    attrib["bl_pez_vs_p"] = _attrib(R[1], c1, "bl_pez", "p", np.float32(t1["bl_pez"]["eer_thr"]),
                                    np.float32(t1["p"]["eer_thr"]), n1s)
    adm = {k: {"d_eer_vs_base": t1[k]["eer_full"] - t1[BASE.get(k, "a")]["eer_full"],
               "admitted": t1[k]["eer_full"] <= t1[BASE.get(k, "a")]["eer_full"] - K_GAIN} for k in PRIMARY}
    ok = [k for k in PRIMARY if adm[k]["admitted"]]
    t1_best = min(ok, key=lambda k: t1[k]["eer_full"]) if ok else None
    build = []
    if adm["bl_pez"]["admitted"] and bl_check["bl_pez"]["t2_all_column_accept_changes"] == 0 \
            and bl_check["bl_pez"]["t1_thr_equal"]:
        build.append("bl_pez")
    if adm["bl_nh"]["admitted"] and bl_check["bl_nh"]["t2_all_column_accept_changes"] == 0 \
            and bl_check["bl_nh"]["t1_thr_equal"] and \
            (("bl_pez" not in build) or t1["bl_nh"]["eer_full"] <= t1["bl_pez"]["eer_full"] - K_GAIN):
        build.append("bl_nh")
    for k in ("def_nh", "def_pez"):
        base = BASE.get(k, "a")
        if adm[k]["admitted"] and t2[k]["acc_share"] <= t2[base]["acc_share"] + 0.01:
            build.append(k)
    if t1_best is not None and t1_best not in build and \
            t1[t1_best]["eer_full"] <= t1["mean"]["eer_full"] - K_GAIN:
        build.append(t1_best)
    verdict = "STOP" if not all(stop.values()) else ("BUILD:" + ",".join(build) if build else "NONE")
    print(json.dumps({"stop": stop, "repro": rep, "t1": t1, "t2": t2, "bl_check": bl_check, "admission": adm,
                      "t1_best_admitted": t1_best}), flush=True)
    print(json.dumps({"attribution": attrib}), flush=True)
    print(json.dumps({"verdict": verdict, "seconds": round(time.time() - t0, 1)}), flush=True)


def synth():
    import analyze_peznh_stage0 as NHM
    import analyze_pez_stage0 as PZ
    from analyze_fusion_943955 import f_mean
    ok = {}
    for seed, kind in ((3, "switch"), (5, "mng")):
        x, g = R4._synth_data(seed, 240, 60, kind, norm_hi=26.0)
        R = rules(x, g)
        ok[f"a_eq_peznh_{seed}"] = bool(np.array_equal(R["a"], NHM.peznh_blvs_n_fco(x, g).astype(np.float32)))
        ok[f"p_eq_pez_{seed}"] = bool(np.array_equal(R["p"], PZ.pez_blvs_n_fco(x, g).astype(np.float32)))
        ok[f"a0_eq_fco_{seed}"] = bool(np.array_equal(core_parts(x, g)[2], R4.fco(x, g)))
        ok[f"mean_eq_fmean_{seed}"] = bool(np.array_equal(R["mean"], f_mean(R["a"], R["b"])))
        ok[f"o_eq_ospc_{seed}"] = bool(np.array_equal(R["o"], np.asarray(H.OSPC(x, g), np.float32)))
        a = R["a"].astype(np.float64)
        ok[f"bl_keeps_hi_{seed}"] = bool(np.array_equal(R["bl_nh"][a >= C0], R["a"][a >= C0]))
        ok[f"bl_lo_below_{seed}"] = bool((R["bl_nh"][a < C0] < (1 + C0) / 2 + 1e-7).all())
        a64, p64, a0, tcf = core_parts(x, g)
        ok[f"d_le_0_{seed}"] = bool((_at(a64) - _at(a0) <= 1e-12).all() and (_at(p64) - _at(a0) <= 1e-12).all())
        ok[f"def_le_mean_{seed}"] = bool((R["def_nh"].astype(np.float64) <= R["mean"].astype(np.float64) + 1e-7).all())
        ok[f"finite_in_range_{seed}"] = all(bool(np.isfinite(v).all() and (np.abs(v) <= 1).all()) for v in R.values())
        h = 150
        Rh = rules(x[:h], g)
        ok[f"prefix_invariant_{seed}"] = all(bool(np.allclose(Rh[k], R[k][:h], rtol=0, atol=1e-6)) for k in R)
        ok[f"tcf_changes_{seed}"] = bool(np.abs(R["tcf0"] - R["a"]).max() > 0)
        ok[f"shapes_{seed}"] = all(v.shape == (240, 60) and v.dtype == np.float32 for v in R.values())
    # preq weights: identical models give w = 1/2 everywhere and output = a
    a = np.tanh(np.random.default_rng(1).normal(size=(30, 7))).astype(np.float32)
    ok["preq_identity"] = bool(np.allclose(_preq(a, a), a, rtol=0, atol=1e-7))
    print(json.dumps({"synth": ok, "all_ok": all(ok.values())}), flush=True)
    return all(ok.values())


if __name__ == "__main__":
    if sys.argv[1:] == ["--synth"]:
        sys.exit(0 if synth() else 1)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
