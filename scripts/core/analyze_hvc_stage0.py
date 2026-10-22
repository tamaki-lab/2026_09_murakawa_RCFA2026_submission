"""HVC stage 0: premise check for the hard frame-verification cap on ESR-N + FCO (analysis only).

Plan, user ruling and frozen decision rule: docs/experiments/20261008_ogata_hvc_stage0.md.

HVC (round 6 design D1): o = ESR-N + FCO output (967770's submitted float32 content, read from its ZIP);
  r_tj = raw cosine of frame t to gallery row j (float64; a zero-norm frame is unverified);
  z_tj = leave-one-out row z of r (population variance over the other K - 1 columns; sd = 0 gives z = 0);
  v_tj = [z_tj >= Z0], Z0 = Phi^-1(1 - 1e-5);
  o'_tj = o_tj if v_tj else min(o_tj, 0).
The rule is per frame, has no state and no Track logic. Stage 0 applies it to the stored 967770 scores; since the
cast to float32 is monotone and 0 is exact, this equals the float64-then-cast design.

(a) Track 1, evaluator-only pseudo targets (= official T1 labels): exact EER invariance, HVC's own FAR=1e-5
    threshold, FRR@FAR1e-5, genuine unverified share, impostor mass above 0.
(b) Track 2, evaluator-side diagnostic layout (never inside a processor): every sequence is 2000 frames, frames
    t < 1400 genuine and t >= 1400 attack; claimed column = argmax of the whole-sequence mean raw cosine (the harness
    pseudo target). Calibration on 967770 first (pooled, then per type with a preregistered type rule), then the
    HVC prediction with the same layout at HVC's own T1 FAR=1e-5 threshold.

Prints JSON aggregates only.
Usage (server): python scripts/core/analyze_hvc_stage0.py ZIP > results/ogata/hvc_stage0.log 2>&1
                python scripts/core/analyze_hvc_stage0.py --synth      # random data only
"""
import hashlib
import io
import json
import os
import statistics
import sys
import time
import zipfile

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import numpy as np  # noqa: E402

Z0 = statistics.NormalDist().inv_cdf(1 - 1e-5)          # 4.264890793922825
ZIP_SHA = "95be504d2cefe7184edbce17a485bce0c6690a039163a57896a0423af194280f"   # 967770 artifact
EER_967770 = 0.003009209994988715                        # recomputed from the 967770 ZIP (submission log)
THR_967770 = 0.843121                                    # same, rounded to 6 decimals
EER_TOL = 1e-12
THR_TOL = 5e-6
THR_MIN = 0.1                                            # STOP if HVC's own FAR=1e-5 threshold <= .1
NPOS_MIN = 2922                                          # STOP if fewer T1 impostor entries have o' > 0 (3 x 974)
GEN_UNVER_MAX = 0.45                                     # KILL if the T1 genuine unverified share exceeds this
T2_LEN, T2_GEN = 2000, 1400                              # diagnostic layout
N_TYPE = {"it": 70, "diffusion": 80, "reenact": 50}
OFFICIAL = {"pooled_frr": 0.01753571, "pooled_aar": 0.38998333,          # detailed page (8 decimals)
            "it_frr": 0.049255102040816325, "it_aar": 0.069,
            "diffusion_frr": 0.0007053571428571428, "diffusion_aar": 0.303375,
            "reenact_frr": 5.714285714285714e-05, "reenact_aar": 0.9779333333333333,
            "be": 0.23338765589569163}
OFFICIAL_RUN = {"it": 24.71428571, "diffusion": 85.725, "reenact": 374.48}
POOLED_TOL = 0.002
TYPE_TOL = 0.005
PASS = {"be_pred_max": 0.215, "it_gen_unver_max": 0.25, "diff_att_ver_max": 0.15, "reenact_att_ver_min": 0.85}
COS40 = 0.40


def verify(x, g):
    """v (T x K bool), z (T x K) and r (T x K), float64 from float32 inputs."""
    xb = np.asarray(x, np.float64)
    gb = np.asarray(g, np.float64)
    K = gb.shape[0]
    n = np.linalg.norm(xb, axis=1)
    u = xb / np.where(n > 0, n, 1.0)[:, None]
    r = u @ (gb / np.linalg.norm(gb, axis=1, keepdims=True)).T
    mu = (r.sum(1, keepdims=True) - r) / (K - 1)
    var = ((r * r).sum(1, keepdims=True) - r * r) / (K - 1) - mu * mu
    sd = np.sqrt(np.maximum(var, 0.0))
    z = np.where(sd > 0, (r - mu) / np.where(sd > 0, sd, 1.0), 0.0)
    v = (z >= Z0) & (n > 0)[:, None]
    return v, z, r


def hvc(o, v):
    o = np.asarray(o, np.float32)
    return np.where(v, o, np.minimum(o, np.float32(0.0))).astype(np.float32)


def run_len(acc):
    best = cur = 0
    for a in acc:
        cur = cur + 1 if a else 0
        best = max(best, cur)
    return best


def _verify_naive(x, g):
    xb = np.asarray(x, np.float64)
    gb = np.asarray(g, np.float64)
    T, K = xb.shape[0], gb.shape[0]
    v = np.zeros((T, K), bool)
    for t in range(T):
        nt = np.linalg.norm(xb[t])
        if nt == 0:
            continue
        r = np.array([xb[t] @ gb[j] / (nt * np.linalg.norm(gb[j])) for j in range(K)])
        for j in range(K):
            rest = np.delete(r, j)
            sd = rest.std()
            z = 0.0 if sd == 0 else (r[j] - rest.mean()) / sd
            v[t, j] = z >= Z0
    return v


# =====================================================================================
# Track 1
# =====================================================================================
def t1_block(O, X, g, tg):
    import analyze_t1_residual as ar
    import common_check as cc
    O2, gun, gun40, iver, ni = [], [], [], 0, 0
    for o, x, t in zip(O, X, tg):
        v, _, r = verify(x, g)
        O2.append(hvc(o, v))
        gun.append(~v[:, t])
        gun40.append(r[:, t] < COS40)
        m = np.ones(g.shape[0], bool)
        m[t] = False
        iver += int(v[:, m].sum())
        ni += int(m.sum()) * x.shape[0]
    res = {}
    for name, outs in (("o", O), ("hvc", O2)):
        gen, imp = ar._gen_imp(outs, tg)
        met = ar._metrics(gen, imp)
        res[name] = {"eer_full": cc._eer_full(gen, imp), "eer_full_ar": met["eer_full"], "eer_thr": met["eer_thr"],
                     "far1e5_thr": met["far1e5_thr"], "frr_far1e5": met["frr_at_far1e5"],
                     "far_at_frr1e5": met["far_at_frr1e5"], "far_at_frr1e3": met["far_at_frr1e3"],
                     "imp_gt0": int(imp.size - np.searchsorted(imp, np.float32(0.0), side="right")),
                     "imp_eq0": int(np.searchsorted(imp, np.float32(0.0), side="right")
                                    - np.searchsorted(imp, np.float32(0.0), side="left")),
                     "imp_ge_far1e5_thr": int(imp.size - np.searchsorted(imp, np.float32(met["far1e5_thr"]),
                                                                         side="left")),
                     "n_imp": int(imp.size), "n_gen": int(gen.size)}
        thr = met["far1e5_thr"]
        res[name]["macro_seq_frr_far1e5"] = float(np.mean([(oo[:, t] < thr).mean() for oo, t in zip(outs, tg)]))
        n_ge_eer = int(imp.size - np.searchsorted(imp, np.float32(met["eer_thr"]), side="left"))
        res[name]["imp_ge_eer_thr"] = n_ge_eer
        res[name]["gen_ge_eer_thr"] = int((gen >= met["eer_thr"]).sum())
        del imp
    res["gen_unverified_share"] = float(np.concatenate(gun).mean())
    res["gen_raw_lt_040_share"] = float(np.concatenate(gun40).mean())
    res["imp_verified_share"] = iver / ni
    return res, O2


# =====================================================================================
# Track 2 (evaluator-side diagnostic layout)
# =====================================================================================
def t2_features(o, x, g):
    v, _, r = verify(x, g)
    c = int(r.mean(0).argmax())
    cg = int(r[:T2_GEN].mean(0).argmax())
    att = slice(T2_GEN, None)
    return {"o": np.asarray(o, np.float32)[:, c], "h": hvc(o, v)[:, c], "v": v[:, c], "r40": r[:, c] >= COS40,
            "c": c, "cg": cg, "med_att": float(np.median(r[att, c])),
            "other_att": float((r[att].argmax(1) != c).mean())}


def type_rule(F, rule):
    n = len(F)
    idx = np.arange(n)
    med = np.array([f["med_att"] for f in F])
    lab = np.empty(n, object)
    if rule == "A":
        o = np.lexsort((idx, med))
        lab[o[:70]], lab[o[70:150]], lab[o[150:]] = "it", "diffusion", "reenact"
        gaps = {"it_diff": float(med[o[70]] - med[o[69]]), "diff_reenact": float(med[o[150]] - med[o[149]])}
    else:
        oth = np.array([f["other_att"] for f in F])
        o = np.lexsort((idx, -oth))
        lab[o[:70]] = "it"
        rest = o[70:]
        o2 = rest[np.lexsort((rest, med[rest]))]
        lab[o2[:80]], lab[o2[80:]] = "diffusion", "reenact"
        gaps = {"it_diff": float(oth[o[69]] - oth[o[70]]), "diff_reenact": float(med[o2[80]] - med[o2[79]])}
    return lab, gaps


def t2_metrics(F, key, thr, lab):
    gen = slice(0, T2_GEN)
    att = slice(T2_GEN, None)
    acc = [f[key] >= thr for f in F]
    out = {"pooled_frr": float(np.mean([~a[gen] for a in acc])), "pooled_aar": float(np.mean([a[att] for a in acc]))}
    if lab is None:
        return out
    frr_m, aar_m = [], []
    for ty in N_TYPE:
        ii = [i for i in range(len(F)) if lab[i] == ty]
        frr = float(np.mean([(~acc[i][gen]).mean() for i in ii]))
        aar = float(np.mean([acc[i][att].mean() for i in ii]))
        runs = np.array([run_len(acc[i][att]) for i in ii], np.float64)
        out.update({f"{ty}_frr": frr, f"{ty}_aar": aar, f"{ty}_run": float(runs.mean()),
                    f"{ty}_s10": float((runs >= 10).mean()), f"{ty}_n": len(ii)})
        frr_m.append(frr)
        aar_m.append(aar)
    out["macro_frr"] = float(np.mean(frr_m))
    out["macro_aar"] = float(np.mean(aar_m))
    out["be"] = 0.5 * (out["macro_frr"] + out["macro_aar"])
    return out


def t2_shares(F, lab):
    gen = slice(0, T2_GEN)
    att = slice(T2_GEN, None)
    out = {"pooled_unverified_all": float(np.mean([~f["v"] for f in F])),
           "pooled_raw_lt_040_all": float(np.mean([~f["r40"] for f in F])),
           "claimed_eq_genuine_part_argmax": int(sum(f["c"] == f["cg"] for f in F))}
    for ty in N_TYPE:
        ii = [i for i in range(len(F)) if lab[i] == ty]
        out[f"{ty}_gen_unverified"] = float(np.mean([~F[i]["v"][gen] for i in ii]))
        out[f"{ty}_att_verified"] = float(np.mean([F[i]["v"][att] for i in ii]))
        out[f"{ty}_gen_raw_lt_040"] = float(np.mean([~F[i]["r40"][gen] for i in ii]))
        out[f"{ty}_att_raw_ge_040"] = float(np.mean([F[i]["r40"][att] for i in ii]))
        out[f"{ty}_att_verified_seq_q"] = np.quantile([F[i]["v"][att].mean() for i in ii],
                                                      [0, .1, .5, .9, 1]).round(4).tolist()
    return out


def _within(got, ref, tol):
    return {k: [got[k], ref[k], abs(got[k] - ref[k]) <= tol] for k in ref}


# =====================================================================================
# Main
# =====================================================================================
def _read_zip(path):
    S = {}
    with zipfile.ZipFile(path) as outer:
        for track in ("track1", "track2"):
            with zipfile.ZipFile(io.BytesIO(outer.read(f"{track}.zip"))) as z:
                S[track] = {m: np.load(io.BytesIO(z.read(m)), allow_pickle=False)["score"] for m in z.namelist()}
    return S


def main(zpath):
    from proxy_lib import load_track, pseudo_targets
    t0 = time.time()
    sha = hashlib.sha256(open(zpath, "rb").read()).hexdigest()
    print(json.dumps({"tag": "hvc_stage0", "zip": os.path.basename(zpath), "zip_sha256": sha, "z0": Z0}), flush=True)
    stop = {"zip_sha_ok": sha == ZIP_SHA}
    S = _read_zip(zpath)
    # ---- (a) Track 1 ----
    g1, s1, n1 = load_track("track1")
    tg1, _ = pseudo_targets(g1, s1)
    O1 = [S["track1"][f"score_{n}.npz"] for n in n1]
    stop["t1_files_and_shapes_ok"] = (len(O1) == len(s1) == len(S["track1"]) and
                                      all(o.shape == (x.shape[0], g1.shape[0]) for o, x in zip(O1, s1)))
    a, _ = t1_block(O1, s1, g1, tg1)
    del s1
    stop["967770_eer_repro"] = abs(a["o"]["eer_full"] - EER_967770) <= EER_TOL
    stop["967770_thr_repro"] = abs(a["o"]["far1e5_thr"] - THR_967770) <= THR_TOL
    stop["hvc_eer_exact"] = (abs(a["hvc"]["eer_full"] - a["o"]["eer_full"]) <= EER_TOL and
                             a["hvc"]["eer_thr"] == a["o"]["eer_thr"] and
                             a["hvc"]["imp_ge_eer_thr"] == a["o"]["imp_ge_eer_thr"] and
                             a["hvc"]["gen_ge_eer_thr"] == a["o"]["gen_ge_eer_thr"])
    stop["hvc_thr_gt_0.1"] = a["hvc"]["far1e5_thr"] > THR_MIN
    stop["hvc_imp_gt0_ge_2922"] = a["hvc"]["imp_gt0"] >= NPOS_MIN
    kill = {"t1_gen_unverified_le_.45": a["gen_unverified_share"] <= GEN_UNVER_MAX}
    print(json.dumps({"t1": a}), flush=True)
    # ---- (b) Track 2 ----
    g2, s2, n2 = load_track("track2")
    O2 = [S["track2"][f"score_{n}.npz"] for n in n2]
    stop["t2_files_and_shapes_ok"] = (len(O2) == len(s2) == len(S["track2"]) and
                                      all(o.shape == (x.shape[0], g2.shape[0]) for o, x in zip(O2, s2)))
    stop["t2_all_len_2000"] = all(x.shape[0] == T2_LEN for x in s2) and len(s2) == sum(N_TYPE.values())
    b = {"status": "UNRESOLVED"}
    if not all(stop.values()):
        print(json.dumps({"stop": stop, "verdict": "STOP", "seconds": round(time.time() - t0, 1)}), flush=True)
        return
    F = [t2_features(o, x, g2) for o, x in zip(O2, s2)]
    del s2, O2
    thr_o, thr_h = a["o"]["far1e5_thr"], a["hvc"]["far1e5_thr"]
    pooled = t2_metrics(F, "o", thr_o, None)
    b["pooled_967770"] = _within(pooled, {k: OFFICIAL[k] for k in ("pooled_frr", "pooled_aar")}, POOLED_TOL)
    b["pooled_ok"] = all(v[2] for v in b["pooled_967770"].values())
    b["rules"] = {}
    chosen = None
    for rule in ("A", "B"):
        lab, gaps = type_rule(F, rule)
        m = t2_metrics(F, "o", thr_o, lab)
        rep = _within(m, {k: OFFICIAL[k] for k in OFFICIAL if not k.startswith("pooled")}, TYPE_TOL)
        ok = all(v[2] for v in rep.values())
        b["rules"][rule] = {"gaps": gaps, "repro_967770": rep, "runs_967770": {t: [m[f"{t}_run"], OFFICIAL_RUN[t]]
                                                                              for t in N_TYPE},
                            "s10_967770": {t: m[f"{t}_s10"] for t in N_TYPE}, "repro_ok": ok}
        if ok and chosen is None:
            chosen = (rule, lab)
    if b["pooled_ok"] and chosen is not None:
        rule, lab = chosen
        h = t2_metrics(F, "h", thr_h, lab)
        sh = t2_shares(F, lab)
        crit = {"be_pred<=.215": h["be"] <= PASS["be_pred_max"],
                "it_gen_unverified<=.25": sh["it_gen_unverified"] <= PASS["it_gen_unver_max"],
                "diffusion_att_verified<=.15": sh["diffusion_att_verified"] <= PASS["diff_att_ver_max"],
                "reenact_att_verified>=.85": sh["reenact_att_verified"] >= PASS["reenact_att_ver_min"]}
        b.update({"status": "PASS" if all(crit.values()) else "FAIL", "rule": rule, "hvc_pred": h,
                  "hvc_pooled": t2_metrics(F, "h", thr_h, None), "shares": sh, "criteria": crit,
                  "hvc_at_967770_thr": t2_metrics(F, "h", thr_o, lab)})
    else:
        b["shares_rule_A_report_only"] = t2_shares(F, type_rule(F, "A")[0])
    verdict = ("STOP" if not all(stop.values()) else "KILL" if not all(kill.values()) else b["status"])
    print(json.dumps({"t2": b}), flush=True)
    print(json.dumps({"stop": stop, "kill": kill, "verdict": verdict, "seconds": round(time.time() - t0, 1)}),
          flush=True)


# =====================================================================================
# Random-data checks (no challenge data)
# =====================================================================================
def synth():
    import analyze_t1_residual as ar
    import common_check as cc
    rng = np.random.default_rng(0)
    ok = {}
    # verify() against the per-entry definition, including a zero-norm frame and planted strong matches
    g = rng.normal(size=(60, 256)).astype(np.float32)
    x = rng.normal(size=(30, 256)).astype(np.float32)
    x[5] = 0
    x[7:12] = (3 * g[2] + 0.1 * rng.normal(size=(5, 256))).astype(np.float32)
    v, z, _ = verify(x, g)
    vn = _verify_naive(x, g)
    ok["verify_eq_naive"] = bool(np.array_equal(v, vn))
    ok["zero_norm_unverified"] = bool(not v[5].any())
    ok["planted_verified"] = bool(v[7:12, 2].all())
    # EER exactness and the cap on random scores with a planted genuine column (EER cut < 0 by construction)
    tg, O, X = [], [], []
    G = rng.normal(size=(20, 16)).astype(np.float32)
    for i in range(40):
        t = i % 20
        xx = (rng.normal(size=(60, 16)) + rng.uniform(0, 2.5, size=(60, 1)) * G[t]).astype(np.float32)
        oo = np.tanh(rng.normal(-1.0, 0.6, size=(60, 20))).astype(np.float32)
        oo[:, t] = np.tanh(rng.normal(0.3, 0.6, size=60))
        tg.append(t)
        O.append(oo)
        X.append(xx)
    a, O2 = t1_block(O, X, G, np.array(tg))
    ok["synth_eer_cut_lt_0"] = a["o"]["eer_thr"] < 0
    ok["synth_eer_exact"] = (a["hvc"]["eer_full"] == a["o"]["eer_full"] and a["hvc"]["eer_thr"] == a["o"]["eer_thr"])
    ok["cap_only_lowers"] = all(bool((h <= o).all()) for h, o in zip(O2, O))
    ok["cap_le0_unless_verified"] = all(bool((h[~verify(xx, G)[0]] <= 0).all()) for h, xx in zip(O2, X))
    ok["cap_unchanged_le0"] = all(bool(np.array_equal(np.minimum(h, 0), np.minimum(o, 0))) for h, o in zip(O2, O))
    gen, imp = ar._gen_imp(O, np.array(tg))
    ok["eer_helpers_agree"] = abs(cc._eer_full(gen, imp) - ar._metrics(gen, imp)["eer_full"]) < 1e-15
    # type rules: planted layout (low median -> it, mid -> diffusion, high -> reenact)
    tys = ["it"] * 70 + ["diffusion"] * 80 + ["reenact"] * 50
    rng.shuffle(tys)
    F = [{"_ty": ty, "med_att": {"it": .05, "diffusion": .25, "reenact": .6}[ty] + .01 * rng.random(),
          "other_att": {"it": .9, "diffusion": .3, "reenact": .0}[ty] + .01 * rng.random()} for ty in tys]
    for rule in ("A", "B"):
        lab, _ = type_rule(F, rule)
        ok[f"type_rule_{rule}_recovers"] = bool(all(lab[i] == F[i]["_ty"] for i in range(200)))
    # t2_metrics on a planted accept pattern
    Fm = [{"o": np.r_[np.ones(1400), np.zeros(600)].astype(np.float32)} for _ in range(200)]
    Fm[0]["o"][1400:1410] = 1
    m = t2_metrics(Fm, "o", 0.5, np.array(tys, object))
    ok["t2_metrics_planted"] = (m["pooled_frr"] == 0 and abs(m["pooled_aar"] - 10 / 120000) < 1e-15 and
                                m["macro_frr"] == 0 and m["be"] == 0.5 * m["macro_aar"])
    ok["run_len"] = run_len([1, 1, 0, 1, 1, 1, 0]) == 3 and run_len([]) == 0
    ok["z0"] = abs(Z0 - 4.26489079) < 1e-8
    print(json.dumps({"synth": ok, "all_ok": all(ok.values())}), flush=True)
    return all(ok.values())


if __name__ == "__main__":
    if sys.argv[1:] == ["--synth"]:
        sys.exit(0 if synth() else 1)
    main(sys.argv[1])
