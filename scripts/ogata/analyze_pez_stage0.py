"""PEZ stage 0: premise check for the prefix-energy z-shrink on BLVS-N+FCO (analysis only).

Plan, authorisation and frozen rules: docs/experiments/20261008_ogata_pez_stage0.md (design and judge:
workflow wf_834d27c3-211, "mechanism-skeptic" lens with the critics' fixes).

PEZ-BLVS-N+FCO = x_r5.esrn_fco with omega = omega_E * omega_V * omega_En:
  hist_tj = (#clean verified frames of column j up to t) >= 1  (the same A >= 1 as analyze_blvs_stage0.omega_v)
  e_t     = sum_j S_tj^2 (x_fr_hcp._fe's S row, float64, left-to-right column order)
  Ebar_t  = (sum_{i<=t} e_i) / (t + 1)  (sequential float64 sum, current frame included; the teammate's prefix rule)
  s2_t    = min(1, e_t / Ebar_t), 1 if Ebar_t == 0
  act     = hist & (omega_V == 1) & (zP > 0) & (LS > 1) & (s2 < 1)
  omega_En = s2 / (1 + (1 - s2) zS^2 / (K - 1)) where act, else 1
Calibration pair (teammate, recomputed here): 943251 = dual_ema_cubic_discriminative_scores; 944021 = row-L2 of it;
944151 = prefix-RMS of it. Their s2 (teammate rule on 943251 rows) is compared with ours.

(a) T1, evaluator-only pseudo targets. (b) T2 label-free: claimed column = whole-sequence mean raw-cosine argmax
(evaluator side only). Prints JSON aggregates only.
Usage (server): python scripts/ogata/analyze_pez_stage0.py ZIP_BLVS ZIP_967770 [workers<=6] > results/ogata/pez_stage0.log
                python scripts/ogata/analyze_pez_stage0.py --synth
"""
import hashlib
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
import analyze_hvc_stage0 as HV  # noqa: E402
import analyze_blvs_stage0 as BL  # noqa: E402
from rcfa.methods.gallery_dual_ema_cubic_discriminative import dual_ema_cubic_discriminative_scores  # noqa: E402
from rcfa.methods.gallery_discriminative_row_l2 import row_l2_scores  # noqa: E402
from rcfa.methods.prefix_rms_normalization import prefix_rms_scores  # noqa: E402

SHA_BLVS = "cb81cea2eb1de538897c128e8c011de53ab18d2b91f4cde811bc51e3096c9c09"
SHA_967770 = "95be504d2cefe7184edbce17a485bce0c6690a039163a57896a0423af194280f"
EER_BLVS, THR_BLVS = 0.0030090251291611243, 0.8431214690208435
K1_EER = EER_BLVS + 2e-5
K2_THR, STOP_THR, K2_TAIL = 0.8381215, 0.8231215, 48
K3_FRR, K4_CROSS = 0.30, 10
ID_ACC, ID_TOL = 0.7 * (1 - 0.0175) + 0.3 * 0.39, 0.003
CAL = {"rl2_thr": (0.6141, 1e-3), "prms_thr": (0.5965, 1e-3), "rl2_frr": (0.0255, 5e-3), "prms_frr": (0.1616, 5e-3)}
CAL_FLIP = (0.08, 0.14)
RATIO = 0.117                                   # (.0315 - .0156) / (.1616 - .0255), official 944021 -> 944151
PASS_NET, KILL_NET = 0.03, 0.015
ZS_BINS, S2_BINS = [10.0, 15.0, 30.0], [0.27, 0.5, 0.79]
D = {}


def energy_s2(rows):
    """Teammate prefix rule on row energy: s2_t = min(1, e_t / (E_t / (t+1))), sequential float64."""
    R = np.asarray(rows, np.float64)
    e = np.zeros(R.shape[0])
    for j in range(R.shape[1]):
        e += R[:, j] * R[:, j]
    s2 = np.ones(R.shape[0])
    E = np.float64(0.0)
    for t in range(R.shape[0]):
        E = np.float64(E + e[t])
        Eb = E / np.float64(t + 1)
        if Eb > 0:
            s2[t] = min(1.0, e[t] / Eb)
    return s2


def parts(x, g):
    f = H._fe(x, g)
    K, zP, LP, LS, _, _ = R5._base(x, g, f)
    oE = R5.esr_omega_of(x, g, K, zP, LS)
    oV = BL.omega_v(x, g, zP, LS)
    v, _, r = HV.verify(x, g)
    clean = np.linalg.norm(np.asarray(x, np.float64), axis=1) >= BL.CLEAN
    hist = np.cumsum(v & clean[:, None], axis=0) >= 1
    s2 = energy_s2(f["S"])
    act = hist & (oV == 1.0) & (zP > 0) & (LS > 1) & (s2[:, None] < 1)
    oEn = np.where(act, s2[:, None] / (1.0 + (1.0 - s2[:, None]) * f["zS"] ** 2 / (K - 1)), 1.0)
    return f, oE, oV, oEn, s2, hist, r


def pez_blvs_n_fco(x, g):
    f, oE, oV, oEn, _, _, _ = parts(x, g)
    return R5.esrn_fco(x, g, f, omega=oE * oV * oEn)


def outputs_fn(x, g):
    """Reference for build_candidate's bit-identity check."""
    return {"pez_blvs_n_fco": pez_blvs_n_fco(x, g).astype(np.float32)}


def _job(task):
    tr, i = task
    x, g = D[f"s{tr}"][i], D[f"g{tr}"]
    f, oE, oV, oEn, s2, hist, r = parts(x, g)
    blvs = R5.esrn_fco(x, g, f, omega=oE * oV).astype(np.float32)
    forced = R5.esrn_fco(x, g, f, omega=oE * oV * np.ones_like(oEn)).astype(np.float32)
    pez = R5.esrn_fco(x, g, f, omega=oE * oV * oEn).astype(np.float32)
    ok = {"blvs_eq_zip": bool(np.array_equal(blvs, D[f"B{tr}"][i])), "forced_eq_blvs": bool(np.array_equal(forced, blvs))}
    p943 = dual_ema_cubic_discriminative_scores(x, g)
    rl2, prms = row_l2_scores(p943), prefix_rms_scores(p943)
    s2t = energy_s2(p943)
    if tr == 1:
        t = int(D["tg"][i])
        oth = np.ones(g.shape[0], bool)
        oth[t] = False
        aux = {"imp_oEn_lt1": int((oEn[:, oth] < 1).sum()), "hist": hist, "zS_t": f["zS"][:, t]}
        return tr, i, ok, {"pez": pez, "rl2": rl2, "prms": prms}, (s2, s2t), aux
    c = int(r.mean(0).argmax())
    return tr, i, ok, {"b": blvs[:, c], "p": pez[:, c], "rl2": rl2[:, c], "prms": prms[:, c],
                       "o7": D["O7"][i][:, c], "zS": f["zS"][:, c]}, (s2, s2t), None


def _spearman(a, b):
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(zb, z7, workers=6):
    import analyze_t1_residual as ar
    import common_check as cc
    from proxy_lib import load_track, pseudo_targets
    t0 = time.time()
    stop = {"blvs_zip_sha": _sha(zb) == SHA_BLVS, "967770_zip_sha": _sha(z7) == SHA_967770}
    SB, S7 = HV._read_zip(zb), HV._read_zip(z7)
    g1, s1, n1 = load_track("track1")
    g2, s2_, n2 = load_track("track2")
    tg, _ = pseudo_targets(g1, s1)
    D.update(s1=s1, g1=g1, s2=s2_, g2=g2, tg=tg, B1=[SB["track1"][f"score_{n}.npz"] for n in n1],
             B2=[SB["track2"][f"score_{n}.npz"] for n in n2], O7=[S7["track2"][f"score_{n}.npz"] for n in n2])
    del SB, S7
    tasks = [(1, i) for i in range(len(s1))] + [(2, i) for i in range(len(s2_))]
    T1o, T2o, S2, A1 = {}, {}, {1: {}, 2: {}}, {}
    okall = {"blvs_eq_zip": True, "forced_eq_blvs": True}
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 6)), mp_context=mp.get_context("fork")) as ex:
        for tr, i, ok, res, ss, aux in ex.map(_job, tasks, chunksize=1):
            for k in okall:
                okall[k] &= ok[k]
            S2[tr][i] = ss
            if tr == 1:
                T1o[i], A1[i] = res, aux
            else:
                T2o[i] = res
    stop.update(okall)
    n = len(s1)
    # ---- (a) T1 ----
    t1 = {}
    for k, outs in (("blvs", D["B1"]), ("pez", [T1o[i]["pez"] for i in range(n)]),
                    ("rl2_944021", [T1o[i]["rl2"] for i in range(n)]), ("prms_944151", [T1o[i]["prms"] for i in range(n)])):
        gen, imp = ar._gen_imp(outs, tg)
        met = ar._metrics(gen, imp)
        t1[k] = {"eer_full": cc._eer_full(gen, imp), "eer_thr": met["eer_thr"], "far1e5_thr": met["far1e5_thr"],
                 "frr_far1e5": met["frr_at_far1e5"], "far_at_frr1e5": met["far_at_frr1e5"]}
        del imp
    cut, thb, thp = (np.float32(t1["blvs"]["eer_thr"]), np.float32(t1["blvs"]["far1e5_thr"]),
                     np.float32(t1["pez"]["far1e5_thr"]))
    tail_low = tail_elig = 0
    cross = {"total": 0, "zS_lt3": 0, "leader": 0, "nonleader": 0}
    newrej = []
    for i in range(n):
        b, p, t = D["B1"][i], T1o[i]["pez"], int(tg[i])
        oth = np.ones(b.shape[1], bool)
        oth[t] = False
        tail = (b >= thb) & oth[None, :]
        tail_low += int((tail & (p < b)).sum())
        tail_elig += int((tail & A1[i]["hist"]).sum())
        cr = (b[:, t] >= cut) & (p[:, t] < cut)
        cross["total"] += int(cr.sum())
        cross["zS_lt3"] += int((cr & (A1[i]["zS_t"] < 3)).sum())
        lead = b.argmax(1) == t
        cross["leader"] += int((cr & lead).sum())
        cross["nonleader"] += int((cr & ~lead).sum())
        newrej += BL._runs((b[:, t] >= thb) & (p[:, t] < thp))
    t1.update({"blvs_tail_entries_lowered": tail_low, "blvs_tail_entries_eligible_hist": tail_elig,
               "genuine_eer_crossings": cross,
               "impostor_entries_oEn_lt1": int(sum(A1[i]["imp_oEn_lt1"] for i in range(n))),
               "new_genuine_rejection_runs": {"count": len(newrej), "q50_90_99_max": (
                   np.quantile(newrej, [.5, .9, .99, 1]).round(1).tolist() if newrej else [])}})
    hard = {"pez_thr>=.8231215": t1["pez"]["far1e5_thr"] >= STOP_THR}
    kill = {"K1_eer<=blvs+2e-5": t1["pez"]["eer_full"] <= K1_EER,
            "K2_thr>=.8381215": t1["pez"]["far1e5_thr"] >= K2_THR, "K2_tail_lowered<=48": tail_low <= K2_TAIL,
            "K3_frr<=.30": t1["pez"]["frr_far1e5"] <= K3_FRR, "K4_crossings<=10": cross["total"] <= K4_CROSS}
    cal = {"rl2_thr": t1["rl2_944021"]["far1e5_thr"], "prms_thr": t1["prms_944151"]["far1e5_thr"],
           "rl2_frr": t1["rl2_944021"]["frr_far1e5"], "prms_frr": t1["prms_944151"]["frr_far1e5"]}
    cal_ok = {k: abs(cal[k] - v) <= tol for k, (v, tol) in CAL.items()}
    print(json.dumps({"stop": stop, "hard_stop": hard, "t1": t1, "t1_kill": kill, "cal_t1": cal, "cal_t1_ok": cal_ok}),
          flush=True)
    # ---- (b) T2 ----
    F = [T2o[i] for i in range(len(s2_))]
    acc = lambda key, thr: float(np.mean(np.concatenate([f[key] >= thr for f in F])))
    id_acc = acc("o7", thb)
    flip_cal = acc("rl2", np.float32(cal["rl2_thr"])) - acc("prms", np.float32(cal["prms_thr"]))
    d2 = acc("b", thb) - acc("p", thp)
    g1 = t1["pez"]["frr_far1e5"] - t1["blvs"]["frr_far1e5"]
    dfrr2 = RATIO * g1
    a_est = (d2 - 0.7 * dfrr2) / 0.3
    net = a_est - dfrr2
    a_cons = (d2 - 0.7 * g1) / 0.3
    grid = np.zeros((4, 4))
    tot = np.zeros((4, 4))
    seqflip = []
    for f, i in zip(F, range(len(F))):
        s2o = S2[2][i][0]
        base = f["b"] >= thb
        fl = base & (f["p"] < thp)
        zb_ = np.searchsorted(ZS_BINS, f["zS"], side="right")
        sb_ = np.searchsorted(S2_BINS, s2o, side="right")
        np.add.at(tot, (zb_[base], sb_[base]), 1)
        np.add.at(grid, (zb_[fl], sb_[fl]), 1)
        seqflip.append(fl.mean())
    s2o1 = np.concatenate([S2[1][i][0] for i in range(n)])
    s2t1 = np.concatenate([S2[1][i][1] for i in range(n)])
    s2o2 = np.concatenate([S2[2][i][0] for i in range(len(F))])
    s2t2 = np.concatenate([S2[2][i][1] for i in range(len(F))])
    qs = [0.05, 0.1, 0.25, 0.5]
    t2 = {"identity_967770_acc_share": id_acc, "identity_target": ID_ACC, "identity_ok": abs(id_acc - ID_ACC) <= ID_TOL,
          "cal_flip_share_944021_minus_944151": flip_cal, "cal_flip_ok": CAL_FLIP[0] <= flip_cal <= CAL_FLIP[1],
          "d2": d2, "g1": g1, "dFRR2_est": dfrr2, "a_est": a_est, "net_est": net,
          "a_conservative_report_only": a_cons,
          "flip_grid_zS_by_s2": {"zS_edges": ZS_BINS, "s2_edges": S2_BINS,
                                 "flipped": grid.astype(int).tolist(), "base_accepted": tot.astype(int).tolist()},
          "per_seq_flip_share_q10_50_90_max": np.quantile(seqflip, [.1, .5, .9, 1]).round(4).tolist(),
          "s2_quantiles": {"ours_t1": np.quantile(s2o1, qs).round(4).tolist(),
                           "ours_t2": np.quantile(s2o2, qs).round(4).tolist(),
                           "teammate_t1": np.quantile(s2t1, qs).round(4).tolist(),
                           "teammate_t2": np.quantile(s2t2, qs).round(4).tolist()},
          "excess_low_s2_t2_minus_.7t1": {"ours": float((s2o2 < .27).mean() - .7 * (s2o1 < .27).mean()),
                                         "teammate": float((s2t2 < .27).mean() - .7 * (s2t1 < .27).mean())},
          "spearman_ours_vs_teammate": {"t1": _spearman(s2o1, s2t1), "t2": _spearman(s2o2, s2t2)}}
    if not all(stop.values()) or not all(hard.values()):
        verdict = "STOP"
    elif not all(kill.values()):
        verdict = "KILL"
    elif not (all(cal_ok.values()) and t2["identity_ok"] and t2["cal_flip_ok"]):
        verdict = "UNRESOLVED"
    else:
        verdict = "PASS" if net >= PASS_NET else ("KILL" if net < KILL_NET else "UNRESOLVED")
    print(json.dumps({"t2": t2}), flush=True)
    print(json.dumps({"verdict": verdict, "seconds": round(time.time() - t0, 1)}), flush=True)


# ---------------- random-data checks (no challenge data) ----------------
def _s2_naive(rows):
    R = np.asarray(rows, np.float64)
    out = []
    E = 0.0
    for t in range(R.shape[0]):
        e = 0.0
        for j in range(R.shape[1]):
            e += R[t, j] * R[t, j]
        E += e
        Eb = E / (t + 1)
        out.append(1.0 if Eb <= 0 else min(1.0, e / Eb))
    return np.array(out)


def synth():
    rng = np.random.default_rng(7)
    ok = {}
    K, T = 60, 180
    g = rng.normal(size=(K, 256)).astype(np.float32)
    amp = np.where(np.arange(T) < 120, 3.0, 1.2)[:, None]
    x = rng.normal(size=(T, 256)) + amp * g[2]
    x = (x / np.linalg.norm(x, axis=1, keepdims=True) * rng.uniform(17, 26, size=(T, 1))).astype(np.float32)
    R = rng.normal(size=(40, 7))
    ok["s2_eq_naive"] = bool(np.array_equal(energy_s2(R), _s2_naive(R)))
    ok["s2_const_rows_is_1"] = bool((energy_s2(np.ones((30, 5))) == 1).all())
    f, oE, oV, oEn, s2, hist, _ = parts(x, g)
    ok["oEn_in_0_1"] = bool(((oEn > 0) & (oEn <= 1)).all())
    ok["oV_oEn_disjoint"] = bool(not ((oV < 1) & (oEn < 1)).any())
    blvs = R5.esrn_fco(x, g, f, omega=oE * oV)
    ok["forced_1_eq_blvs"] = bool(np.array_equal(R5.esrn_fco(x, g, f, omega=oE * oV * np.ones_like(oEn)), blvs))
    ok["blvs_eq_module"] = bool(np.array_equal(blvs, BL.blvs_n_fco(x, g)))
    pez = pez_blvs_n_fco(x, g)
    ok["pez_le_blvs"] = bool((pez <= blvs + 1e-15).all())
    ok["low_energy_segment_discounted"] = bool((oEn[120:, 2] < 1).any())
    h = T // 2
    _, _, _, oEnh, s2h, _, _ = parts(x[:h], g)
    ok["prefix_invariant"] = bool(np.array_equal(oEnh, oEn[:h]) and np.array_equal(s2h, s2[:h]))
    ok["spearman_identity"] = abs(_spearman(np.arange(10.0), np.arange(10.0)) - 1) < 1e-12
    print(json.dumps({"synth": ok, "all_ok": all(ok.values())}), flush=True)
    return all(ok.values())


if __name__ == "__main__":
    if sys.argv[1:] == ["--synth"]:
        sys.exit(0 if synth() else 1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
