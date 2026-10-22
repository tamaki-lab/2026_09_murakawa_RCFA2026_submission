"""PEZ residual (stage A): where PEZ-BLVS-N+FCO still accepts Track 2 attack frames (analysis only).

Plan and frozen rules: docs/experiments/20261009_ogata_pez_residual.md.

L. Layout. Claimed column = whole-sequence mean raw-cosine argmax (proxy_lib.pseudo_targets, evaluator side only).
   Accept indicators of 11 official submissions at their official operating thresholds; scan one global contiguous
   600-frame attack window [s, s + 600), s = 0..1400, against the official pooled frame FRR / AAR.
D. Decomposition of PEZ's accepted claimed-column frames by PEZ gate cell (attack / T2 genuine under the identified
   layout; T1 genuine target column at the same threshold).
G. Single-gate counterfactuals: omega_En applied with exactly one PEZ gate removed (hist, zP > 0, omega_V == 1).
Prints JSON aggregates only.
Usage (server): python scripts/core/analyze_pez_residual.py [workers<=6] > results/ogata/pez_residual.log
                python scripts/core/analyze_pez_residual.py --synth
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
import analyze_pez_stage0 as PZ  # noqa: E402

ART = HERE.parents[1] / "results" / "ogata" / "artifacts"
# name: (artifact dir, SHA-256, official operating threshold, official pooled frame FRR, pooled frame AAR)
SUBS = {
    "dveto_d0005": ("dveto_g1_q005p4_d0005", "a5f765799175d154ecc066e3e22409a2c28a3ba716cecffc9cbadb750800361f",
                    0.02643672, 0.01440357, 0.53625833),
    "dveto_d002": ("dveto_g1_q005p4_d002", "32f411413a731389042326bf12aff726ba5c51404bfb9c1b583ed48c3e764cee",
                   0.02833549, 0.01922143, 0.72743333),
    "rg2s_m0": ("rg2s_s005_f02_p4_ds0005_df001_m0", "8ad75b5bc2caa11ed2c0aacb7864bf7e76a22f08e4cf1309e4b1e69d52cfcf99",
                0.02606382, 0.01215357, 0.63719167),
    "ospc_vtn": ("ospc_vtn", "ccd18a849a586e3eef488fbd0d3e4873533f89b6051b52d69e8dba1de86abec2",
                 0.97213173, 0.01255000, 0.72074167),
    "vtn_mem": ("vtn_mem", "ed77dd3e930e949c0bb902552482b517b5c7eacbe6275a13c085b6c28c0ce295",
                0.56225514, 0.01260357, 0.69441667),
    "esr_hcp": ("esr_hcp", "26c1772c982b1746d1acf3cdac1f268f175ac13d653396f50a28639431bfc4f0",
                0.84484005, 0.01455357, 0.38851667),
    "fco": ("fco", "4bd691fded1051f1878cee8edcd98af92b717469e221888fa87ad0ef153cccdb",
            0.85195476, 0.01589643, 0.57143333),
    "fr_hcp": ("fr_hcp", "775c42b9c059f28951afa6995ede6cc7cfc7ec6b419be0992ebc1191ca1cec6e",
               0.85247308, 0.01349286, 0.57136667),
    "blvs": ("blvs_n_fco", "cb81cea2eb1de538897c128e8c011de53ab18d2b91f4cde811bc51e3096c9c09",
             0.84297073, 0.02754286, 0.32061667),
    "967770": ("esrn_fco", "95be504d2cefe7184edbce17a485bce0c6690a039163a57896a0423af194280f",
               0.84297073, 0.01753571, 0.38998333),
    "pez": ("pez_blvs_n_fco", "f28ce5876ad7e18c0598c07e0a637059958f8d62b52140e6578d7080fc1445b0",
            0.84297073, 0.05121786, 0.14808333),
}
T2_LEN, ATT_LEN = 2000, 600
THP = np.float32(0.84297073)                    # PEZ official operating threshold (= T2 threshold)
CUT = np.float32(-0.15930954)                   # PEZ official T1 EER threshold
LAYOUT_TOL, LAYOUT_UNIQ = 2e-3, 5               # frozen: identified iff min loss <= tol and unique within +-5 frames
REPRO_NDIFF, REPRO_MAX = 10, 1e-6               # recomputed PEZ vs ZIP (cross-process 1-ulp jitter, see PEZ stage 0)
GATES = ("hist", "zP", "oV")
PASS_LO, PASS_HI, K_GEN1 = 0.015, 0.027, 0.03   # net bands: < .015 fail, [.015, .027) indeterminate, >= .027 pass
EER_BLVS_PROXY, K2_THR = 0.0030090251291611243, 0.8381215   # admission (T1 <= BLVS) on the proxy; PEZ stage-0 floor
C5_ENERGY_BLIND = 0.5
S2_EDGES = [0.27, 0.5, 0.79, 1.0]               # bins: <.27, .27-.5, .5-.79, .79-1, ==1
CELLS = ("C1_oV_lt1", "C2_no_hist", "C3_zP_le0", "C3_LS_eq1", "C4_discounted", "C5_s2_eq1")
D = {}


def states(x, g):
    """PEZ intermediate states (same calls as analyze_pez_stage0.parts) plus zP, LS and per-gate omega_En."""
    f = H._fe(x, g)
    K, zP, LP, LS, _, _ = R5._base(x, g, f)
    oE = R5.esr_omega_of(x, g, K, zP, LS)
    oV = BL.omega_v(x, g, zP, LS)
    v, _, _ = HV.verify(x, g)
    clean = np.linalg.norm(np.asarray(x, np.float64), axis=1) >= BL.CLEAN
    hist = np.cumsum(v & clean[:, None], axis=0) >= 1
    s2 = PZ.energy_s2(f["S"])
    low = s2[:, None] < 1
    shrink = s2[:, None] / (1.0 + (1.0 - s2[:, None]) * f["zS"] ** 2 / (K - 1))
    conds = {"hist": hist, "oV": oV == 1.0, "zP": zP > 0, "LS": LS > 1, "low": low}
    act = conds["hist"] & conds["oV"] & conds["zP"] & conds["LS"] & low
    oEn = np.where(act, shrink, 1.0)
    alt = {}
    for gname in GATES:
        a = low & conds["LS"]
        for c in ("hist", "oV", "zP"):
            if c != gname:
                a = a & conds[c]
        alt[gname] = np.where(a, shrink, 1.0)
    cell = np.select([oV < 1, ~hist, zP <= 0, LS <= 1, low], [0, 1, 2, 3, 4], 5).astype(np.int8)
    return f, oE, oV, oEn, alt, cell, s2


def _job(task):
    tr, i = task
    x, g = D[f"s{tr}"][i], D[f"g{tr}"]
    f, oE, oV, oEn, alt, cell, s2 = states(x, g)
    pez = R5.esrn_fco(x, g, f, omega=oE * oV * oEn).astype(np.float32)
    dif = np.abs(pez.astype(np.float64) - D[f"P{tr}"][i].astype(np.float64))
    rep = (int((dif > 0).sum()), float(dif.max()))
    c = int(D[f"c{tr}"][i])
    out = {"cell": cell[:, c], "s2": s2}
    full = {"pez": pez}
    for gname in GATES:
        full[gname] = R5.esrn_fco(x, g, f, omega=oE * oV * alt[gname]).astype(np.float32)
    if tr == 1:
        out["full"] = full                                  # T1: full rows for exact EER / FAR=1e-5 thresholds
        for gname in GATES:
            q = full[gname]
            out[f"up_{gname}"] = int((q > pez).sum())
            out[f"cross_{gname}"] = int(((pez[:, c] >= CUT) & (q[:, c] < CUT)).sum())
    else:
        out["col"] = {k: v[:, c].copy() for k, v in full.items()}
        out.update({f"up_{gname}": int((full[gname] > pez).sum()) for gname in GATES})
    return tr, i, rep, out


def by_cell(acc, cell, s2):
    sb = np.searchsorted(S2_EDGES, s2, side="right")          # 0..3 for s2 < 1, 4 for s2 == 1
    out = {}
    for ci, cn in enumerate(CELLS):
        m = acc & (cell == ci)
        out[cn] = {"n": int(m.sum()), "s2_bins": np.bincount(sb[m], minlength=5).astype(int).tolist()}
    return out


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def layout_scan(prof, official, n_seq):
    """prof[name] = accepted counts per position (summed over sequences). Returns per-s loss and per-sub predictions."""
    s_all = np.arange(0, T2_LEN - ATT_LEN + 1)
    loss = np.zeros(s_all.size)
    preds = {}
    for name, p in prof.items():
        cs = np.concatenate([[0], np.cumsum(p, dtype=np.float64)])
        att = cs[s_all + ATT_LEN] - cs[s_all]
        gen_acc = cs[-1] - att
        aar = att / (n_seq * ATT_LEN)
        frr = 1.0 - gen_acc / (n_seq * (T2_LEN - ATT_LEN))
        frr_o, aar_o = official[name]
        loss = np.maximum(loss, np.maximum(np.abs(frr - frr_o), np.abs(aar - aar_o)))
        preds[name] = (frr, aar)
    return s_all, loss, preds


def layout_decide(s_all, loss):
    k = int(loss.argmin())
    near = s_all[loss <= 2 * loss[k]]
    ok = bool(loss[k] <= LAYOUT_TOL and np.all(np.abs(near - s_all[k]) <= LAYOUT_UNIQ))
    return ok, int(s_all[k]), float(loss[k])


def main(workers=6):
    import analyze_t1_residual as ar
    import common_check as cc
    from proxy_lib import load_track, pseudo_targets
    t0 = time.time()
    shas = {k: _sha(ART / v[0] / "submission.zip") for k, v in SUBS.items()}
    stop = {f"sha_{k}": shas[k] == SUBS[k][1] for k in SUBS}
    g1, s1, n1 = load_track("track1")
    g2, s2_, n2 = load_track("track2")
    c1, _ = pseudo_targets(g1, s1)
    c2, _ = pseudo_targets(g2, s2_)
    stop["t2_len_2000"] = all(x.shape[0] == T2_LEN for x in s2_)
    # ---- L: layout, one ZIP at a time, claimed column only ----
    prof, official = {}, {}
    for k, (d, _, thr, frr, aar) in SUBS.items():
        S = HV._read_zip(ART / d / "submission.zip")
        T2 = [S["track2"][f"score_{n}.npz"] for n in n2]
        if k == "pez":
            D.update(P1=[S["track1"][f"score_{n}.npz"] for n in n1], P2=T2)
        acc = np.stack([T2[i][:, int(c2[i])] >= np.float32(thr) for i in range(len(n2))])
        prof[k] = acc.sum(0)
        official[k] = (frr, aar)
        del S, T2
    s_all, loss, preds = layout_scan(prof, official, len(n2))
    lay_ok, s_star, l_star = layout_decide(s_all, loss)
    lay = {"identified": lay_ok, "s_star": s_star, "loss_star": l_star,
           "loss_at_s0": float(loss[0]), "loss_at_s1400": float(loss[-1]),
           "loss_q_min_10_50": [float(np.quantile(loss, q)) for q in (0, .1, .5)],
           "pred_at_s_star": {k: [round(float(p[0][s_star]), 6), round(float(p[1][s_star]), 6)]
                              for k, p in preds.items()},
           "pez_accept_profile_100": [round(float(prof["pez"][b:b + 100].sum() / (100 * len(n2))), 4)
                                      for b in range(0, T2_LEN, 100)],
           "967770_accept_profile_100": [round(float(prof["967770"][b:b + 100].sum() / (100 * len(n2))), 4)
                                         for b in range(0, T2_LEN, 100)]}
    print(json.dumps({"stop": stop, "layout": lay}), flush=True)
    if not all(stop.values()):
        print(json.dumps({"verdict": "STOP", "seconds": round(time.time() - t0, 1)}), flush=True)
        return
    # ---- D, G: recompute PEZ states ----
    D.update(s1=s1, g1=g1, s2=s2_, g2=g2, c1=c1, c2=c2)
    tasks = [(1, i) for i in range(len(s1))] + [(2, i) for i in range(len(s2_))]
    R = {1: {}, 2: {}}
    nd, mx = 0, 0.0
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 6)), mp_context=mp.get_context("fork")) as ex:
        for tr, i, rep, out in ex.map(_job, tasks, chunksize=1):
            nd, mx = nd + rep[0], max(mx, rep[1])
            R[tr][i] = out
    repro = {"ndiff": nd, "maxdiff": mx, "ok": nd <= REPRO_NDIFF and mx <= REPRO_MAX}
    n1s, n2s = len(s1), len(s2_)
    # exact T1 metrics of the recomputed PEZ and of each gate counterfactual (own FAR=1e-5 threshold = T2 threshold)
    t1 = {}
    for k in ("pez",) + GATES:
        gen, imp = ar._gen_imp([R[1][i]["full"][k] for i in range(n1s)], c1)
        met = ar._metrics(gen, imp)
        t1[k] = {"eer_full": cc._eer_full(gen, imp), "far1e5_thr": met["far1e5_thr"],
                 "frr_far1e5": met["frr_at_far1e5"]}
        del gen, imp
    A1 = np.concatenate([R[1][i]["full"]["pez"][:, int(c1[i])] >= THP for i in range(n1s)])
    C1 = np.concatenate([R[1][i]["cell"] for i in range(n1s)])
    S21 = np.concatenate([R[1][i]["s2"] for i in range(n1s)])
    for i in range(n1s):
        del R[1][i]["full"]
    thr = {k: np.float32(t1[k]["far1e5_thr"]) for k in t1}
    att = np.zeros(T2_LEN, bool)
    att[s_star:s_star + ATT_LEN] = True
    col = {k: np.stack([R[2][i]["col"][k] for i in range(n2s)]) for k in ("pez",) + GATES}
    A2 = col["pez"] >= THP                                   # decomposition at the official threshold
    C2 = np.stack([R[2][i]["cell"] for i in range(n2s)])
    S22 = np.stack([R[2][i]["s2"] for i in range(n2s)])
    n_att, n_gen2 = n2s * ATT_LEN, n2s * (T2_LEN - ATT_LEN)
    seq_aar = (A2 & att[None, :]).sum(1) / ATT_LEN
    hi = seq_aar >= 0.3
    dec = {"pooled_frr_pred": float(1 - (A2 & ~att[None, :]).sum() / n_gen2),
           "pooled_aar_pred": float((A2 & att[None, :]).sum() / n_att),
           "attack_accepted": by_cell(A2 & att[None, :], C2, S22),
           "t2_genuine_accepted": by_cell(A2 & ~att[None, :], C2, S22),
           "t1_genuine_accepted": by_cell(A1, C1, S21),
           "seq_aar_q10_50_90_max": np.quantile(seq_aar, [.1, .5, .9, 1]).round(4).tolist(),
           "n_seq_aar_ge_.3": int(hi.sum()),
           "attack_accepted_in_seq_aar_ge_.3": by_cell((A2 & att[None, :])[hi], C2[hi], S22[hi])}
    tot_att_acc = sum(v["n"] for v in dec["attack_accepted"].values())
    c5_share = dec["attack_accepted"]["C5_s2_eq1"]["n"] / max(1, tot_att_acc)
    # each configuration at its OWN threshold (pez recomputed at its own threshold as the reference)
    acc_own = {k: col[k] >= thr[k] for k in col}
    ref_att = (acc_own["pez"] & att[None, :]).sum() / n_att
    ref_frr = 1 - (acc_own["pez"] & ~att[None, :]).sum() / n_gen2
    gates = {}
    for gname in GATES:
        a = acc_own[gname]
        d_aar = ref_att - (a & att[None, :]).sum() / n_att
        d_frr2 = (1 - (a & ~att[None, :]).sum() / n_gen2) - ref_frr
        net = float(d_aar - d_frr2)
        wa = np.where(hi, 4 / 3, 8 / 9)[:, None]                # RE-proxy macro weights, report only
        net_w = float(((acc_own["pez"] & ~a & att[None, :]) * wa).sum() / n_att
                      - ((acc_own["pez"] & ~a & ~att[None, :]) * wa).sum() / n_gen2)
        up = sum(R[1][i][f"up_{gname}"] for i in range(n1s)) + sum(R[2][i][f"up_{gname}"] for i in range(n2s))
        cross = sum(R[1][i][f"cross_{gname}"] for i in range(n1s))
        d_frr1 = t1[gname]["frr_far1e5"] - t1["pez"]["frr_far1e5"]
        ok = {"eer<=blvs_proxy": t1[gname]["eer_full"] <= EER_BLVS_PROXY,
              "thr>=.8381215": t1[gname]["far1e5_thr"] >= K2_THR,
              "d_frr1<=.03": d_frr1 <= K_GEN1, "no_upward": up == 0}
        band = ("pass" if net >= PASS_HI else "indeterminate" if net >= PASS_LO else "fail")
        gates[gname] = {"t1": t1[gname], "d_frr1": d_frr1, "eer_crossings_at_cut": int(cross), "upward": int(up),
                        "d_aar_pooled": float(d_aar), "d_frr2_pooled": float(d_frr2), "net": net,
                        "net_re_weighted_report_only": net_w, "net_band": band, "checks": ok,
                        "pass": all(ok.values()) and band == "pass",
                        "indeterminate": all(ok.values()) and band == "indeterminate"}
    passing = sorted([k for k in GATES if gates[k]["pass"]],
                     key=lambda k: (-gates[k]["net"], gates[k]["eer_crossings_at_cut"]))
    choice = None
    if not repro["ok"]:
        verdict = "STOP"
    elif not lay_ok:
        verdict = "LAYOUT_UNRESOLVED"
    elif passing:
        choice = passing[0]
        verdict = f"B:{choice}"
    elif any(gates[k]["indeterminate"] for k in GATES):
        verdict = "B_INDETERMINATE"
    else:
        verdict = "C:energy_blind" if c5_share >= C5_ENERGY_BLIND else "NO_GATE:mixed"
    print(json.dumps({"repro": repro, "t1_pez_recomputed": t1["pez"], "ref_own_thr": {"aar": float(ref_att),
                      "frr2": float(ref_frr)}, "decomposition": dec, "c5_share_of_attack_accepted": c5_share,
                      "gates": gates}), flush=True)
    print(json.dumps({"verdict": verdict, "choice": choice, "seconds": round(time.time() - t0, 1)}), flush=True)


# ---------------- random-data checks (no challenge data) ----------------
def synth():
    rng = np.random.default_rng(11)
    ok = {}
    K, T = 60, 180
    g = rng.normal(size=(K, 256)).astype(np.float32)
    amp = np.where(np.arange(T) < 120, 3.0, 1.2)[:, None]
    x = rng.normal(size=(T, 256)) + amp * g[2]
    x = (x / np.linalg.norm(x, axis=1, keepdims=True) * rng.uniform(17, 26, size=(T, 1))).astype(np.float32)
    f, oE, oV, oEn, alt, cell, s2 = states(x, g)
    _, oE0, oV0, oEn0, s20, _, _ = PZ.parts(x, g)
    ok["states_eq_pez_parts"] = bool(np.array_equal(oEn, oEn0) and np.array_equal(oV, oV0) and np.array_equal(s2, s20))
    pez = R5.esrn_fco(x, g, f, omega=oE * oV * oEn)
    ok["pez_eq_module"] = bool(np.array_equal(pez, PZ.pez_blvs_n_fco(x, g)))
    _, zP, _, LS, _, _ = R5._base(x, g, f)
    v, _, _ = HV.verify(x, g)
    clean = np.linalg.norm(np.asarray(x, np.float64), axis=1) >= BL.CLEAN
    conds = {"hist": np.cumsum(v & clean[:, None], axis=0) >= 1, "oV": oV == 1.0, "zP": zP > 0}
    base = (s2[:, None] < 1) & (LS > 1)
    ok["pez_active_nonvacuous"] = bool((oEn < 1).any())
    exact = True
    for k in GATES:
        m = base.copy()
        for c in conds:
            if c != k:
                m &= conds[c]
        exact &= bool(np.array_equal(alt[k] < 1, m))
        exact &= bool(np.array_equal(alt[k][conds[k] & m], oEn[conds[k] & m]))
    ok["alt_drops_exactly_one_gate"] = exact
    ok["alt_le_1"] = all(bool(((alt[k] > 0) & (alt[k] <= 1)).all()) for k in GATES)
    ok["cells_partition"] = bool(np.isin(cell, np.arange(6)).all())
    ok["c4_iff_oEn_lt1"] = bool(np.array_equal(cell == 4, oEn < 1))
    bc = by_cell(np.ones(T, bool), cell[:, 2], s2)
    ok["by_cell_counts_sum"] = sum(v_["n"] for v_ in bc.values()) == T and bool(json.dumps(bc))
    q = R5.esrn_fco(x, g, f, omega=oE * oV * alt["hist"])
    ok["gate_cf_lowers_only"] = bool((q <= pez + 1e-15).all())
    # planted layout: 30 sequences, attack window at s=700, reject attacks w.p. .6, genuine w.p. .05
    n = 30
    official, prof = {}, {}
    for k, (pa, pg) in enumerate([(0.6, 0.05), (0.3, 0.02), (0.8, 0.1)]):
        acc = np.ones((n, T2_LEN), bool)
        acc[:, 700:1300] = rng.random((n, 600)) > pa
        acc[:, :700] = rng.random((n, 700)) > pg
        acc[:, 1300:] = rng.random((n, 700)) > pg
        prof[k] = acc.sum(0)
        official[k] = (1 - (acc[:, :700].sum() + acc[:, 1300:].sum()) / (n * 1400), acc[:, 700:1300].sum() / (n * 600))
    s_all, loss, _ = layout_scan(prof, official, n)
    lok, sst, lst = layout_decide(s_all, loss)
    ok["layout_planted_recovered"] = lok and sst == 700 and lst < 1e-12
    prof_flat = {k: np.full(T2_LEN, n * 0.7) for k in range(3)}
    s_all, loss, _ = layout_scan(prof_flat, {k: (0.3, 0.7) for k in range(3)}, n)
    ok["layout_flat_unresolved"] = not layout_decide(s_all, loss)[0]
    print(json.dumps({"synth": ok, "all_ok": all(ok.values())}), flush=True)
    return all(ok.values())


if __name__ == "__main__":
    if sys.argv[1:] == ["--synth"]:
        sys.exit(0 if synth() else 1)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
