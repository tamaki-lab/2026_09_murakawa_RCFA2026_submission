"""BLVS stage 0: premise check for the Beta-Laplace verification-survival discount on ESR-N + FCO (analysis only).

Plan, user ruling and frozen decision rule: docs/experiments/20261008_ogata_blvs_stage0.md.
Design origin: round 6 ideation (20261008_ogata_r6_ideation.md, design #2, lens 5) with the critic's fixes.

BLVS-N+FCO = x_r5.esrn_fco with omega = omega_E * omega_V (omega_E = ESR's weight, unchanged):
  v_tj   = [LOO row z of the raw cosine >= Z0], Z0 = Phi^-1(1 - 1e-5) (analyze_hvc_stage0.verify; zero-norm: unverified)
  clean  c_t = [|x_t| >= 20]
  state per column (fresh per sequence): A verified clean count, B past unverified clean count, n current
  unverified clean run, p = 1. On a clean frame: if v: B += n, A += 1, n = 0, p = 1;
  elif A >= 1: p *= (B + 1 + n) / (A + B + 2 + n), n += 1; else nothing. Non-clean frames hold the state.
  omega_V = min(1, e p) if n >= 1 and zP > 0 and LS > 1, else 1.
BLVS-FCO (report-only ablation, never built): x_r5.esrn_fco with omega = omega_V (no ESR).

(a) Track 1 (evaluator-only pseudo targets): eer_full, own FAR=1e-5 threshold, FRR@FAR1e-5, impostor entries touched
    by omega_V, genuine entries lowered below the EER cut, genuine omega_V < .05 share, new genuine rejection runs.
(b) Track 2 (evaluator-side diagnostic layout of analyze_hvc_stage0: 2000 frames, t < 1400 genuine, claimed column =
    whole-sequence mean raw cosine argmax; calibration on 967770 first, type rules A then B), then BLVS's predicted
    per-type values and BE at its own T1 FAR=1e-5 threshold.
STOP if ESR-N + FCO recomputed here (omega_V = 1) is not bit-identical to 967770's ZIP on every sequence.
Prints JSON aggregates only.
Usage (server): python scripts/core/analyze_blvs_stage0.py ZIP [workers<=6] > results/ogata/blvs_stage0.log 2>&1
                python scripts/core/analyze_blvs_stage0.py --synth
"""
import io
import json
import math
import multiprocessing as mp
import os
import sys
import time
import zipfile

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
import analyze_hvc_stage0 as HV  # noqa: E402  (verify, T2 layout helpers and 967770's official values)

E = math.e
CLEAN = 20.0
EER_967770 = 0.003009209994988715
THR_967770 = 0.8431214690208435
KILL_EER = EER_967770 + 0.00002
KILL_THR = THR_967770 - 0.02
KILL_FRR5 = 0.20
PASS_BE = 0.225
D = {}


def omega_v(x, g, zP, LS):
    v, _, _ = HV.verify(x, g)
    nrm = np.linalg.norm(np.asarray(x, np.float64), axis=1)
    T, K = v.shape
    A = np.zeros(K)
    B = np.zeros(K)
    n = np.zeros(K)
    p = np.ones(K)
    om = np.ones((T, K))
    for t in range(T):
        if nrm[t] >= CLEAN:
            vt = v[t]
            B = np.where(vt, B + n, B)
            A = np.where(vt, A + 1, A)
            run = ~vt & (A >= 1)
            p = np.where(vt, 1.0, np.where(run, p * (B + 1 + n) / (A + B + 2 + n), p))
            n = np.where(vt, 0.0, np.where(run, n + 1, n))
        act = (n >= 1) & (zP[t] > 0) & (LS[t] > 1)
        om[t] = np.where(act, np.minimum(1.0, E * p), 1.0)
    return om


def blvs_n_fco(x, g):
    """BLVS-N+FCO (float64): x_r5.esrn_fco with omega = omega_E * omega_V; the submission design of this stage 0."""
    f = H._fe(x, g)
    K, zP, LP, LS, _, _ = R5._base(x, g, f)
    return R5.esrn_fco(x, g, f, omega=R5.esr_omega_of(x, g, K, zP, LS) * omega_v(x, g, zP, LS))


def outputs_fn(x, g):
    """Reference for build_candidate's bit-identity check (same expression as outputs()'s blvs)."""
    return {"blvs_n_fco": blvs_n_fco(x, g).astype(np.float32)}


def outputs(x, g):
    f = H._fe(x, g)
    K, zP, LP, LS, _, _ = R5._base(x, g, f)
    oE = R5.esr_omega_of(x, g, K, zP, LS)
    oV = omega_v(x, g, zP, LS)
    ref = R5.esrn_fco(x, g, f, omega=oE).astype(np.float32)
    blvs = R5.esrn_fco(x, g, f, omega=oE * oV).astype(np.float32)
    abl = R5.esrn_fco(x, g, f, omega=oV).astype(np.float32)
    return ref, blvs, abl, oV


def _job(task):
    tr, i = task
    x = D[f"s{tr}"][i]
    g = D[f"g{tr}"]
    o_zip = D[f"O{tr}"][i]
    ref, blvs, abl, oV = outputs(x, g)
    same = bool(np.array_equal(ref, o_zip))
    if tr == 1:
        t = int(D["tg"][i])
        oth = np.ones(g.shape[0], bool)
        oth[t] = False
        aux = {"imp_touched": int((oV[:, oth] < 1).sum()), "gen_omega_lt_05": int((oV[:, t] < 0.05).sum()),
               "gen_omega_lt_1": int((oV[:, t] < 1).sum())}
        return tr, i, same, {"blvs": blvs, "abl": abl}, aux
    feats = HV.verify(x, g)
    r = feats[2]
    c = int(r.mean(0).argmax())
    att = slice(HV.T2_GEN, None)
    F = {"o": o_zip[:, c], "b": blvs[:, c], "f": abl[:, c], "c": c, "cg": int(r[:HV.T2_GEN].mean(0).argmax()),
         "med_att": float(np.median(r[att, c])), "other_att": float((r[att].argmax(1) != c).mean()),
         "flag": oV[:, c] < 0.05}
    return tr, i, same, F, None


def _runs(mask):
    out, cur = [], 0
    for m in mask:
        if m:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def main(zpath, workers=6):
    import analyze_t1_residual as ar
    import common_check as cc
    from proxy_lib import load_track, pseudo_targets
    t0 = time.time()
    S = HV._read_zip(zpath)
    g1, s1, n1 = load_track("track1")
    g2, s2, n2 = load_track("track2")
    tg, _ = pseudo_targets(g1, s1)
    D.update(s1=s1, g1=g1, s2=s2, g2=g2, tg=tg, O1=[S["track1"][f"score_{n}.npz"] for n in n1],
             O2=[S["track2"][f"score_{n}.npz"] for n in n2])
    del S
    tasks = [(1, i) for i in range(len(s1))] + [(2, i) for i in range(len(s2))]
    O1b, O1f, F2, aux1, same = {}, {}, {}, [], True
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 6)), mp_context=mp.get_context("fork")) as ex:
        for tr, i, sm, res, aux in ex.map(_job, tasks, chunksize=1):
            same &= sm
            if tr == 1:
                O1b[i], O1f[i] = res["blvs"], res["abl"]
                aux1.append(aux)
            else:
                F2[i] = res
    stop = {"esrn_fco_recomputed_bit_identical_all_400": bool(same)}
    n = len(s1)
    # ---- (a) Track 1 ----
    t1 = {}
    for k, outs in (("967770", D["O1"]), ("blvs", [O1b[i] for i in range(n)]), ("blvs_fco_ablation",
                                                                                [O1f[i] for i in range(n)])):
        gen, imp = ar._gen_imp(outs, tg)
        met = ar._metrics(gen, imp)
        thr = met["far1e5_thr"]
        t1[k] = {"eer_full": cc._eer_full(gen, imp), "eer_thr": met["eer_thr"], "far1e5_thr": thr,
                 "frr_far1e5": met["frr_at_far1e5"], "far_at_frr1e5": met["far_at_frr1e5"],
                 "macro_seq_frr_far1e5": float(np.mean([(o[:, t] < thr).mean() for o, t in zip(outs, tg)]))}
        del imp
    cut = np.float32(t1["967770"]["eer_thr"])
    thr_o, thr_b = np.float32(t1["967770"]["far1e5_thr"]), np.float32(t1["blvs"]["far1e5_thr"])
    lowered = 0
    newrej = []
    for i in range(n):
        o, b, t = D["O1"][i][:, tg[i]], O1b[i][:, tg[i]], int(tg[i])
        lowered += int(((o >= cut) & (b < cut)).sum())
        newrej += _runs((o >= thr_o) & (b < thr_b))
    ngen = sum(o.shape[0] for o in D["O1"])
    t1["genuine_lowered_below_eer_cut"] = lowered
    t1["new_genuine_rejection_runs_at_own_far_cut"] = {"count": len(newrej), "q50_90_99_max": (
        np.quantile(newrej, [.5, .9, .99, 1]).round(1).tolist() if newrej else [])}
    t1["impostor_entries_with_omega_v_lt_1"] = int(sum(a["imp_touched"] for a in aux1))
    t1["genuine_omega_v_lt_05_share"] = sum(a["gen_omega_lt_05"] for a in aux1) / ngen
    t1["genuine_omega_v_lt_1_share"] = sum(a["gen_omega_lt_1"] for a in aux1) / ngen
    print(json.dumps({"stop": stop, "t1": t1}), flush=True)
    kill = {"eer_full<=967770+.00002": t1["blvs"]["eer_full"] <= KILL_EER,
            "own_far1e5_thr>=.8431-.02": t1["blvs"]["far1e5_thr"] >= KILL_THR,
            "frr_far1e5<=.20": t1["blvs"]["frr_far1e5"] <= KILL_FRR5}
    # ---- (b) Track 2 ----
    F = [F2[i] for i in range(len(s2))]
    b = {"status": "UNRESOLVED", "lengths_ok": all(x.shape[0] == HV.T2_LEN for x in s2) and len(s2) == 200}
    pooled = HV.t2_metrics(F, "o", thr_o, None)
    b["pooled_967770"] = HV._within(pooled, {k: HV.OFFICIAL[k] for k in ("pooled_frr", "pooled_aar")},
                                    HV.POOLED_TOL)
    b["pooled_ok"] = all(v[2] for v in b["pooled_967770"].values())
    chosen, b["rules"] = None, {}
    for rule in ("A", "B"):
        lab, gaps = HV.type_rule(F, rule)
        m = HV.t2_metrics(F, "o", thr_o, lab)
        rep = HV._within(m, {k: HV.OFFICIAL[k] for k in HV.OFFICIAL if not k.startswith("pooled")}, HV.TYPE_TOL)
        ok = all(v[2] for v in rep.values())
        b["rules"][rule] = {"gaps": gaps, "repro_967770": rep, "repro_ok": ok,
                            "runs_967770": {ty: [m[f"{ty}_run"], HV.OFFICIAL_RUN[ty]] for ty in HV.N_TYPE}}
        if ok and chosen is None:
            chosen = (rule, lab)
    b["claimed_eq_genuine_part_argmax"] = int(sum(f["c"] == f["cg"] for f in F))
    if b["lengths_ok"] and b["pooled_ok"] and chosen is not None:
        rule, lab = chosen
        pb = HV.t2_metrics(F, "b", thr_b, lab)
        b.update({"rule": rule, "blvs_pred": pb,
                  "ablation_pred_report_only": HV.t2_metrics(F, "f", np.float32(t1["blvs_fco_ablation"]["far1e5_thr"]),
                                                             lab),
                  "blvs_at_967770_thr": HV.t2_metrics(F, "b", thr_o, lab)})
        fl = {ty: [float(np.mean([F[i]["flag"][:HV.T2_GEN].mean() for i in range(len(F)) if lab[i] == ty])),
                   float(np.mean([F[i]["flag"][HV.T2_GEN:].mean() for i in range(len(F)) if lab[i] == ty]))]
              for ty in HV.N_TYPE}
        b["flag_share_genuine_part_vs_attack_part"] = fl
        b["status"] = "PASS" if pb["be"] <= PASS_BE else "FAIL"
    verdict = ("STOP" if not all(stop.values()) else "KILL" if not all(kill.values()) else b["status"])
    print(json.dumps({"t2": b}), flush=True)
    print(json.dumps({"kill": kill, "verdict": verdict, "seconds": round(time.time() - t0, 1)}), flush=True)


# ---------------- random-data checks (no challenge data) ----------------
def _omega_naive(x, g, zP, LS):
    v, _, _ = HV.verify(x, g)
    nrm = np.linalg.norm(np.asarray(x, np.float64), axis=1)
    T, K = v.shape
    om = np.ones((T, K))
    for j in range(K):
        A = B = nn = 0
        for t in range(T):
            if nrm[t] >= CLEAN:
                if v[t, j]:
                    B += nn
                    A += 1
                    nn = 0
                elif A >= 1:
                    nn += 1
            if nn >= 1 and zP[t, j] > 0 and LS[t, j] > 1:
                p = 1.0
                for k in range(nn):
                    p *= (B + 1 + k) / (A + B + 2 + k)
                om[t, j] = min(1.0, E * p)
    return om


def synth():
    rng = np.random.default_rng(5)
    ok = {}
    K, T = 60, 160
    g = rng.normal(size=(K, 256)).astype(np.float32)
    x = rng.normal(size=(T, 256)) + np.where(np.arange(T)[:, None] < 100, 3.0, 0.0) * g[2]
    x = (x / np.linalg.norm(x, axis=1, keepdims=True) * rng.uniform(17, 26, size=(T, 1))).astype(np.float32)
    x[50] = 0
    f = H._fe(x, g)
    K_, zP, LP, LS, _, _ = R5._base(x, g, f)
    om = omega_v(x, g, zP, LS)
    ok["omega_eq_naive"] = bool(np.allclose(om, _omega_naive(x, g, zP, LS), rtol=1e-12, atol=1e-15))
    ok["omega_in_0_1"] = bool(((om > 0) & (om <= 1)).all())
    clean_after = [t for t in range(100, T) if np.linalg.norm(x[t]) >= CLEAN]
    ok["switch_discounted_within_3_clean_frames"] = bool(om[clean_after[2], 2] < 1)
    oE = R5.esr_omega_of(x, g, K_, zP, LS)
    ref = R5.esrn_fco(x, g, f, omega=oE)
    ok["omega_v_1_is_esrn_fco"] = bool(np.array_equal(R5.esrn_fco(x, g, f, omega=oE * np.ones_like(om)), ref))
    blvs = R5.esrn_fco(x, g, f, omega=oE * om)
    ok["blvs_le_esrn_fco"] = bool((blvs <= ref + 1e-15).all())
    h = T // 2
    fh = H._fe(x[:h], g)
    _, zPh, _, LSh, _, _ = R5._base(x[:h], g, fh)
    ok["prefix_invariant"] = bool(np.array_equal(omega_v(x[:h], g, zPh, LSh), om[:h]))
    ok["runs"] = _runs([1, 1, 0, 1]) == [2, 1]
    print(json.dumps({"synth": ok, "all_ok": all(ok.values())}), flush=True)
    return all(ok.values())


if __name__ == "__main__":
    if sys.argv[1:] == ["--synth"]:
        sys.exit(0 if synth() else 1)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
