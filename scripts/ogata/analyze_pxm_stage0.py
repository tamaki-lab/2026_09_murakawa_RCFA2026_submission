"""PXM stage 0: PEZ-NH with the teammate's max-prefix energy reference (analysis only).

Plan, provenance and frozen rules: docs/experiments/20261009_ogata_pxm_stage0.md.

PXM = PEZ-NH with s2 replaced by the max-prefix rule of rcfa.methods.max_prefix_rms_normalization (squared form):
  e_t = sum_j S_tj^2 (sequential float64), Ebar_t = (sum_{i<=t} e_i)/(t+1), P_t = max_{i<=t} Ebar_i,
  s2_t = min(1, e_t / P_t) (1 if P_t == 0).
omega_En = s2/(1 + (1 - s2) zS^2/(K - 1)) where (omega_V == 1) & (zP > 0) & (LS > 1) & (s2 < 1)  (PEZ-NH's act).
Pseudo targets on the evaluator side only. Prints JSON aggregates only.
Usage (server): python scripts/ogata/analyze_pxm_stage0.py [workers<=6] > results/ogata/pxm_stage0.log
                python scripts/ogata/analyze_pxm_stage0.py --synth
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
import analyze_pez_residual as RS  # noqa: E402
import analyze_gate_scan as GS  # noqa: E402
import analyze_t1_drp_cla as DC  # noqa: E402

NH = DC.NH
PEZ_PROXY_EER = 0.0030089891830279987
K_THR, K_FRR = 0.8381215, 0.30
REPRO_EER = 1e-9
D = {}


def energy_s2_maxprefix(rows):
    """Teammate's max-prefix rule in squared form: s2_t = min(1, e_t / max_{i<=t} Ebar_i), sequential float64."""
    R = np.asarray(rows, np.float64)
    e = np.zeros(R.shape[0])
    for j in range(R.shape[1]):
        e += R[:, j] * R[:, j]
    s2 = np.ones(R.shape[0])
    E = np.float64(0.0)
    P = np.float64(0.0)
    for t in range(R.shape[0]):
        E = np.float64(E + e[t])
        P = max(P, E / np.float64(t + 1))
        if P > 0:
            s2[t] = min(1.0, e[t] / P)
    return s2


def pxm_parts(x, g):
    f, oE, oV, _, alt, _, _ = RS.states(x, g)
    K, zP, LP, LS, _, _ = R5._base(x, g, f)
    s2m = energy_s2_maxprefix(f["S"])
    oEn_m = GS.oen_nh(s2m, f["zS"], K, LS, zP, oV)
    return f, oE, oV, alt["hist"], oEn_m, s2m


def pxm_peznh(x, g):
    f, oE, oV, _, oEn_m, _ = pxm_parts(x, g)
    return R5.esrn_fco(x, g, f, omega=oE * oV * oEn_m)


def outputs_fn(x, g):
    """Reference for build_candidate's bit-identity check."""
    return {"pxm_peznh": pxm_peznh(x, g).astype(np.float32)}


def _job(task):
    tr, i = task
    x, g = D[f"s{tr}"][i], D[f"g{tr}"]
    f, oE, oV, oEn, oEn_m, s2m = pxm_parts(x, g)
    par = R5.esrn_fco(x, g, f, omega=oE * oV * oEn).astype(np.float32)
    pxm = R5.esrn_fco(x, g, f, omega=oE * oV * oEn_m).astype(np.float32)
    if tr == 1:
        return tr, i, {"par": par, "pxm": pxm, "s2m_lt1": int((s2m < 1).sum())}
    c = int(D["c2"][i])
    return tr, i, {"par": par[:, c].copy(), "pxm": pxm[:, c].copy(), "s2m_lt1": int((s2m < 1).sum())}


def _runs_ge(b, L):
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    return int(((np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0]) >= L).sum())


def main(workers=6):
    import analyze_t1_residual as ar
    import common_check as cc
    from proxy_lib import load_track, pseudo_targets
    t0 = time.time()
    g1, s1, n1 = load_track("track1")
    g2, s2_, n2 = load_track("track2")
    c1, _ = pseudo_targets(g1, s1)
    c2, _ = pseudo_targets(g2, s2_)
    D.update(s1=s1, g1=g1, s2=s2_, g2=g2, c2=c2)
    tasks = [(1, i) for i in range(len(s1))] + [(2, i) for i in range(len(s2_))]
    R = {1: {}, 2: {}}
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 6)), mp_context=mp.get_context("fork")) as ex:
        for tr, i, res in ex.map(_job, tasks, chunksize=1):
            R[tr][i] = res
    t1 = {}
    for k in ("par", "pxm"):
        gen, imp = ar._gen_imp([R[1][i][k] for i in range(len(s1))], c1)
        met = ar._metrics(gen, imp)
        t1[k] = {"eer_full": cc._eer_full(gen, imp), "eer_thr": met["eer_thr"], "far1e5_thr": met["far1e5_thr"],
                 "frr_far1e5": met["frr_at_far1e5"], "far_at_frr1e5": met["far_at_frr1e5"]}
        del gen, imp
    stop = {"parent_repro_eer": abs(t1["par"]["eer_full"] - NH["eer_full"]) <= REPRO_EER}
    n2s = len(s2_)
    acc_p = np.stack([R[2][i]["par"] >= np.float32(t1["par"]["far1e5_thr"]) for i in range(n2s)])
    acc_x = np.stack([R[2][i]["pxm"] >= np.float32(t1["pxm"]["far1e5_thr"]) for i in range(n2s)])
    d2 = float(acc_p.mean() - acc_x.mean())
    g1_ = t1["pxm"]["frr_far1e5"] - t1["par"]["frr_far1e5"]
    est = {}
    for k, r in {"944021_944151": 0.117, "967770_blvs": 0.15166, "blvs_pez": 0.25627,
                 "pez_peznh_observed": 1.2}.items():
        dfrr2 = r * g1_
        a = (d2 - 0.7 * dfrr2) / 0.3
        est[k] = {"dFRR2": dfrr2, "dAAR": a, "dBE_pooled_est": -(a - dfrr2) / 2}
    t2 = {"d2_accepted_share_drop": d2, "upward_frames": int((acc_x & ~acc_p).sum()), "g1": g1_,
          "est_report_only": est, "worst_case_dBE_all_genuine": d2 / 0.7 / 2,
          "seqs_with_more_rejection": int(((~acc_x).sum(1) > (~acc_p).sum(1)).sum()),
          "rejection_runs_ge20_parent_vs_pxm": [sum(_runs_ge(~acc_p[i], 20) for i in range(n2s)),
                                                sum(_runs_ge(~acc_x[i], 20) for i in range(n2s))],
          "frames_s2max_lt1": [int(sum(R[1][i]["s2m_lt1"] for i in R[1])), int(sum(R[2][i]["s2m_lt1"] for i in R[2]))]}
    kill = {"eer<=pez_proxy": t1["pxm"]["eer_full"] <= PEZ_PROXY_EER, "thr>=.8381215": t1["pxm"]["far1e5_thr"] >= K_THR,
            "frr<=.30": t1["pxm"]["frr_far1e5"] <= K_FRR, "d2>0": d2 > 0}
    verdict = "STOP" if not all(stop.values()) else ("PASS" if all(kill.values()) else "KILL")
    print(json.dumps({"stop": stop, "t1": t1, "t2": t2, "kill": kill}), flush=True)
    print(json.dumps({"verdict": verdict, "seconds": round(time.time() - t0, 1)}), flush=True)


def synth():
    import analyze_peznh_stage0 as NHM
    import analyze_pez_stage0 as PZ
    import x_r4 as R4
    rng = np.random.default_rng(41)
    ok = {}
    # s2 rule against a naive loop, and the max-prefix property s2max <= s2 (prefix-mean rule)
    Rr = rng.normal(size=(60, 7))
    e = (Rr * Rr).sum(1)
    naive, P, E = [], 0.0, 0.0
    for t in range(60):
        E += e[t]
        P = max(P, E / (t + 1))
        naive.append(1.0 if P <= 0 else min(1.0, e[t] / P))
    ok["s2max_eq_naive"] = bool(np.allclose(energy_s2_maxprefix(Rr), naive, rtol=0, atol=1e-12))
    ok["s2max_le_s2"] = bool((energy_s2_maxprefix(Rr) <= PZ.energy_s2(Rr) + 1e-15).all())
    ok["s2max_const_is_1"] = bool((energy_s2_maxprefix(np.ones((20, 4))) == 1).all())
    for seed, kind in ((3, "switch"), (5, "mng")):
        x, g = R4._synth_data(seed, 240, 60, kind, norm_hi=26.0)
        out = pxm_peznh(x, g)
        par = NHM.peznh_blvs_n_fco(x, g)
        ok[f"finite_in_range_{seed}"] = bool(np.isfinite(out).all() and (np.abs(out) <= 1).all())
        ok[f"pxm_le_peznh_{seed}"] = bool((out <= par + 1e-15).all())
        ok[f"prefix_invariant_{seed}"] = bool(np.allclose(pxm_peznh(x[:150], g), out[:150], rtol=0, atol=1e-12))
        ok[f"changes_something_{seed}"] = bool(np.abs(out - par).max() > 0)
    print(json.dumps({"synth": ok, "all_ok": all(ok.values())}), flush=True)
    return all(ok.values())


if __name__ == "__main__":
    if sys.argv[1:] == ["--synth"]:
        sys.exit(0 if synth() else 1)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
