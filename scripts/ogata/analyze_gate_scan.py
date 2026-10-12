"""Single-gate ablation scan of PEZ-NH's remaining structural gates (T1, analysis only).

Plan and frozen rules: docs/experiments/20261009_ogata_gate_scan.md.

Parent PEZ-NH = esrn_fco with omega = omega_E * omega_V * omega_En(no hist). Each scan item removes exactly one gate:
  E_zP     omega_E without its zP > 0 activity gate          (x_r3.esr_omega called with a positive zP stand-in)
  V_zP     omega_V without its zP > 0 activity gate          (analyze_blvs_stage0.omega_v with a positive zP stand-in)
  V_clean  omega_V updated on every frame, not only |x| >= 20 (analyze_blvs_stage0.CLEAN = 0 inside omega_v only)
Exact T1 metrics for each; T2 claimed-column accepted share at each own threshold is reported only.
Pseudo targets on the evaluator side only. Prints JSON aggregates only.
Usage (server): python scripts/ogata/analyze_gate_scan.py [workers<=6] > results/ogata/gate_scan.log
                python scripts/ogata/analyze_gate_scan.py --synth
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
import analyze_hvc_stage0 as HV  # noqa: E402
import analyze_blvs_stage0 as BL  # noqa: E402
import analyze_pez_residual as RS  # noqa: E402
import analyze_t1_drp_cla as DC  # noqa: E402

ITEMS = ("E_zP", "V_zP", "V_clean")
NH = DC.NH
K_GAIN, K_THR, K_DFRR = 1e-5, 0.8381215, 0.03
REPRO_EER = 1e-9
D = {}


def omega_v_noclean(x, g, zP, LS):
    saved = BL.CLEAN
    BL.CLEAN = 0.0
    try:
        return BL.omega_v(x, g, zP, LS)
    finally:
        BL.CLEAN = saved


def oen_nh(s2, zS, K, LS, zP, oV):
    """PEZ-NH's omega_En for a given omega_V (same formula as analyze_pez_residual.states' alt['hist'])."""
    low = s2[:, None] < 1
    shrink = s2[:, None] / (1.0 + (1.0 - s2[:, None]) * zS ** 2 / (K - 1))
    act = low & (LS > 1) & (zP > 0) & (oV == 1.0)
    return np.where(act, shrink, 1.0)


def variants(x, g):
    """Parent and the three single-gate ablations (float64). omega_En is recomputed with each item's own omega_V."""
    f, oE, oV, _, alt, _, s2 = RS.states(x, g)
    K, zP, LP, LS, _, _ = R5._base(x, g, f)
    oEn = oen_nh(s2, f["zS"], K, LS, zP, oV)
    assert np.array_equal(oEn, alt["hist"])
    pos = np.ones_like(zP)
    out = {"parent": R5.esrn_fco(x, g, f, omega=oE * oV * oEn)}
    oE_nz = R5.esr_omega_of(x, g, K, pos, LS)
    out["E_zP"] = R5.esrn_fco(x, g, f, omega=oE_nz * oV * oEn)
    oV_nz = BL.omega_v(x, g, pos, LS)
    out["V_zP"] = R5.esrn_fco(x, g, f, omega=oE * oV_nz * oen_nh(s2, f["zS"], K, LS, zP, oV_nz))
    oV_nc = omega_v_noclean(x, g, zP, LS)
    out["V_clean"] = R5.esrn_fco(x, g, f, omega=oE * oV_nc * oen_nh(s2, f["zS"], K, LS, zP, oV_nc))
    return out


def _job(task):
    tr, i = task
    x, g = D[f"s{tr}"][i], D[f"g{tr}"]
    v = variants(x, g)
    if tr == 1:
        return tr, i, {k: a.astype(np.float32) for k, a in v.items()}
    c = int(D["c2"][i])
    return tr, i, {k: a.astype(np.float32)[:, c].copy() for k, a in v.items()}


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
    for k in ("parent",) + ITEMS:
        gen, imp = ar._gen_imp([R[1][i][k] for i in range(len(s1))], c1)
        met = ar._metrics(gen, imp)
        t1[k] = {"eer_full": cc._eer_full(gen, imp), "eer_thr": met["eer_thr"], "far1e5_thr": met["far1e5_thr"],
                 "frr_far1e5": met["frr_at_far1e5"], "far_at_frr1e5": met["far_at_frr1e5"]}
        del gen, imp
    stop = {"parent_repro_eer": abs(t1["parent"]["eer_full"] - NH["eer_full"]) <= REPRO_EER}
    acc = {k: float(np.concatenate([R[2][i][k] >= np.float32(t1[k]["far1e5_thr"]) for i in range(len(s2_))]).mean())
           for k in ("parent",) + ITEMS}
    items = {}
    for k in ITEMS:
        ok = {"eer<=parent-1e-5": t1[k]["eer_full"] <= t1["parent"]["eer_full"] - K_GAIN,
              "thr>=.8381215": t1[k]["far1e5_thr"] >= K_THR,
              "dfrr<=.03": t1[k]["frr_far1e5"] - t1["parent"]["frr_far1e5"] <= K_DFRR}
        items[k] = {"t1": t1[k], "d_eer": t1[k]["eer_full"] - t1["parent"]["eer_full"],
                    "t2_acc_share_drop_report_only": acc["parent"] - acc[k], "checks": ok, "lead": all(ok.values())}
    leads = sorted([k for k in ITEMS if items[k]["lead"]], key=lambda k: items[k]["d_eer"])
    verdict = "STOP" if not all(stop.values()) else (f"LEAD:{leads[0]}" if leads else "NONE")
    print(json.dumps({"stop": stop, "parent": t1["parent"], "items": items}), flush=True)
    print(json.dumps({"verdict": verdict, "leads": leads, "seconds": round(time.time() - t0, 1)}), flush=True)


def synth():
    import analyze_peznh_stage0 as NHM
    import x_r4 as R4
    ok = {}
    changes = {k: 0.0 for k in ITEMS}
    for seed, kind in ((3, "switch"), (5, "mng"), (8, "mng")):
        x, g = R4._synth_data(seed, 240, 60, kind, norm_hi=26.0)        # norms 10..26: frames below and above 20
        v = variants(x, g)
        ok[f"parent_eq_peznh_{seed}"] = bool(np.array_equal(v["parent"], NHM.peznh_blvs_n_fco(x, g)))
        ok[f"finite_in_range_{seed}"] = all(bool(np.isfinite(a).all() and (np.abs(a) <= 1).all()) for a in v.values())
        h = 150
        vh = variants(x[:h], g)
        ok[f"prefix_invariant_{seed}"] = all(bool(np.allclose(vh[k], v[k][:h], rtol=0, atol=1e-12)) for k in v)
        for k in ITEMS:
            changes[k] = max(changes[k], float(np.abs(v[k] - v["parent"]).max()))
    ok["clean_restored"] = BL.CLEAN == 20.0
    ok["v_clean_active"] = changes["V_clean"] > 0          # E_zP / V_zP need zP <= 0 with zS > 0 inside their
    # structural state (an ongoing excursion / a verified history): rare on these fixtures; reported, not required
    print(json.dumps({"synth": ok, "all_ok": all(ok.values()), "max_abs_change": changes}), flush=True)
    return all(ok.values())


if __name__ == "__main__":
    if sys.argv[1:] == ["--synth"]:
        sys.exit(0 if synth() else 1)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
