"""T1 check on bank4: rows still before the jump at t >= 10, and rows after an early jump at t < 10 (aggregates only).

bank4's ledger (`analyze_t1_bank4_tune.py`): owner-less rows 10 <= t < J (the opening segment still running) cost 263.8
gen-eq in 473 rows against a floor of ~243 for zP, and rows J <= t < 10 of the 11 sequences with an early jump cost 33.5
in 62 rows under the opening map although they follow the jump. "Before the jump" is causal: at frame t no jump has
been detected in frames 1..t and t < 60 (for Track 1 every sequence has J <= 17, so these are the rows t < J).
Variants (one change each; rows chosen from bank4's own masks, leader runs from the bank output as in bank4):
  pre_open        : free pre-jump rows 10 <= t < 70 get the opening map on cmz continued (cumulative mean of unit
                    frames 0..t, LOO z; clip(TH0 + .05 (cmz - 1)))
  pre_tau<t>      : the mid map with tau t (.75 / 1.25 / 1.5) on those rows (kappa .03)
  pre_k<k>        : the mid map with kappa k (.02 / .04 / .05) on those rows (tau 1)
  early_mid       : owner-less rows J <= t < 10 (J < 10) get the mid map on the restarted memory z
Rows changed against bank4 are swapped exactly into its pooled sets; the best four on 12 subsets. Pseudo targets are
evaluator-only.
Usage (server): python -B scripts/t1_bank/analyze_t1_bank5.py PARENT_ZIP > log
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
sys.path.insert(0, str(HERE.parent / "core"))
sys.path.insert(0, str(HERE.parents[1]))
import x_r5  # noqa: E402,F401  (pins BLAS threads)
import numpy as np  # noqa: E402

import analyze_hvc_stage0 as HV  # noqa: E402
import t1_bank as B  # noqa: E402
import t1_bank2 as B2  # noqa: E402
import t1_bank3 as B3  # noqa: E402
import t1_bank4 as B4  # noqa: E402
from analyze_t1_open2 import post4_open  # noqa: E402,F401
from analyze_t1_restart2 import eer_swap, mid_map  # noqa: E402

VARIANTS = ["pre_open", "pre_tau0.75", "pre_tau1.25", "pre_tau1.5", "pre_k0.02", "pre_k0.04", "pre_k0.05", "early_mid"]
D = {}


def masks4(base, x, g, zP):
    s = np.asarray(base, np.float32).copy()
    cmz = B.early_cmz(x, g)
    s[:cmz.shape[0]] = np.clip(B.TH0 + B.KAPPA * (cmz - B.TAU), -1.0, 1.0)
    own = B4.owners_r(x, g, zP)
    has = np.nonzero(own >= 0)[0]
    if has.size:
        keep = s[has, own[has]].copy()
        s[has] = -1.0
        s[has, own[has]] = keep
    a, run = B2.leader_runs(s)
    t = np.arange(s.shape[0])
    lead_ok = (zP[t, a] >= B2.ZLEAD_OWN) | (run >= B2.RUN_OWN)
    return own, lead_ok


def _job(i):
    x, g, c = D["s1"][i], D["g1"], int(D["c1"][i])
    base0 = np.asarray(D["base"][i], np.float32)
    J = B4.jump_index(x)
    zP = B4.zp_restart(x, g, J)
    v4 = B4.post4(base0, x, g)
    own, lead_ok = masks4(base0, x, g, zP)
    T = v4.shape[0]
    t = np.arange(T)
    pre = (t < J) if J > 0 else (t < B4.JUMP_UNTIL)
    free_pre = (own < 0) & (t >= B2.EXT_FROM) & (t < B3.MID_UNTIL) & ~lead_ok & pre
    early = (own < 0) & (t < B.EARLY) & (t >= J) & (J > 0)
    xb = np.asarray(x, np.float64)
    hi = min(T, B4.JUMP_UNTIL)
    cm = B._loo_z(B._unit(np.cumsum(B._unit(xb[:hi]), axis=0)) @ B._unit(g).T).astype(np.float32)
    keep = np.ones(g.shape[0], bool)
    keep[c] = False
    res = {"c": c, "gen": v4[:, c].astype(np.float64), "imp": v4[:, keep].ravel(), "chg": {},
           "n_pre": int(free_pre.sum()), "n_early": int(early.sum())}
    for name in VARIANTS:
        s = v4.copy()
        if name == "early_mid":
            idx = np.nonzero(early)[0]
            if idx.size:
                s[idx] = mid_map(zP[idx].astype(np.float32), 1.0, 0.03)
        else:
            idx = np.nonzero(free_pre)[0]
            if idx.size:
                if name == "pre_open":
                    s[idx] = np.clip(B.TH0 + B.KAPPA * (cm[idx] - B.TAU), -1.0, 1.0)
                elif name.startswith("pre_tau"):
                    s[idx] = mid_map(zP[idx].astype(np.float32), float(name[7:]), 0.03)
                else:
                    s[idx] = mid_map(zP[idx].astype(np.float32), 1.0, float(name[5:]))
        rows = np.nonzero((s.view(np.uint32) != v4.view(np.uint32)).any(1))[0]
        res["chg"][name] = (rows, s[rows][:, c].astype(np.float64), v4[rows][:, keep].ravel(), s[rows][:, keep].ravel())
    return i, res


def main(zpath, workers=24):
    from proxy_lib import load_track, pseudo_targets
    t0 = time.time()
    g1, s1, n1 = load_track("track1")
    c1, _ = pseudo_targets(g1, s1)
    Z = HV._read_zip(zpath)
    base = [Z["track1"][f"score_{n}.npz"] for n in n1]
    del Z
    D.update(s1=s1, g1=g1, c1=c1, base=base)
    R = [None] * len(s1)
    with ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=mp.get_context("fork")) as ex:
        for i, out in ex.map(_job, range(len(s1)), chunksize=1):
            R[i] = out
    n = len(R)

    def ev(key, ids, imp_sorted):
        gen, rem, add = [], [], []
        for j in ids:
            gj = R[j]["gen"]
            if key != "bank4":
                rows, gnew, iold, inew = R[j]["chg"][key]
                if rows.size:
                    gj = gj.copy()
                    gj[rows] = gnew
                    rem.append(iold)
                    add.append(inew)
            gen.append(gj)
        gen = np.concatenate(gen)
        if not rem:
            return eer_swap(gen, imp_sorted, imp_sorted.size, None, None)
        return eer_swap(gen, imp_sorted, imp_sorted.size, np.sort(np.concatenate(rem)), np.sort(np.concatenate(add)))

    allids = list(range(n))
    imp_all = np.concatenate([o["imp"] for o in R])
    imp_all.sort()
    e4, thr = ev("bank4", allids, imp_all)
    res = {"bank4": e4, "pre_rows": int(sum(o["n_pre"] for o in R)), "early_rows": int(sum(o["n_early"] for o in R)),
           "all": {}, "gain_geq": {}}
    for k in VARIANTS:
        e = ev(k, allids, imp_all)[0]
        res["all"][k] = e
        res["gain_geq"][k] = round((e4 - e) * 978572, 1)
    top = sorted(VARIANTS, key=lambda k: res["all"][k])[:4]
    th32 = np.float32(thr)
    cost = np.array([(o["gen"] < thr).sum() + (o["imp"] >= th32).sum() / 199.0 for o in R])
    order = np.argsort(-cost)
    subsets = {"even": list(range(0, n, 2)), "odd": list(range(1, n, 2)),
               "minus_top1": sorted(set(allids) - set(order[:1].tolist())),
               "minus_top3": sorted(set(allids) - set(order[:3].tolist()))}
    for seed in range(4):
        p = np.random.default_rng(seed).permutation(n)
        subsets[f"r{seed}a"] = sorted(p[: n // 2].tolist())
        subsets[f"r{seed}b"] = sorted(p[n // 2:].tolist())
    del imp_all
    res["subsets"] = {}
    for sname, ids in subsets.items():
        imp = np.concatenate([R[j]["imp"] for j in ids])
        imp.sort()
        b4 = ev("bank4", ids, imp)[0]
        res["subsets"][sname] = {k: round((b4 - ev(k, ids, imp)[0]) * 978572, 1) for k in top}
        del imp
    res["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
