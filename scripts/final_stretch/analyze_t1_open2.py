"""T1 check on bank4: the opening map on a gallery-direction-removed statistic (aggregates only, exact pooled T1-EER).

`analyze_t1_galfe.py` found the lowest opening floor so far with abtt1 — the cumulative mean of unit frames and the unit
gallery both centred by the unit-gallery mean, the top principal direction of the centred gallery removed, then the
row LOO z (pooled opening floor 1,067.0 against cmz's 1,090.4; sum of positions 1,058.2 against 1,077.7). The opening
segment shares a direction across sequences that the gallery's top direction may partly carry. Variants: bank4 with
the opening map clip(TH0 + kappa (stat - tau)) for t < 10 on stat = abtt1 (tau .75 / 1 / 1.25 x kappa .04 / .05 / .06)
and on cmz with the same grid (the calibration control); everything else bank4's (ownership of opening rows, the
extension's leader runs from the bank output, the mid map on the restarted memory). Self-check: cmz with tau 1,
kappa .05 reproduces t1_bank4.post4 bitwise. Rows changed against bank4 are swapped exactly into its pooled sets;
the best four on 12 subsets. Pseudo targets are evaluator-only.
Usage (server): python -B scripts/final_stretch/analyze_t1_open2.py PARENT_ZIP > log
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
sys.path.insert(0, str(HERE.parent / "ogata"))
sys.path.insert(0, str(HERE.parents[1]))
import x_r5  # noqa: E402,F401  (pins BLAS threads)
import numpy as np  # noqa: E402

import analyze_hvc_stage0 as HV  # noqa: E402
import t1_bank as B  # noqa: E402
import t1_bank2 as B2  # noqa: E402
import t1_bank3 as B3  # noqa: E402
import t1_bank4 as B4  # noqa: E402
from analyze_t1_restart2 import eer_swap  # noqa: E402

GRID = [(tau, k) for tau in (0.75, 1.0, 1.25) for k in (0.04, 0.05, 0.06)]
VARIANTS = [(st, tau, k) for st in ("abtt1", "cmz") for tau, k in GRID if not (st == "cmz" and (tau, k) == (1.0, 0.05))]
D = {}


def gallery_pc1(g):
    gu = B._unit(g)
    mu = gu.mean(0)
    G0 = gu - mu
    _, _, vt = np.linalg.svd(G0, full_matrices=False)
    P = np.eye(gu.shape[1]) - np.outer(vt[0], vt[0])
    Gt = G0 @ P
    return mu, P, Gt / np.linalg.norm(Gt, axis=1, keepdims=True)


def open_stat(x, name):
    xb = np.asarray(x, np.float64)
    e = min(B.EARLY, xb.shape[0])
    C = np.cumsum(B._unit(xb[:e]), axis=0)
    if name == "cmz":
        return B.early_cmz(x, D["g1"])
    mu, P, Gn = D["pc1"]
    m = B._unit(C)
    pm = (m - mu) @ P.T
    return B._loo_z(pm @ Gn.T).astype(np.float32)


def post4_open(base, x, g, zP, stat, tau, kappa):
    s = np.asarray(base, np.float32).copy()
    e = stat.shape[0]
    s[:e] = np.clip(B.TH0 + np.float32(kappa) * (stat - np.float32(tau)), -1.0, 1.0)
    own = B4.owners_r(x, g, zP)
    has = np.nonzero(own >= 0)[0]
    if has.size:
        keep = s[has, own[has]].copy()
        s[has] = -1.0
        s[has, own[has]] = keep
    a, run = B2.leader_runs(s)
    t = np.arange(s.shape[0])
    lead_ok = (zP[t, a] >= B2.ZLEAD_OWN) | (run >= B2.RUN_OWN)
    out = s.copy()
    idx = np.nonzero((own < 0) & (t >= B2.EXT_FROM) & lead_ok)[0]
    if idx.size:
        keep = out[idx, a[idx]].copy()
        out[idx] = -1.0
        out[idx, a[idx]] = keep
    idx = np.nonzero((own < 0) & (t >= B2.EXT_FROM) & (t < B3.MID_UNTIL) & ~lead_ok)[0]
    if idx.size:
        out[idx] = B3.mid_map(zP[idx].astype(np.float32))
    return out


def _job(i):
    x, g, c = D["s1"][i], D["g1"], int(D["c1"][i])
    base0 = np.asarray(D["base"][i], np.float32)
    v4 = B4.post4(base0, x, g)
    zP = B4.zp_restart(x, g, B4.jump_index(x))
    stats = {"cmz": open_stat(x, "cmz"), "abtt1": open_stat(x, "abtt1")}
    selfcheck = bool(np.array_equal(post4_open(base0, x, g, zP, stats["cmz"], 1.0, 0.05).view(np.uint32),
                                    v4.view(np.uint32)))
    keep = np.ones(g.shape[0], bool)
    keep[c] = False
    res = {"c": c, "gen": v4[:, c].astype(np.float64), "imp": v4[:, keep].ravel(), "chg": {}, "selfcheck": selfcheck}
    for st, tau, k in VARIANTS:
        s = post4_open(base0, x, g, zP, stats[st], tau, k)
        rows = np.nonzero((s.view(np.uint32) != v4.view(np.uint32)).any(1))[0]
        res["chg"][(st, tau, k)] = (rows, s[rows][:, c].astype(np.float64), v4[rows][:, keep].ravel(),
                                    s[rows][:, keep].ravel())
    return i, res


def main(zpath, workers=24):
    from proxy_lib import load_track, pseudo_targets
    t0 = time.time()
    g1, s1, n1 = load_track("track1")
    c1, _ = pseudo_targets(g1, s1)
    Z = HV._read_zip(zpath)
    base = [Z["track1"][f"score_{n}.npz"] for n in n1]
    del Z
    D.update(s1=s1, g1=g1, c1=c1, base=base, pc1=gallery_pc1(g1))
    R = [None] * len(s1)
    with ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=mp.get_context("fork")) as ex:
        for i, out in ex.map(_job, range(len(s1)), chunksize=1):
            R[i] = out
    n = len(R)
    name = lambda k: "bank4" if k == "bank4" else f"{k[0]}_tau{k[1]:g}_k{k[2]:g}"

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
    res = {"selfcheck": all(o["selfcheck"] for o in R), "bank4": e4, "all": {}, "gain_geq": {}}
    for k in VARIANTS:
        e = ev(k, allids, imp_all)[0]
        res["all"][name(k)] = e
        res["gain_geq"][name(k)] = round((e4 - e) * 978572, 1)
    top = sorted(VARIANTS, key=lambda k: res["all"][name(k)])[:4]
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
        res["subsets"][sname] = {name(k): round((b4 - ev(k, ids, imp)[0]) * 978572, 1) for k in top}
        del imp
    res["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
