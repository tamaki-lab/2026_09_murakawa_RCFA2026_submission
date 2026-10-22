"""T1 check around the jump restart (aggregates only, exact pooled T1-EER).

V2 of `analyze_t1_restart.py` = bank3 with the HCP memory z restarted at the regime jump J in the zP ownership rule,
the extension and the mid map (+74.5 gen-eq, every subset positive). Variants relative to V2 (one change each):
  cut<c>   : jump cut c .4 / .45 / .55 / .6 instead of .5
  ref<r>   : the detector's reference = mean of the first min(t, r) unit frames, r 8 / 12 instead of 10
  cnt      : the clean-frame count of the ownership rule also restarts at J
  opJ      : the opening map's cumulative mean restarts at J (rows J <= t < 10, J < 10 only)
  W<w>     : mid window 10 <= t < w, w 60 / 85 / 100 / 150 instead of 70
  tau<t> / k<k> : mid map clip(TH0 + k (z - t)), t .75 / 1.25 with k .03; k .02 / .04 with t 1
  mx<b>    : the mid map on z - b max(z) of the row, b .05 / .1
  gap<b>   : the mid map on non-leaders z - b (max(z) - z), leader unchanged, b .1
Rows changed against V2 are swapped exactly into V2's pooled sets. Exact pooled T1-EER for every variant, then the
best four on 12 subsets (even / odd, four random halves pairs, without the costliest 1 / 3 sequences of V2).
Pseudo targets are evaluator-only.
Usage (server): python -B scripts/t1_bank/analyze_t1_restart2.py PARENT_ZIP > log
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
import x_fr_hcp  # noqa: E402

D = {}


def jump_index(u, cut=0.5, ref=10, until=60):
    for t in range(1, min(u.shape[0], until)):
        m = u[: min(t, ref)].mean(0)
        nm = np.linalg.norm(m)
        if nm > 0 and float(u[t] @ m / nm) < cut:
            return t
    return -1


def owners_z(x, g, zP, J=-1):
    xb = np.asarray(x, np.float64)
    nrm = np.linalg.norm(xb, axis=1)
    v, _, _ = HV.verify(x, g)
    inc = v & (nrm >= B.CLEAN_NORM)[:, None]
    cnt = np.cumsum(inc, axis=0)
    if J > 0:
        cnt[J:] = np.cumsum(inc[J:], axis=0)
    arg, top, second = B._top2(cnt)
    own = np.where((top >= 1) & (top > second), arg, -1)
    a2, t2, s2 = B._top2(zP)
    alt = np.where((t2 >= B.ZP_OWN) & (t2 > s2) & (s2 <= B.ZP_RUNNER), a2, -1)
    return np.where(own >= 0, own, alt).astype(np.int64)


def early_cmz_J(x, g, J):
    xb = np.asarray(x, np.float64)
    e = min(B.EARLY, xb.shape[0])
    u = B._unit(xb[:e])
    C = np.cumsum(u, axis=0)
    if 0 < J < e:
        C[J:] = np.cumsum(u[J:], axis=0)
    return B._loo_z(B._unit(C) @ B._unit(g).T).astype(np.float32)


def mid_map(zp, tau=1.0, kappa=0.03):
    zp = np.asarray(zp, np.float32)
    return np.clip(B.TH0 + np.float32(kappa) * (zp - np.float32(tau)), -1.0, 1.0).astype(np.float32)


def pipeline(base, x, g, zP, J=-1, cnt_restart=False, open_restart=False, w_end=None, tau=1.0, kappa=0.03, ztf=None):
    s = np.asarray(base, np.float32).copy()
    cmz = early_cmz_J(x, g, J) if open_restart else B.early_cmz(x, g)
    e = cmz.shape[0]
    s[:e] = np.clip(B.TH0 + B.KAPPA * (cmz - B.TAU), -1.0, 1.0)
    own = owners_z(x, g, zP, J if cnt_restart else -1)
    has = np.nonzero(own >= 0)[0]
    if has.size:
        keep = s[has, own[has]].copy()
        s[has] = -1.0
        s[has, own[has]] = keep
    b = s
    s = b.copy()
    T = s.shape[0]
    a, run = B2.leader_runs(b)
    t = np.arange(T)
    extm = (zP[t, a] >= B2.ZLEAD_OWN) | (run >= B2.RUN_OWN)
    ext = (own < 0) & (t >= B2.EXT_FROM) & extm
    idx = np.nonzero(ext)[0]
    if idx.size:
        keep = s[idx, a[idx]].copy()
        s[idx] = -1.0
        s[idx, a[idx]] = keep
    free = (own < 0) & (t >= B2.EXT_FROM) & (t < (B3.MID_UNTIL if w_end is None else w_end)) & ~extm
    idx = np.nonzero(free)[0]
    if idx.size:
        z = zP[idx] if ztf is None else ztf(zP[idx])
        s[idx] = mid_map(z.astype(np.float32), tau, kappa)
    return s


def tf_max(b):
    return lambda z: z - b * z.max(1, keepdims=True)


def tf_gap(b):
    def f(z):
        mx = z.max(1, keepdims=True)
        return np.where(z >= mx, z, z - b * (mx - z))
    return f


VARIANTS = (["cut0.4", "cut0.45", "cut0.55", "cut0.6", "ref8", "ref12", "cnt", "opJ", "W60", "W85", "W100", "W150",
             "tau0.75", "tau1.25", "k0.02", "k0.04", "mx0.05", "mx0.1", "gap0.1"])


def _job(i):
    x, g, c = D["s1"][i], D["g1"], int(D["c1"][i])
    base0 = np.asarray(D["base"][i], np.float32)
    u = B._unit(np.asarray(x, np.float64))
    zP0 = x_fr_hcp._fe(x, g)["zP"]
    cache = {}

    def zr_for(J):
        if J <= 0:
            return zP0
        if J not in cache:
            zr = zP0.copy()
            zr[J:] = x_fr_hcp._fe(x[J:], g)["zP"]
            cache[J] = zr
        return cache[J]

    J = jump_index(u)
    v2 = pipeline(base0, x, g, zr_for(J), J)
    out = {}
    for name in VARIANTS:
        if name.startswith("cut"):
            Jv = jump_index(u, cut=float(name[3:]))
            s = pipeline(base0, x, g, zr_for(Jv), Jv)
        elif name.startswith("ref"):
            Jv = jump_index(u, ref=int(name[3:]))
            s = pipeline(base0, x, g, zr_for(Jv), Jv)
        elif name == "cnt":
            s = pipeline(base0, x, g, zr_for(J), J, cnt_restart=True)
        elif name == "opJ":
            s = pipeline(base0, x, g, zr_for(J), J, open_restart=True)
        elif name.startswith("W"):
            s = pipeline(base0, x, g, zr_for(J), J, w_end=int(name[1:]))
        elif name.startswith("tau"):
            s = pipeline(base0, x, g, zr_for(J), J, tau=float(name[3:]))
        elif name.startswith("k"):
            s = pipeline(base0, x, g, zr_for(J), J, kappa=float(name[1:]))
        elif name.startswith("mx"):
            s = pipeline(base0, x, g, zr_for(J), J, ztf=tf_max(float(name[2:])))
        elif name.startswith("gap"):
            s = pipeline(base0, x, g, zr_for(J), J, ztf=tf_gap(float(name[3:])))
        rows = np.nonzero((s.view(np.uint32) != v2.view(np.uint32)).any(1))[0]
        out[name] = (rows, s[rows])
    keep = np.ones(g.shape[0], bool)
    keep[c] = False
    res = {"c": c, "gen": v2[:, c].astype(np.float64), "imp": v2[:, keep].ravel(), "chg": {}}
    for name, (rows, S) in out.items():
        res["chg"][name] = (rows, S[:, c].astype(np.float64), v2[rows][:, keep].ravel(), S[:, keep].ravel())
    return i, res


def eer_swap(gen_full, imp_sorted, n_imp, rem_sorted, add_sorted):
    gs = np.sort(gen_full)
    cand = np.unique(gs)
    c32 = cand.astype(np.float32)
    acc = (imp_sorted.size - np.searchsorted(imp_sorted, c32, side="left"))
    if rem_sorted is not None:
        acc = acc - (rem_sorted.size - np.searchsorted(rem_sorted, c32, side="left")) \
            + (add_sorted.size - np.searchsorted(add_sorted, c32, side="left"))
    far = acc / n_imp
    frr = np.searchsorted(gs, cand, side="left") / gs.size
    k = int(np.argmin(np.abs(far - frr)))
    return float((far[k] + frr[k]) / 2), float(cand[k])


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

    def ev(name, ids, imp_sorted):
        gen = []
        rem, add = [], []
        for j in ids:
            gj = R[j]["gen"]
            if name != "V2":
                rows, gnew, iold, inew = R[j]["chg"][name]
                if rows.size:
                    gj = gj.copy()
                    gj[rows] = gnew
                    rem.append(iold)
                    add.append(inew)
            gen.append(gj)
        gen = np.concatenate(gen)
        if name == "V2" or not rem:
            return eer_swap(gen, imp_sorted, imp_sorted.size, None, None)
        rem = np.sort(np.concatenate(rem))
        add = np.sort(np.concatenate(add))
        return eer_swap(gen, imp_sorted, imp_sorted.size, rem, add)

    allids = list(range(n))
    imp_all = np.concatenate([o["imp"] for o in R])
    imp_all.sort()
    e2, thr = ev("V2", allids, imp_all)
    res = {"V2": e2, "V2_vs_bank3_geq": round((0.0026272874655607955 - e2) * 978572, 1), "all": {}, "gain_geq": {}}
    for name in VARIANTS:
        e = ev(name, allids, imp_all)[0]
        res["all"][name] = e
        res["gain_geq"][name] = round((e2 - e) * 978572, 1)
    res["changed_rows"] = {name: int(sum(o["chg"][name][0].size for o in R)) for name in VARIANTS}
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
        b2 = ev("V2", ids, imp)[0]
        res["subsets"][sname] = {k: round((b2 - ev(k, ids, imp)[0]) * 978572, 1) for k in top}
        del imp
    res["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
