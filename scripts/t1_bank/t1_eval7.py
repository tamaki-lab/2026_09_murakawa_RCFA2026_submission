"""Analysis helpers on top of bank7 (aggregates only; pseudo targets are evaluator-only, never inside a processor).

- `load(zpath)` reads Track 1 (gallery, sequences, pseudo targets) and the parent's base scores into `D`.
- `state7(base0, x, g)` returns bank7's output and the causal row states of `analyze_t1_bank7_ledger.py`
  (pre_open, pre_free_t10, post_open, owned, extended, mid, late_new, late_gated, late_after_held, flat).
- A job `job(i)` returns `entry(s7, c)` plus, per variant key, `swap(s7, rows, new, c)`: the rows whose scores change,
  the target's new scores and the impostor scores removed / added (as `analyze_t1_openpc.py` does).
- `run(job)` maps a job over the Track 1 sequences with a fork pool; `report(R, keys)` gives the exact pooled T1 proxy
  gain of every key against bank7 (gen-eq = EER x 978,572), the 12 robustness subsets for the best keys and the
  per-sequence deltas at bank7's threshold.
"""
import multiprocessing as mp
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

import analyze_hvc_stage0 as HV  # noqa: E402
import t1_bank as B  # noqa: E402
import t1_bank2 as B2  # noqa: E402
import t1_bank4 as B4  # noqa: E402
import t1_bank5 as B5  # noqa: E402
import t1_bank7 as B7  # noqa: E402
from analyze_t1_bank5 import masks4  # noqa: E402
from analyze_t1_restart2 import eer_swap  # noqa: E402

GEQ = 978572
D = {}


def load(zpath):
    from proxy_lib import load_track, pseudo_targets
    g1, s1, n1 = load_track("track1")
    c1, _ = pseudo_targets(g1, s1)
    Z = HV._read_zip(zpath)
    base = [Z["track1"][f"score_{n}.npz"] for n in n1]
    del Z
    D.update(s1=s1, g1=g1, c1=c1, base=base)
    return len(s1)


def state7(base0, x, g):
    base0 = np.asarray(base0, np.float32)
    J = B4.jump_index(x)
    j0 = J if J > 0 else 0
    zP = B4.zp_restart(x, g, J)
    s7 = B7.post7(base0, x, g)
    own, lead_ok = masks4(base0, x, g, zP)
    T = s7.shape[0]
    t = np.arange(T)
    ext = (own < 0) & (t >= B2.EXT_FROM) & lead_ok
    held = (own >= 0) | ext
    before = np.concatenate([[False], np.cumsum(held)[:-1] > 0])
    rn = B5.recent_norm(x)
    pre = t < j0
    free = (own < 0) & ~ext
    late_new = free & (t >= 70) & ~before & (rn >= B5.NORM_GATE)
    base_rows = free & (t >= 70) & ~late_new
    flat = base_rows & (2 * np.count_nonzero(base0 >= B.TH0, axis=1) > base0.shape[1])
    state = np.full(T, "other", dtype=object)
    state[(t < B.EARLY) & pre] = "pre_open"
    state[(t >= B.EARLY) & pre & free] = "pre_free_t10"
    state[(t < B.EARLY) & ~pre] = "post_open"
    state[own >= 0] = np.where((t < B.EARLY)[own >= 0], state[own >= 0], "owned")
    state[ext] = "extended"
    state[free & (t >= B.EARLY) & (t < 70) & ~pre] = "mid"
    state[late_new] = "late_new"
    state[base_rows & ~before & ~flat] = "late_gated"
    state[base_rows & before & ~flat] = "late_after_held"
    state[flat] = "flat"
    return {"J": J, "j0": j0, "zP": zP, "s7": s7, "own": own, "lead_ok": lead_ok, "state": state, "rn": rn, "t": t}


def entry(s7, c):
    keep = np.ones(s7.shape[1], bool)
    keep[c] = False
    return {"gen": s7[:, c].astype(np.float64), "imp": s7[:, keep].ravel(), "chg": {}}


def swap(s7, rows, new, c):
    """Changed rows of `new` (scores for `rows`) against bank7: (rows, target new, impostors old, impostors new)."""
    rows = np.asarray(rows)
    new = np.asarray(new, np.float32)
    if rows.size == 0:
        return (rows, np.zeros(0), np.zeros(0, np.float32), np.zeros(0, np.float32))
    keep = np.ones(s7.shape[1], bool)
    keep[c] = False
    ch = (new.view(np.uint32) != s7[rows].view(np.uint32)).any(1)
    r = rows[ch]
    return (r, new[ch][:, c].astype(np.float64), s7[r][:, keep].ravel(), new[ch][:, keep].ravel())


def run(job, workers=24):
    n = len(D["s1"])
    R = [None] * n
    with ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=mp.get_context("fork")) as ex:
        for i, out in ex.map(job, range(n), chunksize=1):
            R[i] = out
    return R


def _ev(R, key, ids, imp_sorted):
    gen, rem, add = [], [], []
    for j in ids:
        gj = R[j]["gen"]
        if key is not None and key in R[j]["chg"]:
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


def report(R, keys, name=str, top=5, subsets=True):
    n = len(R)
    allids = list(range(n))
    imp_all = np.concatenate([o["imp"] for o in R])
    imp_all.sort()
    e7, thr = _ev(R, None, allids, imp_all)
    res = {"bank7": e7, "thr": thr, "gain_geq": {}}
    allv = {}
    for k in keys:
        allv[k] = _ev(R, k, allids, imp_all)[0]
        res["gain_geq"][name(k)] = round((e7 - allv[k]) * GEQ, 1)
    best = sorted(keys, key=lambda k: allv[k])[:top]
    th32 = np.float32(thr)
    if subsets:
        cost = np.array([(o["gen"] < thr).sum() + (o["imp"] >= th32).sum() / 199.0 for o in R])
        order = np.argsort(-cost)
        sub = {"even": list(range(0, n, 2)), "odd": list(range(1, n, 2)),
               "minus_top1": sorted(set(allids) - set(order[:1].tolist())),
               "minus_top3": sorted(set(allids) - set(order[:3].tolist()))}
        for seed in range(4):
            p = np.random.default_rng(seed).permutation(n)
            sub[f"r{seed}a"] = sorted(p[: n // 2].tolist())
            sub[f"r{seed}b"] = sorted(p[n // 2:].tolist())
        del imp_all
        res["subsets"] = {}
        for sname, ids in sub.items():
            imp = np.concatenate([R[j]["imp"] for j in ids])
            imp.sort()
            b7 = _ev(R, None, ids, imp)[0]
            res["subsets"][sname] = {name(k): round((b7 - _ev(R, k, ids, imp)[0]) * GEQ, 1) for k in best}
            del imp
    res["fixed_thr"] = {}
    for k in best:
        d = []
        for o in R:
            if k not in o["chg"] or not o["chg"][k][0].size:
                d.append(0.0)
                continue
            rows, gnew, iold, inew = o["chg"][k]
            d.append(float((o["gen"][rows] < thr).sum() + (iold >= th32).sum() / 199.0
                           - (gnew < thr).sum() - (inew >= th32).sum() / 199.0))
        d = np.array(d)
        res["fixed_thr"][name(k)] = {"net": round(float(d.sum()), 1), "changed_seqs": int((d != 0).sum()),
                                     "better": int((d > 0.5).sum()), "worse": int((d < -0.5).sum()),
                                     "largest": [round(float(v), 1) for v in np.sort(d)[::-1][:3]],
                                     "smallest": [round(float(v), 1) for v in np.sort(d)[:3]]}
    return res
