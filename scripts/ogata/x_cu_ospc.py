"""OSP-C: open-set cohort posterior (one-sided Cauchy GLRT) on the VTN-M memory statistic,
with the OSP-G Gaussian contrast as report-only.

Preregistered in docs/experiments/20261006_ogata_cu_ospc.md.
Frozen design (selector key cu_ospc; sources evidence_scaling#1 = OSP-C, evidence_scaling#2 =
OSP-G; see docs/experiments/20261006_ogata_catchup_gpt.md for the gpt-6.1-sol judge round).

outputs_fn(x, g) is a strictly causal, per-sequence processor with fresh state per call,
float64 internals and identical constants on both tracks. It receives only the raw probe
sequence and the raw gallery of the same track. K = c.shape[0] is the gallery size, not a
Track switch. No labels, pseudo targets, track names, attack types or future frames. Every
row is a function of the causal states at t only. The front end (FE) is the VTN-M float64
arithmetic of candidates.vtn_mem, unchanged up to its LOO z; only the per-frame output map
is new. 'tn_ref' and 'vtnm_ref' are non-candidate references from the frozen functions.

  python scripts/ogata/x_cu_ospc.py               # single preregistered run
  python scripts/ogata/x_cu_ospc.py --synth       # random-data pre-run checks (i)-(iv)
  python scripts/ogata/x_cu_ospc.py --stats       # label-free numerical stats (print only)
  python scripts/ogata/x_cu_ospc.py --report LOG  # post-run report (evaluator-only targets)

--synth uses random data only. --stats reads probe sequences but uses no labels or pseudo
targets. --report uses evaluator-only pseudo targets. All print aggregates only.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_check as cc  # noqa: E402  (before numpy: pins BLAS threads)
import numpy as np  # noqa: E402

import candidates  # noqa: E402
from cubic_fast import _prep, cumulative_cubic_raw, ema_cubic_raw  # noqa: E402
from rcfa.methods.gallery_discriminative_readout import discriminative_mapping  # noqa: E402
import x_t1_tnorm_cohort as tn  # noqa: E402

SETTINGS = ["ospc_vtn", "ospg_vtn", "ospc_tn"]
REFS = ["tn_ref", "vtnm_ref"]
PRIMARY = "ospc_vtn"
VTN = candidates.vtn_mem()                            # frozen VTN-M processor (stateless closure)


# ---------------- frozen design ----------------
def _fe(x, g):
    """VTN-M front end, float64, same expressions as candidates.vtn_mem up to its LOO z."""
    u, a, c = _prep(x, g)
    K = c.shape[0]
    M, beta, _ = discriminative_mapping(c)
    nrm = np.linalg.norm(np.asarray(x, np.float64), axis=1)
    al = np.minimum(1.0, 0.05 * (nrm / 20.0) ** 4)
    sm = np.tanh(cumulative_cubic_raw(u, a, c, 1.0) @ M.T + beta)
    ss = np.tanh(ema_cubic_raw(u, a, c, al, 1.0) @ M.T + beta)
    P = (sm - sm.mean(1, keepdims=True)) / 2
    S = (ss - ss.mean(1, keepdims=True)) / 2
    V = np.minimum(P, S + 0.005)
    mu = (P.sum(1, keepdims=True) - P) / (K - 1)
    var = ((P * P).sum(1, keepdims=True) - P * P) / (K - 1) - mu * mu
    sd = np.sqrt(np.maximum(var, 0.0))
    zP = np.where(sd > 0, (P - mu) / np.where(sd > 0, sd, 1.0), 0.0)
    zV = np.where(sd > 0, (V - mu) / np.where(sd > 0, sd, 1.0), 0.0)   # == vtn_mem's z
    return {"K": K, "P": P, "V": V, "sd": sd, "zP": zP, "zV": zV}


def _cohort_m(zc, K):
    w = np.maximum(zc, 0.0) ** 2
    return (w.sum(1, keepdims=True) - w) / (K - 1)


def ospc(zown, zc, K):
    """(R-1)/(R+1), R = A/B: posterior odds of H_j vs background (1/2) + other K-1 ids (1/2),
    one-sided Cauchy GLRT likelihood LR = 1 + z+^2. Strictly inside (-1,1); no clip."""
    m = _cohort_m(zc, K)
    A = 2.0 * (1.0 + np.maximum(zown, 0.0) ** 2)
    B = 2.0 + m
    return (A - B) / (A + B)


def _lse_loo(zc):
    """lse_j = log sum_{k != j} exp(zc_k), leader-exact."""
    T = zc.shape[0]
    rows = np.arange(T)
    i1 = zc.argmax(1)
    m1 = zc[rows, i1][:, None]
    E = np.exp(zc - m1)
    Ssum = E.sum(1, keepdims=True)
    with np.errstate(divide="ignore"):
        lse = m1 + np.log(Ssum - E)                   # >= m1 for j != i1 (contains the leader's 1)
    zo = zc.copy()
    zo[rows, i1] = -np.inf
    m2 = zo.max(1)
    lse[rows, i1] = m2 + np.log(np.exp(zo - m2[:, None]).sum(1))
    assert np.isfinite(lse).all(), "non-finite leave-one-out logsumexp"
    return lse


def ospg(zown, zc, K):
    """Report-only Gaussian unit-shift cohort LLR, L/(1+|L|)."""
    L = zown + np.log(2.0) - np.logaddexp(0.5, _lse_loo(zc) - np.log(K - 1))
    return L / (1.0 + np.abs(L))


def _all64(x, g):
    f = _fe(x, g)
    K, zP, zV = f["K"], f["zP"], f["zV"]
    return f, {"ospc_vtn": ospc(zV, zP, K), "ospg_vtn": ospg(zV, zP, K), "ospc_tn": ospc(zP, zP, K)}


def outputs_fn(x, g):
    _, o = _all64(x, g)
    out = {k: v.astype(np.float32) for k, v in o.items()}
    # non-candidate references from the frozen functions, no gate
    out["tn_ref"] = tn._tnorm64(x, g)[0].astype(np.float32)
    out["vtnm_ref"] = np.asarray(VTN(x, g), np.float32)
    return out


# ---------------- random-data pre-run checks (no challenge data) ----------------
def _synth_data(seed=0, T=300, K=200):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(K, 512)).astype(np.float32)
    h = T // 2
    ident = np.r_[np.repeat(g[7][None], h, 0), np.repeat(g[30][None], T - h, 0)]  # identity switch
    x = rng.normal(size=(T, 512)) + 0.3 * ident
    x = x / np.linalg.norm(x, axis=1, keepdims=True) * rng.uniform(10.0, 32.0, size=(T, 1))
    return x.astype(np.float32), g


def _brute(zown, zc, K):
    """Per-column np.delete loops (reference for (ii))."""
    oc, og = np.empty_like(zc), np.empty_like(zc)
    for j in range(K):
        oth = np.delete(zc, j, axis=1)
        m = (np.maximum(oth, 0.0) ** 2).sum(1) / (K - 1)
        A = 2.0 * (1.0 + np.maximum(zown[:, j], 0.0) ** 2)
        B = 2.0 + m
        oc[:, j] = (A - B) / (A + B)
        mx = oth.max(1)
        lse = mx + np.log(np.exp(oth - mx[:, None]).sum(1))
        L = zown[:, j] + np.log(2.0) - np.logaddexp(0.5, lse - np.log(K - 1))
        og[:, j] = L / (1.0 + np.abs(L))
    return oc, og


def synth():
    import json
    out = {"synth": "cu_ospc"}
    w = {"i_vtn_mismatch_f32": 0, "ii_ospc_vtn": 0.0, "ii_ospg_vtn": 0.0, "ii_ospc_tn": 0.0,
         "ii_ospg_tn": 0.0, "ii_extreme": 0.0, "iii_prefix": 0.0, "iv_coupling": 0.0}
    rng_ok, strict_ok, rows_iv, n_sw = True, True, 0, 0
    for K in (200, 199):
        for seed, T in ((0, 300), (1, 257), (2, 64)):
            x, g = _synth_data(seed, T, K)
            f, o = _all64(x, g)
            zP, zV, sd = f["zP"], f["zV"], f["sd"]
            # (i) FE zV -> z/sqrt(K+z^2) equals frozen vtn_mem bit-identically in float32
            ref = np.asarray(VTN(x, g), np.float32)
            mine = (zV / np.sqrt(K + zV * zV)).astype(np.float32)
            w["i_vtn_mismatch_f32"] += int((mine != ref).sum())
            # (ii) against brute-force per-column loops
            bc, bg = _brute(zV, zP, K)
            w["ii_ospc_vtn"] = max(w["ii_ospc_vtn"], float(np.abs(bc - o["ospc_vtn"]).max()))
            w["ii_ospg_vtn"] = max(w["ii_ospg_vtn"], float(np.abs(bg - o["ospg_vtn"]).max()))
            bc2, bg2 = _brute(zP, zP, K)
            w["ii_ospc_tn"] = max(w["ii_ospc_tn"], float(np.abs(bc2 - o["ospc_tn"]).max()))
            w["ii_ospg_tn"] = max(w["ii_ospg_tn"], float(np.abs(bg2 - ospg(zP, zP, K)).max()))
            n_sw += int((zV < zP).sum())
            # (iii) finite, range, strictness (float64), prefix invariance after float32 cast
            full = outputs_fn(x, g)
            half = outputs_fn(x[: T // 2], g)
            for k in SETTINGS + REFS:
                v = full[k]
                rng_ok &= bool(v.shape == (T, K) and np.isfinite(v).all() and v.min() >= -1 and v.max() <= 1)
                d = float(np.abs(half[k].astype(np.float64) - v[: T // 2].astype(np.float64)).max())
                w["iii_prefix"] = max(w["iii_prefix"], d)
            for k in SETTINGS:
                strict_ok &= bool(np.isfinite(o[k]).all() and np.abs(o[k]).max() < 1)
            # (iv) LOO coupling identity per row with sd > 0
            ok = (sd > 0).all(1)
            rows_iv += int(ok.sum())
            s = (zP * zP / (K + zP * zP)).sum(1)
            w["iv_coupling"] = max(w["iv_coupling"], float(np.abs(s[ok] - K / (K - 1)).max()))
        # (ii) extra: a deterministic extreme row (dominant leader, ties, flat row) per K
        zc = np.zeros((4, K))
        zc[0, 3] = 800.0
        zc[0, 5] = -40.0
        zc[1, 3] = zc[1, 9] = 30.0
        zc[2] = np.linspace(-1, 1, K)
        zc[3, 0] = 1e3
        zc[3, 1] = 999.0
        zo = zc - 0.25
        bc, bg = _brute(zo, zc, K)
        w["ii_extreme"] = max(w["ii_extreme"], float(np.abs(bc - ospc(zo, zc, K)).max()),
                              float(np.abs(bg - ospg(zo, zc, K)).max()))
    out.update(w)
    out.update({"range_ok": rng_ok, "strict_inside_f64": strict_ok, "iv_rows": rows_iv,
                "veto_binding_entries": n_sw})
    out["pass"] = bool(w["i_vtn_mismatch_f32"] == 0 and max(w["ii_ospc_vtn"], w["ii_ospg_vtn"],
                       w["ii_ospc_tn"], w["ii_ospg_tn"], w["ii_extreme"]) <= 1e-9
                       and w["iii_prefix"] < 1e-6 and rng_ok and strict_ok and rows_iv > 0
                       and w["iv_coupling"] <= 1e-9)
    print(json.dumps(out), flush=True)


# ---------------- label-free stats pass (printed only; never changes anything) ----------------
STD_EDGES = [0, 10, 50, 200, 1000, np.inf]
STD_NAMES = ["t<10", "10-50", "50-200", "200-1000", ">=1000"]
MQ = [0.01, 0.1, 0.5, 0.9, 0.99]


def _stats_job(task):
    tr, i = task
    x, g = (cc.D["s1"][i], cc.D["g1"]) if tr == "t1" else (cc.D["s2"][i], cc.D["g2"])
    f, o = _all64(x, g)
    m = _cohort_m(f["zP"], f["K"]).astype(np.float32)
    p = np.digitize(np.arange(m.shape[0]), STD_EDGES) - 1
    r = {"n": int(m.shape[0]), "sd0_entries": int((f["sd"] <= 0).sum()),
         "sd0_frames": int((f["sd"] <= 0).any(1).sum()),
         "m_bins": [m[p == b].ravel() for b in range(len(STD_NAMES))]}
    for k, v in o.items():
        v32 = v.astype(np.float32)
        r[k] = {"max64": float(np.abs(v).max()), "max32": float(np.abs(v32.astype(np.float64)).max()),
                "ge1_32": int((np.abs(v32) >= 1).sum()), "le_m999_32": int((v32 <= -0.999).sum())}
    return tr, r


def stats(workers=6):
    import json
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from proxy_lib import load_track
    g1, s1, _ = load_track("track1")
    g2, s2, _ = load_track("track2")
    cc.D.update(g1=g1, s1=s1, g2=g2, s2=s2)
    tasks = [("t1", i) for i in range(len(s1))] + [("t2", i) for i in range(len(s2))]
    acc = {"t1": [], "t2": []}
    with ProcessPoolExecutor(max_workers=min(workers, 6), mp_context=mp.get_context("fork")) as ex:
        for tr, r in ex.map(_stats_job, tasks, chunksize=1):
            acc[tr].append(r)
    out = {"stats": "cu_ospc", "m_quantiles": MQ}
    for tr, rs in acc.items():
        d = {"frames": sum(r["n"] for r in rs), "sd_eq0_entries": sum(r["sd0_entries"] for r in rs),
             "sd_eq0_frames": sum(r["sd0_frames"] for r in rs), "m_by_t": {}}
        for b, nm in enumerate(STD_NAMES):
            v = np.concatenate([r["m_bins"][b] for r in rs])
            d["m_by_t"][nm] = {"n": int(v.size),
                               "q": (np.quantile(v, MQ).round(4).tolist() if v.size else None),
                               "max": (round(float(v.max()), 3) if v.size else None)}
            del v
        for k in SETTINGS:
            d[k] = {"max_abs_float64": max(r[k]["max64"] for r in rs),
                    "max_abs_float32": max(r[k]["max32"] for r in rs),
                    "n_abs_ge1_float32": sum(r[k]["ge1_32"] for r in rs),
                    "n_le_-0.999_float32": sum(r[k]["le_m999_32"] for r in rs)}
        out[tr] = d
    print(json.dumps(out), flush=True)


# ---------------- post-run report (evaluator side) ----------------
R = {}
FULL = ["ospc_vtn", "tn_ref", "vtnm_ref"]          # recomputed in full on T1
TOPK_ONLY = ["ospg_vtn", "ospc_tn", "955399"]       # T1 FAR=1e-5 threshold only (exact top-k merge)
SPLICE = SETTINGS + REFS + ["955399"]
TOPK = 1500
REPRO = {"tn_ref": {"eer_full": 0.004631, "frr_far1e5": 0.0104},
         "vtnm_ref": {"eer_full": 0.004635, "frr_far1e5": 0.0177, "splice_aar": 0.1234, "splice_run": 74.0,
                      "splice_s10": 0.98, "t2_all_reject": 0.1006},
         "955399": {"eer_full": 0.007879, "frr_far1e5": 0.0489, "splice_aar": 0.088, "splice_run": 52.8},
         "955690": {"eer_full": 0.007879, "frr_far1e5": 0.0252, "splice_aar": 0.0717, "splice_run": 43.0}}


def _unenrolled_fixtures(seqs, tg, gen_len=600, att_len=600, offset=100):
    """splice_fixtures' rule; also returns the attacker's target t_B for gallery deletion."""
    fx = []
    n = len(seqs)
    for i in range(n):
        j = (i + offset) % n
        while tg[j] == tg[i]:
            j = (j + 1) % n
        a, b = seqs[i], seqs[j]
        ga, fb = min(gen_len, a.shape[0]), min(att_len, b.shape[0])
        x = np.concatenate([a[:ga], b[:fb]], 0)
        lab = np.r_[np.zeros(ga, bool), np.ones(fb, bool)]
        fx.append((x, int(tg[i]), int(tg[j]), lab))
    return fx


def _rep_job(task):
    kind, i = task
    a399 = R["a399"]
    if kind == "t1":
        x, g, t = R["s1"][i], R["g1"], R["t1"][i]
        o = outputs_fn(x, g)
        o["955399"] = np.asarray(a399(x, g), np.float32)
        d = {k: o[k] for k in FULL}
        mask = np.ones(g.shape[0], bool)
        mask[t] = False
        for k in TOPK_ONLY:
            imp = o[k][:, mask].ravel()
            d["top_" + k] = np.sort(imp)[-TOPK:] if imp.size > TOPK else np.sort(imp)
        for k in FULL:   # same top-k path, to verify it against the full computation
            imp = o[k][:, mask].ravel()
            d["top_" + k] = np.sort(imp)[-TOPK:] if imp.size > TOPK else np.sort(imp)
        return kind, i, d
    x, ta, tb, lab = R["usp"][i]
    gp = np.delete(R["g1"], tb, 0)
    tc = ta - int(ta > tb)
    o = outputs_fn(x, gp)
    o["955399"] = np.asarray(a399(x, gp), np.float32)
    return kind, i, {k: o[k][:, tc] for k in SPLICE}


def _thr_from_top(tops, ni):
    kth = min(int(np.ceil(ni * (1 - 1e-5))), ni - 1)
    r = ni - 1 - kth                                  # number of impostors strictly above index kth
    assert r + 1 <= TOPK, "top-k too small"
    v = np.sort(np.concatenate(tops))[::-1]
    return float(v[r])


def report(log_path, workers=6):
    import json
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from proxy_lib import load_track, pseudo_targets, splice_fixtures, t1_metrics, run_metrics
    import analyze_t1_residual as ar
    rows = {}
    for line in open(log_path):
        if line.startswith("{\"setting\": "):
            row = json.loads(line)
            rows[row["setting"]] = row
    g1, s1, _ = load_track("track1")
    t1, _ = pseudo_targets(g1, s1)
    sp = splice_fixtures(s1, t1)
    usp = _unenrolled_fixtures(s1, t1)
    for (x, t, lab), (xu, ta, tb, labu) in zip(sp, usp):      # same fixtures as the harness
        assert t == ta and tb != ta and np.array_equal(x, xu) and np.array_equal(lab, labu)
    R.update(g1=g1, s1=s1, t1=t1, usp=usp, a399=cc._anchor_fns()["955399"])
    n1 = len(s1)
    tasks = [("t1", i) for i in range(n1)] + [("usp", i) for i in range(len(usp))]
    res = {"t1": {}, "usp": {}}
    with ProcessPoolExecutor(max_workers=min(workers, 6), mp_context=mp.get_context("fork")) as ex:
        for kind, i, d in ex.map(_rep_job, tasks, chunksize=1):
            res[kind][i] = d
    T1 = [res["t1"][i] for i in range(n1)]
    ni = int(sum(d["ospc_vtn"].shape[0] * (d["ospc_vtn"].shape[1] - 1) for d in T1))
    out = {"report": "cu_ospc", "log": os.path.basename(log_path), "n_imp": ni}
    thr = {}
    t1rep = {}
    for k in FULL:
        outs = [d[k] for d in T1]
        gen, imp = ar._gen_imp(outs, t1)
        met = ar._metrics(gen, imp)
        eer_cc = cc._eer_full(gen, imp)
        del imp
        tm = t1_metrics(outs, t1)
        thr_top = _thr_from_top([d["top_" + k] for d in T1], ni)
        thr[k] = met["far1e5_thr"]
        eb = ar._err_bins(outs, t1, met["eer_thr"])
        fb = ar._err_bins(outs, t1, met["far1e5_thr"])
        bins = {r_["bin"]: r_ for r_ in eb["by_pos"]}
        late_imp = bins["200-1000"]["imp_err"] + bins[">=1000"]["imp_err"]
        gen_ge10 = sum(bins[b]["gen_err"] for b in ("10-50", "50-200", "200-1000", ">=1000"))
        lr = rows.get(k, {})
        t1rep[k] = {
            "eer_full": round(eer_cc, 8), "eer_full_ar": round(met["eer_full"], 8),
            "eer_full_run_row": lr.get("eer_full"),
            "eer_full_match_1e-6": (lr.get("eer_full") is not None and abs(eer_cc - lr["eer_full"]) <= 1e-6),
            "eer_thr": round(met["eer_thr"], 6), "far1e5_thr": round(met["far1e5_thr"], 6),
            "thr_run_row": lr.get("thr"), "thr_topk_equal": bool(thr_top == met["far1e5_thr"]),
            "thr_t1_metrics_equal": bool(tm["thr"] == met["far1e5_thr"]),
            "frr_far1e5": round(met["frr_at_far1e5"], 6),
            "far_at_frr1e5": round(met["far_at_frr1e5"], 6), "frr1e5_thr": round(met["frr1e5_thr"], 6),
            "far_at_frr1e3": round(met["far_at_frr1e3"], 6), "frr1e3_thr": round(met["frr1e3_thr"], 6),
            "macro_seq_frr": round(tm["macro_seq_frr"], 6),
            "eer_thr_bins": {"gen": [r_["gen_err"] for r_ in eb["by_pos"]],
                             "imp": [r_["imp_err"] for r_ in eb["by_pos"]]},
            "far1e5_thr_bins": {"gen": [r_["gen_err"] for r_ in fb["by_pos"]],
                                "imp": [r_["imp_err"] for r_ in fb["by_pos"]]},
            "imp_err_t_ge200_at_eer_thr": int(late_imp), "gen_err_t_ge10_at_eer_thr": int(gen_ge10)}
    out["pos_bins"] = ar.POS_NAMES
    out["t1"] = t1rep
    for k in TOPK_ONLY:
        thr[k] = _thr_from_top([d["top_" + k] for d in T1], ni)
    out["thr_far1e5_all"] = {k: {"thr": round(v, 7), "run_row_thr": rows.get(k, {}).get("thr"),
                                 "match_5e-6": (rows.get(k, {}).get("thr") is not None
                                                and abs(v - rows[k]["thr"]) <= 5e-6)}
                             for k, v in thr.items()}
    del T1, res["t1"]
    # unenrolled-splice diagnostic (K = 199), each setting at its own full-gallery T1 FAR=1e-5 thr
    U = [res["usp"][i] for i in range(len(usp))]
    un = {}
    for k in SPLICE:
        v = []
        for d, (_, _, _, lab) in zip(U, usp):
            acc = d[k] >= np.float32(thr[k])
            v.append(((~acc[~lab]).mean(), acc[lab].mean(), run_metrics(acc[lab])))
        v = np.array(v, np.float64)
        un[k] = {"frr": round(v[:, 0].mean(), 4), "aar": round(v[:, 1].mean(), 4),
                 "run": round(v[:, 2].mean(), 1), "s10": round((v[:, 2] >= 10).mean(), 3)}
    out["unenrolled_splice"] = un
    out["U_label_pass"] = bool(un[PRIMARY]["aar"] <= un["vtnm_ref"]["aar"] and un[PRIMARY]["run"] <= un["vtnm_ref"]["run"])
    out["M1_pass"] = bool(t1rep[PRIMARY]["imp_err_t_ge200_at_eer_thr"] <= 29422)
    out["M2_pass"] = bool(t1rep[PRIMARY]["gen_err_t_ge10_at_eer_thr"] <= 1531)
    # gate and reference reproduction from the run rows (applied once; printed for the record)
    p, vr = rows.get(PRIMARY, {}), rows.get("vtnm_ref", {})
    chk_ok = {}
    for line in open(log_path):
        if line.startswith("{\"check\": "):
            c = json.loads(line)["check"]
            chk_ok[c["setting"]] = c["checks_pass"]
    repro = {k: {m: (rows.get(k, {}).get(m), v, rows.get(k, {}).get(m) == v) for m, v in d.items()}
             for k, d in REPRO.items()}
    out["repro"] = repro
    out["repro_all_equal"] = all(t_[2] for d in repro.values() for t_ in d.values())
    if p and vr:
        gate = {"eer_full<=0.90*vtnm": (p["eer_full"], round(0.90 * vr["eer_full"], 8), p["eer_full"] <= 0.90 * vr["eer_full"]),
                "frr_far1e5<=vtnm": (p["frr_far1e5"], vr["frr_far1e5"], p["frr_far1e5"] <= vr["frr_far1e5"]),
                "checks_pass": (chk_ok.get(PRIMARY), True, bool(chk_ok.get(PRIMARY))),
                "t1_neg1_share<=.001": (p["t1_neg1_share"], 0.001, p["t1_neg1_share"] <= 0.001),
                "splice_aar<=vtnm": (p["splice_aar"], vr["splice_aar"], p["splice_aar"] <= vr["splice_aar"]),
                "splice_run<=vtnm": (p["splice_run"], vr["splice_run"], p["splice_run"] <= vr["splice_run"]),
                "t2_all_reject>=vtnm-.01": (p["t2_all_reject"], round(vr["t2_all_reject"] - 0.01, 6),
                                            p["t2_all_reject"] >= vr["t2_all_reject"] - 0.01)}
        out["gate"] = gate
        t1_ok = all(gate[k][2] for k in ("eer_full<=0.90*vtnm", "frr_far1e5<=vtnm", "checks_pass", "t1_neg1_share<=.001"))
        t2_ok = all(gate[k][2] for k in ("splice_aar<=vtnm", "splice_run<=vtnm", "t2_all_reject>=vtnm-.01"))
        out["gate_T1"], out["gate_T2"] = t1_ok, t2_ok
        keys = ("eer_full", "frr_far1e5", "splice_aar", "t2_all_reject")

        def vec(r_):
            return (r_["eer_full"], r_["frr_far1e5"], r_["splice_aar"], -r_["t2_all_reject"])
        par = {}
        for k in SETTINGS:
            if k not in rows:
                continue
            a = vec(rows[k])
            par[k] = {o_: (all(b_ <= a_ for a_, b_ in zip(a, vec(rows[o_]))) and vec(rows[o_]) != a)
                      for o_ in ("955399", "955690", "tn_ref", "vtnm_ref") if o_ in rows}
        out["pareto_dominated_by"] = {"axes": keys, **par}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--synth":
        synth()
    elif arg == "--stats":
        stats()
    elif arg == "--report":
        report(sys.argv[2])
    else:
        cc.run_check(outputs_fn, SETTINGS + REFS, "cu_ospc", workers=6)
