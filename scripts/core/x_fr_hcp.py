"""HCP: joint history/current switching posterior that replaces OSP-C's veto and one-look map.

Preregistered in docs/experiments/20261007_ogata_fr_hcp.md (selector key fr_hcp; source
posterior_model#2 of the final round, docs/experiments/20261007_ogata_final_gpt.md).

outputs_fn(x, g) is a strictly causal, per-sequence processor with fresh state per call,
float64 internals and identical constants on both tracks. It receives only the raw probe
sequence and the raw gallery of the same track. K = c.shape[0] is the gallery size, not a
Track switch. No labels, pseudo targets, track names, attack types or future frames. Every
row is a function of the causal cumulative and EMA states at t only. The front end (FE) is the
float64 arithmetic of candidates.ospc_vtn, verbatim, plus zS (the short state against its own
leave-one-out cohort). 'ospc_ref' and 'vtnm_ref' come from the frozen candidates functions.

Settings: 'hcp' = hcp(zP, zS) PRIMARY (only gated setting); 'hcp_v' = hcp(zV, zS) report-only;
'cur_only' = ospc(zS, zS) report-only (current-look posterior alone).

  python scripts/core/x_fr_hcp.py                          # single preregistered run_check
  python scripts/core/x_fr_hcp.py --synth                  # random-data pre-run checks (i)-(iv)
  python scripts/core/x_fr_hcp.py --stats                  # label-free stats (print only)
  python scripts/core/x_fr_hcp.py --report RUNLOG CALLOG   # post-run report + frozen gate

--synth uses random data only. --stats reads probe sequences but uses no labels or pseudo
targets. --report uses evaluator-only pseudo targets and calibrate_t2_proxy's exact fixture and
proxy definitions. All print aggregates only.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_check as cc  # noqa: E402  (before numpy: pins BLAS threads; adds repo root to sys.path)
import numpy as np  # noqa: E402

import candidates  # noqa: E402
from cubic_fast import _prep, cumulative_cubic_raw, ema_cubic_raw  # noqa: E402
from rcfa.methods.gallery_discriminative_readout import discriminative_mapping  # noqa: E402

TAG = "fr_hcp"
SETTINGS = ["hcp", "hcp_v", "cur_only"]
REFS = ["ospc_ref", "vtnm_ref"]
PRIMARY = "hcp"
OSPC = candidates.ospc_vtn()                         # frozen OSP-C processor (stateless closure)
VTN = candidates.vtn_mem()                           # frozen VTN-M processor


# ---------------- frozen design ----------------
def _fe(x, g):
    """OSP-C front end (candidates.ospc_vtn verbatim, float64) plus zS."""
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
    zV = np.where(sd > 0, (V - mu) / np.where(sd > 0, sd, 1.0), 0.0)
    muS = (S.sum(1, keepdims=True) - S) / (K - 1)
    varS = ((S * S).sum(1, keepdims=True) - S * S) / (K - 1) - muS * muS
    sdS = np.sqrt(np.maximum(varS, 0.0))
    zS = np.where(sdS > 0, (S - muS) / np.where(sdS > 0, sdS, 1.0), 0.0)   # own LOO cohort
    return {"K": K, "P": P, "S": S, "sd": sd, "sdS": sdS, "zP": zP, "zV": zV, "zS": zS}


def hcp_parts(zh, zs, K):
    """num, den of the HCP odds (leader-exact Q)."""
    LP = 1.0 + np.maximum(zh, 0.0) ** 2
    LS = 1.0 + np.maximum(zs, 0.0) ** 2
    wS = LS - 1.0
    mS = (wS.sum(1, keepdims=True) - wS) / (K - 1)
    Pr = LP * LS
    Q = (Pr.sum(1, keepdims=True) - Pr) / (K - 1)
    r = np.arange(Pr.shape[0])
    i1 = Pr.argmax(1)
    Po = Pr.copy()
    Po[r, i1] = 0.0
    Q[r, i1] = Po.sum(1) / (K - 1)                   # leader-exact
    num = 4.0 * LP * LS
    den = LP * (2.0 + mS) + 1.0 + Q
    return num, den, {"mS": mS, "Q": Q}


def hcp(zh, zs, K):
    num, den, _ = hcp_parts(zh, zs, K)
    rn = np.sqrt(num)
    rd = np.sqrt(den)
    return (rn - rd) / (rn + rd)


def ospc(zown, zc, K):
    """x_cu_ospc.ospc verbatim: (A-B)/(A+B), A = 2(1+zown+^2), B = 2 + LOO mean of zc+^2."""
    w = np.maximum(zc, 0.0) ** 2
    m = (w.sum(1, keepdims=True) - w) / (K - 1)
    A = 2.0 * (1.0 + np.maximum(zown, 0.0) ** 2)
    B = 2.0 + m
    return (A - B) / (A + B)


def _all64(x, g):
    f = _fe(x, g)
    K, zP, zV, zS = f["K"], f["zP"], f["zV"], f["zS"]
    return f, {"hcp": hcp(zP, zS, K), "hcp_v": hcp(zV, zS, K), "cur_only": ospc(zS, zS, K)}


def outputs_fn(x, g):
    _, o = _all64(x, g)
    out = {k: v.astype(np.float32) for k, v in o.items()}
    out["ospc_ref"] = np.asarray(OSPC(x, g), np.float32)
    out["vtnm_ref"] = np.asarray(VTN(x, g), np.float32)
    return out


# ---------------- random-data pre-run checks (no challenge data) ----------------
def _synth_data(seed, T, K, kind):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(K, 512)).astype(np.float32)
    if kind == "switch":                              # identity switch at T/2
        h = T // 2
        ident = np.r_[0.3 * np.repeat(g[7][None], h, 0), 0.3 * np.repeat(g[30][None], T - h, 0)]
    else:                                             # MNG: strong memory leader L, then weaker T
        h = (3 * T) // 5
        ident = np.r_[0.6 * np.repeat(g[11][None], h, 0), 0.25 * np.repeat(g[40][None], T - h, 0)]
    x = rng.normal(size=(T, 512)) + ident
    x = x / np.linalg.norm(x, axis=1, keepdims=True) * rng.uniform(10.0, 32.0, size=(T, 1))
    return x.astype(np.float32), g


def _brute_hcp(zh, zs, K):
    """Per-column np.delete loops: num/den reference for (ii)."""
    LP = 1.0 + np.maximum(zh, 0.0) ** 2
    LS = 1.0 + np.maximum(zs, 0.0) ** 2
    num = 4.0 * LP * LS
    den = np.empty_like(num)
    for j in range(K):
        mS = np.delete(LS - 1.0, j, axis=1).sum(1) / (K - 1)
        Q = np.delete(LP * LS, j, axis=1).sum(1) / (K - 1)
        den[:, j] = LP[:, j] * (2.0 + mS) + 1.0 + Q
    rn, rd = np.sqrt(num), np.sqrt(den)
    return num / den, (rn - rd) / (rn + rd)


def _brute_ospc(zown, zc, K):
    o = np.empty_like(zc)
    for j in range(K):
        m = (np.maximum(np.delete(zc, j, axis=1), 0.0) ** 2).sum(1) / (K - 1)
        A = 2.0 * (1.0 + np.maximum(zown[:, j], 0.0) ** 2)
        B = 2.0 + m
        o[:, j] = (A - B) / (A + B)
    return o


def _extreme_rows(K):
    zh, zs = np.zeros((6, K)), np.zeros((6, K))
    zh[0, 3] = zs[0, 3] = 2000.0                      # dominant leader in both looks
    zh[0, 5] = zs[0, 5] = -40.0
    zh[1, 3] = 2000.0                                 # P leader != S leader
    zs[1, 9] = 2000.0
    zh[2, 3] = zh[2, 9] = zs[2, 3] = zs[2, 9] = 30.0  # ties
    zh[3] = zs[3] = 0.0                               # flat row
    zh[4] = np.linspace(-1, 1, K)
    zs[4] = np.linspace(1, -1, K)
    zh[5, 0], zs[5, 0], zh[5, 1], zs[5, 1] = 1e3, 999.0, 999.0, 1e3   # near-tie leaders
    return zh, zs


def synth():
    w = {"i_ospc_mismatch_f32": 0, "ii_hcp_odds_rel": 0.0, "ii_hcp_out_abs": 0.0, "ii_hcp_v_odds_rel": 0.0,
         "ii_hcp_v_out_abs": 0.0, "ii_cur_only_abs": 0.0, "ii_extreme_odds_rel": 0.0, "ii_extreme_out_abs": 0.0,
         "iii_odds_rel": 0.0, "iv_prefix": 0.0}
    rng_ok, strict_ok, n_mng, n_rows = True, True, 0, 0

    def rel(a, b):
        return float((np.abs(a - b) / np.abs(b)).max())

    for K in (200, 199):
        for kind in ("switch", "mng"):
            for seed, T in ((0, 300), (1, 257), (2, 64)):
                x, g = _synth_data(seed, T, K, kind)
                f, o = _all64(x, g)
                zP, zV, zS = f["zP"], f["zV"], f["zS"]
                n_rows += T
                n_mng += int((f["P"].argmax(1) != f["S"].argmax(1)).sum())
                # (i) FE o_h = ospc(zV, zP) is bit-identical in float32 to candidates.ospc_vtn()
                ref = np.asarray(OSPC(x, g), np.float32)
                w["i_ospc_mismatch_f32"] += int((ospc(zV, zP, K).astype(np.float32) != ref).sum())
                # (ii) brute force
                bo, bout = _brute_hcp(zP, zS, K)
                num, den, _ = hcp_parts(zP, zS, K)
                w["ii_hcp_odds_rel"] = max(w["ii_hcp_odds_rel"], rel(num / den, bo))
                w["ii_hcp_out_abs"] = max(w["ii_hcp_out_abs"], float(np.abs(o["hcp"] - bout).max()))
                bo2, bout2 = _brute_hcp(zV, zS, K)
                num2, den2, _ = hcp_parts(zV, zS, K)
                w["ii_hcp_v_odds_rel"] = max(w["ii_hcp_v_odds_rel"], rel(num2 / den2, bo2))
                w["ii_hcp_v_out_abs"] = max(w["ii_hcp_v_out_abs"], float(np.abs(o["hcp_v"] - bout2).max()))
                w["ii_cur_only_abs"] = max(w["ii_cur_only_abs"],
                                           float(np.abs(o["cur_only"] - _brute_ospc(zS, zS, K)).max()))
                # (iii) ((1+out)/(1-out))^2 == num/den
                for k, (nn, dd) in (("hcp", (num, den)), ("hcp_v", (num2, den2))):
                    v = o[k]
                    w["iii_odds_rel"] = max(w["iii_odds_rel"], rel(((1 + v) / (1 - v)) ** 2, nn / dd))
                # (iv) finite, |out| < 1 strictly in float32, prefix invariance
                full = outputs_fn(x, g)
                half = outputs_fn(x[: T // 2], g)
                for k in SETTINGS + REFS:
                    v = full[k]
                    rng_ok &= bool(v.shape == (T, K) and np.isfinite(v).all() and v.min() >= -1 and v.max() <= 1)
                    d = float(np.abs(half[k].astype(np.float64) - v[: T // 2].astype(np.float64)).max())
                    w["iv_prefix"] = max(w["iv_prefix"], d)
                for k in SETTINGS:
                    strict_ok &= bool(np.isfinite(full[k]).all() and np.abs(full[k].astype(np.float64)).max() < 1)
        # (ii)/(iii) deterministic extreme rows
        zh, zs = _extreme_rows(K)
        bo, bout = _brute_hcp(zh, zs, K)
        num, den, _ = hcp_parts(zh, zs, K)
        v = hcp(zh, zs, K)
        w["ii_extreme_odds_rel"] = max(w["ii_extreme_odds_rel"], rel(num / den, bo))
        w["ii_extreme_out_abs"] = max(w["ii_extreme_out_abs"], float(np.abs(v - bout).max()),
                                      float(np.abs(ospc(zs, zs, K) - _brute_ospc(zs, zs, K)).max()))
        w["iii_odds_rel"] = max(w["iii_odds_rel"], rel(((1 + v) / (1 - v)) ** 2, num / den))
        strict_ok &= bool(np.abs(v.astype(np.float32).astype(np.float64)).max() < 1)
    out = {"synth": TAG, **w, "range_ok": rng_ok, "strict_inside_f32": strict_ok,
           "rows": n_rows, "rows_P_leader_ne_S_leader": n_mng}
    out["pass"] = bool(w["i_ospc_mismatch_f32"] == 0
                       and max(w[k] for k in w if k.startswith("ii_") or k.startswith("iii_")) <= 1e-9
                       and w["iv_prefix"] < 1e-6 and rng_ok and strict_ok and n_mng > 0)
    print(json.dumps(out), flush=True)
    return out["pass"]


# ---------------- label-free stats pass (printed only; never changes anything) ----------------
STD_EDGES = [0, 10, 50, 200, 1000, np.inf]
STD_NAMES = ["t<10", "10-50", "50-200", "200-1000", ">=1000"]
QS = [0.01, 0.1, 0.5, 0.9, 0.99]


def _stats_job(task):
    tr, i = task
    x, g = (cc.D["s1"][i], cc.D["g1"]) if tr == "t1" else (cc.D["s2"][i], cc.D["g2"])
    f, o = _all64(x, g)
    K = f["K"]
    _, _, aux = hcp_parts(f["zP"], f["zS"], K)
    T = x.shape[0]
    p = np.digitize(np.arange(T), STD_EDGES) - 1
    r = np.arange(T)
    lead = o["hcp"].argmax(1)
    ok = (f["sd"] > 0) & (f["sdS"] > 0)
    ratio = np.where(ok, f["sdS"] / np.where(ok, f["sd"], 1.0), np.nan)[:, 0]   # sd is per column (LOO)
    sub = np.arange(0, T, 10)                         # every 10th frame for all-entry quantiles
    res = {"n": T, "bins": []}
    for b in range(len(STD_NAMES)):
        m = p == b
        ms = m[sub]
        res["bins"].append({
            "ratio_leadcol": (f["sdS"][r, lead] / np.where(f["sd"][r, lead] > 0, f["sd"][r, lead], np.nan))[m],
            "ratio_col0": ratio[m],
            "zS_lead": f["zS"][r, lead][m], "zP_lead": f["zP"][r, lead][m],
            "mS_lead": aux["mS"][r, lead][m], "Q_lead": aux["Q"][r, lead][m],
            "mS_all": aux["mS"][sub][ms].ravel().astype(np.float32),
            "Q_all": aux["Q"][sub][ms].ravel().astype(np.float32)})
    for k, v in o.items():
        v32 = v.astype(np.float32)
        res[k] = {"max64": float(np.abs(v).max()), "max32": float(np.abs(v32.astype(np.float64)).max()),
                  "ge1_32": int((np.abs(v32) >= 1).sum()), "le_m999_32": int((v32 <= -0.999).sum())}
    res["sdS0"] = int((f["sdS"] <= 0).sum())
    return tr, res


def _q(v):
    v = v[np.isfinite(v)]
    return None if v.size == 0 else np.quantile(v, QS).round(5).tolist()


def stats(workers=6):
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
    out = {"stats": TAG, "quantiles": QS,
           "note": "leader = argmax of the hcp row; ratio = sdS/sd (LOO sds) at the leader column and at column 0"}
    for tr, rs in acc.items():
        d = {"frames": sum(r["n"] for r in rs), "sdS_eq0_entries": sum(r["sdS0"] for r in rs), "by_t": {}}
        for b, nm in enumerate(STD_NAMES):
            e = {}
            for key in ("ratio_leadcol", "ratio_col0", "zS_lead", "zP_lead", "mS_lead", "Q_lead", "mS_all", "Q_all"):
                v = np.concatenate([r["bins"][b][key] for r in rs]).astype(np.float64)
                e[key] = _q(v)
            e["n"] = int(sum(r["bins"][b]["zS_lead"].size for r in rs))
            d["by_t"][nm] = e
        for k in SETTINGS:
            d[k] = {"max_abs_float64": max(r[k]["max64"] for r in rs),
                    "max_abs_float32": max(r[k]["max32"] for r in rs),
                    "n_abs_ge1_float32": sum(r[k]["ge1_32"] for r in rs),
                    "n_le_-0.999_float32": sum(r[k]["le_m999_32"] for r in rs)}
        out[tr] = d
    print(json.dumps(out), flush=True)


# ---------------- post-run report (evaluator side) ----------------
TOPK = 1500
FULL = SETTINGS + REFS                               # full T1 metrics
PROX = SETTINGS + REFS + ["955399"]                  # calibrated T2 proxies
A399 = {}
# Frozen calibration outcome (applied mechanically to calibrate_t2_proxy anchor rows only, before
# this script ran on challenge data; see the preregistration doc).
D_STAR, B_STAR = "t2_all_reject_st", "t2_cp_reject_pt"    # both: larger = better (expected sign -1)
BE_A, BE_B = 0.5097574683128512, -0.7748887487409217      # OLS be = a + b * B* over 13 rows
BE_LOO_MAE = 0.0163
RUN_REPRO = {  # setting -> metric -> expected (run row unless noted 'usp_*' = report proxy row)
    "ospc_ref": {"eer_full": 0.00368857, "frr_far1e5": 0.015721, "thr": 0.9721794, "splice_aar": 0.1324,
                 "splice_run": 79.2, "t2_all_reject": 0.0926, "usp_aar": 0.238, "usp_run": 133.6,
                 "usp_frr": 0.0455},
    "vtnm_ref": {"eer_full": 0.004635, "frr_far1e5": 0.0177, "splice_aar": 0.1234, "splice_run": 74.0,
                 "t2_all_reject": 0.1006, "usp_aar": 0.1973, "usp_run": 110.8},
    "955399": {"eer_full": 0.007879, "frr_far1e5": 0.0489, "splice_aar": 0.088, "splice_run": 52.8,
               "usp_aar": 0.0901},
    "955690": {"eer_full": 0.007879, "frr_far1e5": 0.0252, "splice_aar": 0.0717, "splice_run": 43.0}}
CAL_MAP = {"ospc_ref": "OSPC_964718", "vtnm_ref": "VTNM", "955399": "V_955399"}


def _tol_ok(k, got, exp):
    if got is None:
        return False
    if k == "thr":
        return bool(abs(got - exp) <= 5e-6)
    return bool(abs(got - exp) <= max(5e-5, 0.0051 * abs(exp)))


def _t(a, b, ok):
    """JSON-safe (value, bar, pass) triple."""
    f = (lambda v: None if v is None else (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)))
    return (f(a), f(b), bool(ok))


def _rep_job(task):
    import calibrate_t2_proxy as cal
    D = cal.D
    kind, i = task

    def outs(x, g):
        o = outputs_fn(x, g)
        o["955399"] = np.asarray(A399["f"](x, g), np.float32)
        bad = {k: int((~np.isfinite(v)).sum() + (np.abs(v) > 1).sum()) for k, v in o.items()}
        return o, bad

    if kind == "t1":
        x, g, t = D["s1"][i], D["g1"], D["t1"][i]
        o, bad = outs(x, g)
        mask = np.ones(g.shape[0], bool)
        mask[t] = False
        res = {}
        for k in PROX:
            imp = o[k][:, mask].ravel()
            top = np.sort(np.partition(imp, imp.size - TOPK)[-TOPK:]) if imp.size > TOPK else np.sort(imp)
            res[k] = {"gen": o[k][:, t].copy(), "top": top, "ni": int(imp.size)}
        return kind, i, res, {k: o[k] for k in FULL}, bad
    if kind == "t2":
        o, bad = outs(D["s2"][i], D["g2"])
        return kind, i, {k: {"pt": o[k][:, D["t2"][i]].copy(), "st": o[k][:, D["st2"][i]].copy()}
                         for k in PROX}, None, bad
    if kind == "usp":
        x, ta, _ = D["sp"][i]
        tb = D["utb"][i]
        g = np.delete(D["g1"], tb, 0)
        tc = ta - int(ta > tb)
    else:
        x, tc, _ = D[kind][i]
        g = D["g1"]
    o, bad = outs(x, g)
    return kind, i, {k: o[k][:, tc].copy() for k in PROX}, None, bad


def report(run_log, cal_log, workers=6):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from proxy_lib import t1_metrics
    import analyze_t1_residual as ar
    import calibrate_t2_proxy as cal
    rows, chk, syn = {}, {}, None
    for line in open(run_log):
        if line.startswith("{\"setting\": "):
            r_ = json.loads(line)
            rows[r_["setting"]] = r_
        elif line.startswith("{\"check\": "):
            c_ = json.loads(line)["check"]
            chk[c_["setting"]] = c_["checks_pass"]
        elif line.startswith("{\"synth\": "):
            syn = json.loads(line)
    calrows = {}
    for line in open(cal_log):
        if line.startswith("{\"method\": "):
            r_ = json.loads(line)
            calrows[r_["method"]] = r_
    cal.DF_LEVELS = ()                                # df fixtures are structurally blind; not used
    cal._load()
    A399["f"] = cc._anchor_fns()["955399"]
    D = cal.D
    n1, n2, nf = len(D["s1"]), len(D["s2"]), len(D["sp"])
    kinds = ["sp", "usp"] + [f"bl{lv:.2f}" for lv in cal.BL_LEVELS]
    tasks = [("t1", i) for i in range(n1)] + [("t2", i) for i in range(n2)] + \
        [(k, i) for k in kinds for i in range(nf)]
    R = {k: {} for k in ["t1", "t2"] + kinds}
    F = {}
    bad = {k: 0 for k in PROX}
    with ProcessPoolExecutor(max_workers=min(workers, 6), mp_context=mp.get_context("fork")) as ex:
        for kind, i, res, full, b in ex.map(_rep_job, tasks, chunksize=1):
            R[kind][i] = res
            if full is not None:
                F[i] = full
            for k, v in b.items():
                bad[k] += v
    t1 = D["t1"]
    out = {"report": TAG, "run_log": os.path.basename(run_log), "cal_log": os.path.basename(cal_log),
           "nonfinite_or_out_of_range_entries": bad, "pos_bins": ar.POS_NAMES}
    # ---- full-precision T1 ----
    t1rep = {}
    for k in FULL:
        outs_ = [F[i][k] for i in range(n1)]
        gen, imp = ar._gen_imp(outs_, t1)
        met = ar._metrics(gen, imp)
        eer_cc = cc._eer_full(gen, imp)
        del imp
        tm = t1_metrics(outs_, t1)
        eb = ar._err_bins(outs_, t1, met["eer_thr"])
        fb = ar._err_bins(outs_, t1, met["far1e5_thr"])
        bins = {r_["bin"]: r_ for r_ in eb["by_pos"]}
        early = ("t=0", "1-3", "4-9")
        lr = rows.get(k, {})
        t1rep[k] = {
            "eer_full": eer_cc, "eer_full_ar": met["eer_full"], "eer_full_run_row": lr.get("eer_full"),
            "eer_full_match_1e-6": (lr.get("eer_full") is not None and abs(eer_cc - lr["eer_full"]) <= 1e-6),
            "eer_thr": met["eer_thr"], "far1e5_thr": met["far1e5_thr"], "thr_run_row": lr.get("thr"),
            "frr_far1e5": met["frr_at_far1e5"], "far_at_frr1e5": met["far_at_frr1e5"],
            "far_at_frr1e3": met["far_at_frr1e3"], "macro_seq_frr": tm["macro_seq_frr"],
            "t1_neg1_share": float((gen <= -0.999).mean()),
            "eer_thr_bins": {"gen": [r_["gen_err"] for r_ in eb["by_pos"]],
                             "imp": [r_["imp_err"] for r_ in eb["by_pos"]]},
            "far1e5_thr_bins": {"gen": [r_["gen_err"] for r_ in fb["by_pos"]],
                                "imp": [r_["imp_err"] for r_ in fb["by_pos"]]},
            "gen_err_t_ge10_at_eer_thr": int(sum(bins[b]["gen_err"] for b in ar.POS_NAMES if b not in early)),
            "early_err_at_eer_thr": round(float(sum(bins[b]["gen_err"] + bins[b]["imp_err"] / 199.0
                                                    for b in early)), 1)}
    out["t1"] = t1rep
    del F
    print(json.dumps(out), flush=True)
    # ---- calibrated T2 proxies (calibrate_t2_proxy._row, exact definitions; own T1 FAR=1e-5 thr) ----
    prox = {}
    for k in PROX:
        cal.OFFICIAL.setdefault(k, {})
        r_ = cal._row(k, R)
        r_.pop("official", None)
        r_.pop("group", None)
        prox[k] = r_
        print(json.dumps({"proxy_row": r_}), flush=True)
    for k in FULL:
        t1rep[k]["thr_topk_equal"] = bool(prox[k]["thr"] == round(t1rep[k]["far1e5_thr"], 7))
    # ---- STOP checks ----
    stop = {}
    rr = {}
    for s, d in RUN_REPRO.items():
        rr[s] = {}
        for m, v in d.items():
            got = (prox.get(s, {}).get(m) if m.startswith("usp_") else rows.get(s, {}).get(m))
            rr[s][m] = (got, v, _tol_ok(m, got, v))
    stop["run_repro_ok"] = all(t_[2] for d in rr.values() for t_ in d.values())
    cr = {}
    for s, cn in CAL_MAP.items():
        cr[s] = {}
        for m, v in calrows.get(cn, {}).items():
            if m in ("method", "group", "official") or v is None or not isinstance(v, (int, float)):
                continue
            if m.startswith("df0"):
                continue
            got = prox[s].get(m)
            cr[s][m] = (got, v, _tol_ok(m, got, v))
    stop["cal_repro_ok"] = bool(cr) and all(cr[s] for s in CAL_MAP) and \
        all(t_[2] for d in cr.values() for t_ in d.values())
    stop["synth_ok"] = bool(syn and syn.get("pass"))
    stop["primary_checks_ok"] = bool(chk.get(PRIMARY)) and bad[PRIMARY] == 0
    stop["eer_full_report_vs_row_ok"] = bool(t1rep[PRIMARY]["eer_full_match_1e-6"])
    out2 = {"report_gate": TAG, "run_repro": rr, "cal_repro": cr, "stop_checks": stop}
    any_stop = not all(stop.values())
    p, o_ = t1rep[PRIMARY], t1rep["ospc_ref"]
    pp, po = prox[PRIMARY], prox["ospc_ref"]
    gate_t1 = {
        "eer_full<=ospc": _t(p["eer_full"], o_["eer_full"], p["eer_full"] <= o_["eer_full"]),
        "frr_far1e5<=ospc+.001": _t(p["frr_far1e5"], o_["frr_far1e5"] + 0.001, p["frr_far1e5"] <= o_["frr_far1e5"] + 0.001),
        "checks_pass": _t(chk.get(PRIMARY), True, bool(chk.get(PRIMARY))),
        "t1_neg1_share<=.001": _t(rows.get(PRIMARY, {}).get("t1_neg1_share"), 0.001,
                                rows.get(PRIMARY, {}).get("t1_neg1_share") is not None
                                and rows[PRIMARY]["t1_neg1_share"] <= 0.001)}
    gate_t2 = {
        "a_usp_aar<=ospc": _t(pp["usp_aar"], po["usp_aar"], pp["usp_aar"] <= po["usp_aar"]),
        "a_usp_run<=ospc": _t(pp["usp_run"], po["usp_run"], pp["usp_run"] <= po["usp_run"]),
        f"b_D*({D_STAR})>=ospc": _t(pp[D_STAR], po[D_STAR], pp[D_STAR] >= po[D_STAR]),
        f"c_B*({B_STAR})>=ospc": _t(pp[B_STAR], po[B_STAR], pp[B_STAR] >= po[B_STAR]),
        "d_usp_frr<=ospc+.005": _t(pp["usp_frr"], po["usp_frr"] + 0.005, pp["usp_frr"] <= po["usp_frr"] + 0.005)}
    fbe = {k: BE_A + BE_B * prox[k][B_STAR] for k in PROX}
    margins = {"M-T1": _t(p["eer_full"], o_["eer_full"] - 0.0002, p["eer_full"] <= o_["eer_full"] - 0.0002),
               "M-T2": _t(fbe[PRIMARY], fbe["ospc_ref"] - 0.010, fbe[PRIMARY] <= fbe["ospc_ref"] - 0.010)}
    t1_ok = all(v[2] for v in gate_t1.values())
    t2_fail = [k for k, v in gate_t2.items() if not v[2]]
    m_ok = any(v[2] for v in margins.values())
    if any_stop:
        verdict = "STOP (review; no rerun): " + ", ".join(k for k, v in stop.items() if not v)
    elif not t1_ok:
        verdict = "T1 FAIL: " + ", ".join(k for k, v in gate_t1.items() if not v[2])
    elif t2_fail:
        verdict = "T1 PASS / T2 FAIL: " + ", ".join(t2_fail)
    elif not m_ok:
        verdict = "T1 PASS / T2 PASS / margin FAIL (neither M-T1 nor M-T2)"
    else:
        verdict = "PROXY PASS"
    labels = {"M-MNG": _t(p["gen_err_t_ge10_at_eer_thr"], 744, p["gen_err_t_ge10_at_eer_thr"] <= 744),
              "M-early": _t(p["early_err_at_eer_thr"], 1248, p["early_err_at_eer_thr"] <= 1248)}
    if verdict == "PROXY PASS" and not labels["M-MNG"][2]:
        verdict += " (mechanism unconfirmed)"
    # report-only forecasts from calibrate rows (OLS over rows with the official value)

    def ols(px, tgt):
        ms = [m for m in calrows if calrows[m]["official"].get(tgt) is not None and calrows[m].get(px) is not None]
        b, a = np.polyfit([calrows[m][px] for m in ms], [calrows[m]["official"][tgt] for m in ms], 1)
        return float(a), float(b), len(ms)
    fc = {}
    for tgt, px in (("diff_aar", D_STAR), ("it_aar", "usp_aar")):
        a, b, n = ols(px, tgt)
        fc[tgt] = {"proxy": px, "a": a, "b": b, "n": n, **{k: round(a + b * prox[k][px], 4) for k in PROX}}
    a, b, n = ols(B_STAR, "be")
    fc["be"] = {"proxy": B_STAR, "a_frozen": BE_A, "b_frozen": BE_B, "a_refit": a, "b_refit": b, "n": n,
                "loo_mae": BE_LOO_MAE, **{k: round(fbe[k], 4) for k in PROX}}
    out2.update({"gate_T1": gate_t1, "gate_T2": gate_t2, "margins": margins, "labels": labels,
                 "gate_T1_pass": t1_ok, "gate_T2_pass": not t2_fail, "verdict": verdict, "forecasts": fc,
                 "report_only_t2_pre_reject": {k: (prox[k]["t2_pre_reject_pt"], prox[k]["t2_pre_reject_st"]) for k in PROX}})
    print(json.dumps(out2), flush=True)
    print(json.dumps({"verdict": verdict}), flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--synth":
        sys.exit(0 if synth() else 1)
    elif arg == "--stats":
        stats()
    elif arg == "--report":
        report(sys.argv[2], sys.argv[3])
    else:
        cc.run_check(outputs_fn, SETTINGS + REFS, TAG, workers=6)
