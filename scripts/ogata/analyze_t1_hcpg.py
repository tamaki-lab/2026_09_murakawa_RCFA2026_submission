"""T1 premise search stage 1: likelihood form (HCP-G) and S horizon on the HCP core (analysis only).

Plan, provenance and frozen rules: docs/experiments/20261009_ogata_t1_premise_search.md.

Per T1 sequence, float64 internals, fresh state per call, the same constants on both tracks:
  front end = x_fr_hcp._fe verbatim (cubic P; quality-EMA cubic S, alpha_t = min(1, a0 (|x_t|/20)^4), a0 = .05).
  S variants: the same S path with a0 in A0; zP unchanged.
  maps on (zP, zS), the HCP odds algebra in log domain with leader-exact leave-one-out sums:
    num = 4 LP LS, den = LP (1 + mean_{k!=j} LS_k) + 1 + mean_{k!=j} LP_k LS_k, out = tanh((ln num - ln den) / 4)
    hcp   LR = 1 + z+^2            control (x_fr_hcp.hcp itself)
    hcpg  LR = exp(z - 1/2)        PRIMARY (OSP-G's unit shift, null mean 1)
    exp1  LR = exp(z+)             report-only
    cau2  LR = (1 + z^2)^sign(z)   report-only
  b = 943955 (gallery_discriminative_row_l2_scores); bl(a | b) = a if a >= .5 else (a + b) / 2.
Pseudo targets on the evaluator side only. Prints JSON aggregates only.
Usage (server): python scripts/ogata/analyze_t1_hcpg.py [workers<=6] > results/ogata/t1_hcpg.log
                python scripts/ogata/analyze_t1_hcpg.py --synth
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
import numpy as np  # noqa: E402

import x_fr_hcp as H  # noqa: E402
from cubic_fast import _prep, ema_cubic_raw  # noqa: E402
from x_cu_ospc import _lse_loo  # noqa: E402
from rcfa.methods.gallery_discriminative_readout import discriminative_mapping  # noqa: E402
from rcfa.methods.gallery_discriminative_row_l2 import gallery_discriminative_row_l2_scores  # noqa: E402

C0 = 0.5
A0 = (0.1, 0.025, 0.0125)
HCP_PROXY = 0.003092348265774678
G_BAR, BL_BAR, S_BAR = 5e-5, 3e-5, 2e-5
LN4 = np.log(4.0)
EDGES = [0, 10, 50, 200, 1000, np.inf]
NAMES = ["t<10", "10-50", "50-200", "200-1000", ">=1000"]
MAPS = ["hcp", "hcpg", "exp1", "cau2"]
SETTINGS = MAPS + ["b", "bl_hcp", "bl_hcpg"] + [f"{m}_a{a0}" for a0 in A0 for m in ("hcp", "hcpg")]
D = {}


def _lr(z, kind):
    """Log-likelihood ratio of a column from its leave-one-out z."""
    if kind == "cau1":
        return np.log1p(np.maximum(z, 0.0) ** 2)
    if kind == "gau":
        return z - 0.5
    if kind == "exp1":
        return np.maximum(z, 0.0)
    if kind == "cau2":
        return np.sign(z) * np.log1p(z * z)
    raise ValueError(kind)


def hcp_log(lP, lS, K):
    """HCP odds algebra for any real log-likelihood ratios, in log domain (leader-exact LOO)."""
    lnk1 = np.log(K - 1)
    ln_num = LN4 + lP + lS
    ln_mls = _lse_loo(lS) - lnk1                     # ln mean_{k!=j} LS_k
    ln_q = _lse_loo(lP + lS) - lnk1                  # ln mean_{k!=j} LP_k LS_k
    a = lP + np.logaddexp(0.0, ln_mls)               # ln LP (1 + mean LS)
    m = np.maximum(np.maximum(a, 0.0), ln_q)
    ln_den = m + np.log(np.exp(a - m) + np.exp(-m) + np.exp(ln_q - m))
    return np.tanh((ln_num - ln_den) / 4.0)


def maps(zP, zS, K):
    return {"hcp": H.hcp(zP, zS, K),
            "hcpg": hcp_log(_lr(zP, "gau"), _lr(zS, "gau"), K),
            "exp1": hcp_log(_lr(zP, "exp1"), _lr(zS, "exp1"), K),
            "cau2": hcp_log(_lr(zP, "cau2"), _lr(zS, "cau2"), K)}


def zs_variants(x, g, a0s):
    """zS of x_fr_hcp._fe with the S rate a0 replaced (same expressions and order)."""
    u, a, c = _prep(x, g)
    K = c.shape[0]
    M, beta, _ = discriminative_mapping(c)
    nrm = np.linalg.norm(np.asarray(x, np.float64), axis=1)
    out = {}
    for a0 in a0s:
        al = np.minimum(1.0, a0 * (nrm / 20.0) ** 4)
        ss = np.tanh(ema_cubic_raw(u, a, c, al, 1.0) @ M.T + beta)
        S = (ss - ss.mean(1, keepdims=True)) / 2
        muS = (S.sum(1, keepdims=True) - S) / (K - 1)
        varS = ((S * S).sum(1, keepdims=True) - S * S) / (K - 1) - muS * muS
        sdS = np.sqrt(np.maximum(varS, 0.0))
        out[a0] = np.where(sdS > 0, (S - muS) / np.where(sdS > 0, sdS, 1.0), 0.0)
    return out


def _bl(a32, b32):
    a64 = np.asarray(a32, np.float64)
    return np.where(a64 >= C0, a64, (a64 + np.asarray(b32, np.float64)) / 2).astype(np.float32)


def outputs(x, g):
    """Every setting (float32) for one sequence, plus the identity check of the log algebra."""
    f = H._fe(x, g)
    K, zP, zS = f["K"], f["zP"], f["zS"]
    o = maps(zP, zS, K)
    ident = float(np.abs(hcp_log(_lr(zP, "cau1"), _lr(zS, "cau1"), K) - o["hcp"]).max()) if zP.size else 0.0
    out = {k: v.astype(np.float32) for k, v in o.items()}
    out["b"] = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    out["bl_hcp"] = _bl(out["hcp"], out["b"])
    out["bl_hcpg"] = _bl(out["hcpg"], out["b"])
    zv = zs_variants(x, g, A0)
    for a0 in A0:
        out[f"hcp_a{a0}"] = H.hcp(zP, zv[a0], K).astype(np.float32)
        out[f"hcpg_a{a0}"] = hcp_log(_lr(zP, "gau"), _lr(zv[a0], "gau"), K).astype(np.float32)
    return out, ident


def _job(i):
    out, ident = outputs(D["s"][i], D["g"])
    return i, out, ident


def _counts(R, tg, k, thr):
    """Genuine misses and accepted impostors at the cut thr, by position bin and by sequence."""
    gm, ia, per = np.zeros(len(NAMES)), np.zeros(len(NAMES)), []
    for i, t in enumerate(tg):
        o = R[i][k]
        pos = np.digitize(np.arange(o.shape[0]), EDGES) - 1
        g_ = o[:, t] < thr
        mask = np.ones(o.shape[1], bool)
        mask[t] = False
        i_ = (o[:, mask] >= thr).sum(1)
        np.add.at(gm, pos, g_)
        np.add.at(ia, pos, i_)
        per.append((int(g_.sum()), int(i_.sum())))
    return gm, ia, per


def main(workers=6):
    import analyze_t1_residual as ar
    import common_check as cc
    from proxy_lib import load_track, pseudo_targets
    t0 = time.time()
    g, s, _ = load_track("track1")
    tg, _ = pseudo_targets(g, s)
    D.update(s=s, g=g)
    n = len(s)
    R, ident = {}, 0.0
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 6)), mp_context=mp.get_context("fork")) as ex:
        for i, out, idn in ex.map(_job, range(n), chunksize=1):
            R[i] = out
            ident = max(ident, idn)
    rng_ok = all(bool(np.isfinite(v).all() and (np.abs(v) <= 1).all()) for i in range(n) for v in R[i].values())
    print(json.dumps({"stage": "outputs", "sequences": n, "identity_max_abs": ident, "finite_in_range": rng_ok,
                      "seconds": round(time.time() - t0, 1)}), flush=True)

    def eer(idx, k):
        gen, imp = ar._gen_imp([R[i][k] for i in idx], tg[idx])
        return cc._eer_full(gen, imp)

    t1 = {}
    for k in SETTINGS:
        gen, imp = ar._gen_imp([R[i][k] for i in range(n)], tg)
        m = ar._metrics(gen, imp)
        del gen, imp
        t1[k] = {"eer_full": m["eer_full"], "eer_thr": m["eer_thr"], "far1e5_thr": m["far1e5_thr"],
                 "frr_far1e5": m["frr_at_far1e5"]}
    _, _, per = _counts(R, tg, "hcp", np.float32(t1["hcp"]["eer_thr"]))
    hard = sorted(set(sorted(range(n), key=lambda i: -per[i][1])[:5]) | set(sorted(range(n), key=lambda i: -per[i][0])[:3]))
    even, odd = np.arange(0, n, 2), np.arange(1, n, 2)
    rest = np.array([i for i in range(n) if i not in hard])
    for k in SETTINGS:
        gm, ia, _ = _counts(R, tg, k, np.float32(t1[k]["eer_thr"]))
        t1[k].update({"even": eer(even, k), "odd": eer(odd, k), "no_hard": eer(rest, k),
                      "gen_eq_by_bin": {nm: round(float(gm[j] + ia[j] / 199.0), 1) for j, nm in enumerate(NAMES)},
                      "gen_misses": int(gm.sum()), "imp_accepts": int(ia.sum())})
        print(json.dumps({"setting": k, **t1[k]}), flush=True)

    def better(k, base, bar):
        return {"d_eer": t1[k]["eer_full"] - t1[base]["eer_full"],
                "d_even": t1[k]["even"] - t1[base]["even"], "d_odd": t1[k]["odd"] - t1[base]["odd"],
                "d_no_hard": t1[k]["no_hard"] - t1[base]["no_hard"],
                "pass": bool(t1[k]["eer_full"] - t1[base]["eer_full"] <= -bar and t1[k]["even"] < t1[base]["even"]
                             and t1[k]["odd"] < t1[base]["odd"])}

    stop = {"hcp_repro": abs(t1["hcp"]["eer_full"] - HCP_PROXY) <= 1e-9, "identity_le_1e-12": ident <= 1e-12,
            "finite_in_range": rng_ok}
    g_core = better("hcpg", "hcp", G_BAR)
    g_bl = {"d_eer": t1["bl_hcpg"]["eer_full"] - t1["bl_hcp"]["eer_full"],
            "pass": bool(t1["bl_hcpg"]["eer_full"] <= t1["bl_hcp"]["eer_full"] - BL_BAR)}
    hcpg_pass = g_core["pass"] and g_bl["pass"]
    s_rule = {}
    for a0 in A0:
        for m in ("hcp", "hcpg"):
            r = better(f"{m}_a{a0}", m, S_BAR)
            r["pass"] = bool(r["pass"] and r["d_no_hard"] < 0)
            s_rule[f"{m}_a{a0}"] = r
    verdict = "STOP" if not all(stop.values()) else \
        ("HCPG_PASS" if hcpg_pass else "HCPG_KILL") + ";S_PASS:" + (",".join(k for k, r in s_rule.items() if r["pass"]) or "none")
    print(json.dumps({"stop": stop, "hard_sequences": hard, "hcpg_core": g_core, "hcpg_bl": g_bl, "s_rule": s_rule,
                      "verdict": verdict, "seconds": round(time.time() - t0, 1)}), flush=True)


def _brute_log(lP, lS, K):
    LP, LS = np.exp(lP), np.exp(lS)
    num = 4.0 * LP * LS
    den = np.empty_like(num)
    for j in range(K):
        den[:, j] = LP[:, j] * (1.0 + np.delete(LS, j, axis=1).sum(1) / (K - 1)) + 1.0 \
            + np.delete(LP * LS, j, axis=1).sum(1) / (K - 1)
    rn, rd = np.sqrt(num), np.sqrt(den)
    return (rn - rd) / (rn + rd)


def synth():
    ok = {}
    K = 200
    rng = np.random.default_rng(0)
    zh, zs = H._extreme_rows(K)
    zr, zsr = 3.0 * rng.normal(size=(40, K)), 3.0 * rng.normal(size=(40, K))
    for nm, (a, b_) in {"extreme": (zh, zs), "random": (zr, zsr)}.items():
        ok[f"identity_{nm}"] = bool(np.abs(hcp_log(_lr(a, "cau1"), _lr(b_, "cau1"), K) - H.hcp(a, b_, K)).max() <= 1e-12)
        o = maps(a, b_, K)
        ok[f"finite_{nm}"] = all(bool(np.isfinite(v).all() and (np.abs(v) <= 1).all()) for v in o.values())
    for kind in ("gau", "exp1", "cau2"):
        lP, lS = _lr(zr, kind), _lr(zsr, kind)
        ok[f"brute_{kind}"] = bool(np.abs(hcp_log(lP, lS, K) - _brute_log(lP, lS, K)).max() <= 1e-12)
    up = zr.copy()
    up[:, 17] += 1.0
    ok["hcpg_monotone_own"] = bool((maps(up, zsr, K)["hcpg"][:, 17] > maps(zr, zsr, K)["hcpg"][:, 17]).all())
    for seed, kind in ((3, "switch"), (5, "mng")):
        x, g = H._synth_data(seed, 240, 60, kind)
        out, ident = outputs(x, g)
        ok[f"identity_seq_{seed}"] = bool(ident <= 1e-12)
        ok[f"zs_a005_equals_fe_{seed}"] = bool(np.array_equal(zs_variants(x, g, [0.05])[0.05], H._fe(x, g)["zS"]))
        ok[f"hcp_equals_parent_{seed}"] = bool(np.array_equal(out["hcp"], H._all64(x, g)[1]["hcp"].astype(np.float32)))
        ok[f"settings_{seed}"] = sorted(out) == sorted(SETTINGS)
        ok[f"shapes_{seed}"] = all(v.shape == (240, 60) and v.dtype == np.float32 for v in out.values())
        ok[f"finite_in_range_{seed}"] = all(bool(np.isfinite(v).all() and (np.abs(v) <= 1).all()) for v in out.values())
        hi = out["hcpg"].astype(np.float64) >= C0
        ok[f"bl_keeps_hi_{seed}"] = bool(np.array_equal(out["bl_hcpg"][hi], out["hcpg"][hi]))
        ok[f"a0_changes_{seed}"] = all(bool(np.abs(out[f"hcp_a{a0}"] - out["hcp"]).max() > 0) for a0 in A0)
        h = 150
        oh, _ = outputs(x[:h], g)
        ok[f"prefix_invariant_{seed}"] = all(bool(np.allclose(oh[k], out[k][:h], rtol=0, atol=1e-6)) for k in out)
        ok[f"prefix_exact_{seed}"] = all(bool(np.array_equal(oh[k], out[k][:h])) for k in out)
    print(json.dumps({"synth": ok, "all_ok": all(v for k, v in ok.items() if not k.startswith("prefix_exact"))}),
          flush=True)
    return all(v for k, v in ok.items() if not k.startswith("prefix_exact"))


if __name__ == "__main__":
    if "--synth" in sys.argv:
        sys.exit(0 if synth() else 1)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
