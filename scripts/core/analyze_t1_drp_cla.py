"""T1 premise checks DRP and CLA-diag on the PEZ-NH parent (analysis only).

Plan and frozen rules: docs/experiments/20261009_ogata_t1_drp_cla.md (specs from the synthesis in
docs/experiments/20261009_ogata_t1_new_angles.md).

DRP  per-column dose-response of the in-span cosine to ln|x| (partialled on ln(1+t), weights a_t, Newey-West HAC),
     standardised across columns (LOO z) = zD, in windows [15,200) and [200,1000), on the 8 hard sequences of the
     parent (top-5 impostor-error and top-3 genuine-error sequences at its EER cut) and 24 rest sequences.
CLA  PEZ-NH with the numerator multiplied by min(1, r), r = LP*LS/Q (HCP's undiscounted parts, leader-exact Q):
     cell accounting, matched-score likelihood-ratio ratio, exact counterfactual T1 metrics.
Pseudo targets on the evaluator side only. Prints JSON aggregates only.
Usage (server): python scripts/core/analyze_t1_drp_cla.py [workers<=6] > results/ogata/t1_drp_cla.log
                python scripts/core/analyze_t1_drp_cla.py --synth
"""
import json
import math
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
import x_r4 as R4  # noqa: E402
import analyze_hvc_stage0 as HV  # noqa: E402
import analyze_blvs_stage0 as BL  # noqa: E402
import analyze_pez_residual as RS  # noqa: E402
from cubic_fast import _prep  # noqa: E402

ZIP_NH = RS.ART / "peznh_blvs_n_fco" / "submission.zip"
SHA_NH = "4e441b54fff3d1c1bee3085a098e5727cc79b066524cfc518a043db1d78ffae5"
NH = {"eer_full": 0.002989023673650839, "eer_thr": -0.16143716871738434, "far1e5_thr": 0.8430233597755432,
      "frr_far1e5": 0.22602935706314917, "far_at_frr1e5": 0.02754954779045593}
REPRO_NDIFF, REPRO_MAX, REPRO_EER = 10, 1e-6, 1e-9
WINDOWS = ((15, 200), (200, 1000))
N_REST, MIN_N = 24, 30
Z_EXC, LOOK, CONF = 2.5758, 0.25, 50.0
# DRP frozen kills
K_RANK, K_RANK_SEQ, K_NULL, K_LOOK, K_HEAD = 3, 5, 0.03, 0.10, 60.0
# CLA frozen kills
K_DEER, K_LR, K_FAR, K_THR, K_HEAD_CLA, BAND = -0.00003, 0.6, 0.005, 0.002, 29.0, 0.05
W_IMP = 199.0
D = {}


# ---------------- DRP ----------------
def loo_z(v):
    K = v.shape[-1]
    mu = (v.sum(-1, keepdims=True) - v) / (K - 1)
    var = ((v * v).sum(-1, keepdims=True) - v * v) / (K - 1) - mu * mu
    sd = np.sqrt(np.maximum(var, 0.0))
    return np.where(sd > 0, (v - mu) / np.where(sd > 0, sd, 1.0), 0.0)


def hac_slopes(X, W, Y):
    """WLS of every column of Y (n x K) on X (n x p) with weights W (n); Newey-West (Bartlett) HAC.
    Returns (beta[1], se[1]) for regressor 1, each of shape (K,)."""
    n, p = X.shape
    Xw = X * W[:, None]
    B = X.T @ Xw
    Bi = np.linalg.inv(B)
    beta = Bi @ (Xw.T @ Y)                                  # p x K
    E = Y - X @ beta                                        # n x K
    G = Xw[:, :, None] * E[:, None, :]                      # n x p x K
    L = int(math.ceil(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    S = np.einsum("tik,tjk->ijk", G, G)
    for lag in range(1, min(L, n - 1) + 1):
        kw = 1.0 - lag / (L + 1.0)
        Gl = np.einsum("tik,tjk->ijk", G[lag:], G[:-lag])
        S += kw * (Gl + Gl.transpose(1, 0, 2))
    V = np.einsum("ai,ijk,jb->abk", Bi, S, Bi)
    return beta[1], np.sqrt(np.maximum(V[1, 1], 0.0))


def drp_seq(x, g, tgt, parent, cut, zP_t):
    u, a, c = _prep(x, g)
    cos = u @ c.T                                           # in-span cosine, T x K
    T = cos.shape[0]
    psi = np.log(np.maximum(np.linalg.norm(np.asarray(x, np.float64), axis=1), 1e-300))
    gn = np.asarray(g, np.float64)
    gn = gn / np.linalg.norm(gn, axis=1, keepdims=True)
    G = gn @ gn.T
    hub = (G.sum(1) - 1.0) / (G.shape[0] - 1)
    look = (G[tgt] >= LOOK)
    look[tgt] = False
    oth = np.ones(G.shape[0], bool)
    oth[tgt] = False
    prank = 1 + (parent > parent[:, [tgt]]).sum(1)          # parent rank of the target per row
    err = (parent[:, tgt] < cut).astype(np.float64) + ((parent >= cut) & oth[None, :]).sum(1) / W_IMP
    out = []
    for lo, hi in WINDOWS:
        hi_ = min(hi, T)
        n = hi_ - lo
        if n < MIN_N:
            out.append(None)
            continue
        t = np.arange(lo, hi_)
        X = np.stack([np.ones(n), psi[lo:hi_], np.log1p(t)], 1)
        b, se = hac_slopes(X, a[lo:hi_], cos[lo:hi_])
        tst = np.where(se > 0, b / np.where(se > 0, se, 1.0), 0.0)
        zD = loo_z(tst)
        rank = int(1 + (zD > zD[tgt]).sum())
        bc = b - b.mean()
        hc = hub - hub.mean()
        r2 = float((bc @ hc) ** 2 / max((bc @ bc) * (hc @ hc), 1e-300))
        conf = float(np.median(zP_t[lo:hi_])) >= CONF
        rows = (rank <= K_RANK) & (prank[lo:hi_] >= 20)
        out.append({"rank": rank, "slope_t": float(b[tgt]), "slope_q50_99_nt": np.quantile(b[oth], [.5, .99]).tolist(),
                    "n_exc_nt": int((zD[oth] > Z_EXC).sum()), "n_nt": int(oth.sum()),
                    "n_exc_look": int((zD[look] > Z_EXC).sum()) if conf else 0, "n_look": int(look.sum()) if conf else 0,
                    "hub_r2": r2, "head_rows": int(rows.sum()) if rank <= K_RANK else 0,
                    "head_geq": float(err[lo:hi_][rows].sum()) if rank <= K_RANK else 0.0})
    return out


# ---------------- CLA ----------------
def cla_parts(x, g):
    f, oE, oV, _, alt, _, _ = RS.states(x, g)
    K, zP, LP, LS, num, den = R5._base(x, g, f)
    om = oE * oV * alt["hist"]
    numN = R5._numN(LP, LS, om)
    w, r_, _ = R4.fco_weights(x, g)
    v, _ = R4.fco_votes(x, g, r_)
    C, M = R4.fco_state(w, v)
    parent, _ = R4.fco_core(numN, den, C, M, K)
    _, _, aux = H.hcp_parts(zP, f["zS"], K)
    Q = aux["Q"]
    ratio = np.where(Q > 0, LP * LS / np.where(Q > 0, Q, 1.0), np.inf)
    cla, _ = R4.fco_core(numN * np.minimum(1.0, ratio), den, C, M, K)
    tie = (LP == 1.0) & (LS == 1.0)
    v_, _, _ = HV.verify(x, g)
    clean = np.linalg.norm(np.asarray(x, np.float64), axis=1) >= BL.CLEAN
    hist = np.cumsum(v_ & clean[:, None], axis=0) >= 1
    return parent, cla, ratio, tie, f["zP"], hist, clean


def _job(task):
    tr, i = task
    x, g = D[f"s{tr}"][i], D[f"g{tr}"]
    parent, cla, ratio, tie, zP, hist, clean = cla_parts(x, g)
    zipped = D[f"Z{tr}"][i]
    p32 = parent.astype(np.float32)
    dif = np.abs(p32.astype(np.float64) - zipped.astype(np.float64))
    rep = (int((dif > 0).sum()), float(dif.max()))
    if tr == 2:
        c = int(D["c2"][i])
        return tr, i, rep, {"p": p32[:, c].copy(), "q": cla.astype(np.float32)[:, c].copy()}
    t = int(D["c1"][i])
    res = {"p": p32, "q": cla.astype(np.float32), "cellr": ratio < 1, "tie": tie,
           "hist_t": hist[:, t], "clean": clean, "zP_t": zP[:, t].copy()}
    return tr, i, rep, res


def _drp_job(task):
    i, = task
    return i, drp_seq(D["s1"][i], D["g1"], int(D["c1"][i]), D["PAR"][i], np.float32(NH["eer_thr"]), D["ZPT"][i])


def main(workers=6):
    import analyze_t1_residual as ar
    import common_check as cc
    from proxy_lib import load_track, pseudo_targets
    t0 = time.time()
    stop = {"zip_sha": RS._sha(ZIP_NH) == SHA_NH}
    S = HV._read_zip(ZIP_NH)
    g1, s1, n1 = load_track("track1")
    g2, s2_, n2 = load_track("track2")
    c1, _ = pseudo_targets(g1, s1)
    c2, _ = pseudo_targets(g2, s2_)
    D.update(s1=s1, g1=g1, s2=s2_, g2=g2, c1=c1, c2=c2, Z1=[S["track1"][f"score_{n}.npz"] for n in n1],
             Z2=[S["track2"][f"score_{n}.npz"] for n in n2])
    del S
    tasks = [(1, i) for i in range(len(s1))] + [(2, i) for i in range(len(s2_))]
    R = {1: {}, 2: {}}
    nd, mx = 0, 0.0
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 6)), mp_context=mp.get_context("fork")) as ex:
        for tr, i, rep, out in ex.map(_job, tasks, chunksize=1):
            nd, mx = nd + rep[0], max(mx, rep[1])
            R[tr][i] = out
    n1s, n2s = len(s1), len(s2_)
    stop["repro_zip"] = nd <= REPRO_NDIFF and mx <= REPRO_MAX
    t1 = {}
    for k in ("p", "q"):
        gen, imp = ar._gen_imp([R[1][i][k] for i in range(n1s)], c1)
        met = ar._metrics(gen, imp)
        t1[k] = {"eer_full": cc._eer_full(gen, imp), "eer_thr": met["eer_thr"], "far1e5_thr": met["far1e5_thr"],
                 "frr_far1e5": met["frr_at_far1e5"], "far_at_frr1e5": met["far_at_frr1e5"]}
        del gen, imp
    stop["repro_eer"] = abs(t1["p"]["eer_full"] - NH["eer_full"]) <= REPRO_EER
    cut = np.float32(NH["eer_thr"])
    # per-sequence errors at the parent's cut -> hard set
    imp_err = np.zeros(n1s)
    gen_err = np.zeros(n1s)
    for i in range(n1s):
        p, t = R[1][i]["p"], int(c1[i])
        oth = np.ones(p.shape[1], bool)
        oth[t] = False
        imp_err[i] = ((p >= cut) & oth[None, :]).sum()
        gen_err[i] = (p[:, t] < cut).sum()
    hard = sorted(set(np.argsort(-imp_err, kind="stable")[:5].tolist()) | set(np.argsort(-gen_err, kind="stable")[:3].tolist()))
    rest_pool = [i for i in range(n1s) if i not in hard]
    stride = len(rest_pool) / N_REST
    rest = [rest_pool[int(k * stride)] for k in range(N_REST)]
    # ---- CLA cells, LR, headroom, anatomy ----
    cells = {f"{rn}_{tn}": {"gen_acc": 0, "imp_acc": 0} for rn in ("r_lt1", "r_ge1") for tn in ("tie", "nontie")}
    band = {s: {"gen_lt": 0, "gen_ge": 0, "imp_lt": 0, "imp_ge": 0} for s in ("fit192", "all200")}
    anat = {}
    crowd_top3, crowd_all = 0, 0
    for i in range(n1s):
        res, t = R[1][i], int(c1[i])
        p, rl, tie = res["p"], res["cellr"], res["tie"]
        oth = np.ones(p.shape[1], bool)
        oth[t] = False
        acc = p >= cut
        for rn, rm in (("r_lt1", rl), ("r_ge1", ~rl)):
            for tn, tm in (("tie", tie), ("nontie", ~tie)):
                m = acc & rm & tm
                cells[f"{rn}_{tn}"]["gen_acc"] += int(m[:, t].sum())
                cells[f"{rn}_{tn}"]["imp_acc"] += int((m & oth[None, :]).sum())
        inb = np.abs(p.astype(np.float64) - float(cut)) <= BAND
        for s in (("all200", "fit192") if i not in hard else ("all200",)):
            band[s]["gen_lt"] += int((inb[:, t] & rl[:, t]).sum())
            band[s]["gen_ge"] += int((inb[:, t] & ~rl[:, t]).sum())
            band[s]["imp_lt"] += int((inb & rl & oth[None, :]).sum())
            band[s]["imp_ge"] += int((inb & ~rl & oth[None, :]).sum())
        T = p.shape[0]
        pos = np.digitize(np.arange(T), [10, 50, 200, 1000])
        ge = (p[:, t] < cut) & (np.arange(T) >= 10)
        for h in (0, 1):
            for cl in (0, 1):
                for b in range(1, 5):
                    k = f"hist{h}_clean{cl}_pos{b}"
                    anat[k] = anat.get(k, 0) + int((ge & (res["hist_t"] == h) & (res["clean"] == cl) & (pos == b)).sum())
        lead = p.argmax(1)
        crowd = acc & oth[None, :]
        crowd[np.arange(T), lead] = False
        per_col = crowd.sum(0)
        crowd_all += int(per_col.sum())
        crowd_top3 += int(np.sort(per_col)[::-1][:3].sum())

    def lr_ratio(b):
        G, I = b["gen_lt"] + b["gen_ge"], b["imp_lt"] + b["imp_ge"]
        if min(b["gen_lt"], b["gen_ge"], b["imp_lt"], b["imp_ge"]) == 0:
            return None
        return (b["gen_lt"] / b["imp_lt"]) / (b["gen_ge"] / b["imp_ge"])

    lr = {s: lr_ratio(band[s]) for s in band}
    head_cla = sum(v["imp_acc"] for k, v in cells.items() if k.startswith("r_lt1")) / W_IMP - \
        sum(v["gen_acc"] for k, v in cells.items() if k.startswith("r_lt1"))
    d_eer = t1["q"]["eer_full"] - t1["p"]["eer_full"]
    thr_p = np.float32(t1["p"]["far1e5_thr"])
    thr_q = np.float32(t1["q"]["far1e5_thr"])
    acc_p = np.concatenate([R[2][i]["p"] >= thr_p for i in range(n2s)])
    acc_q = np.concatenate([R[2][i]["q"] >= thr_q for i in range(n2s)])
    cla_kill = {"dEER<=-.00003": d_eer <= K_DEER, "LR_fit192<=.6": lr["fit192"] is not None and lr["fit192"] <= K_LR,
                "dFAR@FRR1e-5<=.005": t1["q"]["far_at_frr1e5"] - t1["p"]["far_at_frr1e5"] <= K_FAR,
                "|dthr|<=.002": abs(t1["q"]["far1e5_thr"] - t1["p"]["far1e5_thr"]) <= K_THR,
                "headroom>=29": head_cla >= K_HEAD_CLA}
    cla = {"t1_parent": t1["p"], "t1_cla": t1["q"], "dEER": d_eer, "cells_at_cut": cells, "band": band, "lr_ratio": lr,
           "headroom_geq": head_cla, "t2_claimed_acc_share_parent_cla": [float(acc_p.mean()), float(acc_q.mean())],
           "anatomy_gen_err_t_ge10": anat, "crowd_top3_share": crowd_top3 / max(1, crowd_all), "kill": cla_kill,
           "verdict": "PASS" if all(cla_kill.values()) else "KILL"}
    for i in range(n1s):
        R[1][i].pop("q")
    print(json.dumps({"stop": stop, "repro": {"ndiff": nd, "maxdiff": mx}, "n_hard": len(hard),
                      "cla": cla}), flush=True)
    if not all(stop.values()):
        print(json.dumps({"verdict": "STOP", "seconds": round(time.time() - t0, 1)}), flush=True)
        return
    # ---- DRP on hard + rest ----
    D.update(PAR={i: R[1][i]["p"] for i in hard + rest}, ZPT={i: R[1][i]["zP_t"] for i in hard + rest})
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 6)), mp_context=mp.get_context("fork")) as ex:
        DR = dict(ex.map(_drp_job, [(i,) for i in hard + rest], chunksize=1))
    drp = {}
    for wi, (lo, hi) in enumerate(WINDOWS):
        hs = [DR[i][wi] for i in hard if DR[i][wi] is not None]
        rs = [DR[i][wi] for i in rest if DR[i][wi] is not None]
        al = hs + rs
        drp[f"{lo}-{hi}"] = {
            "hard_ranks": [h["rank"] for h in hs], "rest_rank_q50_90": np.quantile([r["rank"] for r in rs], [.5, .9]).tolist()
            if rs else None,
            "hard_top3": int(sum(h["rank"] <= K_RANK for h in hs)), "n_hard": len(hs),
            "null_exc_share": sum(a["n_exc_nt"] for a in al) / max(1, sum(a["n_nt"] for a in al)),
            "look_exc_share": sum(a["n_exc_look"] for a in al) / max(1, sum(a["n_look"] for a in al)),
            "n_look_conf": sum(a["n_look"] for a in al),
            "hub_r2_median": float(np.median([a["hub_r2"] for a in al])),
            "headroom_geq": float(sum(a["head_geq"] for a in al)), "headroom_rows": int(sum(a["head_rows"] for a in al)),
            "target_slope_t_hard": [round(h["slope_t"], 5) for h in hs]}
    need = math.ceil(K_RANK_SEQ / 8 * len(hard))
    drp_kill = {"i_rank_top3_in_>=5of8_either_window": any(v["hard_top3"] >= need for v in drp.values()),
                "ii_null_exc<=.03": all(v["null_exc_share"] <= K_NULL for v in drp.values()),
                "iii_look_exc<=.10": all(v["look_exc_share"] <= K_LOOK for v in drp.values()),
                "iv_headroom>=60": sum(v["headroom_geq"] for v in drp.values()) >= K_HEAD}
    print(json.dumps({"drp": drp, "n_hard": len(hard), "need_top3": need, "drp_kill": drp_kill,
                      "drp_verdict": "PASS" if all(drp_kill.values()) else "KILL"}), flush=True)
    print(json.dumps({"verdict": {"cla": cla["verdict"], "drp": "PASS" if all(drp_kill.values()) else "KILL"},
                      "seconds": round(time.time() - t0, 1)}), flush=True)


# ---------------- random-data checks ----------------
def synth():
    rng = np.random.default_rng(17)
    ok = {}
    # HAC slope: known linear model with AR(1) noise; slope recovered; HAC se > 0
    n, K = 400, 7
    t = np.arange(n)
    psi = rng.normal(size=n).cumsum() * 0.05 + 3.0
    X = np.stack([np.ones(n), psi, np.log1p(t)], 1)
    true = np.linspace(-0.2, 0.4, K)
    e = np.zeros((n, K))
    for s in range(1, n):
        e[s] = 0.9 * e[s - 1] + 0.01 * rng.normal(size=K)
    Y = X[:, [0]] * 0.1 + psi[:, None] * true[None, :] + 0.02 * np.log1p(t)[:, None] + e
    b, se = hac_slopes(X, np.ones(n), Y)
    ok["hac_slope_recovered"] = bool(np.max(np.abs(b - true)) < 0.1)
    bo = np.linalg.lstsq(X, Y, rcond=None)[0][1]
    ok["unit_weights_eq_ols"] = bool(np.allclose(b, bo, atol=1e-10))
    ok["hac_se_positive"] = bool((se > 0).all())
    v = rng.normal(size=(3, 50))
    zz = loo_z(v)
    j = 7
    m = np.delete(v[0], j)
    ok["loo_z"] = bool(abs(zz[0, j] - (v[0, j] - m.mean()) / m.std()) < 1e-10)
    # CLA parts on a planted stream: parent equals PEZ-NH; cla <= parent; r >= 1 entries unchanged
    import analyze_peznh_stage0 as NHM
    Kg, T = 60, 150
    g = rng.normal(size=(Kg, 256)).astype(np.float32)
    x = rng.normal(size=(T, 256)) + 2.5 * g[3]
    x = (x / np.linalg.norm(x, axis=1, keepdims=True) * rng.uniform(17, 26, size=(T, 1))).astype(np.float32)
    parent, cla, ratio, tie, _, _, _ = cla_parts(x, g)
    ok["parent_eq_peznh"] = bool(np.array_equal(parent, NHM.peznh_blvs_n_fco(x, g)))
    ok["cla_le_parent"] = bool((cla <= parent + 1e-15).all())
    ok["cla_eq_parent_where_r_ge1"] = bool(np.array_equal(cla[ratio >= 1], parent[ratio >= 1]))
    ok["some_r_lt1"] = bool((ratio < 1).any())
    out = drp_seq(x, g, 3, parent.astype(np.float32), np.float32(-0.16), np.full(T, 60.0))
    ok["drp_runs"] = out[0] is not None and bool(json.dumps(out))
    print(json.dumps({"synth": ok, "all_ok": all(ok.values())}), flush=True)
    return all(ok.values())


if __name__ == "__main__":
    if sys.argv[1:] == ["--synth"]:
        sys.exit(0 if synth() else 1)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
