"""Round 2 preregistered job: KAH, ISO-HCP and CVM against same-run HCP and OSP-C references.

Preregistered in docs/experiments/20261007_ogata_r2_kah_iso_cvm.md. Three independent primary
designs, each a single fixed design, never composed with each other:

  kah      lane T1     HCP with the history likelihood LP replaced by the equal-weight model average
                       LH = (LP + LB)/2, LB from the canonical baseline's linear prefix (example.py
                       method 1) scored against its own leave-one-out cohort.
  iso_hcp  lane T1     HCP on gallery-isotropised inputs: C = (1/K) sum_j g_j g_j^T (uncentred second
                       moment of the normalised gallery), lam = tr C / d, W = (C + lam I)^(-1/2),
                       T(v) = Wv/||Wv||, gallery g_j -> T(g_j), frame x -> ||x|| T(x/||x||).
  cvm      lane Joint  HCP with the current look LS replaced by LS' = 1 + pic * zS+^2, where
                       pic = 1 - pi and pi is the posterior of a Shiryaev-Roberts change-point model
                       on the in-span cosines (likelihood temperature q = min(1, (|x|/20)^4)).

References computed in the same run: hcp_ref (x_fr_hcp.hcp on the unchanged front end) and
ospc_ref (candidates.ospc_vtn). run_check also adds the anchors 955399 and 955690.

outputs_fn(x, g) is a strictly causal, per-sequence processor with fresh state per call, float64
internals, identical constants on both tracks and no Track logic. K = c.shape[0] is the gallery
size, not a Track switch. No labels, pseudo targets, Track IDs, attack types or future frames.

  python scripts/ogata/x_r2.py --synth                    # random-data pre-run checks only
  python scripts/ogata/x_r2.py                            # single preregistered run_check
  python scripts/ogata/x_r2.py --report RUNLOG CALLOG     # post-run report + frozen gate

--synth uses random data only. --report uses evaluator-only pseudo targets and
calibrate_t2_proxy's exact fixture and proxy definitions. All print aggregates only.
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

import x_fr_hcp as H  # noqa: E402  (frozen HCP parent: _fe, hcp_parts, hcp, OSPC)
from cubic_fast import _prep  # noqa: E402

TAG = "x_r2"
PRIMARIES = ["kah", "iso_hcp", "cvm"]
REFS = ["hcp_ref", "ospc_ref"]
ALL = PRIMARIES + REFS
LANE = {"kah": "T1", "iso_hcp": "T1", "cvm": "Joint"}
_AUTO = object()


# ---------------- shared HCP map with an arbitrary history / current likelihood ----------------
def hcp_parts_L(LH, LS, K):
    """num, den of the HCP odds (leader-exact Q) from likelihood arrays given directly. Expression order
    identical to x_fr_hcp.hcp_parts, so with LP = 1 + zP+^2 and LS = 1 + zS+^2 the result is bit-identical."""
    wS = LS - 1.0
    mS = (wS.sum(1, keepdims=True) - wS) / (K - 1)
    Pr = LH * LS
    Q = (Pr.sum(1, keepdims=True) - Pr) / (K - 1)
    r = np.arange(Pr.shape[0])
    i1 = Pr.argmax(1)
    Po = Pr.copy()
    Po[r, i1] = 0.0
    Q[r, i1] = Po.sum(1) / (K - 1)                   # leader-exact
    num = 4.0 * LH * LS
    den = LH * (2.0 + mS) + 1.0 + Q
    return num, den


def hcp_L(LH, LS, K):
    num, den = hcp_parts_L(LH, LS, K)
    rn = np.sqrt(num)
    rd = np.sqrt(den)
    return (rn - rd) / (rn + rd)


def _loo_z(M, K):
    """Leave-one-out z of every entry against its own row cohort (x_fr_hcp._fe zS formula verbatim;
    z = 0 when sd = 0)."""
    mu = (M.sum(1, keepdims=True) - M) / (K - 1)
    var = ((M * M).sum(1, keepdims=True) - M * M) / (K - 1) - mu * mu
    sd = np.sqrt(np.maximum(var, 0.0))
    return np.where(sd > 0, (M - mu) / np.where(sd > 0, sd, 1.0), 0.0)


# ---------------- design 1: KAH ----------------
def kah_B(x, g):
    """Canonical example.py method 1 (cumulative mean embedding, cosine) in float64, strictly causal.
    The 1/n factor of the example cancels under row normalisation."""
    xb = np.asarray(x, np.float64)
    m = np.cumsum(xb, 0)
    g64 = np.asarray(g, np.float64)
    mn = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    gn = g64 / np.maximum(np.linalg.norm(g64, axis=1, keepdims=True), 1e-12)
    return mn @ gn.T


def kah_core(f, zB):
    """HCP map with LH = (LP + LB)/2 (equal-weight model average of the cubic and linear history looks)."""
    K = f["K"]
    LP = 1.0 + np.maximum(f["zP"], 0.0) ** 2
    LB = 1.0 + np.maximum(zB, 0.0) ** 2
    LH = (LP + LB) / 2.0
    LS = 1.0 + np.maximum(f["zS"], 0.0) ** 2
    return hcp_L(LH, LS, K)


def kah(x, g, fe=None):
    f = H._fe(x, g) if fe is None else fe
    return kah_core(f, _loo_z(kah_B(x, g), f["K"]))


# ---------------- design 2: ISO-HCP ----------------
def _unit_rows(v):
    """Row-normalise in float64; zero rows stay zero. Returns (unit rows, norms)."""
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n > 0, v / np.where(n > 0, n, 1.0), 0.0), n[:, 0]


def iso_W(g):
    """W = (C + lam I)^(-1/2), C = (1/K) sum_j gn_j gn_j^T (uncentred), lam = tr C / d.
    Returns None (identity transform) when lam == 0."""
    gn, _ = _unit_rows(g)
    K, d = gn.shape
    C = gn.T @ gn / K
    lam = float(np.trace(C)) / d
    if not lam > 0.0:
        return None
    ev, V = np.linalg.eigh(C + lam * np.eye(d))
    return (V / np.sqrt(ev)) @ V.T


def _apply_T(Vu, W):
    """T(v) = W v / ||W v|| on unit (or zero) rows. If W v is exactly zero the original row is kept."""
    Y = Vu @ W.T
    n = np.linalg.norm(Y, axis=1, keepdims=True)
    ok = n > 0
    return np.where(ok, Y / np.where(ok, n, 1.0), Vu)


def iso_inputs(x, g, W=_AUTO):
    """Transformed (frames, gallery). Gallery g_j -> T(g_j); frame x -> ||x|| T(x/||x||) (zero frames
    unchanged). W None (lam == 0) returns the inputs unchanged."""
    if W is _AUTO:
        W = iso_W(g)
    if W is None:
        return x, g
    gn, _ = _unit_rows(g)
    xn, nrm = _unit_rows(x)
    return nrm[:, None] * _apply_T(xn, W), _apply_T(gn, W)


def iso_hcp(x, g, W=_AUTO):
    xi, gi = iso_inputs(x, g, W)
    f = H._fe(xi, gi)
    return H.hcp(f["zP"], f["zS"], f["K"])


# ---------------- design 3: CVM ----------------
def cvm_state(u, a, c, nrm, K, force_pi_zero=False):
    """Self-referenced Shiryaev-Roberts change posterior per column (float64, strictly causal).

    u, a, c = cubic_fast._prep(x, g); nrm = ||x_t||. Returns dict(ell, hist, logO, pi, pic).
    Pre-run definitions of undefined cases (see the preregistration):
      * no history (t = 0, or the cumulative past weight cw_{t-1} == 0): ell = 0, the SR state is not
        advanced and pi = 0 (pic = 1), i.e. HCP behaviour;
      * s2 <= 0 (no cross-sectional spread): ell = 0, as in the design spec.
    """
    V = u @ c.T
    T = V.shape[0]
    sV = V.sum(1, keepdims=True)
    sV2 = (V * V).sum(1, keepdims=True)
    b = (sV - V) / (K - 1)
    s2 = (sV2 - V * V) / (K - 1) - b * b
    cw = np.cumsum(a)
    cs = np.cumsum(a[:, None] * V, 0)
    hist = np.zeros(T, bool)
    hist[1:] = cw[:-1] > 0
    mu = b.copy()                                    # rows without history: mu := b  =>  ell = 0
    h = np.nonzero(hist)[0]
    mu[h] = cs[h - 1] / cw[h - 1][:, None]           # strictly past, a-weighted
    q = np.minimum(1.0, (nrm / 20.0) ** 4)
    pos = s2 > 0
    ell = np.where(pos, q[:, None] * ((V - mu) ** 2 - (V - b) ** 2) / (2.0 * np.where(pos, s2, 1.0)), 0.0)
    ell[~hist] = 0.0
    logO = np.full((T, K), -np.inf)
    if not force_pi_zero:
        L = np.full(K, -np.inf)
        for t in range(1, T):
            if not hist[t]:
                continue
            L = ell[t] + np.logaddexp(0.0, L)
            logO[t] = L - np.log(2.0 * t)
    lse = np.logaddexp(0.0, logO)
    return {"ell": ell, "hist": hist, "logO": logO, "pi": np.exp(logO - lse), "pic": np.exp(-lse)}


def cvm_core(f, pic):
    """HCP map with the validity-mixed current look LS' = 1 + pic * zS+^2 (same operation order)."""
    LP = 1.0 + np.maximum(f["zP"], 0.0) ** 2
    LSp = 1.0 + pic * np.maximum(f["zS"], 0.0) ** 2
    return hcp_L(LP, LSp, f["K"])


def cvm_pic(x, g, K, force_pi_zero=False):
    u, a, c = _prep(x, g)
    nrm = np.linalg.norm(np.asarray(x, np.float64), axis=1)
    return cvm_state(u, a, c, nrm, K, force_pi_zero)


def cvm(x, g, fe=None, force_pi_zero=False):
    f = H._fe(x, g) if fe is None else fe
    return cvm_core(f, cvm_pic(x, g, f["K"], force_pi_zero)["pic"])


# ---------------- harness entry ----------------
def _all64(x, g):
    f = H._fe(x, g)
    return {"kah": kah(x, g, f), "iso_hcp": iso_hcp(x, g), "cvm": cvm(x, g, f),
            "hcp_ref": H.hcp(f["zP"], f["zS"], f["K"])}


def outputs_fn(x, g):
    out = {k: v.astype(np.float32) for k, v in _all64(x, g).items()}
    out["ospc_ref"] = np.asarray(H.OSPC(x, g), np.float32)
    return out


# ---------------- random-data pre-run checks (no challenge data) ----------------
def _brute_L(LH, LS, K):
    """Per-column np.delete loops: num/den reference for hcp_L."""
    num = 4.0 * LH * LS
    den = np.empty_like(num)
    for j in range(K):
        mS = np.delete(LS - 1.0, j, axis=1).sum(1) / (K - 1)
        Q = np.delete(LH * LS, j, axis=1).sum(1) / (K - 1)
        den[:, j] = LH[:, j] * (2.0 + mS) + 1.0 + Q
    rn, rd = np.sqrt(num), np.sqrt(den)
    return num / den, (rn - rd) / (rn + rd)


def _brute_logO(ell, hist):
    """Explicit sum over the change time tau of prod exp(ell), in log space, for every t and column."""
    T, K = ell.shape
    S = np.vstack([np.zeros((1, K)), np.cumsum(ell, 0)])          # S[t+1] = sum_{s<=t} ell_s
    out = np.full((T, K), -np.inf)
    for t in range(1, T):
        if not hist[t]:
            continue
        lo = int(np.argmax(hist[1:])) + 1                          # first frame with history
        taus = np.arange(lo, t + 1)
        vals = S[t + 1][None, :] - S[taus][:, :]                   # sum_{s=tau}^{t} ell_s
        mx = vals.max(0)
        out[t] = mx + np.log(np.exp(vals - mx).sum(0)) - np.log(2.0 * t)
    return out


def _extreme_L(K):
    """Likelihood arrays with a dominant leader, ties, near-ties and a flat row."""
    LH, LS = np.ones((6, K)), np.ones((6, K))
    LH[0, 3], LS[0, 3] = 1.0 + 2000.0 ** 2, 1.0 + 2000.0 ** 2
    LH[1, 3], LS[1, 9] = 1.0 + 2000.0 ** 2, 1.0 + 2000.0 ** 2
    LH[2, 3] = LH[2, 9] = LS[2, 3] = LS[2, 9] = 1.0 + 30.0 ** 2
    LH[4] = 1.0 + np.linspace(0, 1, K) ** 2
    LS[4] = 1.0 + np.linspace(1, 0, K) ** 2
    LH[5, 0], LS[5, 0], LH[5, 1], LS[5, 1] = 1e6, 999.0 ** 2, 999.0 ** 2, 1e6
    return LH, LS


def _stream(seed, T, K, kind, norm=None):
    x, g = H._synth_data(seed, T, K, kind)
    if norm is not None:
        x = (x / np.linalg.norm(x, axis=1, keepdims=True) * norm).astype(np.float32)
    return x, g


def synth():
    import example
    w = {}

    def upd(key, v):
        w[key] = max(w.get(key, 0.0), float(v))

    def cnt(key, v):
        w[key] = w.get(key, 0) + int(v)

    def rel(a, b):
        return float((np.abs(a - b) / np.abs(b)).max())

    rng_ok, strict_ok = True, True
    rows = 0
    for K in (200, 199):
        for kind in ("switch", "mng"):
            for seed, T in ((0, 300), (1, 257), (2, 64)):
                x, g = H._synth_data(seed, T, K, kind)
                rows += T
                f = H._fe(x, g)
                zP, zS = f["zP"], f["zS"]
                ref64 = H.hcp(zP, zS, K)
                # ---------- KAH ----------
                cnt("kah_i_LB_eq_LP_mismatch", (kah_core(f, zP) != ref64).sum())          # bitwise
                B = kah_B(x, g)
                zB = _loo_z(B, K)
                LH = (1.0 + np.maximum(zP, 0.0) ** 2 + 1.0 + np.maximum(zB, 0.0) ** 2) / 2.0
                LS = 1.0 + np.maximum(zS, 0.0) ** 2
                bo, bout = _brute_L(LH, LS, K)
                num, den = hcp_parts_L(LH, LS, K)
                o = kah_core(f, zB)
                upd("kah_ii_out_abs", np.abs(o - bout).max())
                upd("kah_ii_odds_rel", rel(num / den, bo))
                bex = example.normalize(example.your_algorithm(x, method=1)) @ example.normalize(
                    np.asarray(g, np.float32)).T
                upd("kah_iii_B_vs_example_abs", np.abs(B - bex.astype(np.float64)).max())
                upd("kah_B_range_excess", max(0.0, float(np.abs(B).max()) - 1.0))
                # ---------- ISO-HCP ----------
                d = x.shape[1]
                eye = np.eye(d)
                xi, gi = iso_inputs(x, g, eye)
                xn, nrm = _unit_rows(x)
                gn, _ = _unit_rows(g)
                upd("iso_a_identity_transform_rel", max(np.abs(xi - x.astype(np.float64)).max() / np.abs(x).max(),
                                                        np.abs(gi - gn).max()))
                xe = nrm[:, None] * (xn / np.linalg.norm(xn, axis=1, keepdims=True))
                ge = gn / np.linalg.norm(gn, axis=1, keepdims=True)
                fe_ = H._fe(xe, ge)
                cnt("iso_a_W_I_vs_explicit_mismatch", (iso_hcp(x, g, eye) != H.hcp(fe_["zP"], fe_["zS"], K)).sum())
                upd("iso_a_info_W_I_vs_raw_hcp_abs", np.abs(iso_hcp(x, g, eye) - ref64).max())   # float32-cast noise
                W = iso_W(g)
                C = gn.T @ gn / K
                lam = float(np.trace(C)) / d
                upd("iso_b_W_inverse_sq_rel", np.abs(W @ (C + lam * eye) @ W - eye).max())
                upd("iso_b_W_asym", np.abs(W - W.T).max())
                upd("iso_b_lam_vs_1_over_d", abs(lam - 1.0 / d))
                xi2, gi2 = iso_inputs(x, g)
                upd("iso_b_gallery_unit_err", np.abs(np.linalg.norm(gi2, axis=1) - 1.0).max())
                upd("iso_b_frame_norm_rel", (np.abs(np.linalg.norm(xi2, axis=1) - nrm) / nrm).max())
                upd("iso_b_info_hcp_vs_iso_abs", np.abs(iso_hcp(x, g) - ref64).max())
                # ---------- CVM ----------
                cnt("cvm_i_pi0_mismatch", (cvm(x, g, f, force_pi_zero=True) != ref64).sum())   # bitwise
                st = cvm_pic(x, g, K)
                pic = st["pic"]
                LSp = 1.0 + pic * np.maximum(zS, 0.0) ** 2
                LP = 1.0 + np.maximum(zP, 0.0) ** 2
                bo, bout = _brute_L(LP, LSp, K)
                num, den = hcp_parts_L(LP, LSp, K)
                o = cvm_core(f, pic)
                upd("cvm_ii_out_abs", np.abs(o - bout).max())
                upd("cvm_ii_odds_rel", rel(num / den, bo))
                bl = _brute_logO(st["ell"], st["hist"])
                fin = np.isfinite(bl)
                cnt("cvm_iii_logO_finite_pattern_mismatch", (fin != np.isfinite(st["logO"])).sum())
                upd("cvm_iii_brute_SR_logO_rel", (np.abs(bl[fin] - st["logO"][fin]) / np.maximum(1.0, np.abs(bl[fin]))).max())
                cnt("cvm_pic_out_of_unit", ((pic < 0) | (pic > 1) | ~np.isfinite(pic)).sum())
                cnt("cvm_pic_eq1_where_no_history", (pic[~st["hist"]] != 1.0).sum())
                # ---------- all: range, finiteness, prefix invariance ----------
                full = outputs_fn(x, g)
                half = outputs_fn(x[: T // 2], g)
                for k in ALL:
                    v = full[k]
                    rng_ok &= bool(v.shape == (T, K) and np.isfinite(v).all() and v.min() >= -1 and v.max() <= 1)
                    upd(f"iv_prefix_{k}", np.abs(half[k].astype(np.float64) - v[: T // 2].astype(np.float64)).max())
                for k in PRIMARIES:
                    strict_ok &= bool(np.abs(full[k].astype(np.float64)).max() < 1)
        # deterministic extreme rows through hcp_L (dominant leaders, ties, near-ties, flat row)
        LH, LS = _extreme_L(K)
        bo, bout = _brute_L(LH, LS, K)
        num, den = hcp_parts_L(LH, LS, K)
        v = hcp_L(LH, LS, K)
        upd("extreme_out_abs", np.abs(v - bout).max())
        upd("extreme_odds_rel", rel(num / den, bo))
        strict_ok &= bool(np.abs(v.astype(np.float32).astype(np.float64)).max() < 1)
        # CVM behaviour checks (K = 200 only is enough; repeated for 199)
        # stable stream: strong identity (0.6 g[11]) throughout, frames at the cap norm 25 (q = 1)
        rng = np.random.default_rng(5)
        g = rng.normal(size=(K, 512)).astype(np.float32)
        T = 160
        xs = rng.normal(size=(T, 512)) + 0.6 * g[11]
        xs = (xs / np.linalg.norm(xs, axis=1, keepdims=True) * 25.0).astype(np.float32)
        st = cvm_pic(xs, g, K)
        upd("cvm_stable_late_median_pi_target", np.median(st["pi"][T // 2:, 11]))
        # identity switch at T/2: 0.6 g[7] then 0.6 g[30], q = 1
        h = T // 2
        ident = np.r_[0.6 * np.repeat(g[7][None], h, 0), 0.6 * np.repeat(g[30][None], T - h, 0)]
        xw = rng.normal(size=(T, 512)) + ident
        xw = (xw / np.linalg.norm(xw, axis=1, keepdims=True) * 25.0).astype(np.float32)
        st = cvm_pic(xw, g, K)
        f = H._fe(xw, g)
        o = cvm_core(f, st["pic"])
        hit = np.nonzero(st["pi"][h:, 7] > 0.99)[0]
        w["cvm_switch_frames_to_pi_gt_.99"] = int(hit[0]) + 1 if hit.size else 10 ** 6
        w["cvm_switch_out_target_at_hit"] = float(o[h + hit[0], 7]) if hit.size else 1.0
        upd("cvm_switch_pre_pi_max_target", st["pi"][max(1, h - 20):h, 7].max())
        # zero-history definition (cvm_state on arrays): first four frames carry zero weight
        u_, a_, c_ = _prep(*_stream(7, 40, K, "switch"))
        nrm_ = np.full(40, 20.0)
        a0 = a_.copy()
        a0[:4] = 0.0
        s0 = cvm_state(u_, a0, c_, nrm_, K)
        w["cvm_zero_history_frames_pi0"] = bool((s0["pi"][:5] == 0.0).all() and (s0["pic"][:5] == 1.0).all()
                                                 and (~s0["hist"][:5]).all() and s0["hist"][5]
                                                 and bool(np.isfinite(s0["ell"]).all())
                                                 and bool(np.isfinite(s0["logO"][5:]).all()))
        bl = _brute_logO(s0["ell"], s0["hist"])
        fin = np.isfinite(bl)
        upd("cvm_zero_history_brute_SR_rel", (np.abs(bl[fin] - s0["logO"][fin]) / np.maximum(1.0, np.abs(bl[fin]))).max())
        # s2 <= 0 rows: V identical across columns -> ell = 0
        u1 = np.zeros((6, c_.shape[1]))
        s1 = cvm_state(u1, np.ones(6), c_, np.full(6, 20.0), K)
        w["cvm_s2_zero_ell_zero"] = bool((s1["ell"] == 0.0).all() and np.isfinite(s1["pic"]).all())
    # extreme sequences through the whole pipeline: tiny/huge norms, constant frame, T = 1, 2
    ex_ok = True
    rng = np.random.default_rng(11)
    for K in (200, 199):
        g = rng.normal(size=(K, 512)).astype(np.float32)
        seqs = []
        x = rng.normal(size=(40, 512)).astype(np.float32)
        x[:, :] = x * rng.choice([1e-3, 1.0, 1e4], size=(40, 1)).astype(np.float32)
        seqs.append(x)
        seqs.append(np.repeat(rng.normal(size=(1, 512)).astype(np.float32), 30, 0))
        seqs.append(rng.normal(size=(1, 512)).astype(np.float32) * 20)
        seqs.append(rng.normal(size=(2, 512)).astype(np.float32) * 20)
        z = rng.normal(size=(40, 512)).astype(np.float32) * 20
        z[0] = 0.0
        z[17] = 0.0
        seqs.append(z)
        for xs_ in seqs:
            try:
                o = outputs_fn(xs_, g)
                for k in ALL:
                    v = o[k]
                    ex_ok &= bool(v.shape == (xs_.shape[0], K) and np.isfinite(v).all() and v.min() >= -1
                                  and v.max() <= 1)
            except Exception as e:                    # an unhandled degenerate input is a synthetic failure
                ex_ok = False
                w.setdefault("extreme_seq_errors", []).append(f"{type(e).__name__}: {str(e)[:80]}")
    out = {"synth": TAG, **w, "range_ok": rng_ok, "strict_inside_f32_primaries": strict_ok,
           "extreme_sequences_ok": ex_ok, "rows": rows}
    ok_zero = ("kah_i_LB_eq_LP_mismatch", "iso_a_W_I_vs_explicit_mismatch", "cvm_i_pi0_mismatch",
               "cvm_iii_logO_finite_pattern_mismatch", "cvm_pic_out_of_unit", "cvm_pic_eq1_where_no_history")
    tight = ("kah_ii_out_abs", "kah_ii_odds_rel", "iso_a_identity_transform_rel", "iso_b_W_inverse_sq_rel",
             "iso_b_W_asym", "iso_b_lam_vs_1_over_d", "iso_b_gallery_unit_err", "iso_b_frame_norm_rel",
             "cvm_ii_out_abs", "cvm_ii_odds_rel", "cvm_iii_brute_SR_logO_rel", "cvm_zero_history_brute_SR_rel",
             "extreme_out_abs", "extreme_odds_rel")
    out["pass"] = bool(
        all(w[k] == 0 for k in ok_zero)
        and all(w[k] <= 1e-9 for k in tight)
        and w["kah_iii_B_vs_example_abs"] <= 1e-5 and w["kah_B_range_excess"] <= 1e-12
        and all(w[f"iv_prefix_{k}"] < 1e-6 for k in ALL)
        and rng_ok and strict_ok and ex_ok
        and w["cvm_stable_late_median_pi_target"] < 1e-3
        and w["cvm_switch_frames_to_pi_gt_.99"] <= 3 and w["cvm_switch_out_target_at_hit"] < 0.5
        and w["cvm_zero_history_frames_pi0"] and w["cvm_s2_zero_ell_zero"])
    print(json.dumps(out), flush=True)
    return out["pass"]


# ---------------- post-run report (evaluator side) ----------------
TOPK = 1500
FULL = ALL                                           # full T1 metrics
PROX = ALL + ["955399"]                              # calibrated T2 proxies
A399 = {}
# Frozen calibration outcome of docs/experiments/20261007_ogata_fr_hcp.md (calibrate_t2_proxy.log,
# anchor rows only), reused unchanged. Both proxies: larger = better.
D_STAR, B_STAR = "t2_all_reject_st", "t2_cp_reject_pt"
BE_A, BE_B = 0.5097574683128512, -0.7748887487409217      # OLS be = a + b * B* over 13 rows
BE_LOO_MAE = 0.0163
# Same-run reproduction. Full-precision eer_full of the references (x_fr_hcp.log, HCP report, 2026-10-07).
HCP_EER, OSPC_EER, EER_REPRO_TOL = 0.003092348265774678, 0.0036885662354849517, 1e-7
HCP_RECORDED = {"eer_full": HCP_EER, "frr_far1e5": 0.025390058166389392, "far_at_frr1e5": 0.032500932494045176,
                "usp_aar": 0.10272, "usp_run": 61.26, "usp_frr": 0.05639, "t2_cp_reject_pt": 0.22867,
                "t2_all_reject_st": 0.14189, "t2_pre_reject_pt": 0.02093}
RUN_REPRO = {  # setting -> metric -> expected (run row unless noted 'usp_*' = report proxy row)
    "ospc_ref": {"eer_full": 0.00368857, "frr_far1e5": 0.015721, "thr": 0.9721794, "splice_aar": 0.1324,
                 "splice_run": 79.2, "t2_all_reject": 0.0926, "usp_aar": 0.238, "usp_run": 133.6,
                 "usp_frr": 0.0455},
    "955399": {"eer_full": 0.007879, "frr_far1e5": 0.0489, "splice_aar": 0.088, "splice_run": 52.8,
               "usp_aar": 0.0901},
    "955690": {"eer_full": 0.007879, "frr_far1e5": 0.0252, "splice_aar": 0.0717, "splice_run": 43.0}}
CAL_MAP = {"ospc_ref": "OSPC_964718", "955399": "V_955399"}
CHECKS_EARLY = ("t=0", "1-3", "4-9")


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


def _top3_share(outs_, t1, thr, tmin=0):
    """Share of genuine errors (o[t] < thr) carried by the three sequences with the most errors."""
    cnt = [int((o[tmin:, t] < thr).sum()) for o, t in zip(outs_, t1)]
    tot = sum(cnt)
    return {"errors": tot, "top3_share": None if tot == 0 else round(float(sum(sorted(cnt)[-3:]) / tot), 4)}


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
            "gen_err_t_ge10_at_eer_thr": int(sum(bins[b]["gen_err"] for b in ar.POS_NAMES if b not in CHECKS_EARLY)),
            "early_err_at_eer_thr": round(float(sum(bins[b]["gen_err"] + bins[b]["imp_err"] / 199.0
                                                    for b in CHECKS_EARLY)), 1),
            "top3_gen_err_at_eer_thr_all": _top3_share(outs_, t1, met["eer_thr"], 0),
            "top3_gen_err_at_eer_thr_t_ge10": _top3_share(outs_, t1, met["eer_thr"], 10)}
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
    rr = {}
    for s, d in RUN_REPRO.items():
        rr[s] = {}
        for m, v in d.items():
            got = (prox.get(s, {}).get(m) if m.startswith("usp_") else rows.get(s, {}).get(m))
            rr[s][m] = (got, v, _tol_ok(m, got, v))
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
    eer_repro = {"hcp_ref": _t(t1rep["hcp_ref"]["eer_full"], HCP_EER,
                               abs(t1rep["hcp_ref"]["eer_full"] - HCP_EER) <= EER_REPRO_TOL),
                 "ospc_ref": _t(t1rep["ospc_ref"]["eer_full"], OSPC_EER,
                                abs(t1rep["ospc_ref"]["eer_full"] - OSPC_EER) <= EER_REPRO_TOL)}
    stop = {"ref_eer_repro_1e-7_ok": all(v[2] for v in eer_repro.values()),
            "run_repro_ok": all(t_[2] for d in rr.values() for t_ in d.values()),
            "cal_repro_ok": bool(cr) and all(cr[s] for s in CAL_MAP) and
            all(t_[2] for d in cr.values() for t_ in d.values()),
            "synth_and_structural_identity_ok": bool(syn and syn.get("pass")),
            "refs_checks_ok": bool(chk.get("hcp_ref")) and bool(chk.get("ospc_ref")) and
            bad["hcp_ref"] == 0 and bad["ospc_ref"] == 0}
    global_stop = not all(stop.values())
    design_stop = {k: {"checks_pass_and_finite_in_range": bool(chk.get(k)) and bad[k] == 0,
                       "eer_full_report_vs_row_ok": bool(t1rep[k]["eer_full_match_1e-6"])} for k in PRIMARIES}
    out2 = {"report_gate": TAG, "run_repro": rr, "cal_repro": cr, "ref_eer_repro": eer_repro,
            "stop_checks_global": stop, "stop_checks_per_design": design_stop}
    # ---- frozen gate (applied once, mechanically; same-run hcp_ref) ----
    fbe = {k: BE_A + BE_B * prox[k][B_STAR] for k in PROX}
    hr = t1rep["hcp_ref"]
    gate, verdict = {}, {}
    for k in PRIMARIES:
        p = t1rep[k]
        common = {"checks_pass": _t(chk.get(k), True, bool(chk.get(k))),
                  "t1_neg1_share<=.001": _t(p["t1_neg1_share"], 0.001, p["t1_neg1_share"] <= 0.001)}
        if LANE[k] == "T1":
            terms = {"eer_full<=hcp_ref-.0002": _t(p["eer_full"], hr["eer_full"] - 0.0002,
                                                  p["eer_full"] <= hr["eer_full"] - 0.0002),
                     "BE_fc<=BE_fc(hcp_ref)+.010": _t(fbe[k], fbe["hcp_ref"] + 0.010,
                                                     fbe[k] <= fbe["hcp_ref"] + 0.010)}
        else:
            terms = {"BE_fc<=BE_fc(hcp_ref)-.020": _t(fbe[k], fbe["hcp_ref"] - 0.020,
                                                     fbe[k] <= fbe["hcp_ref"] - 0.020),
                     "eer_full<=hcp_ref+.0002": _t(p["eer_full"], hr["eer_full"] + 0.0002,
                                                  p["eer_full"] <= hr["eer_full"] + 0.0002)}
        terms.update(common)
        gate[k] = {"lane": LANE[k], "terms": terms}
        if global_stop:
            verdict[k] = "STOP (global, no verdict): " + ", ".join(s for s, v in stop.items() if not v)
        elif not all(design_stop[k].values()):
            verdict[k] = "STOP (design, no verdict): " + ", ".join(s for s, v in design_stop[k].items() if not v)
        else:
            failed = [s for s, v in terms.items() if not v[2]]
            verdict[k] = "PASS" if not failed else "FAIL: " + ", ".join(failed)
    ranking = {"T1_by_eer_full": sorted([k for k in PRIMARIES if LANE[k] == "T1" and verdict[k] == "PASS"],
                                        key=lambda k: t1rep[k]["eer_full"]),
               "Joint_by_BE_fc": sorted([k for k in PRIMARIES if LANE[k] == "Joint" and verdict[k] == "PASS"],
                                        key=lambda k: fbe[k])}
    # ---- report-only disclosures ----
    rep_only = {}
    for k in ALL:
        pk = prox[k]
        rep_only[k] = {"eer_full": t1rep[k]["eer_full"], "frr_far1e5": t1rep[k]["frr_far1e5"],
                       "far_at_frr1e5": t1rep[k]["far_at_frr1e5"], "usp_aar": pk["usp_aar"],
                       "usp_run": pk["usp_run"], "usp_frr": pk["usp_frr"],
                       "t2_pre_reject_pt": pk["t2_pre_reject_pt"], "D*(" + D_STAR + ")": pk[D_STAR],
                       "B*(" + B_STAR + ")": pk[B_STAR], "BE_fc": round(fbe[k], 4),
                       "own_eer_thr": t1rep[k]["eer_thr"], "own_far1e5_thr": t1rep[k]["far1e5_thr"],
                       "eer_thr_bins": t1rep[k]["eer_thr_bins"],
                       "top3_gen_err_at_eer_thr_all": t1rep[k]["top3_gen_err_at_eer_thr_all"],
                       "top3_gen_err_at_eer_thr_t_ge10": t1rep[k]["top3_gen_err_at_eer_thr_t_ge10"]}
    hcp_vs_recorded = {m: (got, v) for m, v in HCP_RECORDED.items()
                       for got in [t1rep["hcp_ref"].get(m, prox["hcp_ref"].get(m))]}
    out2.update({"gate": gate, "verdict": verdict, "ranking": ranking, "be_forecast": {k: round(fbe[k], 4) for k in PROX},
                 "report_only": rep_only, "hcp_ref_vs_recorded_hcp_run": hcp_vs_recorded,
                 "pos_bins": ar.POS_NAMES, "note": "A pass is a candidate for the user's decision, not a submission."})
    print(json.dumps(out2), flush=True)
    print(json.dumps({"verdict": verdict, "ranking": ranking}), flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--synth":
        sys.exit(0 if synth() else 1)
    elif arg == "--report":
        report(sys.argv[2], sys.argv[3])
    else:
        cc.run_check(outputs_fn, ALL, TAG, workers=6)
