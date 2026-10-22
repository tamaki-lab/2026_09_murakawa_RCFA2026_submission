"""Round 5 preregistered job: ESR-N, ESR-N + FCO and ESR-N x RME against same-run HCP, ESR-HCP and FCO.

Preregistered in docs/experiments/20261008_ogata_r5_esrn.md (designs from the round 5 ideation,
docs/experiments/20261007_ogata_r5_ideation.md). Three fixed designs, no new fitted constant, run once:

  esr_n     HCP odds with ESR's excursion-survival weight omega applied to the claimed column's OWN numerator
            only: num = 4 LP LS', LS' = 1 + omega (LS - 1); the denominator is HCP's (mS and leader-exact Q from
            the undiscounted LS). omega is x_r3.esr_omega unchanged. out <= HCP entrywise.
  esrn_fco  x_r4.fco_core unchanged on the ESR-N numerator: Phi(numN + F_j, den + F_0 + sum_{k != j} F_k).
  esrn_rme  ESR-N with omega = omega_E * omega_R. omega_R is the record-minimum majority exceedance channel on
            the frame-level raw cosine: X clean frames of the last W = 20 strictly below the column's minimum
            over the first half of the sequence so far; omega_R = min(1, e p), p = C(n_c, X) / C(N + n_c, X).

References computed in the same run: hcp_ref (x_fr_hcp.hcp), esr_ref (x_r3.esr_hcp), fco_ref (x_r4.fco) and
ospc_ref (candidates.ospc_vtn). run_check also adds the anchors 955399 and 955690.

outputs_fn(x, g) is a strictly causal, per-sequence processor with fresh state per call, float64 internals,
identical constants on both tracks and no Track logic. K = g.shape[0] is the gallery size, not a Track switch.
No labels, pseudo targets, Track IDs, attack types or future frames.

  python scripts/core/x_r5.py --synth                    # random-data pre-run checks only
  python scripts/core/x_r5.py --stats                    # label-free statistics incl. the frozen RME kill rules
  python scripts/core/x_r5.py                            # single preregistered run_check
  python scripts/core/x_r5.py --report RUNLOG CALLOG     # post-run report + frozen gate

--synth uses random data only. --stats reads probe sequences, no labels or pseudo targets. --report uses
evaluator-only pseudo targets and calibrate_t2_proxy's exact fixture and proxy definitions. Aggregates only.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import bisect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_check as cc  # noqa: E402  (before numpy: pins BLAS threads; adds repo root to sys.path)
import numpy as np  # noqa: E402

import x_fr_hcp as H  # noqa: E402  (frozen HCP parent: _fe, hcp_parts, hcp, OSPC)
import x_r2 as R2  # noqa: E402  (shared report constants and helpers)
import x_r3 as R3  # noqa: E402  (frozen ESR: esr_series, esr_omega, esr_hcp)
import x_r4 as R4  # noqa: E402  (frozen FCO: fco_weights, fco_votes, fco_state, fco_core, fco; NET helpers)

TAG = "x_r5"
PRIMARIES = ["esr_n", "esrn_fco", "esrn_rme"]
REFS = ["hcp_ref", "esr_ref", "fco_ref", "ospc_ref"]
ALL = PRIMARIES + REFS
E_CONST = float(np.e)
W_RME = 20                                            # RME window (frames); single-shot constant
X_MIN = 3                                             # RME minimum exceedance count; single-shot constant
NET_W = 199
THR_HCP = 0.8525308                                   # HCP FAR=1e-5 threshold of the HCP run
CUT_EER = -0.153304                                   # HCP EER threshold of the HCP run (odds .539)
ESR_EER = 0.003219037844438148                        # x_r3 run, esr_hcp eer_full (recomputed from the 966631 ZIP)
FCO_EER = 0.003017528957229187                        # x_r4 run, fco eer_full (recomputed from the 967117 ZIP)
GATE_N_BE = 0.005                                     # esr_n, esrn_fco: BE_fc <= BE_fc(esr_ref) + .005
GATE_FCO_STACK = 0.00004                              # esrn_fco: eer_full <= eer_full(esr_n) - .00004
GATE_FCO_HCP = 0.00005                                # esrn_fco: eer_full <= eer_full(hcp_ref) - .00005
GATE_RME_EER = 0.00002                                # esrn_rme: eer_full <= eer_full(esr_n) + .00002
GATE_RME_BE = 0.005                                   # esrn_rme: BE_fc <= BE_fc(esr_n) - .005
UPLOAD_BAR = 0.00316                                  # esr_n: report-only 'upload-worthy' marker
FCO_FLIPS_MIN = 6000                                  # esrn_fco: FCO flips on ESR-N at esr_n's own EER cut
FCO_OUTSIDE_MIN = 0.40                                # esrn_fco: share of those flips outside the worst sequence
RME_KILL = {"t1_flag_share_max": 0.015, "t2_t1_ratio_min": 3.0, "t1_run_seq_share_max": 0.05,
            "t2_clean_share_min": 0.5, "t2_late_share_min": 0.6, "t2_phi_max": 0.6, "t1_prod_share_max": 0.03}


# =====================================================================================
# Designs
# =====================================================================================
def _omap(num, den):
    rn = np.sqrt(num)
    rd = np.sqrt(den)
    return (rn - rd) / (rn + rd)


def _numN(LP, LS, omega):
    """4 LP LS' with LS' = 1 + omega (LS - 1); LS' = LS exactly where omega == 1 (HCP's own num, same operations)."""
    LSp = np.where(omega == 1.0, LS, 1.0 + omega * (LS - 1.0))
    return 4.0 * LP * LSp


def _base(x, g, f):
    K, zP, zS = f["K"], f["zP"], f["zS"]
    num, den, _ = H.hcp_parts(zP, zS, K)
    LP = 1.0 + np.maximum(zP, 0.0) ** 2
    LS = 1.0 + np.maximum(zS, 0.0) ** 2
    return K, zP, LP, LS, num, den


def esr_omega_of(x, g, K, zP, LS):
    D, a, q = R3.esr_series(x, g, K)
    return R3.esr_omega(D, a, q, zP, LS)


def rme_omega(x, g, zP, LS, full=False):
    """omega_R (T x K), float64, strictly causal. For t >= 2W - 1: reference = frames 0..h (h = t // 2, N = h + 1),
    M_j = min over the reference of the raw cosine c[., j]; window = frames t - W + 1..t (disjoint from the
    reference); clean = norm >= median of the reference norms and norm > 0; n_c clean frames; X_j = #clean window
    frames with c < M_j (strict). Active iff n_c >= 3, X_j >= 3, 2 X_j > n_c, zP > 0, LS > 1:
    p = prod_{i < X} (n_c - i) / (N + n_c - i), omega_R = min(1, e p). Everything else: omega_R = 1."""
    xb = np.asarray(x, np.float64)
    g64 = np.asarray(g, np.float64)
    T, K = xb.shape[0], g64.shape[0]
    nrm = np.linalg.norm(xb, axis=1)
    u = xb / np.maximum(nrm, 1e-12)[:, None]
    gn = g64 / np.maximum(np.linalg.norm(g64, axis=1, keepdims=True), 1e-12)
    c = u @ gn.T
    om = np.ones((T, K))
    aux = {"nc": np.zeros(T, np.int64), "X": None}
    if T == 0:
        return (om, aux) if full else om
    cm = np.minimum.accumulate(c, axis=0)
    Xs = np.zeros((T, K), np.int64) if full else None
    srt = []                                          # sorted reference norms 0..h (grown causally)
    h_in = -1
    for t in range(2 * W_RME - 1, T):
        h = t // 2
        while h_in < h:
            h_in += 1
            bisect.insort(srt, float(nrm[h_in]))
        N = h + 1
        med = float(np.median(srt))
        idx = np.arange(t - W_RME + 1, t + 1)
        clean = (nrm[idx] >= med) & (nrm[idx] > 0)
        nc = int(clean.sum())
        aux["nc"][t] = nc
        if nc < X_MIN:
            continue
        X = (c[idx[clean]] < cm[h][None, :]).sum(0)
        if full:
            Xs[t] = X
        act = (X >= X_MIN) & (2 * X > nc) & (zP[t] > 0) & (LS[t] > 1)
        ja = np.nonzero(act)[0]
        if ja.size:
            for j in ja:
                p = 1.0
                for i in range(int(X[j])):
                    p *= (nc - i) / (N + nc - i)
                om[t, j] = min(1.0, E_CONST * p)
    aux["X"] = Xs
    return (om, aux) if full else om


def esr_n(x, g, fe=None, omega=None, full=False):
    f = H._fe(x, g) if fe is None else fe
    K, zP, LP, LS, num, den = _base(x, g, f)
    om = esr_omega_of(x, g, K, zP, LS) if omega is None else omega
    numN = _numN(LP, LS, om)
    out = _omap(numN, den)
    return (out, {"omega": om, "numN": numN, "num": num, "den": den}) if full else out


def esrn_fco(x, g, fe=None, omega=None, A=None, full=False):
    f = H._fe(x, g) if fe is None else fe
    K, zP, LP, LS, num, den = _base(x, g, f)
    om = esr_omega_of(x, g, K, zP, LS) if omega is None else omega
    numN = _numN(LP, LS, om)
    w, r, _ = R4.fco_weights(x, g)
    v, _ = R4.fco_votes(x, g, r)
    C, M = R4.fco_state(w, v)
    out, aux = R4.fco_core(numN, den, C, M, K, A=A)
    return (out, {"omega": om, "numN": numN, "den": den, "C": C, "M": M, **aux}) if full else out


def esrn_rme(x, g, fe=None, omega_e=None, omega_r=None, full=False):
    f = H._fe(x, g) if fe is None else fe
    K, zP, LP, LS, num, den = _base(x, g, f)
    oe = esr_omega_of(x, g, K, zP, LS) if omega_e is None else omega_e
    orr = rme_omega(x, g, zP, LS) if omega_r is None else omega_r
    om = oe * orr
    out = _omap(_numN(LP, LS, om), den)
    return (out, {"omega_e": oe, "omega_r": orr, "omega": om}) if full else out


# =====================================================================================
# Harness entry
# =====================================================================================
def _all64(x, g):
    f = H._fe(x, g)
    K, zP, LP, LS, num, den = _base(x, g, f)
    oe = esr_omega_of(x, g, K, zP, LS)
    orr = rme_omega(x, g, zP, LS)
    return {"esr_n": esr_n(x, g, f, omega=oe), "esrn_fco": esrn_fco(x, g, f, omega=oe),
            "esrn_rme": esrn_rme(x, g, f, omega_e=oe, omega_r=orr), "hcp_ref": H.hcp(zP, f["zS"], K),
            "esr_ref": R3.esr_hcp(x, g, f), "fco_ref": R4.fco(x, g, f)}


def outputs_fn(x, g):
    if np.asarray(x).shape[0] == 0:                   # empty sequence: nothing to score (defined, not an error)
        return {k: np.zeros((0, np.asarray(g).shape[0]), np.float32) for k in ALL}
    out = {k: v.astype(np.float32) for k, v in _all64(x, g).items()}
    out["ospc_ref"] = np.asarray(H.OSPC(x, g), np.float32)
    return out


# =====================================================================================
# Random-data pre-run checks (no challenge data)
# =====================================================================================
def _rme_naive(x, g, zP, LS):
    """Per-frame, per-column loops from the definition; median by np.median of the reference slice; p by exact
    binomial coefficients."""
    xb = np.asarray(x, np.float64)
    g64 = np.asarray(g, np.float64)
    T, K = xb.shape[0], g64.shape[0]
    nrm = np.linalg.norm(xb, axis=1)
    om = np.ones((T, K))
    gj = [g64[j] / max(np.linalg.norm(g64[j]), 1e-12) for j in range(K)]
    cos = [[float((xb[s] / max(nrm[s], 1e-12)) @ gj[j]) for j in range(K)] for s in range(T)]
    for t in range(2 * W_RME - 1, T):
        h = t // 2
        med = np.median(nrm[: h + 1])
        win = [s for s in range(t - W_RME + 1, t + 1) if nrm[s] >= med and nrm[s] > 0]
        nc = len(win)
        if nc < X_MIN:
            continue
        for j in range(K):
            Mj = min(cos[s][j] for s in range(h + 1))
            X = sum(1 for s in win if cos[s][j] < Mj)
            if X >= X_MIN and 2 * X > nc and zP[t, j] > 0 and LS[t, j] > 1:
                p = math.comb(nc, X) / math.comb(h + 1 + nc, X)
                om[t, j] = min(1.0, E_CONST * p)
    return om


def _drift_stream(seed, T, K, tgt, s_hi, s_lo, t_drop, lo=16.0, hi=24.0):
    """Target of strength s_hi until t_drop, then s_lo (a clean, stable fall below the genuine level: the RME/ESR
    diffusion-like case), norms uniform in [lo, hi]."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(K, 512)).astype(np.float32)
    s = np.where(np.arange(T) < t_drop, s_hi, s_lo)[:, None]
    x = rng.normal(size=(T, 512)) + s * g[tgt][None].astype(np.float64)
    x = x / np.linalg.norm(x, axis=1, keepdims=True) * rng.uniform(lo, hi, size=(T, 1))
    return x.astype(np.float32), g


def synth():
    w = {}

    def upd(key, v):
        w[key] = max(w.get(key, 0.0), float(v))

    def cnt(key, v):
        w[key] = w.get(key, 0) + int(v)

    ok_d = {k: True for k in PRIMARIES}
    ok_shared = {"range": True}
    rows = 0
    for K in (200, 199):
        for kind in ("switch", "mng"):
            for seed, T in ((0, 300), (1, 257), (2, 64), (3, 760)):
                x, g = R4._synth_data(seed, T, K, kind)
                rows += T
                f = H._fe(x, g)
                K_, zP, LP, LS, num, den = _base(x, g, f)
                ref_h = H.hcp(zP, f["zS"], K)
                ref_e = R3.esr_hcp(x, g, f)
                ref_f = R4.fco(x, g, f)
                one = np.ones((T, K))
                # ---------- identities (bitwise, float64) ----------
                cnt("id_esrn_omega1_vs_hcp", (esr_n(x, g, f, omega=one) != ref_h).sum())
                cnt("id_numN_omega1_vs_hcp_num", (_numN(LP, LS, one) != num).sum())
                cnt("id_esrnfco_omega1_vs_fco", (esrn_fco(x, g, f, omega=one) != ref_f).sum())
                o_n, sn = esr_n(x, g, f, full=True)
                cnt("id_esrnfco_Ainf_vs_esrn", (esrn_fco(x, g, f, omega=sn["omega"], A=np.inf) != o_n).sum())
                cnt("id_esrnfco_both_neutral_vs_hcp", (esrn_fco(x, g, f, omega=one, A=np.inf) != ref_h).sum())
                cnt("id_esrnrme_omegaR1_vs_esrn",
                    (esrn_rme(x, g, f, omega_e=sn["omega"], omega_r=one) != o_n).sum())
                cnt("id_esrnrme_both1_vs_hcp", (esrn_rme(x, g, f, omega_e=one, omega_r=one) != ref_h).sum())
                om_e = esr_omega_of(x, g, K, zP, LS)
                cnt("id_esr_omega_vs_x_r3", (om_e != R3.esr_hcp(x, g, f, full=True)[1]["omega"]).sum())
                # ---------- monotonicity ----------
                upd("esrn_above_hcp", max(0.0, float((o_n - ref_h).max())))
                o_r, sr = esrn_rme(x, g, f, full=True)
                upd("esrnrme_above_esrn", max(0.0, float((o_r - o_n).max())))
                o_f = esrn_fco(x, g, f, omega=sn["omega"])
                upd("esrnfco_above_fco", max(0.0, float((o_f - ref_f).max())))
                # ESR-N <= ESR is not guaranteed entrywise (ESR's discounted competitors raise other columns, ESR's
                # own den also shrinks); report the share only
                w["esrn_le_esr_share"] = float(np.mean(o_n <= ref_e + 1e-12))
                cnt("esrn_rows_changed", (o_n != ref_h).any(1).sum())
                cnt("esrnrme_rows_changed_vs_esrn", (o_r != o_n).any(1).sum())
                cnt("rme_t_lt_39_active", (sr["omega_r"][: 2 * W_RME - 1] != 1.0).sum())
                upd("rme_omega_r_above_1", max(0.0, float(sr["omega_r"].max()) - 1.0))
                upd("rme_omega_r_below_0", max(0.0, -float(sr["omega_r"].min())))
                # ---------- naive RME (two short sequences per K) ----------
                if seed in (2, 1) and kind == "switch":
                    om_naive = _rme_naive(x, g, zP, LS)
                    upd("rme_naive_abs", float(np.abs(om_naive - sr["omega_r"]).max()))
                # ---------- range, finiteness, prefix invariance (float32 cast, as the harness) ----------
                full = outputs_fn(x, g)
                for k, ref in (("hcp_ref", ref_h), ("esr_ref", ref_e), ("fco_ref", ref_f)):
                    cnt(f"id_{k}_vs_parent_f32", (full[k] != ref.astype(np.float32)).sum())
                for hh in sorted({T // 2} | ({1, 17, 39, 40, T - 1} if seed == 1 else set())):
                    half = outputs_fn(x[:hh], g)
                    for k in ALL:
                        upd(f"iv_prefix_{k}", np.abs(half[k].astype(np.float64) - full[k][:hh].astype(np.float64)).max())
                for k in ALL:
                    v = full[k]
                    good = bool(v.shape == (T, K) and np.isfinite(v).all() and v.min() >= -1 and v.max() <= 1)
                    if k in PRIMARIES:
                        ok_d[k] &= good
                        sin = bool(np.abs(v.astype(np.float64)).max() < 1)
                        w[f"strict_inside_f32_{k}"] = bool(w.get(f"strict_inside_f32_{k}", True) and sin)
                    else:
                        ok_shared["range"] &= good
    # ---------- RME p against the exact binomial ratio ----------
    worst = 0.0
    for nc in range(3, 21):
        for X in range(3, nc + 1):
            for N in (20, 41, 100, 801, 5000):
                p = 1.0
                for i in range(X):
                    p *= (nc - i) / (N + nc - i)
                ex = math.comb(nc, X) / math.comb(N + nc, X)
                worst = max(worst, abs(p - ex) / ex)
    w["rme_p_vs_comb_rel"] = worst
    p = 1.0
    for i in range(4):
        p *= (6 - i) / (801 + 6 - i)
    w["rme_p_example_n6_X4_N801"] = p
    # ---------- mechanism (counts only): a clean fall below the genuine level after a stable genuine start ----------
    K = 200
    xd, gd = _drift_stream(21, 900, K, 7, 0.30, 0.10, 450)
    fd = H._fe(xd, gd)
    hd = H.hcp(fd["zP"], fd["zS"], K)
    nd = esr_n(xd, gd, fd)
    rd, srd = esrn_rme(xd, gd, fd, full=True)
    post = np.arange(900) >= 450
    w["drift_hcp_target_acc_post_far"] = int((hd[post, 7] >= THR_HCP).sum())
    w["drift_esrn_target_acc_post_far"] = int((nd[post, 7] >= THR_HCP).sum())
    w["drift_esrnrme_target_acc_post_far"] = int((rd[post, 7] >= THR_HCP).sum())
    w["drift_rme_active_post_frames"] = int((srd["omega_r"][post, 7] < 1).sum())
    w["drift_rme_active_pre_frames"] = int((srd["omega_r"][~post, 7] < 1).sum())
    xs, gs = R4._stream(22, 900, K, 7, 0.25)
    fs = H._fe(xs, gs)
    rs, srs = esrn_rme(xs, gs, fs, full=True)
    w["stable_rme_active_frames_target"] = int((srs["omega_r"][:, 7] < 1).sum())
    w["stable_esrnrme_vs_esrn_target_flips_far"] = int(((rs[:, 7] >= THR_HCP) != (esr_n(xs, gs, fs)[:, 7] >= THR_HCP)).sum())
    # ---------- extreme sequences through the whole pipeline ----------
    ex_ok = True
    rng = np.random.default_rng(11)
    for K in (200, 199):
        g = rng.normal(size=(K, 512)).astype(np.float32)
        seqs = []
        xx = rng.normal(size=(60, 512)).astype(np.float32)
        xx[:, :] = xx * rng.choice([1e-3, 1.0, 1e4], size=(60, 1)).astype(np.float32)
        seqs.append(xx)
        seqs.append(np.repeat(rng.normal(size=(1, 512)).astype(np.float32), 50, 0))
        seqs.append(rng.normal(size=(1, 512)).astype(np.float32) * 20)
        seqs.append(rng.normal(size=(2, 512)).astype(np.float32) * 20)
        z = rng.normal(size=(60, 512)).astype(np.float32) * 20
        z[0] = 0.0
        z[17] = 0.0
        z[45:50] = 0.0
        seqs.append(z)
        seqs.append(np.zeros((50, 512), np.float32))
        seqs.append(np.zeros((0, 512), np.float32))
        seqs.append(rng.normal(size=(80, 512)).astype(np.float32) * 60)
        for xs_ in seqs:
            try:
                o = outputs_fn(xs_, g)
                for k in ALL:
                    v = o[k]
                    good = bool(v.shape == (xs_.shape[0], K) and np.isfinite(v).all()
                                and (v.size == 0 or (v.min() >= -1 and v.max() <= 1)))
                    ex_ok &= good
                    if k in PRIMARIES:
                        ok_d[k] &= good
            except Exception as e:                    # an unhandled degenerate input is a synthetic failure
                ex_ok = False
                w.setdefault("extreme_seq_errors", []).append(f"{type(e).__name__}: {str(e)[:80]}")
    # ---------- fresh state per call ----------
    xa, ga = R4._stream(8, 220, 200, 3, 0.1)
    xb2, _ = R4._stream(9, 180, 200, 3, 0.1)
    o1 = outputs_fn(xa, ga)
    outputs_fn(xb2, ga)
    o3 = outputs_fn(xa, ga)
    w["fresh_state_mismatch"] = int(sum((o1[k] != o3[k]).sum() for k in ALL))
    # ---------- verdicts ----------
    zero = {"esr_n": ("id_esrn_omega1_vs_hcp", "id_numN_omega1_vs_hcp_num", "id_esr_omega_vs_x_r3"),
            "esrn_fco": ("id_esrnfco_omega1_vs_fco", "id_esrnfco_Ainf_vs_esrn", "id_esrnfco_both_neutral_vs_hcp"),
            "esrn_rme": ("id_esrnrme_omegaR1_vs_esrn", "id_esrnrme_both1_vs_hcp", "rme_t_lt_39_active")}
    tight = {"esr_n": ("esrn_above_hcp",), "esrn_fco": ("esrnfco_above_fco",),
             "esrn_rme": ("esrnrme_above_esrn", "rme_omega_r_above_1", "rme_omega_r_below_0", "rme_naive_abs")}
    for k in PRIMARIES:
        ok_d[k] &= all(w[n] == 0 for n in zero[k]) and all(w[n] <= 1e-12 for n in tight[k])
        ok_d[k] &= w[f"iv_prefix_{k}"] < 1e-6 and w[f"strict_inside_f32_{k}"]
    ok_d["esr_n"] &= w["esrn_rows_changed"] > 0
    ok_d["esrn_rme"] &= w["esrnrme_rows_changed_vs_esrn"] > 0 and w["drift_rme_active_post_frames"] > 0 \
        and w["stable_rme_active_frames_target"] == 0               # the RME checks are not vacuous
    ok_d["esrn_rme"] &= w["rme_p_vs_comb_rel"] <= 1e-12 and abs(w["rme_p_example_n6_X4_N801"] - 8.6e-10) < 0.05e-10
    shared = bool(ok_shared["range"] and ex_ok and w["fresh_state_mismatch"] == 0
                  and all(w[f"id_{k}_vs_parent_f32"] == 0 for k in ("hcp_ref", "esr_ref", "fco_ref"))
                  and all(w[f"iv_prefix_{k}"] < 1e-6 for k in REFS))
    out = {"synth": TAG, **w, "shared_ok": shared, "extreme_sequences_ok": ex_ok, "rows": rows,
           "design_ok": {k: bool(v) for k, v in ok_d.items()}}
    out["pass"] = bool(shared and all(ok_d.values()))
    print(json.dumps(out), flush=True)
    return out["pass"]


# =====================================================================================
# Label-free statistics; the RME kill rules are frozen and applied mechanically by --report
# =====================================================================================
STD_EDGES = [0, 10, 50, 200, 1000, np.inf]
STD_NAMES = ["t<10", "10-50", "50-200", "200-1000", ">=1000"]
QS = [0.01, 0.1, 0.5, 0.9, 0.99]


def _runs_ge(flag, L):
    """True if flag has a run of at least L consecutive True values."""
    if flag.size == 0:
        return False
    d = np.diff(np.r_[0, flag.astype(np.int8), 0])
    s, e = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return bool(((e - s) >= L).any())


def _stats_job(task):
    tr, i = task
    x, g = (cc.D["s1"][i], cc.D["g1"]) if tr == "t1" else (cc.D["s2"][i], cc.D["g2"])
    f = H._fe(x, g)
    K, zP, LP, LS, num, den = _base(x, g, f)
    T = x.shape[0]
    r = np.arange(T)
    lead = H.hcp(zP, f["zS"], K).argmax(1)              # HCP leader column (unadapted row), as in the ESR stats
    oe = esr_omega_of(x, g, K, zP, LS)
    orr, aux = rme_omega(x, g, zP, LS, full=True)
    fr = orr[r, lead] < 0.01
    fe_ = oe[r, lead] < 0.01
    fp = (oe * orr)[r, lead] < 0.01
    late = r >= 200
    res = {"n": T, "late": int(late.sum()), "flag_late": int((fr & late).sum()), "flagE_late": int((fe_ & late).sum()),
           "prod_late": int((fp & late).sum()), "both_late": int((fr & fe_ & late).sum()),
           "run20_late": _runs_ge(fr & late, 20),
           "flag_rel_late07": int((fr & late & (r >= 0.7 * T)).sum()),
           "clean_in_flag": float(aux["nc"][fr & late].sum()), "frames_flag": int((fr & late).sum()),
           "any_rme_active": int((orr < 1).sum()), "entries": T * K}
    return tr, res


def _phi(a, b, n11, n):
    """phi coefficient from counts: a = #A, b = #B, n11 = #(A and B), n = total."""
    n10, n01 = a - n11, b - n11
    n00 = n - a - b + n11
    den = math.sqrt(max(a * (n - a) * b * (n - b), 0))
    return None if den == 0 else (n11 * n00 - n10 * n01) / den


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
        for tr, rr in ex.map(_stats_job, tasks, chunksize=1):
            acc[tr].append(rr)
    out = {"stats": TAG, "note": ("label-free; leader = argmax of the unadapted HCP row; flag = omega_R < .01 at the "
                                  "leader column; late = t >= 200; the RME kill terms are evaluated here and applied "
                                  "mechanically by --report")}
    agg = {}
    for tr, rs in acc.items():
        s = {k: sum(r_[k] for r_ in rs) for k in ("late", "flag_late", "flagE_late", "prod_late", "both_late",
                                                  "flag_rel_late07", "clean_in_flag", "frames_flag",
                                                  "any_rme_active", "entries")}
        s["seq"] = len(rs)
        s["seq_run20_late"] = int(sum(r_["run20_late"] for r_ in rs))
        s["flag_share_late"] = s["flag_late"] / max(s["late"], 1)
        s["flagE_share_late"] = s["flagE_late"] / max(s["late"], 1)
        s["prod_share_late"] = s["prod_late"] / max(s["late"], 1)
        s["seq_run20_share"] = s["seq_run20_late"] / max(s["seq"], 1)
        s["clean_share_in_flag_windows"] = s["clean_in_flag"] / max(W_RME * s["frames_flag"], 1)
        s["late07_share_of_flags"] = s["flag_rel_late07"] / max(s["frames_flag"], 1)
        s["phi_flagR_flagE_late"] = _phi(s["flag_late"], s["flagE_late"], s["both_late"], s["late"])
        s["rme_active_entry_share"] = s["any_rme_active"] / max(s["entries"], 1)
        agg[tr] = s
    t1, t2 = agg["t1"], agg["t2"]
    ratio = t2["flag_share_late"] / t1["flag_share_late"] if t1["flag_share_late"] > 0 else float("inf")
    kill = {"t1_flag_share>.015": (t1["flag_share_late"], RME_KILL["t1_flag_share_max"],
                                   t1["flag_share_late"] > RME_KILL["t1_flag_share_max"]),
            "t2_t1_ratio<3": (ratio, RME_KILL["t2_t1_ratio_min"], ratio < RME_KILL["t2_t1_ratio_min"]),
            "t1_seq_run20_share>.05": (t1["seq_run20_share"], RME_KILL["t1_run_seq_share_max"],
                                       t1["seq_run20_share"] > RME_KILL["t1_run_seq_share_max"]),
            "t2_clean_share<.5": (t2["clean_share_in_flag_windows"], RME_KILL["t2_clean_share_min"],
                                  t2["frames_flag"] > 0 and t2["clean_share_in_flag_windows"] < RME_KILL["t2_clean_share_min"]),
            "t2_late07_share<.6": (t2["late07_share_of_flags"], RME_KILL["t2_late_share_min"],
                                   t2["late07_share_of_flags"] < RME_KILL["t2_late_share_min"]),
            "t2_phi>=.6": (t2["phi_flagR_flagE_late"], RME_KILL["t2_phi_max"],
                           t2["phi_flagR_flagE_late"] is not None and t2["phi_flagR_flagE_late"] >= RME_KILL["t2_phi_max"]),
            "t1_prod_share>.03": (t1["prod_share_late"], RME_KILL["t1_prod_share_max"],
                                  t1["prod_share_late"] > RME_KILL["t1_prod_share_max"])}
    out.update({"t1": t1, "t2": t2, "rme_kill_terms": {k: [None if v[0] is None else float(v[0]), v[1], bool(v[2])]
                                                       for k, v in kill.items()},
                "rme_killed": bool(any(v[2] for v in kill.values()))})
    print(json.dumps(out), flush=True)


# =====================================================================================
# Post-run report (evaluator side): frozen gate applied once, mechanically
# =====================================================================================
TOPK = R2.TOPK
FULL = ALL
PROX = ALL + ["955399"]
A399 = R2.A399
D_STAR, B_STAR = R2.D_STAR, R2.B_STAR
BE_A, BE_B, BE_LOO_MAE = R2.BE_A, R2.BE_B, R2.BE_LOO_MAE
HCP_EER, OSPC_EER, EER_REPRO_TOL = R2.HCP_EER, R2.OSPC_EER, R2.EER_REPRO_TOL
HCP_RECORDED = dict(R2.HCP_RECORDED, thr=THR_HCP)
RUN_REPRO, CAL_MAP, CHECKS_EARLY = R2.RUN_REPRO, R2.CAL_MAP, R2.CHECKS_EARLY
_tol_ok, _t, _top3_share = R2._tol_ok, R2._t, R2._top3_share
_seq_errors, _net_block = R4._seq_errors, R4._net_block


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
    rows, chk, syn, stt = {}, {}, None, None
    for line in open(run_log):
        if line.startswith("{\"setting\": "):
            r_ = json.loads(line)
            rows[r_["setting"]] = r_
        elif line.startswith("{\"check\": "):
            c_ = json.loads(line)["check"]
            chk[c_["setting"]] = c_["checks_pass"]
        elif line.startswith("{\"synth\": "):
            syn = json.loads(line)
        elif line.startswith("{\"stats\": "):
            stt = json.loads(line)
    calrows = {}
    for line in open(cal_log):
        if line.startswith("{\"method\": "):
            r_ = json.loads(line)
            calrows[r_["method"]] = r_
    cal.DF_LEVELS = ()
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
    # ---- full-precision T1 (hcp_ref first: its EER threshold and per-sequence errors define NET) ----
    t1rep = {}
    thr_ref, ge_ref, ie_ref, worst = None, None, None, None
    for k in ["hcp_ref"] + [s for s in FULL if s != "hcp_ref"]:
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
        if k == "hcp_ref":
            thr_ref = met["eer_thr"]
        ge, ie = _seq_errors(outs_, t1, thr_ref)
        if k == "hcp_ref":
            ge_ref, ie_ref = ge, ie
            worst = np.argsort(-ie_ref, kind="stable")
        rb = ar._err_bins(outs_, t1, thr_ref)
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
            "hcp_ref_eer_thr_bins": {"gen": [r_["gen_err"] for r_ in rb["by_pos"]],
                                     "imp": [r_["imp_err"] for r_ in rb["by_pos"]]},
            "gen_err_t_ge10_at_eer_thr": int(sum(bins[b]["gen_err"] for b in ar.POS_NAMES if b not in CHECKS_EARLY)),
            "top3_gen_err_at_eer_thr_all": _top3_share(outs_, t1, met["eer_thr"], 0),
            "at_hcp_ref_eer_thr": _net_block(ge, ie, ge_ref, ie_ref, worst)}
    # ---- FCO stacking on ESR-N: impostor entries esr_n accepts and esrn_fco rejects at esr_n's own EER cut ----
    cut_n = t1rep["esr_n"]["eer_thr"]
    fl = np.zeros(n1, np.int64)
    gl = 0
    for i in range(n1):
        a_n = F[i]["esr_n"] >= cut_n
        a_f = F[i]["esrn_fco"] >= cut_n
        tt = t1[i]
        flip = a_n & ~a_f
        gl += int(flip[:, tt].sum())
        flip[:, tt] = False
        fl[i] = int(flip.sum())
    wfl = int(np.argmax(fl)) if n1 else 0
    fco_stack = {"cut": cut_n, "imp_flips": int(fl.sum()), "gen_flips": gl,
                 "outside_worst_share": (float(1.0 - fl[wfl] / fl.sum()) if fl.sum() > 0 else 0.0),
                 "worst_seq_flips": int(fl[wfl]), "worst_seq_is_hcp_worst": bool(wfl == int(worst[0]))}
    out["t1"] = t1rep
    out["fco_stack_on_esrn"] = fco_stack
    out["hcp_ref_eer_thr_used_for_net"] = thr_ref
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
                                abs(t1rep["ospc_ref"]["eer_full"] - OSPC_EER) <= EER_REPRO_TOL),
                 "esr_ref": _t(t1rep["esr_ref"]["eer_full"], ESR_EER,
                               abs(t1rep["esr_ref"]["eer_full"] - ESR_EER) <= EER_REPRO_TOL),
                 "fco_ref": _t(t1rep["fco_ref"]["eer_full"], FCO_EER,
                               abs(t1rep["fco_ref"]["eer_full"] - FCO_EER) <= EER_REPRO_TOL)}
    hcp_thr_b = _t(prox["hcp_ref"]["thr"], HCP_RECORDED["thr"], abs(prox["hcp_ref"]["thr"] - HCP_RECORDED["thr"]) <= 5e-6)
    hcp_B = _t(prox["hcp_ref"][B_STAR], HCP_RECORDED["t2_cp_reject_pt"],
               abs(prox["hcp_ref"][B_STAR] - HCP_RECORDED["t2_cp_reject_pt"]) <= 2e-5)
    stop = {"ref_eer_repro_1e-7_ok": all(v[2] for v in eer_repro.values()),
            "hcp_ref_thr_and_Bstar_repro_ok": bool(hcp_thr_b[2] and hcp_B[2]),
            "run_repro_ok": all(t_[2] for d in rr.values() for t_ in d.values()),
            "cal_repro_ok": bool(cr) and all(cr[s] for s in CAL_MAP) and
            all(t_[2] for d in cr.values() for t_ in d.values()),
            "synth_line_present_and_shared_checks_ok": bool(syn and syn.get("shared_ok")),
            "stats_line_present": stt is not None,
            "refs_checks_ok": all(bool(chk.get(k)) and bad[k] == 0 for k in REFS)}
    global_stop = not all(stop.values())
    design_stop = {k: {"checks_pass_and_finite_in_range": bool(chk.get(k)) and bad[k] == 0,
                       "eer_full_report_vs_row_ok": bool(t1rep[k]["eer_full_match_1e-6"]),
                       "structural_identities_ok_in_synth_line": bool(syn and syn.get("design_ok", {}).get(k))}
                   for k in PRIMARIES}
    out2 = {"report_gate": TAG, "run_repro": rr, "cal_repro": cr, "ref_eer_repro": eer_repro,
            "hcp_ref_thr_repro": hcp_thr_b, "hcp_ref_Bstar_repro": hcp_B,
            "stop_checks_global": stop, "stop_checks_per_design": design_stop}
    # ---- frozen gate (applied once, mechanically; same-run references) ----
    fbe = {k: BE_A + BE_B * prox[k][B_STAR] for k in PROX}
    E = {k: t1rep[k]["eer_full"] for k in FULL}
    rme_killed = bool(stt and stt.get("rme_killed"))

    def base_terms(k):
        return {"checks_pass": _t(chk.get(k), True, bool(chk.get(k))),
                "t1_neg1_share<=.001": _t(t1rep[k]["t1_neg1_share"], 0.001, t1rep[k]["t1_neg1_share"] <= 0.001)}
    gate = {
        "esr_n": {"eer_full<=esr_ref": _t(E["esr_n"], E["esr_ref"], E["esr_n"] <= E["esr_ref"]),
                  "BE_fc<=BE_fc(esr_ref)+.005": _t(fbe["esr_n"], fbe["esr_ref"] + GATE_N_BE,
                                                  fbe["esr_n"] <= fbe["esr_ref"] + GATE_N_BE),
                  **base_terms("esr_n")},
        "esrn_fco": {"eer_full<=esr_n-.00004": _t(E["esrn_fco"], E["esr_n"] - GATE_FCO_STACK,
                                                 E["esrn_fco"] <= E["esr_n"] - GATE_FCO_STACK),
                     "eer_full<=hcp_ref-.00005": _t(E["esrn_fco"], E["hcp_ref"] - GATE_FCO_HCP,
                                                   E["esrn_fco"] <= E["hcp_ref"] - GATE_FCO_HCP),
                     "BE_fc<=BE_fc(esr_ref)+.005": _t(fbe["esrn_fco"], fbe["esr_ref"] + GATE_N_BE,
                                                     fbe["esrn_fco"] <= fbe["esr_ref"] + GATE_N_BE),
                     "fco_flips_on_esrn>=6000": _t(fco_stack["imp_flips"], FCO_FLIPS_MIN,
                                                   fco_stack["imp_flips"] >= FCO_FLIPS_MIN),
                     "fco_flips_outside_worst>=.40": _t(fco_stack["outside_worst_share"], FCO_OUTSIDE_MIN,
                                                        fco_stack["outside_worst_share"] >= FCO_OUTSIDE_MIN),
                     **base_terms("esrn_fco")},
        "esrn_rme": {"rme_premise_not_killed": _t(not rme_killed, True, not rme_killed),
                     "eer_full<=esr_ref": _t(E["esrn_rme"], E["esr_ref"], E["esrn_rme"] <= E["esr_ref"]),
                     "eer_full<=esr_n+.00002": _t(E["esrn_rme"], E["esr_n"] + GATE_RME_EER,
                                                 E["esrn_rme"] <= E["esr_n"] + GATE_RME_EER),
                     "BE_fc<=BE_fc(esr_n)-.005": _t(fbe["esrn_rme"], fbe["esr_n"] - GATE_RME_BE,
                                                   fbe["esrn_rme"] <= fbe["esr_n"] - GATE_RME_BE),
                     **base_terms("esrn_rme")}}
    verdict = {}
    for k in PRIMARIES:
        if global_stop:
            verdict[k] = "STOP (global, no verdict): " + ", ".join(s for s, v in stop.items() if not v)
        elif not all(design_stop[k].values()):
            verdict[k] = "STOP (design, no verdict): " + ", ".join(s for s, v in design_stop[k].items() if not v)
        else:
            failed = [s for s, v in gate[k].items() if not v[2]]
            verdict[k] = "PASS" if not failed else "FAIL: " + ", ".join(failed)
    markers = {"esr_n_upload_worthy_eer<=.00316": E["esr_n"] <= UPLOAD_BAR,
               **{f"{k}_eer<fco_ref": bool(E[k] < E["fco_ref"]) for k in PRIMARIES},
               **{f"{k}_eer<hcp_ref": bool(E[k] < E["hcp_ref"]) for k in PRIMARIES}}

    def ols(px, tgt):
        ms = [m for m in calrows if calrows[m].get("official", {}).get(tgt) is not None
              and calrows[m].get(px) is not None]
        b, a = np.polyfit([calrows[m][px] for m in ms], [calrows[m]["official"][tgt] for m in ms], 1)
        return float(a), float(b), len(ms)
    fc = {}
    for tgt, px in (("diff_aar", D_STAR), ("it_aar", "usp_aar")):
        a, b, n = ols(px, tgt)
        fc[tgt] = {"proxy": px, "a": a, "b": b, "n": n, **{k: round(a + b * prox[k][px], 4) for k in PROX}}
    rep_only = {}
    for k in ALL:
        pk = prox[k]
        rep_only[k] = {"eer_full": E[k], "eer_full_minus_hcp_ref": E[k] - E["hcp_ref"],
                       "eer_full_minus_esr_ref": E[k] - E["esr_ref"], "frr_far1e5": t1rep[k]["frr_far1e5"],
                       "far_at_frr1e5": t1rep[k]["far_at_frr1e5"], "usp_aar": pk["usp_aar"], "usp_run": pk["usp_run"],
                       "usp_frr": pk["usp_frr"], "t1_band_reject": pk["t1_band_reject"],
                       "t2_pre_reject_pt": pk["t2_pre_reject_pt"], "D*(" + D_STAR + ")": pk[D_STAR],
                       "B*(" + B_STAR + ")": pk[B_STAR], "BE_fc": round(fbe[k], 4),
                       "own_eer_thr": t1rep[k]["eer_thr"], "own_far1e5_thr": t1rep[k]["far1e5_thr"],
                       "own_far1e5_odds": (float(((1 + t1rep[k]["far1e5_thr"]) / (1 - t1rep[k]["far1e5_thr"])) ** 2)
                                           if t1rep[k]["far1e5_thr"] < 1 else None),
                       "eer_thr_bins": t1rep[k]["eer_thr_bins"], "hcp_ref_eer_thr_bins": t1rep[k]["hcp_ref_eer_thr_bins"],
                       "far1e5_thr_bins": t1rep[k]["far1e5_thr_bins"],
                       "gen_err_t_ge10_at_eer_thr": t1rep[k]["gen_err_t_ge10_at_eer_thr"],
                       "top3_gen_err_at_eer_thr_all": t1rep[k]["top3_gen_err_at_eer_thr_all"],
                       "at_hcp_ref_eer_thr": t1rep[k]["at_hcp_ref_eer_thr"]}
    # ESR's original Joint-gate terms, printed for disclosure; expected to FAIL by structure (not gate terms here)
    esr_orig = {k: {"usp_frr<=usp_frr(hcp_ref)+.010": _t(prox[k]["usp_frr"], prox["hcp_ref"]["usp_frr"] + 0.010,
                                                        prox[k]["usp_frr"] <= prox["hcp_ref"]["usp_frr"] + 0.010),
                    "t1_band_reject(hcp_ref)": prox["hcp_ref"]["t1_band_reject"], "t1_band_reject": prox[k]["t1_band_reject"]}
                for k in PRIMARIES + ["esr_ref"]}
    out2.update({"gate": gate, "verdict": verdict, "markers_report_only": markers,
                 "rme_premise_from_stats": None if stt is None else {"killed": rme_killed, "terms": stt.get("rme_kill_terms")},
                 "be_forecast": {k: round(fbe[k], 4) for k in PROX}, "forecasts_report_only": fc,
                 "report_only": rep_only, "esr_original_joint_terms_report_only": esr_orig,
                 "fco_stack_on_esrn": fco_stack, "pos_bins": ar.POS_NAMES,
                 "note": ("A pass is a candidate for the user's decision, not a submission; no artifact is built. "
                          "BE_fc cannot resolve BE differences under about .016; official BE decides.")})
    print(json.dumps(out2), flush=True)
    print(json.dumps({"verdict": verdict, "markers": markers}), flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--synth":
        sys.exit(0 if synth() else 1)
    elif arg == "--stats":
        stats()
    elif arg == "--report":
        report(sys.argv[2], sys.argv[3])
    else:
        cc.run_check(outputs_fn, ALL, TAG, workers=6)
