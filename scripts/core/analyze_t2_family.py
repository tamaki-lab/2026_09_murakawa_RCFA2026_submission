"""T2 family stage 0: fixed variants of the PEZ-NH defence, judged label-free before any upload (analysis only).

Plan, provenance and frozen rules: docs/experiments/20261009_ogata_t2_family.md.

Shared float64 internals per sequence (frozen research code): front end f, omega_E, omega_V, the PEZ energy ratio s2,
FCO state, raw cosine r, verification v (row z >= 4.265), clean = |x| >= 20, hist (verified clean history).
PEZ form for any ratio s2c in [0, 1]: shrink = s2c / (1 + (1 - s2c) zS^2 / (K - 1)), applied where
act(s2c) = (s2c < 1) & (LS > 1) & (zP > 0) & (omega_V == 1). PEZ-NH = omega_E omega_V shrink(s2) on act(s2).
Every variant is Phi(numN(omega) + F, den + G) with one changed omega (or, for lx_nh, one changed FCO denominator).
Quality EMA of a frame cue q: q~_0 = q_0, q~_t = (1 - al_t) q~_{t-1} + al_t q_t, al_t = min(1, .05 (|x_t| / 20)^4)
(the rate of PEZ's S state); ratio(q) = min(1, q~_t / mean_{s <= t} q~_s) (1 if the mean is not positive).

  D (energy reference / source; replace PEZ-NH's s2 unless noted)
    pxm_hist   PEZ (hist gate) with the max-prefix reference (base PEZ)
    slowref_nh 959765's reference: prefix mean through 45 frames, then ref = .978 ref + .022 e_t
    ts_nh      two-sided (949766): s2 = min(e / E, E / e), E = prefix mean
    med_nh     prefix median reference
    hold_nh    reference = mean of e over frames whose e was >= the reference when they arrived
    ver_nh     reference = mean of e over frames with a verified clean column
    dframe_nh  stacked: PEZ-NH times the PEZ form of the frame-row energy ratio (memoryless row Y_t)
    damp_nh    amplitude cue: s2 = ratio of V_t = EMA(|Y_t|^2) (prefix-mean form)
    dcoh_nh    coherence cue: s2 = ratio of e_t / V_t (prefix-mean form)
  F (gates and closed-set channels)
    lx_nh      PEZ-NH with the current raw-cosine winner exempt from the FCO cross term in its denominator
    owc_nh     stacked Beta-Laplace survival whose failures are clean frames where another column verifies
    hi_nh      energy discount requires verified history only for the memory leader argmax_j zP
    novg_nh    PEZ-NH without the omega_V == 1 gate;   novg_pez  PEZ without it (base PEZ)
    rel_nh     energy discount released on frames where the column itself verifies on a clean frame
    rme_nh     stacked x_r5.rme_omega
  E (new cues; stacked PEZ-form factor on PEZ-NH; per column where noted)
    prof_nh    per column j: corr over k != j of the EMA raw-cosine row and the past mean row (prefix-mean form)
    cont_nh    corr of consecutive raw-cosine rows, quality EMA, ratio
    rtpl_nh    per column j: cos(EMA of the unit gallery-orthogonal residual, sum of residuals of past verified
               clean frames of j), prefix-mean form over defined frames
    disp_nh    |u_t - u_{t-1}|^2 (frame motion), quality EMA, ratio   (lower motion than own history)
    norm_nh    |x_t|^2, quality EMA, ratio
    span_nh    in-span energy fraction, quality EMA, ratio
    fplo_nh    L1 / (L2 sqrt d) of x_t, quality EMA, ratio;  fphi_nh  its reciprocal
Pseudo targets on the evaluator side only. Prints JSON aggregates only.
Usage (server): python scripts/core/analyze_t2_family.py [workers<=6] > results/ogata/t2_family.log
                python scripts/core/analyze_t2_family.py --synth
"""
import bisect
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
import analyze_pxm_stage0 as PX  # noqa: E402
import analyze_t1_drp_cla as DC  # noqa: E402
from cubic_fast import _prep  # noqa: E402
from rcfa.methods.gallery_discriminative_readout import discriminative_mapping  # noqa: E402
from rcfa.methods.gallery_subspace_residual import GallerySubspace  # noqa: E402
from rcfa.methods.gallery_discriminative_row_l2 import gallery_discriminative_row_l2_scores  # noqa: E402

NH = DC.NH
PEZ_PROXY = {"eer_full": 0.0030089891830279987, "far1e5_thr": 0.8431214690208435,
             "frr_far1e5": 0.22062556459821045}
SLOW_PREFIX, SLOW_ALPHA = 45, 0.022
D_VARS = ["pxm_hist", "slowref_nh", "ts_nh", "med_nh", "hold_nh", "ver_nh", "dframe_nh", "damp_nh", "dcoh_nh"]
F_VARS = ["lx_nh", "owc_nh", "hi_nh", "novg_nh", "novg_pez", "rel_nh", "rme_nh"]
E_VARS = ["prof_nh", "cont_nh", "rtpl_nh", "disp_nh", "norm_nh", "span_nh", "fplo_nh", "fphi_nh"]
VARS = D_VARS + F_VARS + E_VARS
BASE = {"pxm_hist": "pez", "novg_pez": "pez"}           # every other variant is compared with PEZ-NH
DISCOUNT_ONLY = ["pxm_hist", "dframe_nh", "owc_nh", "novg_nh", "novg_pez", "rme_nh"] + E_VARS
R_CAL = (0.25627, 1.2)                                 # T2/T1 genuine FRR ratios: BLVS->PEZ and PEZ->PEZ-NH (official)
GATES = {"eer_max_vs_base": 2e-5, "thr_min": 0.8381215, "frr_max": 0.30, "dbe_opt_max": -0.002}
FAMILY = {**{k: "D" for k in D_VARS}, **{k: "F" for k in F_VARS}, **{k: "E" for k in E_VARS}}
REPRO_NDIFF, REPRO_MAX = 10, 1e-6
NEAR = 50
D = {}


# ---------------------------------------------------------------- shared helpers
def shrink(s2c, zS, K):
    s2c = np.asarray(s2c, np.float64)
    s = s2c[:, None] if s2c.ndim == 1 else s2c
    return s / (1.0 + (1.0 - s) * zS ** 2 / (K - 1))


def act_of(s2c, LS, zP, oV):
    s2c = np.asarray(s2c, np.float64)
    low = (s2c[:, None] if s2c.ndim == 1 else s2c) < 1
    return low & (LS > 1) & (zP > 0) & (oV == 1.0)


def oen(s2c, zS, K, act):
    return np.where(act, shrink(s2c, zS, K), 1.0)


def q_ema(q, al):
    """Quality EMA along axis 0 (al_0 treated as 1, as in cubic_fast.ema_cubic_raw)."""
    q = np.asarray(q, np.float64)
    out = np.empty_like(q)
    if q.shape[0] == 0:
        return out
    out[0] = q[0]
    for t in range(1, q.shape[0]):
        out[t] = (1.0 - al[t]) * out[t - 1] + al[t] * q[t]
    return out


def ratio_prefix(qs, defined=None):
    """min(1, qs_t / mean_{s <= t} qs_s) along axis 0; 1 where the mean is not positive or qs is undefined."""
    qs = np.asarray(qs, np.float64)
    if defined is None:
        cnt = np.arange(1, qs.shape[0] + 1, dtype=np.float64)
        cnt = cnt[:, None] if qs.ndim == 2 else cnt
        mean = np.cumsum(qs, axis=0) / cnt
        ok = mean > 0
    else:
        cnt = np.cumsum(defined, axis=0).astype(np.float64)
        mean = np.cumsum(np.where(defined, qs, 0.0), axis=0) / np.maximum(cnt, 1.0)
        ok = defined & (mean > 0)
    return np.where(ok, np.minimum(1.0, qs / np.where(ok, mean, 1.0)), 1.0)


def row_energy(rows):
    """sum_j rows_tj^2, sequential column order (the PEZ energy arithmetic)."""
    R = np.asarray(rows, np.float64)
    e = np.zeros(R.shape[0])
    for j in range(R.shape[1]):
        e += R[:, j] * R[:, j]
    return e


def s2_ref(e, ref):
    return np.where(ref > 0, np.minimum(1.0, e / np.where(ref > 0, ref, 1.0)), 1.0)


def ref_prefix_mean(e):
    out = np.empty_like(e)
    E = np.float64(0.0)
    for t in range(e.shape[0]):
        E = np.float64(E + e[t])
        out[t] = E / np.float64(t + 1)
    return out


def ref_slow(e):
    out = np.empty_like(e)
    E, V = np.float64(0.0), np.float64(0.0)
    for t in range(e.shape[0]):
        E = np.float64(E + e[t])
        if t + 1 <= SLOW_PREFIX:
            V = E / np.float64(t + 1)
        else:
            V = np.float64(np.float64((1.0 - SLOW_ALPHA) * V) + np.float64(SLOW_ALPHA * e[t]))
        out[t] = V
    return out


def ref_median(e):
    out = np.empty_like(e)
    srt = []
    for t in range(e.shape[0]):
        bisect.insort(srt, float(e[t]))
        n = len(srt)
        out[t] = srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
    return out


def ref_hold(e):
    out = np.empty_like(e)
    sh, n, ref = 0.0, 0, 0.0
    for t in range(e.shape[0]):
        if n == 0 or e[t] >= ref:
            sh += float(e[t])
            n += 1
            ref = sh / n
        out[t] = ref
    return out


def ref_ver(e, anyver):
    out = np.zeros_like(e)
    sv, n = 0.0, 0
    for t in range(e.shape[0]):
        if anyver[t]:
            sv += float(e[t])
            n += 1
        out[t] = sv / n if n else 0.0
    return out


def s2_two_sided(e):
    E = ref_prefix_mean(e)
    lo = np.where(E > 0, e / np.where(E > 0, E, 1.0), 1.0)
    hi = np.where(e > 0, E / np.where(e > 0, e, 1.0), 0.0)
    return np.where(E > 0, np.minimum(lo, hi), 1.0)


def survival_other(v, clean, zP, LS):
    """BLVS recursion with failures = clean frames where the column is unverified and another column verifies."""
    T, K = v.shape
    A, B, n, p = np.zeros(K), np.zeros(K), np.zeros(K), np.ones(K)
    om = np.ones((T, K))
    for t in range(T):
        if clean[t]:
            vt = v[t]
            other = vt.sum() - vt > 0
            fail = ~vt & other & (A >= 1)
            B = np.where(vt, B + n, B)
            A = np.where(vt, A + 1, A)
            p = np.where(vt, 1.0, np.where(fail, p * (B + 1 + n) / (A + B + 2 + n), p))
            n = np.where(vt, 0.0, np.where(fail, n + 1, n))
        actv = (n >= 1) & (zP[t] > 0) & (LS[t] > 1)
        om[t] = np.where(actv, np.minimum(1.0, math.e * p), 1.0)
    return om


def fco_core_lx(num, den, C, M, K, lead):
    """x_r4.fco_core with the cross term sum_{k != j} F_k dropped from the denominator of lead columns."""
    A = (K + 1) / 2.0
    e = np.maximum(M - A, 0.0)
    share = np.zeros_like(C)
    np.divide(C, M[:, None], out=share, where=(M > 0)[:, None])
    F = e[:, None] * share
    F0 = e / K
    pre = np.zeros_like(F)
    suf = np.zeros_like(F)
    pre[:, 1:] = np.cumsum(F, 1)[:, :-1]
    suf[:, :-1] = np.cumsum(F[:, ::-1], 1)[:, ::-1][:, 1:]
    G = F0[:, None] + np.where(lead, 0.0, pre + suf)
    return R4._omap(num + F, den + G)


def _corr_excl(m, h):
    """Pearson correlation over k != j of rows m and h, for every column j (T x K); nan where undefined."""
    K = m.shape[1]
    n = K - 1.0
    Sm, Sh = m.sum(1, keepdims=True), h.sum(1, keepdims=True)
    Smm, Shh, Smh = (m * m).sum(1, keepdims=True), (h * h).sum(1, keepdims=True), (m * h).sum(1, keepdims=True)
    sm, sh = (Sm - m) / n, (Sh - h) / n
    cov = (Smh - m * h) / n - sm * sh
    vm = (Smm - m * m) / n - sm * sm
    vh = (Shh - h * h) / n - sh * sh
    ok = (vm > 0) & (vh > 0)
    return np.where(ok, cov / np.sqrt(np.where(ok, vm * vh, 1.0)), np.nan)


def _corr_rows(m, h):
    mc = m - m.mean(1, keepdims=True)
    hc = h - h.mean(1, keepdims=True)
    den = np.sqrt((mc * mc).sum(1) * (hc * hc).sum(1))
    return np.where(den > 0, (mc * hc).sum(1) / np.where(den > 0, den, 1.0), np.nan)


# ---------------------------------------------------------------- per-sequence variants
def parts(x, g):
    f, oE, oV, oEn, alt, _, s2 = RS.states(x, g)
    K, zP, LP, LS, num, den = R5._base(x, g, f)
    w, rn, inspan = R4.fco_weights(x, g)
    vf, _ = R4.fco_votes(x, g, rn)
    C, M = R4.fco_state(w, vf)
    v, _, rc = HV.verify(x, g)
    xb = np.asarray(x, np.float64)
    nrm = np.linalg.norm(xb, axis=1)
    clean = nrm >= BL.CLEAN
    hist = np.cumsum(v & clean[:, None], axis=0) >= 1
    al = np.minimum(1.0, 0.05 * (nrm / 20.0) ** 4)
    return dict(f=f, oE=oE, oV=oV, oEn=oEn, oEn_nh=alt["hist"], s2=s2, K=K, zP=zP, LP=LP, LS=LS, num=num, den=den,
                C=C, M=M, vf=vf, inspan=inspan, v=v, rc=rc, xb=xb, nrm=nrm, clean=clean, hist=hist, al=al)


def frame_rows(x, g):
    """Memoryless evidence row Y_t: the S readout with the EMA rate fixed at 1 (current frame only)."""
    u, a, c = _prep(x, g)
    Mt, beta, _ = discriminative_mapping(c)
    yr = np.tanh(((u @ c.T) ** 3) @ Mt.T + beta)
    return (yr - yr.mean(1, keepdims=True)) / 2


def cue_s2(P, x, g, which):
    """Ratio s2 (row vector T or matrix T x K) for one cue."""
    al, nrm = P["al"], P["nrm"]
    if which == "prof":
        rc = P["rc"]
        T = rc.shape[0]
        sm = q_ema(rc, al)
        past = np.zeros_like(rc)
        if T > 1:
            past[1:] = np.cumsum(rc, axis=0)[:-1] / np.arange(1, T)[:, None]
        q = _corr_excl(sm, past)
        q[0] = np.nan
        dfn = np.isfinite(q)
        return ratio_prefix(np.maximum(np.nan_to_num(q, nan=0.0), 0.0), defined=dfn)
    if which == "cont":
        rc = P["rc"]
        q = np.full(rc.shape[0], np.nan)
        if rc.shape[0] > 1:
            q[1:] = _corr_rows(rc[1:], rc[:-1])
        q = np.maximum(np.nan_to_num(q, nan=1.0), 0.0)
        return ratio_prefix(q_ema(q, al))
    if which == "rtpl":
        xb = P["xb"]
        u = xb / np.maximum(nrm, 1e-12)[:, None]
        B = np.asarray(GallerySubspace(np.asarray(g, np.float32)).basis, np.float64)
        rr = u - (u @ B.T) @ B
        rn_ = np.linalg.norm(rr, axis=1)
        rh = np.where(rn_[:, None] > 0, rr / np.maximum(rn_, 1e-300)[:, None], 0.0)
        R = q_ema(rh, al)
        T, K = P["v"].shape
        q = np.zeros((T, K))
        dfn = np.zeros((T, K), bool)
        tmpl = {}
        vc = P["v"] & P["clean"][:, None]
        for t in range(T):
            Rn = np.linalg.norm(R[t])
            for j, s in tmpl.items():
                sn = np.linalg.norm(s)
                if Rn > 0 and sn > 0:
                    q[t, j] = max(0.0, float(R[t] @ s) / (Rn * sn))
                    dfn[t, j] = True
            for j in np.nonzero(vc[t])[0]:
                tmpl[int(j)] = tmpl.get(int(j), 0.0) + rh[t]
        return ratio_prefix(q, defined=dfn)
    if which == "disp":
        u = P["xb"] / np.maximum(nrm, 1e-12)[:, None]
        q = np.zeros(u.shape[0])
        if u.shape[0] > 1:
            d = u[1:] - u[:-1]
            q[1:] = (d * d).sum(1)
        return ratio_prefix(q_ema(q, al))
    if which == "norm":
        return ratio_prefix(q_ema(nrm * nrm, al))
    if which == "span":
        return ratio_prefix(q_ema(P["inspan"], al))
    if which in ("fplo", "fphi"):
        xb = P["xb"]
        l1 = np.abs(xb).sum(1)
        l2 = np.sqrt((xb * xb).sum(1)) * math.sqrt(xb.shape[1])
        q = np.where(l2 > 0, l1 / np.where(l2 > 0, l2, 1.0), 1.0)
        if which == "fphi":
            q = np.where(q > 0, 1.0 / np.where(q > 0, q, 1.0), 1.0)
        return ratio_prefix(q_ema(q, al))
    raise KeyError(which)


def _table(x, g, P):
    """name -> zero-argument callable for PEZ-NH, PEZ and every variant; shared intermediates are computed lazily once."""
    f, K, zP, zS, LP, LS, oE, oV = P["f"], P["K"], P["zP"], P["f"]["zS"], P["LP"], P["LS"], P["oE"], P["oV"]
    den, C, M = P["den"], P["C"], P["M"]
    s2, hist = P["s2"], P["hist"]
    vc = P["v"] & P["clean"][:, None]
    w_nh = oE * oV * P["oEn_nh"]
    memo = {}

    def lazy(key, fn):
        if key not in memo:
            memo[key] = fn()
        return memo[key]

    def out(om):
        return R4.fco_core(R5._numN(LP, LS, om), den, C, M, K)[0]

    def nh_with(s2c):
        return out(oE * oV * oen(s2c, zS, K, act_of(s2c, LS, zP, oV)))

    def stacked(s2c):
        return out(w_nh * oen(s2c, zS, K, act_of(s2c, LS, zP, oV)))

    def e():
        return lazy("e", lambda: row_energy(f["S"]))

    def s2m():
        return lazy("s2m", lambda: PX.energy_s2_maxprefix(f["S"]))

    def eY():
        return lazy("eY", lambda: row_energy(frame_rows(x, g)))

    def V():
        return lazy("V", lambda: q_ema(eY(), P["al"]))

    def coh():
        return lazy("coh", lambda: np.where(V() > 0, e() / np.where(V() > 0, V(), 1.0), 1.0))

    def lead_p():
        def mk():
            m = np.zeros_like(zP, bool)
            m[np.arange(zP.shape[0]), zP.argmax(1)] = True
            return m
        return lazy("lead_p", mk)

    def novg():
        return lazy("novg", lambda: (np.asarray(s2)[:, None] < 1) & (LS > 1) & (zP > 0))

    T = {"nh": lambda: out(w_nh), "pez": lambda: out(oE * oV * P["oEn"]),
         "pxm_hist": lambda: out(oE * oV * oen(s2m(), zS, K, act_of(s2m(), LS, zP, oV) & hist)),
         "slowref_nh": lambda: nh_with(s2_ref(e(), ref_slow(e()))),
         "ts_nh": lambda: nh_with(s2_two_sided(e())),
         "med_nh": lambda: nh_with(s2_ref(e(), ref_median(e()))),
         "hold_nh": lambda: nh_with(s2_ref(e(), ref_hold(e()))),
         "ver_nh": lambda: nh_with(s2_ref(e(), ref_ver(e(), vc.any(1)))),
         "dframe_nh": lambda: stacked(s2_ref(eY(), ref_prefix_mean(eY()))),
         "damp_nh": lambda: nh_with(s2_ref(V(), ref_prefix_mean(V()))),
         "dcoh_nh": lambda: nh_with(s2_ref(coh(), ref_prefix_mean(coh()))),
         "lx_nh": lambda: fco_core_lx(R5._numN(LP, LS, w_nh), den, C, M, K, P["vf"] > 0),
         "owc_nh": lambda: out(w_nh * survival_other(P["v"], P["clean"], zP, LS)),
         "hi_nh": lambda: out(oE * oV * oen(s2, zS, K, act_of(s2, LS, zP, oV) & (hist | ~lead_p()))),
         "novg_nh": lambda: out(oE * oV * oen(s2, zS, K, novg())),
         "novg_pez": lambda: out(oE * oV * oen(s2, zS, K, novg() & hist)),
         "rel_nh": lambda: out(oE * oV * oen(s2, zS, K, act_of(s2, LS, zP, oV) & ~vc)),
         "rme_nh": lambda: out(w_nh * R5.rme_omega(x, g, zP, LS))}
    for k in ("prof", "cont", "rtpl", "disp", "norm", "span", "fplo", "fphi"):
        T[f"{k}_nh"] = (lambda kk: (lambda: stacked(cue_s2(P, x, g, kk))))(k)
    return T


def variants(x, g, P=None, only=None):
    """float64 outputs of PEZ-NH, PEZ and every variant (or only the named ones) for one sequence."""
    P = parts(x, g) if P is None else P
    T = _table(x, g, P)
    return {k: T[k]() for k in (list(T) if only is None else list(only))}


def outputs_fn(x, g):
    """Reference for build_candidate's bit-identity check: every variant and its band-limited version (bl_<name>)."""
    import analyze_fusion_family as FF
    out = {k: v.astype(np.float32) for k, v in variants(x, g).items()}
    b = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    out.update({f"bl_{k}": FF._bl(out[k], b).astype(np.float32) for k in VARS})
    return out


def bl_variant(x, g, name):
    """The fusion family's band-limited rule (C0 = .5, partner 943955) on top of one variant (the build path)."""
    import analyze_fusion_family as FF
    a = variant_one(x, g, name).astype(np.float32)
    b = np.asarray(gallery_discriminative_row_l2_scores(x, g), np.float32)
    return FF._bl(a, b).astype(np.float32)


def variant_one(x, g, name):
    """One variant only (the build path); the same table entry as variants(), so the output is bit-identical."""
    return variants(x, g, only=[name])[name]


# ---------------------------------------------------------------- run
def _job(task):
    tr, i = task
    x, g = D[f"s{tr}"][i], D[f"g{tr}"]
    out = {k: v.astype(np.float32) for k, v in variants(x, g).items()}
    rep = {}
    for k, z in (("nh", "ZA"), ("pez", "ZP")):
        ref = D[f"{z}{tr}"][i]
        rep[k] = (int((out[k] != ref).sum()), float(np.abs(out[k].astype(np.float64) - ref.astype(np.float64)).max()))
    if tr == 1:
        return tr, i, out, rep
    c = int(D["c2"][i])
    return tr, i, {k: v[:, c].copy() for k, v in out.items()}, rep


def _runs(b):
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    s, e = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return s, e


def _near_mask(rej):
    """Frames within NEAR frames of a rejection run of length >= 20 (diagnostic only)."""
    s, e = _runs(rej)
    m = np.zeros(rej.shape[0], bool)
    for a, b in zip(s, e):
        if b - a >= 20:
            m[max(0, a - NEAR):min(rej.shape[0], b + NEAR)] = True
    return m


def main(workers=6):
    import analyze_t1_residual as ar
    import common_check as cc
    from proxy_lib import load_track, pseudo_targets
    t0 = time.time()
    zp = RS.ART / "pez_blvs_n_fco" / "submission.zip"
    stop = {"zip_sha_nh": RS._sha(DC.ZIP_NH) == DC.SHA_NH,
            "zip_sha_pez": RS._sha(zp) == "f28ce5876ad7e18c0598c07e0a637059958f8d62b52140e6578d7080fc1445b0"}
    if not all(stop.values()):
        print(json.dumps({"stop": stop, "verdict": "STOP"}), flush=True)
        return
    g1, s1, n1 = load_track("track1")
    g2, s2_, n2 = load_track("track2")
    c1, _ = pseudo_targets(g1, s1)
    c2, _ = pseudo_targets(g2, s2_)
    D.update(s1=s1, g1=g1, s2=s2_, g2=g2, c2=c2)
    for z, p in (("ZA", DC.ZIP_NH), ("ZP", zp)):
        S = HV._read_zip(p)
        D[f"{z}1"] = [S["track1"][f"score_{n}.npz"] for n in n1]
        D[f"{z}2"] = [S["track2"][f"score_{n}.npz"] for n in n2]
        del S
    tasks = [(1, i) for i in range(len(s1))] + [(2, i) for i in range(len(s2_))]
    R = {1: {}, 2: {}}
    rep = {k: [0, 0.0] for k in ("nh", "pez")}
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 6)), mp_context=mp.get_context("fork")) as ex:
        for tr, i, res, rp in ex.map(_job, tasks, chunksize=1):
            R[tr][i] = res
            for k, (nd, mx) in rp.items():
                rep[k][0] += nd
                rep[k][1] = max(rep[k][1], mx)
    for z in ("ZA", "ZP"):
        D.pop(f"{z}1")
        D.pop(f"{z}2")
    stop.update({f"repro_{k}": rep[k][0] <= REPRO_NDIFF and rep[k][1] <= REPRO_MAX for k in rep})
    n1s, n2s = len(s1), len(s2_)
    t1 = {}
    for k in ["nh", "pez"] + VARS:
        gen, imp = ar._gen_imp([R[1][i][k] for i in range(n1s)], c1)
        met = ar._metrics(gen, imp)
        t1[k] = {"eer_full": cc._eer_full(gen, imp), "eer_thr": met["eer_thr"], "far1e5_thr": met["far1e5_thr"],
                 "frr_far1e5": met["frr_at_far1e5"], "far_at_frr1e5": met["far_at_frr1e5"]}
        del gen, imp
    stop["nh_eer"] = abs(t1["nh"]["eer_full"] - NH["eer_full"]) <= 1e-9
    stop["pez_eer"] = abs(t1["pez"]["eer_full"] - PEZ_PROXY["eer_full"]) <= 1e-9
    acc = {}
    for k in ["nh", "pez"] + VARS:
        thr = np.float32(t1[k]["far1e5_thr"])
        acc[k] = [R[2][i][k] >= thr for i in range(n2s)]
    near = [_near_mask(~acc["nh"][i]) for i in range(n2s)]
    t2, est, gate = {}, {}, {}
    for k in VARS:
        b = BASE.get(k, "nh")
        A, Bb = np.concatenate(acc[k]), np.concatenate(acc[b])
        down, up = Bb & ~A, A & ~Bb
        nr = np.concatenate(near)
        dr = [Bb_ & ~A_ for A_, Bb_ in zip(acc[k], acc[b])]
        run20 = 0
        for m in dr:
            s, e = _runs(m)
            run20 += int(((e - s) * ((e - s) >= 20)).sum())
        d2 = float(Bb.mean() - A.mean())
        g1_ = t1[k]["frr_far1e5"] - t1[b]["frr_far1e5"]
        t2[k] = {"base": b, "acc_share": float(A.mean()), "d2": d2, "down": int(down.sum()), "up": int(up.sum()),
                 "down_in_runs_ge20": run20, "down_near_share": float(down[nr].sum() / max(down.sum(), 1)),
                 "base_acc_near_share": float(Bb[nr].sum() / max(Bb.sum(), 1)), "g1": g1_}
        est[k] = {f"r{r}": 5.0 / 3.0 * (r * g1_ - d2) for r in R_CAL}
        est[k]["opt"], est[k]["pess"] = min(est[k].values()), max(est[k].values())
        gate[k] = {"eer": t1[k]["eer_full"] <= t1[b]["eer_full"] + GATES["eer_max_vs_base"],
                   "thr": t1[k]["far1e5_thr"] >= GATES["thr_min"], "frr": t1[k]["frr_far1e5"] <= GATES["frr_max"],
                   "dbe_opt": est[k]["opt"] <= GATES["dbe_opt_max"]}
    passing = sorted([k for k in VARS if all(gate[k].values())], key=lambda k: est[k]["opt"])
    build = [min([k for k in passing if FAMILY[k] == fam], key=lambda k: est[k]["opt"])
             for fam in ("D", "F", "E") if any(FAMILY[k] == fam for k in passing)]
    verdict = "STOP" if not all(stop.values()) else ("BUILD:" + ",".join(build) if build else "NONE")
    print(json.dumps({"stop": stop, "repro": rep, "t1": t1, "t2": t2, "est_report": est, "gate": gate,
                      "passing_ranked": passing}), flush=True)
    print(json.dumps({"verdict": verdict, "seconds": round(time.time() - t0, 1)}), flush=True)


def synth():
    import analyze_peznh_stage0 as NHM
    import analyze_pez_stage0 as PZ
    from cubic_fast import ema_cubic_raw
    ok, inert = {}, {}
    for seed, kind in ((3, "switch"), (5, "mng")):
        x, g = R4._synth_data(seed, 240, 60, kind, norm_hi=26.0)
        P = parts(x, g)
        V = variants(x, g, P)
        ok[f"nh_eq_peznh_{seed}"] = bool(np.array_equal(V["nh"].astype(np.float32),
                                                        NHM.peznh_blvs_n_fco(x, g).astype(np.float32)))
        ok[f"pez_eq_pez_{seed}"] = bool(np.array_equal(V["pez"].astype(np.float32),
                                                       PZ.pez_blvs_n_fco(x, g).astype(np.float32)))
        e = row_energy(P["f"]["S"])
        ok[f"prefix_ref_eq_pez_s2_{seed}"] = bool(np.array_equal(s2_ref(e, ref_prefix_mean(e)), P["s2"]))
        K, zP, zS, LS, oV = P["K"], P["zP"], P["f"]["zS"], P["LS"], P["oV"]
        one = np.ones(240)
        nh_one = R4.fco_core(R5._numN(P["LP"], LS, P["oE"] * oV * P["oEn_nh"] * oen(one, zS, K, act_of(one, LS, zP, oV))),
                             P["den"], P["C"], P["M"], K)[0]
        ok[f"stack_neutral_{seed}"] = bool(np.array_equal(nh_one, V["nh"]))
        ok[f"lx_none_eq_nh_{seed}"] = bool(np.array_equal(
            fco_core_lx(R5._numN(P["LP"], LS, P["oE"] * oV * P["oEn_nh"]), P["den"], P["C"], P["M"], K,
                        np.zeros_like(zP, bool)), V["nh"]))
        u, a, c = _prep(x, g)
        Mt, beta, _ = discriminative_mapping(c)
        yr = np.tanh(ema_cubic_raw(u, a, c, np.ones(240), 1.0) @ Mt.T + beta)
        ok[f"frame_rows_eq_ema1_{seed}"] = bool(np.allclose(frame_rows(x, g), (yr - yr.mean(1, keepdims=True)) / 2,
                                                            rtol=0, atol=1e-9))
        for k in DISCOUNT_ONLY:
            b = BASE.get(k, "nh")
            ok[f"le_base_{k}_{seed}"] = bool((V[k] <= V[b] + 1e-12).all())
        ok[f"finite_in_range_{seed}"] = all(bool(np.isfinite(v).all() and (np.abs(v) < 1).all()) for v in V.values())
        ok[f"lean_eq_full_{seed}"] = all(bool(np.array_equal(variant_one(x, g, k), V[k])) for k in VARS)
        if seed == 3:
            O = outputs_fn(x, g)
            ok["bl_lean_eq_full"] = all(bool(np.array_equal(bl_variant(x, g, k), O[f"bl_{k}"])) for k in VARS[:4])
            hi = O["prof_nh"] >= np.float32(0.5)
            ok["bl_keeps_hi"] = bool(np.array_equal(O["bl_prof_nh"][hi], O["prof_nh"][hi])
                                     and (O["bl_prof_nh"][~hi] < np.float32(0.75)).all())
        h = 150
        Vh = variants(x[:h], g)
        bad = [k for k in V if not np.allclose(Vh[k], V[k][:h], rtol=0, atol=1e-9)]
        ok[f"prefix_invariant_{seed}"] = not bad
        if bad:
            ok[f"prefix_bad_{seed}"] = bad
        inert[seed] = [k for k in VARS if np.abs(V[k] - V[BASE.get(k, "nh")]).max() == 0]
    # reference rules against naive loops
    rng = np.random.default_rng(7)
    e = rng.exponential(size=300)
    ok["ref_median_naive"] = bool(np.allclose(ref_median(e), [np.median(e[:t + 1]) for t in range(300)], atol=1e-12))
    nv = []
    E, V_ = 0.0, 0.0
    for t in range(300):
        E += e[t]
        V_ = E / (t + 1) if t + 1 <= 45 else (1 - 0.022) * V_ + 0.022 * e[t]
        nv.append(V_)
    ok["ref_slow_naive"] = bool(np.allclose(ref_slow(e), nv, rtol=1e-12, atol=0))
    m = rng.normal(size=(5, 9))
    h_ = rng.normal(size=(5, 9))
    ce = _corr_excl(m, h_)
    ok["corr_excl_naive"] = bool(np.allclose(ce[2, 4], np.corrcoef(np.delete(m[2], 4), np.delete(h_[2], 4))[0, 1],
                                             atol=1e-12))
    allok = all(v is True for v in ok.values())
    print(json.dumps({"synth": ok, "inert_on_synth_report_only": inert, "all_ok": allok}), flush=True)
    return allok


if __name__ == "__main__":
    if sys.argv[1:] == ["--synth"]:
        sys.exit(0 if synth() else 1)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
