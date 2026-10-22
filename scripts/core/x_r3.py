"""Round 3 preregistered job: SEI-H, ESR-HCP and RSC-HCP against same-run HCP and OSP-C references.

Preregistered in docs/experiments/20261007_ogata_r3_sei_esr_rsc.md. Three independent primary
designs, each a single fixed design, never composed with each other:

  sei_h    lane T1     HCP with the history likelihood LP multiplied by m, the odds multiplier of a
                       full-dimensional self-attribution memory (appearance-neighbour kernel over the
                       sequence's own earlier frames, attribution only where P, S and raw cosine agree
                       on the leader). Q and mS stay at HCP's unadapted values.
  esr_hcp  lane Joint  HCP with the current look LS replaced by LS' = 1 + omega (LS - 1); omega =
                       min(1, e p) is the survival likelihood of the ongoing excursion of the in-span
                       cohort-centred cosine below its own history mean, ranked against the sequence's
                       own completed excursions.
  rsc_hcp  lane Joint  HCP with the numerator's current look LS replaced by 1 + w zS+^2; w = 1/Lambda is a
                       per-frame scalar from a one-sided self-referenced collapse statistic of the
                       gallery-orthogonal residual jitter. Denominator, mS and Q are HCP's own.

References computed in the same run: hcp_ref (x_fr_hcp.hcp on the unchanged front end) and ospc_ref
(candidates.ospc_vtn). run_check also adds the anchors 955399 and 955690.

outputs_fn(x, g) is a strictly causal, per-sequence processor with fresh state per call, float64
internals, identical constants on both tracks and no Track logic. K = g.shape[0] is the gallery size,
not a Track switch. No labels, pseudo targets, Track IDs, attack types or future frames.

  python scripts/core/x_r3.py --synth                    # random-data pre-run checks only
  python scripts/core/x_r3.py --stats                    # label-free premise / kill statistics (print only)
  python scripts/core/x_r3.py                            # single preregistered run_check
  python scripts/core/x_r3.py --report RUNLOG CALLOG     # post-run report + frozen gate

--synth uses random data only. --stats reads probe sequences but uses no labels or pseudo targets and
never changes a verdict (except the RSC activation counters, which are a pre-declared STOP for RSC only).
--report uses evaluator-only pseudo targets and calibrate_t2_proxy's exact fixture and proxy definitions.
All print aggregates only.
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
import x_r2 as R2  # noqa: E402  (shared hcp_L map, report constants and helpers)
from cubic_fast import _prep  # noqa: E402

TAG = "x_r3"
PRIMARIES = ["sei_h", "esr_hcp", "rsc_hcp"]
REFS = ["hcp_ref", "ospc_ref"]
ALL = PRIMARIES + REFS
LANE = {"sei_h": "T1", "esr_hcp": "Joint", "rsc_hcp": "Joint"}
E_CONST = float(np.e)
CHUNK = 1024                                          # SEI kernel row chunk (implementation only)
LAM_BAR = 3.0                                         # RSC kill statistic: discount factor Lambda = 1/w > 3
THR_HCP = 0.8525308                                   # HCP FAR=1e-5 threshold of the HCP run (odds 157.81)


# =====================================================================================
# Design 1: SEI-H
# =====================================================================================
def _norm_floor(v):
    """Float64 row-normalisation with the repository's 1e-12 norm floor (zero rows stay zero)."""
    v = np.asarray(v, np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def sei_attribution(zP, zS, p, xn, gn):
    """Attribution matrix A (T x K): A[t, l] = p[t, l] when the leaders of the P z-row, the S z-row and
    the raw full-dimensional cosine to the gallery all equal l and zP[t, l] > 0 and zS[t, l] > 0 (and the
    frame has non-zero norm); otherwise the row is zero. np.argmax ties go to the first index."""
    T = zP.shape[0]
    r = np.arange(T)
    l1 = zP.argmax(1)
    l2 = zS.argmax(1)
    l3 = (xn @ gn.T).argmax(1)
    nz = (xn * xn).sum(1) > 0                         # zero-norm frame: no attribution (all-tie artefact)
    agree = (l1 == l2) & (l1 == l3) & (zP[r, l1] > 0) & (zS[r, l2] > 0) & nz
    A = np.zeros(zP.shape)
    A[r[agree], l1[agree]] = p[r[agree], l1[agree]]
    return A, agree


def sei_kernel(xn, A):
    """Strictly causal appearance kernel kappa[t, k] = (max(xn_t . xn_k, 0))^3 for k < t (0 for k >= t).
    Returns W[t] = sum_k kappa[t, k] and S[t, j] = sum_k kappa[t, k] A[k, j], chunked over rows."""
    T, K = A.shape
    W = np.zeros(T)
    S = np.zeros((T, K))
    for i0 in range(0, T, CHUNK):
        i1 = min(T, i0 + CHUNK)
        G = xn[i0:i1] @ xn[:i1].T
        np.maximum(G, 0.0, out=G)
        G = G * G * G
        rows = np.arange(i0, i1)[:, None]
        cols = np.arange(i1)[None, :]
        G[cols >= rows] = 0.0                         # frame t itself and the future are excluded
        W[i0:i1] = G.sum(1)
        S[i0:i1] = G @ A[:i1]
    return W, S


def sei_multiplier(W, S):
    """m[t, j] = max(1, (1 + S) / max(1, 1 + W - S)); the inner max is a round-off guard only."""
    return np.maximum(1.0, (1.0 + S) / np.maximum(1.0, 1.0 + W[:, None] - S))


def sei_core(zP, zS, K, m):
    """HCP map with LP' = LP * m; Q and mS stay at the unadapted HCP values (never recomputed from LP').
    Expression order identical to x_fr_hcp.hcp_parts, so m == 1 reproduces x_fr_hcp.hcp bit for bit."""
    num, den, aux = H.hcp_parts(zP, zS, K)
    LP = 1.0 + np.maximum(zP, 0.0) ** 2
    LS = 1.0 + np.maximum(zS, 0.0) ** 2
    LPp = LP * m
    num2 = 4.0 * LPp * LS
    den2 = LPp * (2.0 + aux["mS"]) + 1.0 + aux["Q"]
    rn = np.sqrt(num2)
    rd = np.sqrt(den2)
    return (rn - rd) / (rn + rd), {"LS": LS, "mS": aux["mS"], "num": num, "den": den}


def sei_h(x, g, fe=None, force_A_zero=False, full=False):
    f = H._fe(x, g) if fe is None else fe
    K, zP, zS = f["K"], f["zP"], f["zS"]
    num, den, _ = H.hcp_parts(zP, zS, K)
    theta = num / den
    p = theta / (1.0 + theta)                         # unadapted HCP posterior; internal state only
    xn, gn = _norm_floor(x), _norm_floor(g)
    A, agree = sei_attribution(zP, zS, p, xn, gn)
    if force_A_zero:
        A = np.zeros_like(A)
        agree = np.zeros_like(agree)
    W, S = sei_kernel(xn, A)
    m = sei_multiplier(W, S)
    out, aux = sei_core(zP, zS, K, m)
    if full:
        return out, {"m": m, "agree": agree, "W": W, "S": S, "A": A, "p": p, **aux}
    return out


# =====================================================================================
# Design 2: ESR-HCP
# =====================================================================================
def esr_omega(D, a, q, zP, LS, counters=None):
    """Excursion-survival validity weight omega (T x K), float64, strictly causal, one O(T) loop.

    D: cohort-centred in-span cosine series; a: memory weights; q: quality rate; zP, LS: the unadapted HCP
    history z and current-look likelihood (only used to gate which columns can be active).
    Per column: an excursion starts at the first frame with D < reference (reference = strictly-past a-weighted
    history mean Hbar, frozen at the excursion start), accumulates area R += q (H0 - D), and ends at the first
    frame with D >= H0 (no latch); completed areas (R > 0) are stored. While ongoing and R > 0, zP > 0, LS > 1:
    p = (1 + #{completed >= R}) / (1 + #completed), omega = min(1, e p). Everything else has omega = 1.
    """
    T, K = D.shape
    cw = np.cumsum(a)
    cs = np.cumsum(a[:, None] * D, 0)
    Hbar = np.full((T, K), np.nan)
    ok = np.zeros(T, bool)
    ok[1:] = cw[:-1] > 0                              # no history (t = 0 or zero cumulative weight): NaN
    ii = np.nonzero(ok)[0]
    Hbar[ii] = cs[ii - 1] / cw[ii - 1][:, None]
    inx = np.zeros(K, bool)
    R = np.zeros(K)
    H0 = np.full(K, np.nan)
    cnt = np.zeros(K, np.int64)
    G = np.full((K, T // 2 + 2), -1.0)
    omega = np.ones((T, K))
    n_zero_area = 0
    for t in range(1, T):
        ref = np.where(inx, H0, Hbar[t])
        below = D[t] < ref                            # strict; NaN reference -> False
        ending = inx & ~below
        if ending.any():
            j = np.nonzero(ending & (R > 0))[0]
            n_zero_area += int((ending & ~(R > 0)).sum())
            G[j, cnt[j]] = R[j]                       # a zero-area excursion is not recorded
            cnt[j] += 1
            inx[ending] = False
            R[ending] = 0.0
            H0[ending] = np.nan
        starting = ~inx & below
        if starting.any():
            H0[starting] = Hbar[t][starting]
            inx |= starting
        if inx.any():
            R[inx] += q[t] * (H0[inx] - D[t][inx])    # the excursion's own frames never enter its reference
        act = inx & (R > 0) & (zP[t] > 0) & (LS[t] > 1)
        ja = np.nonzero(act)[0]
        if ja.size:
            kmax = int(cnt.max())
            r = (G[ja, :kmax] >= R[ja, None]).sum(1)  # ties count as >= (conservative)
            p = (1.0 + r) / (1.0 + cnt[ja])
            omega[t, ja] = np.minimum(1.0, E_CONST * p)
    if counters is not None:
        counters["zero_area_excursions"] = n_zero_area
        counters["open_at_end"] = int(inx.sum())
        counters["completed"] = int(cnt.sum())
    return omega


def esr_series(x, g, K):
    """Evidence series: D (T x K), memory weights a, quality rate q (all float64)."""
    u, a, c = _prep(x, g)
    nrm = np.linalg.norm(np.asarray(x, np.float64), axis=1)
    V = u @ c.T
    b = (V.sum(1, keepdims=True) - V) / (K - 1)
    return V - b, a, np.minimum(1.0, (nrm / 20.0) ** 4)


def esr_core(zP, zS, K, omega):
    LP = 1.0 + np.maximum(zP, 0.0) ** 2
    LS = 1.0 + np.maximum(zS, 0.0) ** 2
    return R2.hcp_L(LP, 1.0 + omega * (LS - 1.0), K)


def esr_hcp(x, g, fe=None, force_omega_one=False, full=False):
    f = H._fe(x, g) if fe is None else fe
    K, zP, zS = f["K"], f["zP"], f["zS"]
    LS = 1.0 + np.maximum(zS, 0.0) ** 2
    if force_omega_one:
        omega = np.ones(zP.shape)
        cnt = {}
    else:
        D, a, q = esr_series(x, g, K)
        cnt = {}
        omega = esr_omega(D, a, q, zP, LS, cnt)
    out = esr_core(zP, zS, K, omega)
    return (out, {"omega": omega, "counters": cnt}) if full else out


# =====================================================================================
# Design 3: RSC-HCP
# =====================================================================================
def rsc_observable(x, g):
    """nu_t = (1 - cos(r_t, r_{t-1})) / 2 for t >= 1 (nu[0] is unused), r = unit frame minus its projection on
    the gallery span; al_t = min(1, .05 (|x_t|/20)^4). Degenerate cases: zero residual power in either frame
    gives kappa = 0 (nu = .5, rough, the conservative case); zero-norm frame gives u = 0, pw = 0, nu = .5, al = 0;
    kappa is clipped to [-1, 1] against rounding only."""
    xb = np.asarray(x, np.float64)
    nrm = np.linalg.norm(xb, axis=1)
    u = xb / np.maximum(nrm, 1e-12)[:, None]
    g64 = np.asarray(g, np.float64)
    gn = g64 / np.maximum(np.linalg.norm(g64, axis=1, keepdims=True), 1e-12)
    Q = np.linalg.qr(gn.T)[0]                         # 512 x K orthonormal gallery-span basis
    r = u - (u @ Q) @ Q.T
    pw = (r * r).sum(1)
    T = xb.shape[0]
    nu = np.full(T, 0.5)
    if T > 1:
        den = np.sqrt(pw[1:] * pw[:-1])
        dot = (r[1:] * r[:-1]).sum(1)
        kappa = np.where(den > 0, dot / np.where(den > 0, den, 1.0), 0.0)
        nu[1:] = (1.0 - np.clip(kappa, -1.0, 1.0)) / 2.0
    al = np.minimum(1.0, 0.05 * (nrm / 20.0) ** 4)
    return nu, al


def rsc_dyn(nu, al, counters=None):
    """DYN(obs = nu, al, t0 = 1, mode = 'collapse'): s2 (T,). Scalar recursion, strictly causal.
    E: EMA of nu at rate al (E = nu at t0 = 1); L: cumulative al-weighted mean of nu over frames t0..t-1;
    d = (L - E) / (L + E); z = d+ / sqrt(ss / n) with ss, n over the strictly past d; Wilks statistic
    s2 = Lden log1p(z^2 / Lden), Lden = sum of past al. s2 = 0 for t < 2 and whenever Lden = 0, n = 0 or ss = 0."""
    T = len(nu)
    s2 = np.zeros(T)
    E = Lnum = Lden = ss = 0.0
    n = 0
    n_ss_small = 0
    for t in range(1, T):
        E = nu[t] if t == 1 else (1.0 - al[t]) * E + al[t] * nu[t]
        if Lden > 0.0:
            L = Lnum / Lden
            dn = L + E
            d = (L - E) / dn if dn > 0.0 else 0.0
            if n >= 1:
                if ss < 1e-12:
                    n_ss_small += 1
                if ss > 0.0:
                    z = max(d, 0.0) / np.sqrt(ss / n)
                    s2[t] = Lden * np.log1p(z * z / Lden)
            ss += d * d
            n += 1
        Lnum += al[t] * nu[t]
        Lden += al[t]
    if counters is not None:
        counters["nu_lt_1e-12"] = int((nu[1:] < 1e-12).sum())
        counters["ss_lt_1e-12"] = n_ss_small
    return s2


def rsc_w_from_s2(s2):
    """w = exp(-max(0, s2/4 - .5 ln 2)) = min(1, sqrt2 exp(-s2/4)) in (0, 1] (underflow to 0 is harmless)."""
    return np.exp(-np.maximum(0.0, s2 / 4.0 - 0.5 * np.log(2.0)))


def rsc_w(x, g, counters=None):
    nu, al = rsc_observable(x, g)
    s2 = rsc_dyn(nu, al, counters)
    w = rsc_w_from_s2(s2)
    if counters is not None:
        counters["w_lt_.01"] = int((w < 0.01).sum())
    return w, s2


def rsc_core(zP, zS, K, w):
    """Numerator-only fold-in: LSe = 1 + w zS+^2 in the numerator; the denominator (mS, Q) is HCP's own.
    Expression order identical to x_fr_hcp.hcp_parts, so w == 1 reproduces x_fr_hcp.hcp bit for bit."""
    _, den, _ = H.hcp_parts(zP, zS, K)
    LP = 1.0 + np.maximum(zP, 0.0) ** 2
    LSe = 1.0 + w[:, None] * np.maximum(zS, 0.0) ** 2
    num2 = 4.0 * LP * LSe
    rn = np.sqrt(num2)
    rd = np.sqrt(den)
    return (rn - rd) / (rn + rd)


def rsc_hcp(x, g, fe=None, force_w_one=False, full=False):
    f = H._fe(x, g) if fe is None else fe
    K, zP, zS = f["K"], f["zP"], f["zS"]
    T = zP.shape[0]
    if force_w_one:
        w, s2, cnt = np.ones(T), np.zeros(T), {}
    else:
        cnt = {}
        w, s2 = rsc_w(x, g, cnt)
    out = rsc_core(zP, zS, K, w)
    return (out, {"w": w, "s2": s2, "counters": cnt}) if full else out


# =====================================================================================
# Harness entry
# =====================================================================================
def _all64(x, g):
    f = H._fe(x, g)
    return {"sei_h": sei_h(x, g, f), "esr_hcp": esr_hcp(x, g, f), "rsc_hcp": rsc_hcp(x, g, f),
            "hcp_ref": H.hcp(f["zP"], f["zS"], f["K"])}


def outputs_fn(x, g):
    if np.asarray(x).shape[0] == 0:                   # empty sequence: nothing to score (defined, not an error)
        return {k: np.zeros((0, np.asarray(g).shape[0]), np.float32) for k in ALL}
    out = {k: v.astype(np.float32) for k, v in _all64(x, g).items()}
    out["ospc_ref"] = np.asarray(H.OSPC(x, g), np.float32)
    return out


# =====================================================================================
# Random-data pre-run checks (no challenge data)
# =====================================================================================
def _sei_naive(zP, zS, p, xn, gn):
    """Independent per-frame loops: A, W, S, m (small T only)."""
    T, K = zP.shape
    A = np.zeros((T, K))
    for t in range(T):
        if not (xn[t] @ xn[t] > 0):
            continue
        l1, l2 = int(np.argmax(zP[t])), int(np.argmax(zS[t]))
        l3 = int(np.argmax(gn @ xn[t]))
        if l1 == l2 == l3 and zP[t, l1] > 0 and zS[t, l1] > 0:
            A[t, l1] = p[t, l1]
    W = np.zeros(T)
    S = np.zeros((T, K))
    for t in range(T):
        for k in range(t):
            kap = max(float(xn[t] @ xn[k]), 0.0) ** 3
            W[t] += kap
            S[t] += kap * A[k]
    m = np.empty((T, K))
    for t in range(T):
        for j in range(K):
            m[t, j] = max(1.0, (1.0 + S[t, j]) / max(1.0, 1.0 + W[t] - S[t, j]))
    return A, W, S, m


def _esr_naive(D, a, q, zP, LS):
    """Independent per-column Python loops. Returns omega and the final p of every completed excursion."""
    T, K = D.shape
    omega = np.ones((T, K))
    finals = []
    for j in range(K):
        comp, inx, R, H0, cw, cs = [], False, 0.0, np.nan, 0.0, 0.0
        for t in range(T):
            if t >= 1:
                Hb = cs / cw if cw > 0 else np.nan
                ref = H0 if inx else Hb
                below = bool(D[t, j] < ref)
                if inx and not below:
                    if R > 0:
                        finals.append((1.0 + sum(1 for c in comp if c >= R)) / (1.0 + len(comp)))
                        comp.append(R)
                    inx, R, H0 = False, 0.0, np.nan
                if (not inx) and below:
                    inx, H0 = True, Hb
                if inx:
                    R += q[t] * (H0 - D[t, j])
                if inx and R > 0 and zP[t, j] > 0 and LS[t, j] > 1:
                    r = sum(1 for c in comp if c >= R)
                    omega[t, j] = min(1.0, E_CONST * (1.0 + r) / (1.0 + len(comp)))
            cw += a[t]
            cs += a[t] * D[t, j]
    return omega, finals


def _rsc_brute(nu, al):
    """Per-t recomputation of s2 from explicit sums (no recursion)."""
    T = len(nu)
    E = np.zeros(T)
    for tau in range(1, T):
        v = np.prod(1.0 - al[2:tau + 1]) * nu[1]
        for k in range(2, tau + 1):
            v += al[k] * np.prod(1.0 - al[k + 1:tau + 1]) * nu[k]
        E[tau] = v
    s2 = np.zeros(T)
    d = {}
    for t in range(2, T):
        Lden = float(al[1:t].sum())
        if Lden <= 0.0:
            continue
        Lnum = float((al[1:t] * nu[1:t]).sum())
        L = Lnum / Lden
        dn = L + E[t]
        d[t] = (L - E[t]) / dn if dn > 0 else 0.0
    for t in range(2, T):
        if t not in d:
            continue
        past = [d[tau] for tau in d if tau < t]
        ssum = float(sum(v * v for v in past))
        if len(past) >= 1 and ssum > 0:
            Lden = float(al[1:t].sum())
            z = max(d[t], 0.0) / np.sqrt(ssum / len(past))
            s2[t] = Lden * np.log1p(z * z / Lden)
    return s2


def _ar_noise(rng, T, n, rho):
    """Stationary AR(1) series (T x n), unit variance."""
    e = rng.normal(size=(T, n))
    out = np.empty((T, n))
    out[0] = e[0]
    s = np.sqrt(1.0 - rho * rho)
    for t in range(1, T):
        out[t] = rho * out[t - 1] + s * e[t]
    return out


def _resid_stream(rng, g, T, rho_fn, ident=None, strength=0.0, norm=24.0):
    """Frames whose gallery-orthogonal residual is an AR(1) vector process with per-frame correlation rho_fn(t)
    (a callable); optional in-span identity part strength[t] * g[ident]. Unit-variance residual dims."""
    K = g.shape[0]
    gn = g.astype(np.float64) / np.linalg.norm(g, axis=1, keepdims=True)
    Q = np.linalg.qr(gn.T)[0]
    e = np.empty((T, 512))
    prev = None
    for t in range(T):
        eps = rng.normal(size=512)
        eps = eps - (eps @ Q) @ Q.T
        rho = rho_fn(t)
        e[t] = eps if prev is None else rho * prev + np.sqrt(1.0 - rho * rho) * eps
        prev = e[t]
    x = e * np.sqrt(512.0 / (512.0 - K)) * 1.0
    if ident is not None:
        x = x + np.asarray(strength)[:, None] * g[ident][None].astype(np.float64)
    x = x / np.linalg.norm(x, axis=1, keepdims=True) * norm
    return x.astype(np.float32)


def _target_stream(rng, g, T, tgt, strength_fn, norm_lo=18.0, norm_hi=30.0, shared=None):
    """noise + strength_t * g[tgt] (+ a fixed shared 'appearance' vector), norms in [lo, hi]."""
    x = rng.normal(size=(T, 512))
    s = np.array([strength_fn(t) for t in range(T)])
    x = x + s[:, None] * g[tgt][None].astype(np.float64)
    if shared is not None:
        x = x + shared[None]
    x = x / np.linalg.norm(x, axis=1, keepdims=True) * rng.uniform(norm_lo, norm_hi, size=(T, 1))
    return x.astype(np.float32), s


def synth():
    import example
    w = {}

    def upd(key, v):
        w[key] = max(w.get(key, 0.0), float(v))

    def cnt(key, v):
        w[key] = w.get(key, 0) + int(v)

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
                # ---------- structural identities (neutralised evidence == HCP, bitwise) ----------
                cnt("id_sei_A0_mismatch", (sei_h(x, g, f, force_A_zero=True) != ref64).sum())
                cnt("id_esr_omega1_mismatch", (esr_hcp(x, g, f, force_omega_one=True) != ref64).sum())
                cnt("id_rsc_w1_mismatch", (rsc_hcp(x, g, f, force_w_one=True) != ref64).sum())
                cnt("id_esr_core_omega1_mismatch", (esr_core(zP, zS, K, np.ones(zP.shape)) != ref64).sum())
                cnt("id_rsc_s2zero_mismatch", (rsc_core(zP, zS, K, rsc_w_from_s2(np.zeros(T))) != ref64).sum())
                cnt("id_sei_m1_mismatch", (sei_core(zP, zS, K, np.ones(zP.shape))[0] != ref64).sum())
                # ---------- SEI-H ----------
                o_sei, st = sei_h(x, g, f, full=True)
                m = st["m"]
                upd("sei_m_below_1", max(0.0, 1.0 - float(m.min())))
                w["sei_m_max_minus_bound"] = max(w.get("sei_m_max_minus_bound", -1e9),
                                                 float((m - (1.0 + st["W"][:, None])).max()))
                upd("sei_out_below_hcp", max(0.0, float((ref64 - o_sei).max())))
                eq = (m == 1.0)
                cnt("sei_m1_not_bitwise_hcp", (o_sei[eq] != ref64[eq]).sum())
                odds = ((1.0 + o_sei) / (1.0 - o_sei)) ** 2
                bound = 4.0 * st["LS"] / (2.0 + st["mS"])
                upd("sei_odds_over_bound_rel", float((odds / bound).max()) - 1.0)
                xn, gn = _norm_floor(x), _norm_floor(g)
                if T <= 130:
                    An, Wn, Sn, mn = _sei_naive(zP, zS, st["p"], xn, gn)
                    upd("sei_naive_A_abs", np.abs(An - st["A"]).max())
                    upd("sei_naive_W_rel", (np.abs(Wn - st["W"]) / np.maximum(1.0, np.abs(Wn))).max())
                    upd("sei_naive_S_rel", (np.abs(Sn - st["S"]) / np.maximum(1.0, np.abs(Sn))).max())
                    upd("sei_naive_m_rel", (np.abs(mn - m) / mn).max())
                    upd("sei_naive_out_abs", np.abs(sei_core(zP, zS, K, mn)[0] - o_sei).max())
                w["sei_agree_frames"] = w.get("sei_agree_frames", 0) + int(st["agree"].sum())
                # ---------- ESR-HCP ----------
                D, a, q = esr_series(x, g, K)
                LS = 1.0 + np.maximum(zS, 0.0) ** 2
                om = esr_omega(D, a, q, zP, LS)
                if K == 200:
                    om_n, _ = _esr_naive(D, a, q, zP, LS)
                    upd("esr_naive_vs_vec_abs", np.abs(om - om_n).max())
                upd("esr_omega_above_1", max(0.0, float(om.max()) - 1.0))
                cnt("esr_omega_nonpositive", (om <= 0).sum())
                cnt("esr_omega_below_e_over_T", (om < E_CONST / (T // 2 + 2)).sum())
                lsp = 1.0 + om * (LS - 1.0)
                upd("esr_LSp_above_LS", max(0.0, float((lsp - LS).max())))
                upd("esr_LSp_below_1", max(0.0, 1.0 - float(lsp.min())))
                # ---------- RSC-HCP ----------
                nu, al = rsc_observable(x, g)
                s2 = rsc_dyn(nu, al)
                if T <= 130:
                    upd("rsc_brute_s2_abs", np.abs(_rsc_brute(nu, al) - s2).max())
                upd("rsc_nu_range_excess", max(0.0, float(nu[1:].min()) * -1.0, float(nu.max()) - 1.0))
                upd("rsc_s2_negative", max(0.0, -float(s2.min())))
                cnt("rsc_s2_lt2_nonzero", (s2[:2] != 0).sum())
                wv = rsc_w_from_s2(s2)
                cnt("rsc_w_out_of_unit", ((wv <= 0) | (wv > 1) | ~np.isfinite(wv)).sum())
                cnt("rsc_w_lt1_below_ln2_region", ((wv < 1.0) & (s2 <= 2.0 * np.log(2.0))).sum())
                o_rsc = rsc_core(zP, zS, K, wv)
                upd("rsc_out_above_hcp", max(0.0, float((o_rsc - ref64).max())))
                eqw = wv == 1.0
                cnt("rsc_w1_not_bitwise_hcp", (o_rsc[eqw] != ref64[eqw]).sum() if eqw.any() else 0)
                # ---------- all: range, finiteness, prefix invariance ----------
                full = outputs_fn(x, g)
                half = outputs_fn(x[: T // 2], g)
                for k in ALL:
                    v = full[k]
                    rng_ok &= bool(v.shape == (T, K) and np.isfinite(v).all() and v.min() >= -1 and v.max() <= 1)
                    upd(f"iv_prefix_{k}", np.abs(half[k].astype(np.float64) - v[: T // 2].astype(np.float64)).max())
                for k in PRIMARIES:
                    strict_ok &= bool(np.abs(full[k].astype(np.float64)).max() < 1)
        # ---------- deterministic extreme rows through the three maps ----------
        zh, zs = H._extreme_rows(K)
        Te = zh.shape[0]
        rng = np.random.default_rng(3)
        Wd = rng.uniform(0.0, 2500.0, size=Te)
        Sd = Wd[:, None] * rng.uniform(0.0, 1.0, size=(Te, K))
        me = sei_multiplier(Wd, Sd)
        oe, ae = sei_core(zh, zs, K, me)
        he = H.hcp(zh, zs, K)
        upd("extreme_sei_out_below_hcp", max(0.0, float((he - oe).max())))
        odds_e = ((1.0 + oe) / (1.0 - oe)) ** 2
        upd("extreme_sei_odds_over_bound_rel", float((odds_e / (4.0 * ae["LS"] / (2.0 + ae["mS"]))).max()) - 1.0)
        strict_ok &= bool(np.abs(oe.astype(np.float32).astype(np.float64)).max() < 1)
        om_e = rng.uniform(np.e / 200.0, 1.0, size=(Te, K))
        oe2 = esr_core(zh, zs, K, om_e)
        strict_ok &= bool(np.abs(oe2.astype(np.float32).astype(np.float64)).max() < 1 and np.isfinite(oe2).all())
        for wc in (1.0, 0.5, 1e-3, 1e-300, 0.0):
            o3 = rsc_core(zh, zs, K, np.full(Te, wc))
            strict_ok &= bool(np.abs(o3.astype(np.float32).astype(np.float64)).max() < 1 and np.isfinite(o3).all())
            upd("extreme_rsc_out_above_hcp", max(0.0, float((o3 - he).max())))
    # ---------- ESR iid / regime-shift / stationary sanity (direct on the series) ----------
    rng = np.random.default_rng(21)
    Tn, Kn = 3000, 60
    Di = 0.3 + 0.08 * rng.normal(size=(Tn, Kn))
    ones = np.ones((Tn, Kn))
    oi, finals = _esr_naive(Di, np.ones(Tn), np.ones(Tn), ones, 2.0 * ones)
    fin = np.array(finals)
    w["esr_iid_share_final_p_le_.1"] = float((fin <= 0.1).mean())
    w["esr_iid_completed_excursions"] = int(fin.size)
    # regime shift: D mean -2 sd after t = 500 (AR(.5) noise sd .08)
    Tn, Kn = 1000, 100
    Dn = 0.3 + 0.08 * _ar_noise(rng, Tn, Kn, 0.5)
    Ds = Dn.copy()
    Ds[500:] -= 0.16
    ones = np.ones((Tn, Kn))
    om_s = esr_omega(Ds, np.ones(Tn), np.ones(Tn), ones, 2.0 * ones)
    om_t = esr_omega(Dn, np.ones(Tn), np.ones(Tn), ones, 2.0 * ones)
    w["esr_shift_share_omega_lt_.2_next300"] = float((om_s[500:800] < 0.2).mean())
    w["esr_stationary_share_omega_lt_.2"] = float((om_t[1:] < 0.2).mean())
    w["esr_stationary_share_omega_lt_.2_t_ge_300"] = float((om_t[300:] < 0.2).mean())
    # ---------- mechanism scenarios through the whole pipeline (random data) ----------
    K = 200
    thr = THR_HCP
    rng = np.random.default_rng(31)
    g = rng.normal(size=(K, 512)).astype(np.float32)
    # (a) identity switch (claimed column 7 -> column 30 at T/2): ESR fires within tens of frames
    T = 400
    h = T // 2
    xs, _ = _target_stream(rng, g, T, 7, lambda t: 0.3, 20.0, 28.0)
    xs2, _ = _target_stream(rng, g, T, 30, lambda t: 0.3, 20.0, 28.0)
    xsw = np.concatenate([xs[:h], xs2[h:]]).astype(np.float32)
    f = H._fe(xsw, g)
    o_h = H.hcp(f["zP"], f["zS"], K)
    o_e, se = esr_hcp(xsw, g, f, full=True)
    elig = (f["zP"][:, 7] > 0) & (f["zS"][:, 7] > 0)
    win = np.zeros(T, bool)
    win[h:h + 30] = True
    pre = np.zeros(T, bool)
    pre[100:h] = True
    w["esr_switch_eligible_post30"] = int((elig & win).sum())
    w["esr_switch_share_omega_lt_.2_post30"] = (float((se["omega"][elig & win, 7] < 0.2).mean())
                                                if (elig & win).any() else 0.0)
    w["esr_switch_share_omega_lt_.2_pre"] = (float((se["omega"][elig & pre, 7] < 0.2).mean())
                                             if (elig & pre).any() else 0.0)
    def first_below(o):
        idx = np.nonzero(o[h:, 7] < thr)[0]
        return int(idx[0]) if idx.size else 10 ** 6
    w["esr_switch_frames_to_below_thr_esr"] = first_below(o_e)
    w["esr_switch_frames_to_below_thr_hcp"] = first_below(o_h)
    # (b) slow drift: strength .3 -> 0 linearly over the second half
    T = 400
    xd, sd_ = _target_stream(rng, g, T, 7, lambda t: 0.3 if t < 200 else 0.3 * (1.0 - (t - 200) / 200.0), 20.0, 28.0)
    f = H._fe(xd, g)
    o_h = H.hcp(f["zP"], f["zS"], K)
    o_e, se = esr_hcp(xd, g, f, full=True)
    elig = (f["zP"][:, 7] > 0) & (f["zS"][:, 7] > 0)
    win = np.zeros(T, bool)
    win[260:360] = True
    pre = np.zeros(T, bool)
    pre[100:200] = True
    w["esr_drift_eligible_window"] = int((elig & win).sum())
    w["esr_drift_share_omega_lt_.2_window"] = (float((se["omega"][elig & win, 7] < 0.2).mean())
                                               if (elig & win).any() else 0.0)
    w["esr_drift_share_omega_lt_.2_pre"] = (float((se["omega"][elig & pre, 7] < 0.2).mean())
                                            if (elig & pre).any() else 0.0)
    # (c) genuine dips: identity strength dips to .1 for 5-30 frames after t = 100; target column 11
    T = 600
    rngd = np.random.default_rng(41)
    dip = np.zeros(T, bool)
    t = 100
    while t < T:
        if rngd.random() < 0.04:
            L = int(rngd.integers(5, 31))
            dip[t:t + L] = True
            t += L
        t += 1
    xg, sg = _target_stream(rngd, g, T, 11, lambda t: 0.1 if dip[t] else 0.5, 18.0, 30.0)
    f = H._fe(xg, g)
    o_h = H.hcp(f["zP"], f["zS"], K)
    o_e, se = esr_hcp(xg, g, f, full=True)
    o_r, sr = rsc_hcp(xg, g, f, full=True)
    o_s, ss_ = sei_h(xg, g, f, full=True)
    acc = o_h[:, 11] >= thr
    w["dip_frames"] = int(dip.sum())
    w["dip_hcp_accept_frames_target"] = int(acc.sum())
    w["dip_retention_esr"] = float((o_e[acc, 11] >= thr).mean()) if acc.any() else 0.0
    w["dip_retention_rsc"] = float((o_r[acc, 11] >= thr).mean()) if acc.any() else 0.0
    w["dip_retention_sei"] = float((o_s[acc, 11] >= thr).mean()) if acc.any() else 0.0
    w["dip_esr_share_omega_lt_.2_target_t_ge_100"] = float((se["omega"][100:, 11] < 0.2).mean())
    w["dip_rsc_share_lambda_gt_3_t_ge_100"] = float((sr["w"][100:] < 1.0 / LAM_BAR).mean())
    # (d) SEI: shared-appearance stream with identity dips; the boost fires on dip frames of the target column
    T = 400
    rngs = np.random.default_rng(51)
    shared = rngs.normal(size=512) * 1.2
    dip_s = np.zeros(T, bool)
    t = 100
    while t < T:
        if rngs.random() < 0.05:
            L = int(rngs.integers(3, 12))
            dip_s[t:t + L] = True
            t += L
        t += 1
    xp, _ = _target_stream(rngs, g, T, 11, lambda t: 0.0 if dip_s[t] else 0.5, 20.0, 28.0, shared=shared)
    f = H._fe(xp, g)
    o_h = H.hcp(f["zP"], f["zS"], K)
    o_s, ss_ = sei_h(xp, g, f, full=True)
    late = np.arange(T) >= 100
    mx = ss_["m"].max(1)
    w["sei_boost_agree_share_t_ge_100"] = float(ss_["agree"][late].mean())
    w["sei_boost_share_frames_maxm_gt_1.5_t_ge_100"] = float((mx[late] > 1.5).mean())
    dm = dip_s & late
    w["sei_boost_dip_frames"] = int(dm.sum())
    w["sei_boost_dip_target_m_median"] = float(np.median(ss_["m"][dm, 11])) if dm.any() else 0.0
    w["sei_boost_dip_target_out_gain_median"] = float(np.median(o_s[dm, 11] - o_h[dm, 11])) if dm.any() else 0.0
    # (e) SEI identity transition: appearance kept, identity 7 -> 30; post-switch claimed entry stays below the bound
    T = 400
    h = T // 2
    xa, _ = _target_stream(rngs, g, T, 7, lambda t: 0.5, 20.0, 28.0, shared=shared)
    xb, _ = _target_stream(rngs, g, T, 30, lambda t: 0.5, 20.0, 28.0, shared=shared)
    xt = np.concatenate([xa[:h], xb[h:]]).astype(np.float32)
    f = H._fe(xt, g)
    o_s, st_ = sei_h(xt, g, f, full=True)
    oh_ = H.hcp(f["zP"], f["zS"], K)
    post = np.arange(T) >= h
    odds = ((1.0 + o_s) / (1.0 - o_s)) ** 2
    bound = 4.0 * st_["LS"] / (2.0 + st_["mS"])
    w["sei_transition_odds_over_bound_rel_max"] = float((odds / bound).max()) - 1.0
    thr_odds = ((1.0 + thr) / (1.0 - thr)) ** 2
    bad = post & (bound[:, 7] < thr_odds)
    w["sei_transition_post_claimed_bound_below_thr_frames"] = int(bad.sum())
    w["sei_transition_post_claimed_out_ge_thr_where_bound_below"] = int((o_s[bad, 7] >= thr).sum())
    w["sei_transition_post_claimed_max_out_gain_vs_hcp"] = float((o_s[post, 7] - oh_[post, 7]).max())
    # (f) RSC: residual correlation .5 -> .95 at T/2; null stream; rough spells
    T = 600
    rngr = np.random.default_rng(61)
    xr = _resid_stream(rngr, g, T, lambda t: 0.5 if t < 300 else 0.95, 11, np.full(T, 0.5))
    f = H._fe(xr, g)
    o_r, sr = rsc_hcp(xr, g, f, full=True)
    lam = 1.0 / sr["w"]
    w["rsc_collapse_share_lambda_gt_3_post"] = float((lam[300:] > LAM_BAR).mean())
    w["rsc_collapse_share_lambda_gt_3_pre"] = float((lam[:300] > LAM_BAR).mean())
    nul = []
    nlt = 0
    for sd in range(5):
        xn_ = _resid_stream(np.random.default_rng(70 + sd), g, 1500, lambda t: 0.5, 11, np.full(1500, 0.5))
        wn, _ = rsc_w(xn_, g)
        nul.append(float((wn < 1.0 / LAM_BAR).mean()))
        nlt += int((wn < 0.01).sum())
    w["rsc_null_share_lambda_gt_3_max_over_5_seeds"] = max(nul)
    w["rsc_null_w_lt_.01_frames"] = nlt
    rough = np.zeros(1200, bool)
    for s0 in range(300, 1200, 150):
        rough[s0:s0 + 30] = True
    xq = _resid_stream(np.random.default_rng(81), g, 1200, lambda t: 0.1 if rough[t] else 0.5, 11, np.full(1200, 0.5))
    wq, _ = rsc_w(xq, g)
    w["rsc_rough_spell_share_lambda_gt_3_during"] = float((wq[rough] < 1.0 / LAM_BAR).mean())
    w["rsc_rough_spell_share_lambda_gt_3_overall_t_ge_300"] = float((wq[300:] < 1.0 / LAM_BAR).mean())
    # extreme sequences through the whole pipeline: tiny/huge norms, constant frame, T = 1, 2, zero frames
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
        z2 = rng.normal(size=(40, 512)).astype(np.float32) * 20
        z2[1] = 0.0
        seqs.append(z2)
        z3 = np.zeros((10, 512), np.float32)
        seqs.append(z3)
        seqs.append(np.zeros((0, 512), np.float32))
        for xs_ in seqs:
            try:
                o = outputs_fn(xs_, g)
                for k in ALL:
                    v = o[k]
                    ex_ok &= bool(v.shape == (xs_.shape[0], K) and np.isfinite(v).all()
                                  and (v.size == 0 or (v.min() >= -1 and v.max() <= 1)))
            except Exception as e:                    # an unhandled degenerate input is a synthetic failure
                ex_ok = False
                w.setdefault("extreme_seq_errors", []).append(f"{type(e).__name__}: {str(e)[:80]}")
    out = {"synth": TAG, **w, "range_ok": rng_ok, "strict_inside_f32_primaries": strict_ok,
           "extreme_sequences_ok": ex_ok, "rows": rows}
    ok_zero = ("id_sei_A0_mismatch", "id_esr_omega1_mismatch", "id_rsc_w1_mismatch", "id_esr_core_omega1_mismatch",
               "id_rsc_s2zero_mismatch", "id_sei_m1_mismatch", "sei_m1_not_bitwise_hcp", "rsc_w1_not_bitwise_hcp",
               "esr_omega_nonpositive", "esr_omega_below_e_over_T",
               "rsc_s2_lt2_nonzero", "rsc_w_out_of_unit", "rsc_w_lt1_below_ln2_region")
    tight = ("sei_naive_A_abs", "sei_naive_W_rel", "sei_naive_S_rel", "sei_naive_m_rel", "sei_naive_out_abs",
             "esr_naive_vs_vec_abs", "rsc_brute_s2_abs", "sei_m_below_1", "esr_omega_above_1",
             "esr_LSp_above_LS", "esr_LSp_below_1", "rsc_nu_range_excess", "rsc_s2_negative")
    mono = ("sei_out_below_hcp", "rsc_out_above_hcp", "extreme_sei_out_below_hcp", "extreme_rsc_out_above_hcp")
    out["pass"] = bool(
        all(w[k] == 0 for k in ok_zero)
        and all(w[k] <= 1e-9 for k in tight)
        and all(w[k] <= 1e-15 for k in mono)
        and w["sei_odds_over_bound_rel"] <= 1e-9 and w["extreme_sei_odds_over_bound_rel"] <= 1e-9
        and w["sei_transition_odds_over_bound_rel_max"] <= 1e-9
        and w["sei_m_max_minus_bound"] <= 1e-9
        and w["sei_agree_frames"] > 0
        and all(w[f"iv_prefix_{k}"] < 1e-6 for k in ALL)
        and rng_ok and strict_ok and ex_ok
        and 0.05 <= w["esr_iid_share_final_p_le_.1"] <= 0.2
        and w["esr_shift_share_omega_lt_.2_next300"] >= 0.40 and w["esr_stationary_share_omega_lt_.2"] < 0.10
        and w["esr_switch_eligible_post30"] > 0 and w["esr_switch_share_omega_lt_.2_post30"] >= 0.5
        and w["esr_switch_share_omega_lt_.2_pre"] <= 0.15
        and w["esr_drift_eligible_window"] > 0 and w["esr_drift_share_omega_lt_.2_window"] >= 0.5
        and w["esr_drift_share_omega_lt_.2_pre"] <= 0.15
        and w["dip_retention_esr"] >= 0.95 and w["dip_retention_rsc"] >= 0.99 and w["dip_retention_sei"] >= 0.99
        and w["sei_boost_share_frames_maxm_gt_1.5_t_ge_100"] > 0 and w["sei_boost_dip_target_m_median"] > 1.5
        and w["sei_boost_dip_target_out_gain_median"] > 0
        and w["sei_transition_post_claimed_out_ge_thr_where_bound_below"] == 0
        and w["rsc_collapse_share_lambda_gt_3_post"] >= 0.10 and w["rsc_collapse_share_lambda_gt_3_pre"] <= 0.02
        and w["rsc_null_share_lambda_gt_3_max_over_5_seeds"] <= 0.02 and w["rsc_null_w_lt_.01_frames"] == 0
        and w["rsc_rough_spell_share_lambda_gt_3_during"] <= 0.01)
    print(json.dumps(out), flush=True)
    return out["pass"]


# =====================================================================================
# Label-free premise / kill statistics (printed only; never changes a verdict)
# =====================================================================================
STD_EDGES = [0, 10, 50, 200, 1000, np.inf]
STD_NAMES = ["t<10", "10-50", "50-200", "200-1000", ">=1000"]
QS = [0.01, 0.1, 0.5, 0.9, 0.99]


def _stats_job(task):
    tr, i = task
    x, g = (cc.D["s1"][i], cc.D["g1"]) if tr == "t1" else (cc.D["s2"][i], cc.D["g2"])
    f = H._fe(x, g)
    K, zP, zS = f["K"], f["zP"], f["zS"]
    T = x.shape[0]
    hcp = H.hcp(zP, zS, K)
    lead = hcp.argmax(1)
    r = np.arange(T)
    pb = np.digitize(np.arange(T), STD_EDGES) - 1
    nb = len(STD_NAMES)
    sub = np.arange(0, T, 4)
    res = {"n": T, "entries": T * K, "frames": np.bincount(pb, minlength=nb)}
    # ---- SEI-H ----
    o_s, st = sei_h(x, g, f, full=True)
    mx = st["m"].max(1)
    res["sei_agree"] = np.bincount(pb[st["agree"]], minlength=nb)
    res["sei_maxm_gt_1.5"] = np.bincount(pb[mx > 1.5], minlength=nb)
    res["sei_maxm_gt_1.05"] = np.bincount(pb[mx > 1.05], minlength=nb)
    res["sei_entries_m_gt_1.05"] = np.bincount(pb, weights=(st["m"] > 1.05).sum(1), minlength=nb)
    res["sei_maxm"] = mx.astype(np.float32)
    res["sei_maxm_pos"] = pb
    # ---- ESR-HCP ----
    o_e, se = esr_hcp(x, g, f, full=True)
    om = se["omega"]
    ol = om[r, lead]
    lead_elig = (zP[r, lead] > 0) & (np.maximum(zS[r, lead], 0.0) ** 2 > 0)
    res["esr_lead_lt_.5"] = np.bincount(pb[ol < 0.5], minlength=nb)
    res["esr_lead_lt_.2"] = np.bincount(pb[ol < 0.2], minlength=nb)
    res["esr_lead_lt_.05"] = np.bincount(pb[ol < 0.05], minlength=nb)
    res["esr_lead_elig"] = np.bincount(pb[lead_elig], minlength=nb)
    el = (zP > 0) & (np.maximum(zS, 0.0) ** 2 > 0)
    for nm, thr_ in (("5", 0.5), ("2", 0.2), ("05", 0.05)):
        res[f"esr_entry_lt_.{nm}"] = np.bincount(pb, weights=((om < thr_) & el).sum(1), minlength=nb)
    res["esr_entry_elig"] = np.bincount(pb, weights=el.sum(1), minlength=nb)
    res["esr_omega_sub"] = [om[sub][pb[sub] == b][(zP[sub][pb[sub] == b] > 0)].astype(np.float32) for b in range(nb)]
    res["esr_counters"] = se["counters"]
    # ---- RSC-HCP ----
    o_r, sr = rsc_hcp(x, g, f, full=True)
    wv = sr["w"]
    res["rsc_w_lt_.5"] = np.bincount(pb[wv < 0.5], minlength=nb)
    res["rsc_w_lt_.1"] = np.bincount(pb[wv < 0.1], minlength=nb)
    res["rsc_lambda_gt_3"] = np.bincount(pb[wv < 1.0 / LAM_BAR], minlength=nb)
    res["rsc_flag_relpos"] = (np.nonzero(wv < 1.0 / LAM_BAR)[0] / max(T, 1)).astype(np.float32)
    res["rsc_max_s2"] = float(sr["s2"].max())
    res["rsc_counters"] = sr["counters"]
    for k, v in (("sei_h", o_s), ("esr_hcp", o_e), ("rsc_hcp", o_r)):
        v32 = v.astype(np.float32)
        res[k] = {"max64": float(np.abs(v).max()), "max32": float(np.abs(v32.astype(np.float64)).max()),
                  "ge1_32": int((np.abs(v32) >= 1).sum()), "le_m999_32": int((v32 <= -0.999).sum())}
    return tr, res


def _q(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return None if v.size == 0 else np.quantile(v, QS).round(5).tolist()


def _share(num, den):
    return None if den == 0 else round(float(num) / float(den), 6)


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
    nb = len(STD_NAMES)
    ge200 = [3, 4]                                    # bins 200-1000 and >=1000
    out = {"stats": TAG, "quantiles": QS, "bins": STD_NAMES, "lambda_bar": LAM_BAR,
           "note": ("label-free; ESR 'frame' = HCP-leader column (argmax of the unadapted HCP row), omega read at "
                    "that column; RSC Lambda = 1/w; esr entry shares are over entries with zP>0 and LS>1")}
    summ = {}
    rsc_counts = {"nu_lt_1e-12": 0, "ss_lt_1e-12": 0, "w_lt_.01": 0}
    for tr, rs in acc.items():
        d = {"frames": int(sum(r["n"] for r in rs))}

        def tot(key):
            return np.sum([np.asarray(r[key], np.float64) for r in rs], axis=0)
        fr = tot("frames")
        d["frames_by_bin"] = fr.astype(int).tolist()
        d["sei"] = {"agree_share": _share(tot("sei_agree").sum(), fr.sum()),
                    "agree_share_by_bin": [_share(a, b) for a, b in zip(tot("sei_agree"), fr)],
                    "share_frames_maxm_gt_1.5": _share(tot("sei_maxm_gt_1.5").sum(), fr.sum()),
                    "share_frames_maxm_gt_1.5_by_bin": [_share(a, b) for a, b in zip(tot("sei_maxm_gt_1.5"), fr)],
                    "share_frames_maxm_gt_1.05": _share(tot("sei_maxm_gt_1.05").sum(), fr.sum()),
                    "share_entries_m_gt_1.05": _share(tot("sei_entries_m_gt_1.05").sum(), sum(r["entries"] for r in rs)),
                    "maxm_quantiles": _q(np.concatenate([r["sei_maxm"] for r in rs])),
                    "maxm_quantiles_by_bin": [_q(np.concatenate([r["sei_maxm"][r["sei_maxm_pos"] == b] for r in rs]))
                                              for b in range(nb)]}
        d["esr"] = {"lead_share_omega_lt": {k: _share(tot(f"esr_lead_lt_.{k}").sum(), fr.sum()) for k in ("5", "2", "05")},
                    "lead_share_omega_lt_.2_by_bin": [_share(a, b) for a, b in zip(tot("esr_lead_lt_.2"), fr)],
                    "lead_share_omega_lt_.2_t_ge_200": _share(sum(tot("esr_lead_lt_.2")[b] for b in ge200),
                                                              sum(fr[b] for b in ge200)),
                    "lead_share_omega_lt_.05_t_ge_200": _share(sum(tot("esr_lead_lt_.05")[b] for b in ge200),
                                                               sum(fr[b] for b in ge200)),
                    "entry_share_omega_lt_by_bin": {k: [_share(a, b) for a, b in
                                                        zip(tot(f"esr_entry_lt_.{k}"), tot("esr_entry_elig"))]
                                                    for k in ("5", "2", "05")},
                    "omega_quantiles_by_bin_zP_gt_0": [_q(np.concatenate([r["esr_omega_sub"][b] for r in rs]))
                                                       for b in range(nb)],
                    "completed_excursions": int(sum(r["esr_counters"]["completed"] for r in rs)),
                    "zero_area_excursions": int(sum(r["esr_counters"]["zero_area_excursions"] for r in rs)),
                    "open_at_sequence_end_entries": int(sum(r["esr_counters"]["open_at_end"] for r in rs))}
        d["rsc"] = {"share_w_lt_.5": _share(tot("rsc_w_lt_.5").sum(), fr.sum()),
                    "share_w_lt_.1": _share(tot("rsc_w_lt_.1").sum(), fr.sum()),
                    "share_lambda_gt_3": _share(tot("rsc_lambda_gt_3").sum(), fr.sum()),
                    "share_w_lt_.5_by_bin": [_share(a, b) for a, b in zip(tot("rsc_w_lt_.5"), fr)],
                    "share_w_lt_.1_by_bin": [_share(a, b) for a, b in zip(tot("rsc_w_lt_.1"), fr)],
                    "share_lambda_gt_3_by_bin": [_share(a, b) for a, b in zip(tot("rsc_lambda_gt_3"), fr)],
                    "share_lambda_gt_3_t_ge_200": _share(sum(tot("rsc_lambda_gt_3")[b] for b in ge200),
                                                         sum(fr[b] for b in ge200)),
                    "flag_relative_position_quantiles": _q(np.concatenate([r["rsc_flag_relpos"] for r in rs])),
                    "max_s2_per_sequence_quantiles": _q([r["rsc_max_s2"] for r in rs])}
        for k in ("nu_lt_1e-12", "ss_lt_1e-12", "w_lt_.01"):
            rsc_counts[k] += int(sum(r["rsc_counters"][k] for r in rs))
        d["rsc"]["counters"] = {k: int(sum(r["rsc_counters"][k] for r in rs)) for k in rsc_counts}
        for k in PRIMARIES:
            d[k] = {"max_abs_float64": max(r[k]["max64"] for r in rs), "max_abs_float32": max(r[k]["max32"] for r in rs),
                    "n_abs_ge1_float32": sum(r[k]["ge1_32"] for r in rs),
                    "n_le_-0.999_float32": sum(r[k]["le_m999_32"] for r in rs)}
        out[tr] = d
        summ[tr] = d
    t1, t2 = summ["t1"], summ["t2"]

    def ratio(a, b):
        return None if (a is None or b is None or a == 0) else round(b / a, 4)
    out["kill_statistics"] = {
        "sei_t1_share_frames_maxm_gt_1.5": t1["sei"]["share_frames_maxm_gt_1.5"],
        "sei_premise_ge_2pct_t1": bool((t1["sei"]["share_frames_maxm_gt_1.5"] or 0.0) >= 0.02),
        "esr_t2_over_t1_share_omega_lt_.2_t_ge_200": ratio(t1["esr"]["lead_share_omega_lt_.2_t_ge_200"],
                                                           t2["esr"]["lead_share_omega_lt_.2_t_ge_200"]),
        "rsc_t2_over_t1_share_lambda_gt_3_t_ge_200": ratio(t1["rsc"]["share_lambda_gt_3_t_ge_200"],
                                                           t2["rsc"]["share_lambda_gt_3_t_ge_200"]),
        "rsc_ratio_desired_ge_3_do_not_upload_if_lt_1.5": True}
    out["rsc_activation_counters_both_tracks"] = rsc_counts
    out["rsc_counters_all_zero"] = bool(all(v == 0 for v in rsc_counts.values()))
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
    hcp_thr_b = _t(prox["hcp_ref"]["thr"], HCP_RECORDED["thr"], abs(prox["hcp_ref"]["thr"] - HCP_RECORDED["thr"]) <= 5e-6)
    hcp_B = _t(prox["hcp_ref"][B_STAR], HCP_RECORDED["t2_cp_reject_pt"],
               abs(prox["hcp_ref"][B_STAR] - HCP_RECORDED["t2_cp_reject_pt"]) <= 2e-5)
    stop = {"ref_eer_repro_1e-7_ok": all(v[2] for v in eer_repro.values()),
            "hcp_ref_thr_and_Bstar_repro_ok": bool(hcp_thr_b[2] and hcp_B[2]),
            "run_repro_ok": all(t_[2] for d in rr.values() for t_ in d.values()),
            "cal_repro_ok": bool(cr) and all(cr[s] for s in CAL_MAP) and
            all(t_[2] for d in cr.values() for t_ in d.values()),
            "synth_and_structural_identity_ok": bool(syn and syn.get("pass")),
            "refs_checks_ok": bool(chk.get("hcp_ref")) and bool(chk.get("ospc_ref")) and
            bad["hcp_ref"] == 0 and bad["ospc_ref"] == 0}
    global_stop = not all(stop.values())
    rsc_ctr_ok = bool(stt and stt.get("rsc_counters_all_zero"))
    design_stop = {k: {"checks_pass_and_finite_in_range": bool(chk.get(k)) and bad[k] == 0,
                       "eer_full_report_vs_row_ok": bool(t1rep[k]["eer_full_match_1e-6"])} for k in PRIMARIES}
    design_stop["rsc_hcp"]["rsc_counters_zero_nu_ss_w"] = rsc_ctr_ok
    out2 = {"report_gate": TAG, "run_repro": rr, "cal_repro": cr, "ref_eer_repro": eer_repro,
            "hcp_ref_thr_repro": hcp_thr_b, "hcp_ref_Bstar_repro": hcp_B,
            "stop_checks_global": stop, "stop_checks_per_design": design_stop}
    # ---- frozen gate (applied once, mechanically; same-run hcp_ref) ----
    fbe = {k: BE_A + BE_B * prox[k][B_STAR] for k in PROX}
    hr, hp = t1rep["hcp_ref"], prox["hcp_ref"]
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
                                                  p["eer_full"] <= hr["eer_full"] + 0.0002),
                     "usp_frr<=usp_frr(hcp_ref)+.010": _t(prox[k]["usp_frr"], hp["usp_frr"] + 0.010,
                                                         prox[k]["usp_frr"] <= hp["usp_frr"] + 0.010),
                     "t1_band_reject<=t1_band_reject(hcp_ref)+.005": _t(prox[k]["t1_band_reject"],
                                                                        hp["t1_band_reject"] + 0.005,
                                                                        prox[k]["t1_band_reject"]
                                                                        <= hp["t1_band_reject"] + 0.005)}
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

    # ---- report-only disclosures (including the diffusion / IT forecasts of the HCP report) ----
    def ols(px, tgt):
        ms = [m for m in calrows if calrows[m].get("official", {}).get(tgt) is not None
              and calrows[m].get(px) is not None]
        b, a = np.polyfit([calrows[m][px] for m in ms], [calrows[m]["official"][tgt] for m in ms], 1)
        return float(a), float(b), len(ms)
    fc = {}
    for tgt, px in (("diff_aar", D_STAR), ("it_aar", "usp_aar")):
        a, b, n = ols(px, tgt)
        fc[tgt] = {"proxy": px, "a": a, "b": b, "n": n, **{k: round(a + b * prox[k][px], 4) for k in PROX}}
    a, b, n = ols(B_STAR, "be")
    fc["be"] = {"proxy": B_STAR, "a_frozen": BE_A, "b_frozen": BE_B, "a_refit": a, "b_refit": b, "n": n,
                "loo_mae": BE_LOO_MAE, **{k: round(fbe[k], 4) for k in PROX}}
    rep_only = {}
    for k in ALL:
        pk = prox[k]
        rep_only[k] = {"eer_full": t1rep[k]["eer_full"], "frr_far1e5": t1rep[k]["frr_far1e5"],
                       "far_at_frr1e5": t1rep[k]["far_at_frr1e5"], "usp_aar": pk["usp_aar"],
                       "usp_run": pk["usp_run"], "usp_frr": pk["usp_frr"], "t1_band_reject": pk["t1_band_reject"],
                       "t2_pre_reject_pt": pk["t2_pre_reject_pt"], "D*(" + D_STAR + ")": pk[D_STAR],
                       "B*(" + B_STAR + ")": pk[B_STAR], "BE_fc": round(fbe[k], 4),
                       "own_eer_thr": t1rep[k]["eer_thr"], "own_far1e5_thr": t1rep[k]["far1e5_thr"],
                       "eer_thr_bins": t1rep[k]["eer_thr_bins"],
                       "gen_err_t_ge10_at_eer_thr": t1rep[k]["gen_err_t_ge10_at_eer_thr"],
                       "top3_gen_err_at_eer_thr_all": t1rep[k]["top3_gen_err_at_eer_thr_all"],
                       "top3_gen_err_at_eer_thr_t_ge10": t1rep[k]["top3_gen_err_at_eer_thr_t_ge10"]}
    hcp_vs_recorded = {m: (got, v) for m, v in HCP_RECORDED.items()
                       for got in [t1rep["hcp_ref"].get(m, prox["hcp_ref"].get(m))]}
    out2.update({"gate": gate, "verdict": verdict, "ranking": ranking, "be_forecast": {k: round(fbe[k], 4) for k in PROX},
                 "forecasts_report_only": fc, "report_only": rep_only,
                 "premise_kill_statistics_from_run_log": None if stt is None else
                 {"kill_statistics": stt.get("kill_statistics"), "rsc_counters": stt.get("rsc_activation_counters_both_tracks")},
                 "hcp_ref_vs_recorded_hcp_run": hcp_vs_recorded, "pos_bins": ar.POS_NAMES,
                 "note": "A pass is a candidate for the user's decision, not a submission; no artifact is built."})
    print(json.dumps(out2), flush=True)
    print(json.dumps({"verdict": verdict, "ranking": ranking}), flush=True)


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
