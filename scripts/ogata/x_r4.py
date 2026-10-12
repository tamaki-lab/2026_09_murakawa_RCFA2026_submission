"""Round 4 preregistered job: MUX, RVS and FCO against same-run HCP and OSP-C references.

Preregistered in docs/experiments/20261007_ogata_r4_mux_rvs_fco.md. Three independent primary designs,
each a single fixed design, all lane T1, never composed with each other (user decision 2026-10-07):

  mux  HCP with the posterior mass of every non-exempt column multiplied by kappa = 1 - omega p_inc.
       The incumbent is the current P-leader, omega = tau / (1 + tau) is the Chinese-restaurant-process
       (alpha = 1) predictive weight built from the matured corroborated look mass tau of the
       incumbent's current uninterrupted run, p_inc is HCP's own posterior of the incumbent.
       The P-leader and the S-leader keep HCP's value.
  rvs  HCP with the numerator of every non-exempt column multiplied by psi = (1 + s_j) / (1 + s_max),
       a Laplace / reciprocal-rank visit-share factor from the matured corroborated ledger of the
       sequence's own posterior rows. Declared as a fixed, deterministic score-fusion rule
       (Laplace / reciprocal-rank ledger), NOT as a derived Bayes factor.
  fco  HCP with a finite, shared ownership resource added to the numerator and denominator:
       quality-weighted raw-cosine winner votes C_j (ties split), mass M, excess e = (M - A)+ with
       A = (K + 1) / 2; U_j + F_j against V_j + F_0 + sum_{k != j} F_k (sums, not means).
       Quality weight w = min(1, (|x|/20)^4) * (in-span energy fraction), see fco_weights.

References computed in the same run: hcp_ref (x_fr_hcp.hcp on the unchanged front end) and ospc_ref
(candidates.ospc_vtn). run_check also adds the anchors 955399 and 955690.

outputs_fn(x, g) is a strictly causal, per-sequence processor with fresh state per call, float64
internals, identical constants on both tracks and no Track logic. K = g.shape[0] is the gallery size,
not a Track switch. No labels, pseudo targets, Track IDs, attack types or future frames.

  python scripts/ogata/x_r4.py --synth                    # random-data pre-run checks only
  python scripts/ogata/x_r4.py --stats                    # label-free premise statistics (print only)
  python scripts/ogata/x_r4.py                            # single preregistered run_check
  python scripts/ogata/x_r4.py --report RUNLOG CALLOG     # post-run report + frozen gate

--synth uses random data only. --stats reads probe sequences but uses no labels or pseudo targets and
never changes a verdict. --report uses evaluator-only pseudo targets and calibrate_t2_proxy's exact
fixture and proxy definitions. All print aggregates only.
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
import x_r2 as R2  # noqa: E402  (shared report constants and helpers)
from cubic_fast import _prep  # noqa: E402  (only for the --synth cross-check of the FCO weight)
from rcfa.methods.gallery_subspace_residual import GallerySubspace  # noqa: E402

TAG = "x_r4"
PRIMARIES = ["mux", "rvs", "fco"]
REFS = ["hcp_ref", "ospc_ref"]
ALL = PRIMARIES + REFS
LANE = {k: "T1" for k in PRIMARIES}
GATE_EER_MARGIN = 0.00006                             # frozen gate term (user decision B)
STRETCH_EER_MARGIN = 0.0002                           # original stretch target, report-only
GATE_BE_MARGIN = 0.010
NET_W = 199                                           # impostors per genuine entry on Track 1 (K - 1)
THR_HCP = 0.8525308                                   # HCP FAR=1e-5 threshold of the HCP run (odds 157.81)
CUT_EER = -0.153304                                   # HCP EER threshold of the HCP run (odds .539); --stats only
CAP_M = 0.65                                          # --stats only: MUX 'cap-active' level m = omega p_inc >= 1 - p*
CROWD_N = 10                                          # --stats only: crowd row = >= 10 accepted non-exempt columns


# =====================================================================================
# Shared ledger (MUX and RVS)
# =====================================================================================
def _omap(num, den):
    """HCP output map, same operations and order as x_fr_hcp.hcp."""
    rn = np.sqrt(num)
    rd = np.sqrt(den)
    return (rn - rd) / (rn + rd)


def _al(x):
    """HCP's own EMA rate of the short state S (x_fr_hcp._fe, same expression), per frame; 0 for a zero-norm frame."""
    nrm = np.linalg.norm(np.asarray(x, np.float64), axis=1)
    with np.errstate(over="ignore"):
        return np.minimum(1.0, 0.05 * (nrm / 20.0) ** 4)


def _lag(al):
    """d_s = ceil(1 / al_s), one look window of S (inf when al_s = 0 or 1 / al_s overflows)."""
    with np.errstate(divide="ignore", over="ignore"):
        return np.where(al > 0, np.ceil(1.0 / np.where(al > 0, al, 1.0)), np.inf)


def _corroborated(zP, zS, al):
    """Vote v_s = l_s on E_s: P-leader == S-leader (np.argmax, first index on ties), both z > 0, and s >= d_s;
    otherwise -1. Also returns the P- and S-leader of every row."""
    T = zP.shape[0]
    r = np.arange(T)
    lP, lS = zP.argmax(1), zS.argmax(1)
    ok = (lP == lS) & (zP[r, lP] > 0) & (zS[r, lS] > 0) & (r >= _lag(al))
    return np.where(ok, lP, -1), lP, lS


def _arrival_rows(vote, al):
    """Frames whose vote becomes a mature look at some t <= T - 1 (arrival s + d_s), and that arrival row.
    Votes arriving after the last row can never affect any row of the sequence and are dropped."""
    T = vote.size
    arr = np.arange(T) + _lag(al)
    ok = (vote >= 0) & np.isfinite(arr) & (arr <= T - 1) & (al > 0)
    return ok, arr


# =====================================================================================
# Design 1: MUX
# =====================================================================================
def mux_tau(vote, lP, al):
    """tau_t = sum of al_s over votes s of the CURRENT run of the P-leader (maximal block of consecutive frames
    with the same argmax zP, a new run starts at every change) that are mature at t (s + d_s <= t)."""
    T = vote.size
    run = np.r_[0, np.cumsum(lP[1:] != lP[:-1])]
    ok, arr = _arrival_rows(vote, al)
    order = np.argsort(np.where(ok, arr, np.inf), kind="stable")
    acc = np.zeros(int(run[-1]) + 1)
    tau = np.zeros(T)
    j = 0
    for t in range(T):
        while j < T and ok[order[j]] and arr[order[j]] <= t:
            s = order[j]
            acc[run[s]] += al[s]
            j += 1
        tau[t] = acc[run[t]]
    return tau


def mux_core(num, den, lP, lS, tau):
    """p_j' = kappa p_j exactly for non-exempt j (theta' = kappa theta / (1 + (1 - kappa) theta)); the P-leader and
    the S-leader keep kappa = 1. tau == 0 gives kappa == 1.0 exactly, hence HCP bit for bit."""
    T, K = num.shape
    r = np.arange(T)
    om = tau / (1.0 + tau)                            # CRP (alpha = 1) predictive weight of the incumbent
    p_inc = num[r, lP] / (num[r, lP] + den[r, lP])    # HCP's own posterior of the incumbent
    kap = np.repeat((1.0 - om * p_inc)[:, None], K, 1)
    kap[r, lP] = 1.0
    kap[r, lS] = 1.0
    return _omap(kap * num, den + (1.0 - kap) * num), {"om": om, "p_inc": p_inc, "kap": kap}


def mux(x, g, fe=None, neutral=False, full=False):
    f = H._fe(x, g) if fe is None else fe
    K, zP, zS = f["K"], f["zP"], f["zS"]
    num, den, _ = H.hcp_parts(zP, zS, K)
    al = _al(x)
    vote, lP, lS = _corroborated(zP, zS, al)
    tau = np.zeros(zP.shape[0]) if neutral else mux_tau(vote, lP, al)
    out, aux = mux_core(num, den, lP, lS, tau)
    return (out, {"tau": tau, "lP": lP, "lS": lS, "vote": vote, "al": al, "num": num, "den": den, **aux}) if full else out


# =====================================================================================
# Design 2: RVS
# =====================================================================================
def rvs_rank(Pr_rows):
    """Competition rank 1 + #{k : Pr_k > Pr_j} per row; columns with equal Pr share the best rank."""
    K = Pr_rows.shape[1]
    rk = np.empty(Pr_rows.shape)
    for i in range(Pr_rows.shape[0]):
        rk[i] = K - np.searchsorted(np.sort(Pr_rows[i]), Pr_rows[i], side="right") + 1
    return rk


def rvs_ledger(Pr, vote, al):
    """s[t, j] = sum over mature corroborated frames s' (s' + d_s' <= t) of al_s' / rank[s', j]; M1[t] = sum al_s'."""
    T, K = Pr.shape
    ok, arr = _arrival_rows(vote, al)
    A = np.zeros((T, K))
    M1 = np.zeros(T)
    idx = np.nonzero(ok)[0]
    if idx.size:
        rk = rvs_rank(Pr[idx])
        np.add.at(A, arr[idx].astype(np.int64), al[idx, None] / rk)
        np.add.at(M1, arr[idx].astype(np.int64), al[idx])
    return np.cumsum(A, 0), np.cumsum(M1)


def rvs_core(num, den, s, lP, lS):
    """num' = psi num, den unchanged, psi = (1 + s_j) / (1 + s_max) in (0, 1], psi = 1 for the P- and S-leader.
    s == 0 gives psi == 1.0 exactly, hence HCP bit for bit."""
    T = num.shape[0]
    r = np.arange(T)
    psi = (1.0 + s) / (1.0 + s.max(1, keepdims=True))
    psi[r, lP] = 1.0
    psi[r, lS] = 1.0
    return _omap(psi * num, den), {"psi": psi}


def rvs(x, g, fe=None, neutral=False, full=False):
    f = H._fe(x, g) if fe is None else fe
    K, zP, zS = f["K"], f["zP"], f["zS"]
    num, den, _ = H.hcp_parts(zP, zS, K)
    Pr = (1.0 + np.maximum(zP, 0.0) ** 2) * (1.0 + np.maximum(zS, 0.0) ** 2)
    al = _al(x)
    vote, lP, lS = _corroborated(zP, zS, al)
    if neutral:
        s, M1 = np.zeros(zP.shape), np.zeros(zP.shape[0])
    else:
        s, M1 = rvs_ledger(Pr, vote, al)
    out, aux = rvs_core(num, den, s, lP, lS)
    return (out, {"s": s, "M1": M1, "lP": lP, "lS": lS, "vote": vote, "al": al, "num": num, "den": den, **aux}) if full else out


# =====================================================================================
# Design 3: FCO
# =====================================================================================
def fco_inspan(x, g):
    """In-span energy fraction f_t = |B x_t|^2 / |x_t|^2 in [0, 1], B the orthonormal basis of the gallery row space
    (rcfa GallerySubspace, the same basis cubic_kernel_reference.prepare uses). The min(1, .) is a round-off guard
    only (Bessel's inequality gives f <= 1); f = 0 for a frame whose energy is zero or underflows."""
    xb = np.asarray(x, np.float64)
    basis = GallerySubspace(np.asarray(g, np.float32)).basis
    coef = xb @ basis.T
    e_in = np.einsum("tk,tk->t", coef, coef)
    en = np.einsum("td,td->t", xb, xb)
    return np.where(en > 0, np.minimum(1.0, e_in / np.where(en > 0, en, 1.0)), 0.0)


def fco_weights(x, g):
    """w_t = min(1, (r_t / 20)^4) * f_t for r_t = |x_t| > 0, else 0; 0 <= w_t <= 1. Returns (w, r, f).

    CORRECTION of the specification's second factor. The spec writes (a_t / r_t)^2 and describes it as the share of
    the frame's energy inside the gallery span. a_t from cubic_fast._prep is not that ratio:
    cubic_kernel_reference.prepare forms z = (x max(|x|, 1e-12)) B^T, so a_t = |z_t| = max(|x_t|, 1e-12) |B x_t| =
    r_t |B x_t| (for r_t >= 1e-12) and the literal (a_t / r_t)^2 = |B x_t|^2 = r_t^2 f_t (about 120, not <= 1).
    The in-span energy fraction is f_t = |B x_t|^2 / r_t^2 = (a_t / r_t^2)^2."""
    xb = np.asarray(x, np.float64)
    r = np.linalg.norm(xb, axis=1)
    f = fco_inspan(x, g)
    with np.errstate(over="ignore"):
        q = np.minimum(1.0, (r / 20.0) ** 4)
    return np.where(r > 0, q * f, 0.0), r, f


def fco_votes(x, g, r):
    """Raw-cosine winner votes: v[t, j] = 1 / |T_t| on the exact float64 argmax set T_t of the raw cosine between the
    unit probe and the unit gallery (ties split equally, no index preference); zero rows for a zero-norm frame."""
    xb = np.asarray(x, np.float64)
    g64 = np.asarray(g, np.float64)
    xn = xb / np.maximum(r, 1e-12)[:, None]
    gn = g64 / np.maximum(np.linalg.norm(g64, axis=1, keepdims=True), 1e-12)
    raw = xn @ gn.T
    tie = raw == raw.max(1, keepdims=True)
    v = tie / tie.sum(1, keepdims=True)
    v[r <= 0] = 0.0
    return v, raw


def fco_state(w, v):
    """C[t, j] = sum_{s <= t} w_s v[s, j] (ownership votes), M[t] = sum_{s <= t} w_s (evidence mass)."""
    return np.cumsum(w[:, None] * v, 0), np.cumsum(w)


def fco_core(num, den, C, M, K, A=None):
    """e = (M - A)+, A = (K + 1) / 2 (Jeffreys mass of K + 1 hypotheses); F_j = e C_j / M, F_0 = e / K;
    out_j = Phi(U_j + F_j, V_j + F_0 + sum_{k != j} F_k), U = 4 LP LS, V = HCP's den (leader-exact Q is HCP's own).
    The exclusive sum uses shifted prefix/suffix cumulative sums (no subtraction). e == 0 adds +0.0 and gives HCP bit
    for bit. A = inf neutralises the evidence (identity checks)."""
    A = (K + 1) / 2.0 if A is None else A
    e = np.maximum(M - A, 0.0)
    share = np.zeros_like(C)
    np.divide(C, M[:, None], out=share, where=(M > 0)[:, None])
    F = e[:, None] * share
    F0 = e / K
    pre = np.zeros_like(F)
    suf = np.zeros_like(F)
    pre[:, 1:] = np.cumsum(F, 1)[:, :-1]
    suf[:, :-1] = np.cumsum(F[:, ::-1], 1)[:, ::-1][:, 1:]
    G = F0[:, None] + (pre + suf)
    return _omap(num + F, den + G), {"e": e, "F": F, "G": G}


def fco(x, g, fe=None, neutral=False, full=False):
    f = H._fe(x, g) if fe is None else fe
    K, zP, zS = f["K"], f["zP"], f["zS"]
    num, den, _ = H.hcp_parts(zP, zS, K)
    w, r, fr = fco_weights(x, g)
    v, _ = fco_votes(x, g, r)
    C, M = fco_state(w, v)
    out, aux = fco_core(num, den, C, M, K, A=np.inf if neutral else None)
    return (out, {"w": w, "r": r, "f": fr, "C": C, "M": M, "num": num, "den": den, **aux}) if full else out


# =====================================================================================
# Harness entry
# =====================================================================================
def _all64(x, g):
    f = H._fe(x, g)
    return {"mux": mux(x, g, f), "rvs": rvs(x, g, f), "fco": fco(x, g, f),
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
def _synth_data(seed, T, K, kind, norm_hi=29.0):
    """x_fr_hcp._synth_data with the upper frame norm as a parameter. norm_hi < 29.9 keeps al < .25, so that no
    corroborated vote can mature before t = 10 (2 d_s >= 10)."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(K, 512)).astype(np.float32)
    if kind == "switch":                              # identity switch at T/2
        h = T // 2
        ident = np.r_[0.3 * np.repeat(g[7][None], h, 0), 0.3 * np.repeat(g[30][None], T - h, 0)]
    else:                                             # MNG: strong memory leader, then weaker T
        h = (3 * T) // 5
        ident = np.r_[0.6 * np.repeat(g[11][None], h, 0), 0.25 * np.repeat(g[40][None], T - h, 0)]
    x = rng.normal(size=(T, 512)) + ident
    x = x / np.linalg.norm(x, axis=1, keepdims=True) * rng.uniform(10.0, norm_hi, size=(T, 1))
    return x.astype(np.float32), g


def _stream(seed, T, K, tgt, s, lo=16.0, hi=24.0):
    """noise + s * g[tgt], frame norms uniform in [lo, hi]: one identity of constant strength s."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(K, 512)).astype(np.float32)
    x = rng.normal(size=(T, 512)) + s * g[tgt][None].astype(np.float64)
    x = x / np.linalg.norm(x, axis=1, keepdims=True) * rng.uniform(lo, hi, size=(T, 1))
    return x.astype(np.float32), g


def _mng_stream(seed, T, K, s2, s1=0.6, h=480, lo=16.0, hi=24.0):
    """Wrong strong memory leader (column 11, strength s1) for h frames, then a weaker target (column 40, strength s2)."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(K, 512)).astype(np.float32)
    ident = np.r_[s1 * np.repeat(g[11][None], h, 0), s2 * np.repeat(g[40][None], T - h, 0)]
    x = rng.normal(size=(T, 512)) + ident
    x = x / np.linalg.norm(x, axis=1, keepdims=True) * rng.uniform(lo, hi, size=(T, 1))
    return x.astype(np.float32), g


def _mux_naive(zP, zS, num, den, al):
    """Per-frame loops from the definitions: tau by a double loop over the current run, kappa and out per entry."""
    T, K = zP.shape
    lP = [int(np.argmax(zP[t])) for t in range(T)]
    lS = [int(np.argmax(zS[t])) for t in range(T)]
    d = [np.ceil(1.0 / al[s]) if al[s] > 0 else np.inf for s in range(T)]
    E = [lP[s] == lS[s] and zP[s, lP[s]] > 0 and zS[s, lS[s]] > 0 and s >= d[s] for s in range(T)]
    tau = np.zeros(T)
    out = np.empty((T, K))
    for t in range(T):
        t0 = t
        while t0 > 0 and lP[t0 - 1] == lP[t]:
            t0 -= 1
        tau[t] = sum(al[s] for s in range(t0, t) if E[s] and s + d[s] <= t)
        om = tau[t] / (1.0 + tau[t])
        pinc = num[t, lP[t]] / (num[t, lP[t]] + den[t, lP[t]])
        for j in range(K):
            kap = 1.0 if j in (lP[t], lS[t]) else 1.0 - om * pinc
            a, b = kap * num[t, j], den[t, j] + (1.0 - kap) * num[t, j]
            out[t, j] = (np.sqrt(a) - np.sqrt(b)) / (np.sqrt(a) + np.sqrt(b))
    return out, tau


def _rvs_naive(zP, zS, num, den, al):
    """Per-frame loops: explicit ranks 1 + sum(Pr_k > Pr_j), ledger by a double loop over (t, s), psi per entry."""
    T, K = zP.shape
    LP = 1.0 + np.maximum(zP, 0.0) ** 2
    LS = 1.0 + np.maximum(zS, 0.0) ** 2
    Pr = LP * LS
    lP = [int(np.argmax(zP[t])) for t in range(T)]
    lS = [int(np.argmax(zS[t])) for t in range(T)]
    d = [np.ceil(1.0 / al[s]) if al[s] > 0 else np.inf for s in range(T)]
    E = [lP[s] == lS[s] and zP[s, lP[s]] > 0 and zS[s, lS[s]] > 0 and s >= d[s] for s in range(T)]
    rk = {s: 1.0 + (Pr[s][None, :] > Pr[s][:, None]).sum(1) for s in range(T) if E[s]}
    s_led = np.zeros((T, K))
    out = np.empty((T, K))
    for t in range(T):
        for s in range(t):
            if E[s] and s + d[s] <= t:
                s_led[t] += al[s] / rk[s]
        smax = s_led[t].max()
        for j in range(K):
            psi = 1.0 if j in (lP[t], lS[t]) else (1.0 + s_led[t, j]) / (1.0 + smax)
            a = psi * num[t, j]
            out[t, j] = (np.sqrt(a) - np.sqrt(den[t, j])) / (np.sqrt(a) + np.sqrt(den[t, j]))
    return out, s_led, rk


def _fco_naive(x, g, num, den, v):
    """Per-frame loops: w from a per-frame projection, state updates, exclusion by np.delete, scalar map."""
    T, K = num.shape
    xb = np.asarray(x, np.float64)
    basis = GallerySubspace(np.asarray(g, np.float32)).basis
    A = (K + 1) / 2.0
    C = np.zeros(K)
    M = 0.0
    out = np.empty((T, K))
    wl = np.zeros(T)
    for t in range(T):
        r = float(np.sqrt(xb[t] @ xb[t]))
        if r > 0:
            fr = float(np.sum((basis @ xb[t]) ** 2)) / float(xb[t] @ xb[t])
            wl[t] = min(1.0, (r / 20.0) ** 4) * min(1.0, fr)
        C = C + wl[t] * v[t]
        M = M + wl[t]
        e = max(M - A, 0.0)
        F = e * C / M if M > 0 else np.zeros(K)
        for j in range(K):
            ub = num[t, j] + F[j]
            vb = den[t, j] + e / K + np.delete(F, j).sum()
            out[t, j] = (np.sqrt(ub) - np.sqrt(vb)) / (np.sqrt(ub) + np.sqrt(vb))
    return out, wl


def synth():
    w = {}

    def upd(key, v):
        w[key] = max(w.get(key, 0.0), float(v))

    def cnt(key, v):
        w[key] = w.get(key, 0) + int(v)

    ok_shared = {"range": True}
    ok_d = {k: True for k in PRIMARIES}
    rows = 0
    for K in (200, 199):
        for kind in ("switch", "mng"):
            for seed, T in ((0, 300), (1, 257), (2, 64), (3, 760)):
                x, g = _synth_data(seed, T, K, kind)
                rows += T
                f = H._fe(x, g)
                zP, zS = f["zP"], f["zS"]
                ref64 = H.hcp(zP, zS, K)
                num, den, _ = H.hcp_parts(zP, zS, K)
                al = _al(x)
                r10 = np.arange(T) < 10
                # ---------- shared: the output map and the neutralised identities (bitwise, float64) ----------
                cnt("id_omap_mismatch", (_omap(num, den) != ref64).sum())
                cnt("id_mux_tau0_mismatch", (mux(x, g, f, neutral=True) != ref64).sum())
                cnt("id_rvs_s0_mismatch", (rvs(x, g, f, neutral=True) != ref64).sum())
                cnt("id_fco_e0_mismatch", (fco(x, g, f, neutral=True) != ref64).sum())
                lP0, lS0 = zP.argmax(1), zS.argmax(1)
                cnt("id_mux_core_tau0_mismatch", (mux_core(num, den, lP0, lS0, np.zeros(T))[0] != ref64).sum())
                cnt("id_rvs_core_s0_mismatch", (rvs_core(num, den, np.zeros((T, K)), lP0, lS0)[0] != ref64).sum())
                # ---------- MUX ----------
                o_m, sm = mux(x, g, f, full=True)
                upd("mux_out_above_hcp", max(0.0, float((o_m - ref64).max())))
                exm = np.zeros((T, K), bool)
                exm[np.arange(T), sm["lP"]] = True
                exm[np.arange(T), sm["lS"]] = True
                cnt("mux_exempt_not_bitwise_hcp", (o_m[exm] != ref64[exm]).sum())
                pc = (sm["kap"] * num) / (sm["kap"] * num + (den + (1.0 - sm["kap"]) * num))
                ph = num / (num + den)
                upd("mux_p_eq_kappa_p_rel", float((np.abs(pc - sm["kap"] * ph) / (sm["kap"] * ph)).max()))
                cnt("mux_kap_out_of_unit", ((sm["kap"] <= 0) | (sm["kap"] > 1)).sum())
                cnt("mux_t_lt_10_changed_rows", (np.abs(o_m - ref64)[r10] > 0).any(1).sum())
                cnt("mux_rows_changed", (o_m != ref64).any(1).sum())
                cnt("mux_votes_mature", (_arrival_rows(sm["vote"], al)[0]).sum())
                o_mn, tau_n = _mux_naive(zP, zS, num, den, al)
                upd("mux_naive_tau_abs", np.abs(tau_n - sm["tau"]).max())
                upd("mux_naive_out_abs", np.abs(o_mn - o_m).max())
                # ---------- RVS ----------
                o_r, sr = rvs(x, g, f, full=True)
                upd("rvs_out_above_hcp", max(0.0, float((o_r - ref64).max())))
                psi = sr["psi"]
                cnt("rvs_psi_out_of_unit", ((psi <= 0) | (psi > 1)).sum())
                cnt("rvs_exempt_psi_not_1", (psi[exm] != 1.0).sum())
                cnt("rvs_exempt_not_bitwise_hcp", (o_r[exm] != ref64[exm]).sum())
                cnt("rvs_t_lt_10_changed_rows", (np.abs(o_r - ref64)[r10] > 0).any(1).sum())
                cnt("rvs_rows_changed", (o_r != ref64).any(1).sum())
                o_rn, s_n, rk_n = _rvs_naive(zP, zS, num, den, al)
                upd("rvs_naive_ledger_abs", np.abs(s_n - sr["s"]).max())
                upd("rvs_naive_out_abs", np.abs(o_rn - o_r).max())
                Pr = (1.0 + np.maximum(zP, 0.0) ** 2) * (1.0 + np.maximum(zS, 0.0) ** 2)
                idx = np.nonzero(_arrival_rows(sr["vote"], al)[0])[0]
                if idx.size:
                    upd("rvs_rank_vs_naive_abs", np.abs(rvs_rank(Pr[idx]) - np.stack([rk_n[s] for s in idx])).max())
                # ---------- FCO ----------
                o_f, sf = fco(x, g, f, full=True)
                wv = sf["w"]
                upd("fco_w_above_1", max(0.0, float(wv.max()) - 1.0))
                upd("fco_w_below_0", max(0.0, -float(wv.min())))
                cnt("fco_w_nonfinite", (~np.isfinite(wv)).sum())
                u_, a_, _ = _prep(x, g)
                rr = sf["r"]
                upd("fco_f_vs_a_over_r2_sq_rel", float((np.abs(sf["f"] - (a_ / rr ** 2) ** 2) / sf["f"]).max()))
                w["fco_literal_w_median_unfixed"] = float(np.median((a_ / rr) ** 2))
                w["fco_f_median"] = float(np.median(sf["f"]))
                e0 = sf["e"] == 0
                cnt("fco_e0_rows", e0.sum())
                cnt("fco_e0_rows_not_bitwise_hcp", (o_f[e0] != ref64[e0]).sum())
                cnt("fco_t_lt_10_changed_rows", (o_f != ref64)[r10].any(1).sum())
                cnt("fco_t_lt_A_changed_rows", (o_f != ref64)[: int((K + 1) / 2 - 0.5)].any(1).sum())
                cnt("fco_rows_changed", (o_f != ref64).any(1).sum())
                pos = sf["e"] > 0
                if pos.any():
                    upd("fco_sumF_minus_e_rel", float((np.abs(sf["F"].sum(1) - sf["e"])[pos] / sf["e"][pos]).max()))
                upd("fco_F_below_0", max(0.0, -float(sf["F"].min())))
                vv, _ = fco_votes(x, g, rr)
                o_fn, w_n = _fco_naive(x, g, num, den, vv)
                upd("fco_naive_w_abs", np.abs(w_n - wv).max())
                upd("fco_naive_out_abs", np.abs(o_fn - o_f).max())
                # ---------- all: range, finiteness, prefix invariance (float32 cast, as the harness) ----------
                full = outputs_fn(x, g)
                cnt("id_hcp_ref_vs_x_fr_hcp_all64_mismatch",
                    (full["hcp_ref"] != H._all64(x, g)[1]["hcp"].astype(np.float32)).sum())
                for h in sorted({T // 2} | ({1, 17, T - 1} if seed == 1 else set())):
                    half = outputs_fn(x[:h], g)
                    for k in ALL:
                        upd(f"iv_prefix_{k}", np.abs(half[k].astype(np.float64) - full[k][:h].astype(np.float64)).max())
                for k in ALL:
                    v = full[k]
                    good = bool(v.shape == (T, K) and np.isfinite(v).all() and v.min() >= -1 and v.max() <= 1)
                    ok_shared["range"] &= good if k in REFS else True
                    if k in PRIMARIES:
                        ok_d[k] &= good
                for k in PRIMARIES:
                    sin = bool(np.abs(full[k].astype(np.float64)).max() < 1)
                    ok_d[k] &= sin
                    w[f"strict_inside_f32_{k}"] = bool(w.get(f"strict_inside_f32_{k}", True) and sin)
        # ---------- deterministic extreme rows through the three cores ----------
        zh, zs = H._extreme_rows(K)
        Te = zh.shape[0]
        rng = np.random.default_rng(3)
        he = H.hcp(zh, zs, K)
        nm, dn, _ = H.hcp_parts(zh, zs, K)
        lPe, lSe = zh.argmax(1), zs.argmax(1)
        om_e = mux_core(nm, dn, lPe, lSe, rng.uniform(0.0, 2000.0, size=Te))[0]
        rv_e = rvs_core(nm, dn, rng.uniform(0.0, 2000.0, size=(Te, K)) * rng.uniform(0, 1, size=(Te, K)), lPe, lSe)[0]
        Cd = rng.uniform(0.0, 1.0, size=(Te, K))
        Md = rng.uniform(0.0, 2500.0, size=Te)
        Cd = Cd / Cd.sum(1, keepdims=True) * Md[:, None]
        fc_e = fco_core(nm, dn, Cd, Md, K)[0]
        for k, v in (("mux", om_e), ("rvs", rv_e), ("fco", fc_e)):
            good = bool(np.isfinite(v).all() and np.abs(v.astype(np.float32).astype(np.float64)).max() < 1)
            ok_d[k] &= good
            w[f"extreme_rows_ok_{k}"] = bool(w.get(f"extreme_rows_ok_{k}", True) and good)
        upd("extreme_mux_out_above_hcp", max(0.0, float((om_e - he).max())))
        upd("extreme_rvs_out_above_hcp", max(0.0, float((rv_e - he).max())))
    # ---------- strong-leader rows: unchanged within a stated tolerance ----------
    K = 200
    x, g = _stream(3, 600, K, 7, 0.25)
    f = H._fe(x, g)
    ref64 = H.hcp(f["zP"], f["zS"], K)
    strong = (f["zP"].max(1) >= 100.0) & (np.arange(600) >= 50)
    w["strong_rows"] = int(strong.sum())
    for k, fn in (("mux", mux), ("rvs", rvs), ("fco", fco)):
        o = fn(x, g, f)
        d = np.abs(o - ref64)[strong]
        upd(f"strong_{k}_max_abs_diff", d.max())
        for nm_, thr in (("eer", CUT_EER), ("far", THR_HCP)):
            cnt(f"strong_{k}_decision_flips_{nm_}", ((o[strong] >= thr) != (ref64[strong] >= thr)).sum())
    # ---------- weak persistent leader crowd: counts only (mechanism, not performance) ----------
    T = 1500
    x, g = _stream(0, T, K, 7, 0.03)
    f = H._fe(x, g)
    ref64 = H.hcp(f["zP"], f["zS"], K)
    late = np.arange(T) >= 500
    w["weak_zP_leader_median_t_ge_500"] = float(np.median(f["zP"].max(1)[late]))
    acc_h = ref64 >= CUT_EER
    w["weak_hcp_target_acc_t_ge_500"] = float(acc_h[late, 7].mean())
    w["weak_hcp_crowd_per_frame_t_ge_500"] = float((acc_h[late].sum(1) - acc_h[late, 7]).mean())
    for k, fn in (("mux", mux), ("rvs", rvs), ("fco", fco)):
        o = fn(x, g, f)
        a = o >= CUT_EER
        w[f"weak_{k}_target_acc_t_ge_500"] = float(a[late, 7].mean())
        w[f"weak_{k}_crowd_per_frame_t_ge_500"] = float((a[late].sum(1) - a[late, 7]).mean())
        w[f"weak_{k}_crowd_reduction"] = 1.0 - w[f"weak_{k}_crowd_per_frame_t_ge_500"] / max(w["weak_hcp_crowd_per_frame_t_ge_500"], 1e-9)
        w[f"weak_{k}_mechanism_ok"] = bool(w[f"weak_{k}_crowd_reduction"] >= 0.25
                                           and w[f"weak_{k}_target_acc_t_ge_500"] >= 0.9 * w["weak_hcp_target_acc_t_ge_500"])
    # ---------- MNG rescue (wrong strong memory leader, then weaker target): counts only ----------
    x, g = _mng_stream(5, 1000, K, 0.12)
    f = H._fe(x, g)
    ref64 = H.hcp(f["zP"], f["zS"], K)
    post = np.arange(1000) >= 480
    a_h = ref64 >= CUT_EER
    w["rescue_hcp_target_acc_post"] = int(a_h[post, 40].sum())
    w["rescue_hcp_wrong_acc_post"] = int(a_h[post, 11].sum())
    for k, fn in (("mux", mux), ("rvs", rvs), ("fco", fco)):
        a = fn(x, g, f) >= CUT_EER
        w[f"rescue_{k}_target_acc_post"] = int(a[post, 40].sum())
        w[f"rescue_{k}_wrong_acc_post"] = int(a[post, 11].sum())
    # ---------- extreme sequences through the whole pipeline ----------
    ex_ok = True
    rng = np.random.default_rng(11)
    for K in (200, 199):
        g = rng.normal(size=(K, 512)).astype(np.float32)
        seqs = []
        xx = rng.normal(size=(40, 512)).astype(np.float32)
        xx[:, :] = xx * rng.choice([1e-3, 1.0, 1e4], size=(40, 1)).astype(np.float32)
        seqs.append(xx)
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
        seqs.append(np.zeros((10, 512), np.float32))
        seqs.append(np.zeros((0, 512), np.float32))
        seqs.append(rng.normal(size=(60, 512)).astype(np.float32) * 60)      # al = 1 (norm ~ 1360 / 20 > 1)
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
    # ---------- high-norm stream: t < 10 identity of MUX/RVS needs |x| < 29.9 (reported, FCO is exact) ----------
    xh, gh = _synth_data(4, 120, 200, "switch", norm_hi=60.0)
    fh = H._fe(xh, gh)
    rh = H.hcp(fh["zP"], fh["zS"], 200)
    for k, fn in (("mux", mux), ("rvs", rvs), ("fco", fco)):
        w[f"highnorm_{k}_t_lt_10_changed_rows"] = int((fn(xh, gh, fh) != rh)[:10].any(1).sum())
        w[f"highnorm_{k}_neutral_mismatch"] = int((fn(xh, gh, fh, neutral=True) != rh).sum())
    # ---------- fresh state per call; FCO vote and tie conventions ----------
    xa, ga = _stream(8, 220, 200, 3, 0.1)
    xb2, _ = _stream(9, 180, 200, 3, 0.1)
    o1 = outputs_fn(xa, ga)
    outputs_fn(xb2, ga)                                   # a different sequence in between: no state may leak
    o3 = outputs_fn(xa, ga)
    w["fresh_state_mismatch"] = int(sum((o1[k] != o3[k]).sum() for k in ALL))
    rng = np.random.default_rng(17)
    gt = rng.normal(size=(6, 512))
    gt[4] = gt[1]                                         # two identical gallery rows: an exact tie
    xt = np.stack([gt[1] * 3.0, gt[2] * 0.5 + 1e-3 * rng.normal(size=512), np.zeros(512), rng.normal(size=512)])
    rt = np.linalg.norm(xt, axis=1)
    vt, rawt = fco_votes(xt, gt, rt)
    w["fco_vote_tie_split_ok"] = bool(np.array_equal(vt[0], np.eye(6)[1] * 0.5 + np.eye(6)[4] * 0.5))
    w["fco_vote_single_winner_ok"] = bool(vt[1].argmax() == 2 and vt[1].sum() == 1.0 and vt[3].sum() == 1.0)
    w["fco_vote_zero_norm_row_zero"] = bool((vt[2] == 0).all())
    wt = fco_weights(xt.astype(np.float32), gt.astype(np.float32))[0]
    w["fco_weight_zero_norm_zero"] = bool(wt[2] == 0.0 and (wt >= 0).all() and (wt <= 1).all())
    # ---------- report arithmetic self-test: NET = (impostor errors removed) - 199 (genuine errors added) ----------
    z = np.full((4, 5), -1.0, np.float32)                 # one sequence, 4 frames, 5 columns, target column 0
    ref_o, cand_a, cand_b = z.copy(), z.copy(), z.copy()
    ref_o[:, 0] = [1, 1, 1, -1]                           # one genuine error (frame 3) at thr 0
    ref_o[0, 1] = ref_o[1, 2] = 1.0                       # two impostor errors
    cand_a[:, 0] = [1, 1, 1, -1]                          # both impostors removed, nothing added
    cand_b[:, 0] = [1, 1, -1, -1]                         # one more genuine error ...
    cand_b[0, 1] = 1.0                                    # ... and one impostor removed (net: +1 - 199)
    ge0, ie0 = _seq_errors([ref_o], [0], 0.0)
    ga, ia = _seq_errors([cand_a], [0], 0.0)
    gb, ib = _seq_errors([cand_b], [0], 0.0)
    na, nb_ = _net_block(ga, ia, ge0, ie0, np.array([0]))["net_total"], _net_block(gb, ib, ge0, ie0, np.array([0]))["net_total"]
    w["net_selftest_ok"] = bool(ge0[0] == 1 and ie0[0] == 2 and na == 2 and nb_ == 1 - NET_W)
    # ---------- verdicts ----------
    ok_zero = {
        "mux": ("id_mux_tau0_mismatch", "id_mux_core_tau0_mismatch", "mux_exempt_not_bitwise_hcp", "mux_kap_out_of_unit",
                "mux_t_lt_10_changed_rows"),
        "rvs": ("id_rvs_s0_mismatch", "id_rvs_core_s0_mismatch", "rvs_exempt_not_bitwise_hcp", "rvs_psi_out_of_unit",
                "rvs_exempt_psi_not_1", "rvs_t_lt_10_changed_rows"),
        "fco": ("id_fco_e0_mismatch", "fco_w_nonfinite", "fco_e0_rows_not_bitwise_hcp", "fco_t_lt_10_changed_rows",
                "fco_t_lt_A_changed_rows", "highnorm_fco_t_lt_10_changed_rows", "highnorm_fco_neutral_mismatch")}
    tight = {
        "mux": ("mux_out_above_hcp", "mux_naive_tau_abs", "mux_naive_out_abs", "extreme_mux_out_above_hcp"),
        "rvs": ("rvs_out_above_hcp", "rvs_naive_ledger_abs", "rvs_naive_out_abs", "rvs_rank_vs_naive_abs",
                "extreme_rvs_out_above_hcp"),
        "fco": ("fco_w_above_1", "fco_w_below_0", "fco_F_below_0", "fco_naive_w_abs", "fco_naive_out_abs")}
    for k in PRIMARIES:
        ok_d[k] &= all(w[n] == 0 for n in ok_zero[k]) and all(w[n] <= 1e-12 for n in tight[k])
        ok_d[k] &= w[f"iv_prefix_{k}"] < 1e-6 and w[f"strict_inside_f32_{k}"] and w[f"extreme_rows_ok_{k}"]
        ok_d[k] &= w[f"strong_{k}_max_abs_diff"] <= 1e-3 and w[f"strong_{k}_decision_flips_eer"] == 0 \
            and w[f"strong_{k}_decision_flips_far"] == 0
        ok_d[k] &= w[f"{k}_rows_changed"] > 0 and w["strong_rows"] > 0                   # the checks are not vacuous
    ok_d["mux"] &= w["mux_p_eq_kappa_p_rel"] <= 1e-12 and w["mux_votes_mature"] > 0
    ok_d["rvs"] &= w["rvs_naive_ledger_abs"] <= 1e-12
    ok_d["fco"] &= w["fco_sumF_minus_e_rel"] <= 1e-12 and w["fco_f_vs_a_over_r2_sq_rel"] <= 1e-5 \
        and w["fco_literal_w_median_unfixed"] > 10.0 and w["fco_vote_tie_split_ok"] \
        and w["fco_vote_single_winner_ok"] and w["fco_vote_zero_norm_row_zero"] and w["fco_weight_zero_norm_zero"]
    shared = bool(w["id_omap_mismatch"] == 0 and w["id_hcp_ref_vs_x_fr_hcp_all64_mismatch"] == 0
                  and all(w[f"iv_prefix_{k}"] < 1e-6 for k in REFS) and ex_ok and ok_shared["range"]
                  and w["fresh_state_mismatch"] == 0 and w["net_selftest_ok"])
    out = {"synth": TAG, **w, "shared_ok": shared, "extreme_sequences_ok": ex_ok, "rows": rows,
           "design_ok": {k: bool(v) for k, v in ok_d.items()}}
    out["pass"] = bool(shared and all(ok_d.values()))
    print(json.dumps(out), flush=True)
    return out["pass"]


# =====================================================================================
# Label-free premise statistics (printed only; never changes a verdict)
# =====================================================================================
STD_EDGES = [0, 10, 50, 200, 1000, np.inf]
STD_NAMES = ["t<10", "10-50", "50-200", "200-1000", ">=1000"]
ZP_EDGES = [0, 5, 20, 50, 100, 300, np.inf]
ZP_NAMES = ["<5", "5-20", "20-50", "50-100", "100-300", ">=300"]
QS = [0.01, 0.1, 0.5, 0.9, 0.99]


def _bc(pb, mask, nb):
    return np.bincount(pb[mask], minlength=nb)


def _bw(pb, weights, nb):
    return np.bincount(pb, weights=weights, minlength=nb)


def _stats_job(task):
    tr, i = task
    x, g = (cc.D["s1"][i], cc.D["g1"]) if tr == "t1" else (cc.D["s2"][i], cc.D["g2"])
    f = H._fe(x, g)
    K, zP, zS = f["K"], f["zP"], f["zS"]
    T = x.shape[0]
    r = np.arange(T)
    nb = len(STD_NAMES)
    pb = np.digitize(r, STD_EDGES) - 1
    hcp = H.hcp(zP, zS, K)
    h32 = hcp.astype(np.float32)
    acc_h = hcp >= CUT_EER
    tail_h = hcp >= THR_HCP
    o_m, sm = mux(x, g, f, full=True)
    o_r, sr = rvs(x, g, f, full=True)
    o_f, sf = fco(x, g, f, full=True)
    lP, lS = sm["lP"], sm["lS"]
    exm = np.zeros((T, K), bool)
    exm[r, lP] = True
    exm[r, lS] = True
    n_ne = (acc_h & ~exm).sum(1)
    crowd = n_ne >= CROWD_N
    xb = np.asarray(x, np.float64)
    gb = np.asarray(g, np.float64)
    xn = xb / np.maximum(np.linalg.norm(xb, axis=1, keepdims=True), 1e-12)
    gn = gb / np.maximum(np.linalg.norm(gb, axis=1, keepdims=True), 1e-12)
    cm_lead = (np.cumsum(xn, 0) @ gn.T).argmax(1)         # causal cumulative-mean cosine leader (example.py method 1)
    res = {"n": T, "entries": T * K, "frames": np.bincount(pb, minlength=nb),
           "hcp_pool_crowd": _bw(pb, np.where(crowd, n_ne, 0), nb), "hcp_pool_neigh": _bw(pb, np.where(~crowd, n_ne, 0), nb),
           "frames_crowd": _bc(pb, crowd, nb), "hcp_tail": int(tail_h.sum()),
           "hcp_acc": _bw(pb, acc_h.sum(1), nb), "hcp_acc_exempt": _bw(pb, (acc_h & exm).sum(1), nb)}
    outs = {"mux": o_m, "rvs": o_r, "fco": o_f}
    for k, o in outs.items():
        o32 = o.astype(np.float32)
        a_d = o >= CUT_EER
        res[k] = {"rows_changed": _bc(pb, (o32 != h32).any(1), nb), "entries_changed": _bw(pb, (o32 != h32).sum(1), nb),
                  "flips_crowd": _bw(pb, np.where(crowd, (acc_h & ~a_d).sum(1), 0), nb),
                  "flips_neigh": _bw(pb, np.where(~crowd, (acc_h & ~a_d).sum(1), 0), nb),
                  "raises": _bw(pb, (~acc_h & a_d).sum(1), nb),
                  "tail_lost": int((tail_h & (o < THR_HCP)).sum()), "tail_gained": int((~tail_h & (o >= THR_HCP)).sum()),
                  "max64": float(np.abs(o).max()), "max32": float(np.abs(o32.astype(np.float64)).max()),
                  "ge1_32": int((np.abs(o32) >= 1).sum()), "le_m999_32": int((o32 <= -0.999).sum())}
    # ---- MUX: m_t = omega p_inc, tau, cap-active pools (A = m >= .65, A_partial = 0 < m < .65, U = m == 0) ----
    m = sm["om"] * sm["p_inc"]
    cls = np.where(m >= CAP_M, 2, np.where(m > 0, 1, 0))
    pool = np.zeros((3, 2, nb))                           # [m == 0, 0 < m < .65, m >= .65] x [crowd, neighbour] x bin
    for c in range(3):
        for j, rowsel in enumerate((crowd, ~crowd)):
            pool[c, j] = _bw(pb, np.where((cls == c) & rowsel, n_ne, 0), nb)
    res["mux"].update({"m_zero": _bc(pb, m == 0, nb), "m_cap": _bc(pb, m >= CAP_M, nb), "m_samp": [m[pb == b].astype(np.float32) for b in range(nb)],
                       "tau_samp": [sm["tau"][pb == b].astype(np.float32) for b in range(nb)], "pool": pool,
                       "cap_frames": int((m >= CAP_M).sum()), "cap_disagree_cmean": int(((m >= CAP_M) & (lP != cm_lead)).sum()),
                       "lS_eq_lP": _bc(pb, lS == lP, nb)})
    # ---- RVS: psi on non-exempt entries; ledger share; ledger leader ----
    psi = sr["psi"]
    Mm, smax = sr["M1"], sr["s"].max(1)
    share = np.where(Mm > 0, smax / np.where(Mm > 0, Mm, 1.0), np.nan)
    led = sr["s"].argmax(1)
    ne = ~exm
    nea = ne & acc_h
    res["rvs"].update({"n_ne": _bw(pb, ne.sum(1), nb), "n_ne_psi_eq1": _bw(pb, (ne & (psi == 1.0)).sum(1), nb),
                       "n_ne_psi_lt5": _bw(pb, (ne & (psi < 0.5)).sum(1), nb), "n_ne_psi_lt1": _bw(pb, (ne & (psi < 0.1)).sum(1), nb),
                       "n_nea": _bw(pb, nea.sum(1), nb), "n_nea_psi_lt5": _bw(pb, (nea & (psi < 0.5)).sum(1), nb),
                       "n_nea_psi_lt1": _bw(pb, (nea & (psi < 0.1)).sum(1), nb),
                       "share_samp": [share[(pb == b) & np.isfinite(share)].astype(np.float32) for b in range(nb)],
                       "M_samp": [Mm[pb == b].astype(np.float32) for b in range(nb)],
                       "led_eq_zP": _bc(pb, (smax > 0) & (led == lP), nb), "led_frames": _bc(pb, smax > 0, nb),
                       "inc_frames": int(((share >= 0.5)).sum()), "inc_disagree_cmean": int(((share >= 0.5) & (led != cm_lead)).sum())})
    # ---- FCO: M > A, ownership concentration, owner vs zP top-1, perturbation bound by row-max zP ----
    C, M, e = sf["C"], sf["M"], sf["e"]
    pos = e > 0
    Cs = np.sort(C, 1)
    c1 = np.where(M > 0, Cs[:, -1] / np.where(M > 0, M, 1.0), np.nan)
    c12 = np.where(M > 0, (Cs[:, -1] + Cs[:, -2]) / np.where(M > 0, M, 1.0), np.nan)
    own = C.argmax(1)
    zmax = np.maximum(zP.max(1), 0.0)
    zb = np.digitize(zmax, ZP_EDGES) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        ub = (np.log1p(sf["F"] / sf["num"]) + np.log1p(sf["G"] / sf["den"])).max(1)
    dout = np.abs(o_f - hcp).max(1)
    res["fco"].update({"M_gt_A": _bc(pb, pos, nb), "c1_samp": [c1[(pb == b) & pos].astype(np.float32) for b in range(nb)],
                       "c12_samp": [c12[(pb == b) & pos].astype(np.float32) for b in range(nb)],
                       "own_eq_zP": _bc(pb, (M > 0) & (own == lP), nb), "own_frames": _bc(pb, M > 0, nb),
                       "own_frames_pos": int(pos.sum()), "own_disagree_cmean": int((pos & (own != cm_lead)).sum()),
                       "z_frames": np.bincount(zb, minlength=len(ZP_NAMES)), "z_pos": np.bincount(zb[pos], minlength=len(ZP_NAMES)),
                       "ub_samp": [ub[(zb == b) & pos].astype(np.float32) for b in range(len(ZP_NAMES))],
                       "dout_samp": [dout[(zb == b) & pos].astype(np.float32) for b in range(len(ZP_NAMES))],
                       "w_samp": [sf["w"][pb == b].astype(np.float32) for b in range(nb)]})
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
    out = {"stats": TAG, "quantiles": QS, "bins": STD_NAMES, "zP_bins": ZP_NAMES,
           "note": ("label-free; descriptive only, no kill rule. 'accepted' = HCP output >= CUT_EER (odds .539); crowd row = "
                    ">= 10 accepted non-exempt columns (exempt = P-leader, S-leader); cmean disagreement is not damage; "
                    "flips = HCP-accepted entries the design rejects, raises = the reverse")}
    for tr, rs in acc.items():
        def tot(*keys):
            return np.sum([np.asarray(_get(r, keys), np.float64) for r in rs], axis=0)

        def cat(*keys):
            return [np.concatenate([_get(r, keys)[b] for r in rs]) for b in range(len(_get(rs[0], keys)))]
        fr = tot("frames")
        d = {"frames": int(fr.sum()), "frames_by_bin": fr.astype(int).tolist(),
             "hcp_accepted_nonexempt_entries": {"crowd_rows_by_bin": tot("hcp_pool_crowd").astype(int).tolist(),
                                                "neighbour_rows_by_bin": tot("hcp_pool_neigh").astype(int).tolist(),
                                                "crowd_row_frames_by_bin": tot("frames_crowd").astype(int).tolist()},
             "hcp_accepted_entries_exempt_share_by_bin": [_share(a, b) for a, b in zip(tot("hcp_acc_exempt"), tot("hcp_acc"))],
             "hcp_tail_entries": int(tot("hcp_tail"))}
        for k in PRIMARIES:
            dk = {"rows_changed_share_by_bin": [_share(a, b) for a, b in zip(tot(k, "rows_changed"), fr)],
                  "rows_changed_t_lt_10": int(tot(k, "rows_changed")[0]),
                  "entries_changed_by_bin": tot(k, "entries_changed").astype(int).tolist(),
                  "flips_crowd_rows_by_bin": tot(k, "flips_crowd").astype(int).tolist(),
                  "flips_neighbour_rows_by_bin": tot(k, "flips_neigh").astype(int).tolist(),
                  "flips_total": int(tot(k, "flips_crowd").sum() + tot(k, "flips_neigh").sum()),
                  "raises_by_bin": tot(k, "raises").astype(int).tolist(), "raises_total": int(tot(k, "raises").sum()),
                  "tail_lost": int(tot(k, "tail_lost")), "tail_gained": int(tot(k, "tail_gained")),
                  "max_abs_float64": max(r[k]["max64"] for r in rs), "max_abs_float32": max(r[k]["max32"] for r in rs),
                  "n_abs_ge1_float32": int(tot(k, "ge1_32")), "n_le_-0.999_float32": int(tot(k, "le_m999_32"))}
            if k == "mux":
                dk.update({"m_eq_0_share_by_bin": [_share(a, b) for a, b in zip(tot(k, "m_zero"), fr)],
                           "cap_active_share_by_bin": [_share(a, b) for a, b in zip(tot(k, "m_cap"), fr)],
                           "m_quantiles_by_bin": [_q(v) for v in cat(k, "m_samp")],
                           "tau_quantiles_by_bin": [_q(v) for v in cat(k, "tau_samp")],
                           "pools_hcp_accepted_nonexempt_entries_by_bin": {
                               nm: {"crowd_rows": tot(k, "pool")[c, 0].astype(int).tolist(),
                                    "neighbour_rows": tot(k, "pool")[c, 1].astype(int).tolist()}
                               for c, nm in enumerate(("m_eq_0_unaddressable", "0<m<.65_partial", "m>=.65_cap_active"))},
                           "cap_active_frames": int(tot(k, "cap_frames")),
                           "cap_active_incumbent_ne_cmean_leader_share": _share(tot(k, "cap_disagree_cmean"), tot(k, "cap_frames")),
                           "S_leader_eq_P_leader_share_by_bin": [_share(a, b) for a, b in zip(tot(k, "lS_eq_lP"), fr)]})
            elif k == "rvs":
                dk.update({"psi_eq_1_share_nonexempt_by_bin": [_share(a, b) for a, b in zip(tot(k, "n_ne_psi_eq1"), tot(k, "n_ne"))],
                           "psi_lt_.5_share_nonexempt_by_bin": [_share(a, b) for a, b in zip(tot(k, "n_ne_psi_lt5"), tot(k, "n_ne"))],
                           "psi_lt_.1_share_nonexempt_by_bin": [_share(a, b) for a, b in zip(tot(k, "n_ne_psi_lt1"), tot(k, "n_ne"))],
                           "psi_lt_.5_share_hcp_accepted_nonexempt_by_bin": [_share(a, b) for a, b in zip(tot(k, "n_nea_psi_lt5"), tot(k, "n_nea"))],
                           "psi_lt_.1_share_hcp_accepted_nonexempt_by_bin": [_share(a, b) for a, b in zip(tot(k, "n_nea_psi_lt1"), tot(k, "n_nea"))],
                           "smax_over_M_quantiles_by_bin": [_q(v) for v in cat(k, "share_samp")],
                           "M_quantiles_by_bin": [_q(v) for v in cat(k, "M_samp")],
                           "ledger_leader_eq_zP_top1_share_by_bin": [_share(a, b) for a, b in zip(tot(k, "led_eq_zP"), tot(k, "led_frames"))],
                           "incumbent_frames_smax_over_M_ge_.5": int(tot(k, "inc_frames")),
                           "incumbent_ne_cmean_leader_share": _share(tot(k, "inc_disagree_cmean"), tot(k, "inc_frames"))})
            else:
                z_fr, z_pos = tot(k, "z_frames"), tot(k, "z_pos")
                dk.update({"share_M_gt_A_by_bin": [_share(a, b) for a, b in zip(tot(k, "M_gt_A"), fr)],
                           "share_M_gt_A_overall": _share(tot(k, "M_gt_A").sum(), fr.sum()),
                           "C1_over_M_quantiles_by_bin_M_gt_A": [_q(v) for v in cat(k, "c1_samp")],
                           "C1C2_over_M_quantiles_by_bin_M_gt_A": [_q(v) for v in cat(k, "c12_samp")],
                           "owner_eq_zP_top1_share_by_bin": [_share(a, b) for a, b in zip(tot(k, "own_eq_zP"), tot(k, "own_frames"))],
                           "owner_ne_cmean_leader_share_M_gt_A": _share(tot(k, "own_disagree_cmean"), tot(k, "own_frames_pos")),
                           "w_quantiles_by_bin": [_q(v) for v in cat(k, "w_samp")],
                           "by_row_max_zP": {"frames": z_fr.astype(int).tolist(), "frames_M_gt_A": z_pos.astype(int).tolist(),
                                             "bound_max_logodds_quantiles_M_gt_A": [_q(v) for v in cat(k, "ub_samp")],
                                             "max_abs_out_change_quantiles_M_gt_A": [_q(v) for v in cat(k, "dout_samp")]}})
            d[k] = dk
        out[tr] = d
    print(json.dumps(out), flush=True)


def _get(r, keys):
    for k in keys:
        r = r[k]
    return r


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


def _seq_errors(outs_, tg, thr):
    """Per-sequence genuine errors (target entry < thr) and impostor errors (non-target entries >= thr)."""
    ge, ie = [], []
    for o, t in zip(outs_, tg):
        a = o >= thr
        gt = a[:, t]
        ge.append(int((~gt).sum()))
        ie.append(int(a.sum() - gt.sum()))
    return np.array(ge, np.int64), np.array(ie, np.int64)


def _net_block(ge, ie, ge_ref, ie_ref, worst):
    """NET = (impostor errors removed) - 199 (genuine errors added) at the reference threshold, split by the
    reference's worst-impostor sequences (worst 1, worst 2-5, rest). Positive NET favours the candidate."""
    groups = {"worst1": worst[:1], "worst2-5": worst[1:5], "rest": worst[5:]}
    blk = {"imp_err_total": int(ie.sum()), "gen_err_total": int(ge.sum()),
           "imp_err_worst5": int(ie[worst[:5]].sum()), "imp_err_worst5_ref": int(ie_ref[worst[:5]].sum()),
           "net_total": int((ie_ref.sum() - ie.sum()) - NET_W * (ge.sum() - ge_ref.sum())), "groups": {}}
    for nm, idx in groups.items():
        rem, add = int(ie_ref[idx].sum() - ie[idx].sum()), int(ge[idx].sum() - ge_ref[idx].sum())
        blk["groups"][nm] = {"imp_removed": rem, "gen_added": add, "net": rem - NET_W * add}
    return blk


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
            "early_err_at_eer_thr": round(float(sum(bins[b]["gen_err"] + bins[b]["imp_err"] / 199.0
                                                    for b in CHECKS_EARLY)), 1),
            "top3_gen_err_at_eer_thr_all": _top3_share(outs_, t1, met["eer_thr"], 0),
            "top3_gen_err_at_eer_thr_t_ge10": _top3_share(outs_, t1, met["eer_thr"], 10),
            "at_hcp_ref_eer_thr": _net_block(ge, ie, ge_ref, ie_ref, worst)}
    out["t1"] = t1rep
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
            "synth_line_present_and_shared_checks_ok": bool(syn and syn.get("shared_ok")),
            "refs_checks_ok": bool(chk.get("hcp_ref")) and bool(chk.get("ospc_ref")) and
            bad["hcp_ref"] == 0 and bad["ospc_ref"] == 0}
    global_stop = not all(stop.values())
    design_stop = {k: {"checks_pass_and_finite_in_range": bool(chk.get(k)) and bad[k] == 0,
                       "eer_full_report_vs_row_ok": bool(t1rep[k]["eer_full_match_1e-6"]),
                       "structural_identities_ok_in_synth_line": bool(syn and syn.get("design_ok", {}).get(k))}
                   for k in PRIMARIES}
    out2 = {"report_gate": TAG, "run_repro": rr, "cal_repro": cr, "ref_eer_repro": eer_repro,
            "hcp_ref_thr_repro": hcp_thr_b, "hcp_ref_Bstar_repro": hcp_B,
            "stop_checks_global": stop, "stop_checks_per_design": design_stop}
    # ---- frozen gate (applied once, mechanically; same-run hcp_ref) ----
    fbe = {k: BE_A + BE_B * prox[k][B_STAR] for k in PROX}
    hr = t1rep["hcp_ref"]
    gate, verdict, stretch = {}, {}, {}
    for k in PRIMARIES:
        p = t1rep[k]
        terms = {"eer_full<=hcp_ref-.00006": _t(p["eer_full"], hr["eer_full"] - GATE_EER_MARGIN,
                                                p["eer_full"] <= hr["eer_full"] - GATE_EER_MARGIN),
                 "BE_fc<=BE_fc(hcp_ref)+.010": _t(fbe[k], fbe["hcp_ref"] + GATE_BE_MARGIN,
                                                 fbe[k] <= fbe["hcp_ref"] + GATE_BE_MARGIN),
                 "checks_pass": _t(chk.get(k), True, bool(chk.get(k))),
                 "t1_neg1_share<=.001": _t(p["t1_neg1_share"], 0.001, p["t1_neg1_share"] <= 0.001)}
        gate[k] = {"lane": LANE[k], "terms": terms}
        # report-only: the original stretch target; meeting only the -.00006 term is never a pass of this one
        stretch[k] = _t(p["eer_full"], hr["eer_full"] - STRETCH_EER_MARGIN,
                        p["eer_full"] <= hr["eer_full"] - STRETCH_EER_MARGIN)
        if global_stop:
            verdict[k] = "STOP (global, no verdict): " + ", ".join(s for s, v in stop.items() if not v)
        elif not all(design_stop[k].values()):
            verdict[k] = "STOP (design, no verdict): " + ", ".join(s for s, v in design_stop[k].items() if not v)
        else:
            failed = [s for s, v in terms.items() if not v[2]]
            verdict[k] = "PASS" if not failed else "FAIL: " + ", ".join(failed)
    ranking = {"T1_by_eer_full": sorted([k for k in PRIMARIES if verdict[k] == "PASS"],
                                        key=lambda k: t1rep[k]["eer_full"])}

    # ---- report-only disclosures ----
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
        rep_only[k] = {"eer_full": t1rep[k]["eer_full"], "eer_full_minus_hcp_ref": t1rep[k]["eer_full"] - hr["eer_full"],
                       "frr_far1e5": t1rep[k]["frr_far1e5"],
                       "far_at_frr1e5": t1rep[k]["far_at_frr1e5"], "usp_aar": pk["usp_aar"],
                       "usp_run": pk["usp_run"], "usp_frr": pk["usp_frr"], "t1_band_reject": pk["t1_band_reject"],
                       "t2_pre_reject_pt": pk["t2_pre_reject_pt"], "D*(" + D_STAR + ")": pk[D_STAR],
                       "B*(" + B_STAR + ")": pk[B_STAR], "BE_fc": round(fbe[k], 4),
                       "own_eer_thr": t1rep[k]["eer_thr"], "own_far1e5_thr": t1rep[k]["far1e5_thr"],
                       "eer_thr_bins": t1rep[k]["eer_thr_bins"], "hcp_ref_eer_thr_bins": t1rep[k]["hcp_ref_eer_thr_bins"],
                       "gen_err_t_ge10_at_eer_thr": t1rep[k]["gen_err_t_ge10_at_eer_thr"],
                       "top3_gen_err_at_eer_thr_all": t1rep[k]["top3_gen_err_at_eer_thr_all"],
                       "top3_gen_err_at_eer_thr_t_ge10": t1rep[k]["top3_gen_err_at_eer_thr_t_ge10"],
                       "at_hcp_ref_eer_thr": t1rep[k]["at_hcp_ref_eer_thr"]}
    hcp_vs_recorded = {m: (got, v) for m, v in HCP_RECORDED.items()
                       for got in [t1rep["hcp_ref"].get(m, prox["hcp_ref"].get(m))]}
    out2.update({"gate": gate, "verdict": verdict, "ranking": ranking,
                 "stretch_eer_full<=hcp_ref-.0002_report_only": stretch,
                 "be_forecast": {k: round(fbe[k], 4) for k in PROX},
                 "forecasts_report_only": fc, "report_only": rep_only,
                 "premise_statistics_from_run_log": None if stt is None else {"present": True},
                 "hcp_ref_vs_recorded_hcp_run": hcp_vs_recorded, "pos_bins": ar.POS_NAMES,
                 "note": ("A pass is a candidate for the user's decision, not a submission; no artifact is built. "
                          "A design meeting only the -.00006 term is never a pass of the -.0002 stretch target.")})
    print(json.dumps(out2), flush=True)
    print(json.dumps({"verdict": verdict, "stretch_minus_.0002_report_only": {k: v[2] for k, v in stretch.items()},
                      "ranking": ranking}), flush=True)


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
