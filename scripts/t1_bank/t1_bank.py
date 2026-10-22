"""T1-specialist bank candidate: 971136's processor plus two label-free causal post-steps (same on both tracks).

Evaluated in docs/experiments/20261009_final_stretch_t1_direction.md (section F, `analyze_t1_bank.py` own2 + early 1.0,
T1 proxy .0027886547923785037 on the 971136 scores).

  base  : candidates.t2_family_bl('dframe_nh') — the frozen config that built 971136
  early : for t < 10 every column becomes TH0 + KAPPA * (cmz - TAU), clipped to [-1, 1]; cmz is the raw row LOO z of
          the cumulative mean of unit frames 0..t (float64, cast to float32); TH0 is 971136's EER threshold, used as a
          fixed constant; KAPPA = .05, TAU = 1.0
  own2  : owner = strict leader of the running count of verified clean frames (raw LOO z >= Z0, norm >= 20), m = 1;
          while none exists, the strict leader of the HCP memory z zP if zP >= 16 and the runner-up <= 4.
          Once an owner exists at frame t, every other column of that frame becomes -1 (the owner keeps its score).
The early step is applied first, then ownership, exactly as in the evaluation. All state is per sequence and uses
frames <= t only. `bank` is the vectorised processor; `bank_ref` is an independent per-frame loop used as the
builder's bit-identity reference.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "core"))
import numpy as np  # noqa: E402

import analyze_hvc_stage0 as HV  # noqa: E402
import candidates  # noqa: E402
import x_fr_hcp  # noqa: E402

TH0 = np.float32(-0.08167967945337296)
KAPPA = 0.05
TAU = 1.0
EARLY = 10
CLEAN_NORM = 20.0
ZP_OWN, ZP_RUNNER = 16.0, 4.0
_BASE = candidates.t2_family_bl("dframe_nh")


def _unit(a):
    a = np.asarray(a, np.float64)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.where(n > 0, n, 1.0)


def _loo_z(r):
    K = r.shape[1]
    mu = (r.sum(1, keepdims=True) - r) / (K - 1)
    var = ((r * r).sum(1, keepdims=True) - r * r) / (K - 1) - mu * mu
    sd = np.sqrt(np.maximum(var, 0.0))
    return np.where(sd > 0, (r - mu) / np.where(sd > 0, sd, 1.0), 0.0)


def _top2(st):
    p = np.partition(st, -2, axis=1)
    return st.argmax(1), p[:, -1], p[:, -2]


def owners(x, g):
    """own2 per frame (int64, -1 = no owner)."""
    xb = np.asarray(x, np.float64)
    nrm = np.linalg.norm(xb, axis=1)
    v, _, _ = HV.verify(x, g)
    cnt = np.cumsum(v & (nrm >= CLEAN_NORM)[:, None], axis=0)
    arg, top, second = _top2(cnt)
    own = np.where((top >= 1) & (top > second), arg, -1)
    zP = x_fr_hcp._fe(x, g)["zP"]
    a2, t2, s2 = _top2(zP)
    alt = np.where((t2 >= ZP_OWN) & (t2 > s2) & (s2 <= ZP_RUNNER), a2, -1)
    return np.where(own >= 0, own, alt).astype(np.int64)


def early_cmz(x, g):
    xb = np.asarray(x, np.float64)
    e = min(EARLY, xb.shape[0])
    return _loo_z(_unit(np.cumsum(_unit(xb[:e]), axis=0)) @ _unit(g).T).astype(np.float32)


def post(s, x, g):
    """Apply the early step and ownership to a (T x K) float32 base score matrix."""
    s = np.asarray(s, np.float32).copy()
    cmz = early_cmz(x, g)
    e = cmz.shape[0]
    s[:e] = np.clip(TH0 + KAPPA * (cmz - TAU), -1.0, 1.0)
    ow = owners(x, g)
    has = np.nonzero(ow >= 0)[0]
    if has.size:
        keep = s[has, ow[has]].copy()
        s[has] = -1.0
        s[has, ow[has]] = keep
    return s


def bank(x, g):
    return post(_BASE(x, g), x, g)


def bank_ref(x, g):
    """Per-frame loop reference: independent owner bookkeeping and indexing; the numeric primitives (row norms,
    the block matrix product and the LOO z) are the same calls as in `post`, so the result must be bit-identical."""
    s = np.asarray(_BASE(x, g), np.float32).copy()
    xb = np.asarray(x, np.float64)
    T, K = s.shape
    nrm = np.linalg.norm(xb, axis=1)
    zall = HV.verify(x, g)[1]
    zP = x_fr_hcp._fe(x, g)["zP"]
    e = min(EARLY, T)
    u = _unit(xb[:e])
    C = np.zeros_like(u)
    acc = np.zeros(xb.shape[1])
    for t in range(e):
        acc = acc + u[t]
        C[t] = acc
    early = _loo_z(_unit(C) @ _unit(g).T).astype(np.float32)
    cnt = np.zeros(K, np.int64)
    for t in range(T):
        if t < e:
            s[t] = np.clip(TH0 + KAPPA * (early[t] - TAU), -1.0, 1.0)
        if nrm[t] >= CLEAN_NORM and nrm[t] > 0:
            cnt = cnt + (zall[t] >= HV.Z0)
        order = np.argsort(-cnt, kind="stable")
        owner = -1
        if cnt[order[0]] >= 1 and cnt[order[0]] > cnt[order[1]]:
            owner = int(order[0])
        else:
            zo = np.argsort(-zP[t], kind="stable")
            if zP[t, zo[0]] >= ZP_OWN and zP[t, zo[0]] > zP[t, zo[1]] and zP[t, zo[1]] <= ZP_RUNNER:
                owner = int(zo[0])
        if owner >= 0:
            keep = s[t, owner]
            s[t] = -1.0
            s[t, owner] = keep
    return s


def outputs_fn(x, g):
    return {"bank": bank_ref(x, g)}
