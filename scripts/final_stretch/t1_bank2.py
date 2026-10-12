"""T1-specialist bank2 candidate: the bank processor plus a label-free ownership extension (same on both tracks).

Evaluated in docs/experiments/20261009_final_stretch_t1_direction.md (`analyze_t1_ownext2.py` variant zl_16_or_pe_160,
T1 proxy .0027305 on the 971136 scores, against the bank's .0027887).

  bank : t1_bank.bank — 971136's processor, the opening override and own2 ownership, unchanged
  ext  : in a frame t >= 10 that own2 leaves without an owner, let a be the leader of the bank output row (argmax,
         lowest index on ties) and run the number of consecutive frames up to t whose bank row leader is a. If the HCP
         memory z of column a, zP[t, a], is >= 16 or run >= 160, the frame is owned by a: a keeps its bank score and
         every other column becomes -1.
The leader and its run are read from the bank output (before the extension), so the extension of one frame never
changes another frame. All state is per sequence and uses frames <= t only. `bank2` is the vectorised processor;
`bank2_ref` is an independent per-frame loop (own2 and the extension re-implemented frame by frame) used as the
builder's bit-identity reference.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ogata"))
import numpy as np  # noqa: E402

import analyze_hvc_stage0 as HV  # noqa: E402
import t1_bank as B  # noqa: E402
import x_fr_hcp  # noqa: E402

EXT_FROM = 10
ZLEAD_OWN = 16.0
RUN_OWN = 160


def leader_runs(s):
    """Row leader of a (T x K) score matrix (argmax, lowest index on ties) and its consecutive-run length."""
    a = np.asarray(s, np.float64).argmax(1)
    run = np.ones(a.shape[0], np.int64)
    for t in range(1, a.shape[0]):
        if a[t] == a[t - 1]:
            run[t] = run[t - 1] + 1
    return a, run


def extend(s, x, g):
    """Apply the ownership extension to a bank output s (T x K float32); returns a new float32 array."""
    s = np.asarray(s, np.float32).copy()
    T = s.shape[0]
    own = B.owners(x, g)
    zP = x_fr_hcp._fe(x, g)["zP"]
    a, run = leader_runs(s)
    t = np.arange(T)
    ext = (own < 0) & (t >= EXT_FROM) & ((zP[t, a] >= ZLEAD_OWN) | (run >= RUN_OWN))
    idx = np.nonzero(ext)[0]
    if idx.size:
        keep = s[idx, a[idx]].copy()
        s[idx] = -1.0
        s[idx, a[idx]] = keep
    return s


def bank2(x, g):
    return extend(B.bank(x, g), x, g)


def bank2_ref(x, g):
    """Per-frame loop reference: the opening override, own2 and the extension are applied frame by frame with
    independent bookkeeping; the numeric primitives (base scores, row norms, the block matrix product, the LOO z and
    zP) are the same calls as in `bank2`, so the result must be bit-identical."""
    s = np.asarray(B._BASE(x, g), np.float32).copy()
    xb = np.asarray(x, np.float64)
    T, K = s.shape
    nrm = np.linalg.norm(xb, axis=1)
    zall = HV.verify(x, g)[1]
    zP = x_fr_hcp._fe(x, g)["zP"]
    e = min(B.EARLY, T)
    u = B._unit(xb[:e])
    C = np.zeros_like(u)
    acc = np.zeros(xb.shape[1])
    for t in range(e):
        acc = acc + u[t]
        C[t] = acc
    early = B._loo_z(B._unit(C) @ B._unit(g).T).astype(np.float32)
    cnt = np.zeros(K, np.int64)
    prev, run = -1, 0
    for t in range(T):
        if t < e:
            s[t] = np.clip(B.TH0 + B.KAPPA * (early[t] - B.TAU), -1.0, 1.0)
        if nrm[t] >= B.CLEAN_NORM and nrm[t] > 0:
            cnt = cnt + (zall[t] >= HV.Z0)
        order = np.argsort(-cnt, kind="stable")
        owner = -1
        if cnt[order[0]] >= 1 and cnt[order[0]] > cnt[order[1]]:
            owner = int(order[0])
        else:
            zo = np.argsort(-zP[t], kind="stable")
            if zP[t, zo[0]] >= B.ZP_OWN and zP[t, zo[0]] > zP[t, zo[1]] and zP[t, zo[1]] <= B.ZP_RUNNER:
                owner = int(zo[0])
        if owner >= 0:
            keep = s[t, owner]
            s[t] = -1.0
            s[t, owner] = keep
        row = s[t].astype(np.float64)
        lead = int(np.flatnonzero(row == row.max())[0])
        run = run + 1 if lead == prev else 1
        prev = lead
        if owner < 0 and t >= EXT_FROM and (zP[t, lead] >= ZLEAD_OWN or run >= RUN_OWN):
            keep = s[t, lead]
            s[t] = -1.0
            s[t, lead] = keep
    return s


def outputs_fn(x, g):
    return {"bank2": bank2_ref(x, g)}
