"""T1-specialist candidate b8d: b8v (bank8 + P0) with score-space database augmentation (DBS10) in bank8's free rows
after the jump (same on both tracks).

Round P (docs/experiments/20261009_final_stretch_t1_direction.md, 2026-10-12): E1's designs DBS10
(`analyze_t1_pa02.py`, rows in bank8 states mid / late_new / flat / late_gated, exact pooled +226.6 gen-eq, bankable)
and AH (`analyze_t1_pa03.py`, the same mechanism in late_after_held rows, +27.4, bankable); their union, the evaluated
STACK (+251.0), is every row bank8 leaves free (not owned, not extended) at t >= 10 and at or after the detected jump
(the jump is detected before frame 60 = t1_bank4.JUMP_UNTIL < 70, so these are exactly the union of those states
on Track 1, where every sequence has a detected jump). Causal correction against the analysis labels: `state7` sets
j0 = 0 when no jump is ever detected, which labels rows 10..59 of a no-jump sequence as post-jump although at those
frames a jump could still be detected later (a future dependence); this processor treats a row as post-jump only
when the jump was detected at or before t, or when t >= 60 without a detection (no jump can be detected any more).
On Track 1 the two definitions select the same rows; they differ only in rows 10..59 of Track-2 no-jump sequences.
Disclosure: the family, k = 10 and the rows were chosen after an evaluator-only diagnostic on the released Track-1
sequences with pseudo targets (`analyze_t1_pa01.py`).

  rows   : own < 0 (bank8's label-free ownership, `t1_bank4.owners_r` with the restarted HCP memory), not extended
           (t >= 10 and (the row leader's zP >= 16 or its run >= 160), bank8's own extension rule), t >= 10, and
           post-jump: a jump J detected at or before t (t >= J), or no jump detected and t >= 60.
  score  : s = bank8's row t (float64), z = (s - mean s) / sd s over the 200 columns (population sd),
           r_j = z_j + (1/10) sum_{i in NN10(j)} z_i, NN10(j) = the 10 other gallery columns with the largest cosine to
           column j (unit gallery rows, stable argsort, ties by column index).
  output : `analyze_t1_v13.transplant`: the row's 200 bank8 values re-assigned in descending order of r (ties: bank8
           value, then column), so the row keeps its value multiset and accept count at every threshold; sd 0,
           non-finite or constant r keep the row. Then P0 (`t1_b8v.compose`) unchanged: exact -1 entries become
           -1 + .025 (1 + b), b = 943955.
Every quantity of row t uses frames <= t of the sequence and the gallery only; fresh state per sequence; the same code
and constants on both tracks. `b8d(x, g)` is the build path: bank8 and its row states on the fixed 4096-row block
(`t1_bank8p.pad`), DBS one row at a time. `outputs_fn(x, g)` is the builder's bit-identity reference: bank8 by the
per-frame loop `t1_bank8.bank8_ref` on the unpadded sequence, row states from the unpadded base, DBS on the block of
all selected rows at once (the arithmetic of `analyze_t1_pa02.apply`).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ogata"))
sys.path.insert(0, str(HERE.parents[1]))
import numpy as np  # noqa: E402

import analyze_t1_v13 as V13  # noqa: E402
import t1_b8v as T8V  # noqa: E402
import t1_bank as B  # noqa: E402
import t1_bank2 as B2  # noqa: E402
import t1_bank4 as B4  # noqa: E402
import t1_bank8 as B8  # noqa: E402
import t1_bank8p as B8P  # noqa: E402

K_NN = 10


def gallery_nn(g):
    """NN10 per gallery column (K x 10 int), as in `analyze_t1_pa02.gal`."""
    gu = B._unit(np.asarray(g, np.float64))
    Cm = gu @ gu.T
    np.fill_diagonal(Cm, -np.inf)
    return np.argsort(-Cm, axis=1, kind="stable")[:, :K_NN]


def owner_lead(base, x, g, zP):
    """bank8's ownership and leader rule (`analyze_t1_bank5.masks4`, the same steps as in `t1_bank8.post8`)."""
    s = np.asarray(base, np.float32).copy()
    cmz = B.early_cmz(x, g)
    s[:cmz.shape[0]] = np.clip(B.TH0 + B.KAPPA * (cmz - B.TAU), -1.0, 1.0)
    own = B4.owners_r(x, g, zP)
    has = np.nonzero(own >= 0)[0]
    if has.size:
        keep = s[has, own[has]].copy()
        s[has] = -1.0
        s[has, own[has]] = keep
    a, run = B2.leader_runs(s)
    t = np.arange(s.shape[0])
    lead_ok = (zP[t, a] >= B2.ZLEAD_OWN) | (run >= B2.RUN_OWN)
    return own, lead_ok


def dbs_rows(base, x, g, owner_fn=owner_lead):
    """Boolean mask of the DBS rows (free, t >= 10, post-jump as known at frame t)."""
    J = B4.jump_index(x)
    zP = B4.zp_restart(x, g, J)
    own, lead_ok = owner_fn(base, x, g, zP)
    t = np.arange(np.asarray(x).shape[0])
    ext = (own < 0) & (t >= B2.EXT_FROM) & lead_ok
    post = (t >= J) if J > 0 else (t >= B4.JUMP_UNTIL)
    return (own < 0) & ~ext & (t >= B.EARLY) & post


def dbs_block(s, rows, nn):
    """`analyze_t1_pa02.apply(..., "sm", 10, 1.0, ...)` arithmetic on the given rows (one block)."""
    out = np.array(s, dtype=np.float32, copy=True)
    rows = np.asarray(rows)
    if not rows.size:
        return out
    S8 = out[rows].astype(np.float64)
    sd = S8.std(1, keepdims=True)
    z = np.where(sd > 0, (S8 - S8.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1.0), 0.0)
    r = z + 1.0 * z[:, nn].mean(2)
    for n_, t in enumerate(rows):
        if not sd[n_, 0] > 0:
            continue
        o = V13.transplant(out[t], r[n_])
        if o is not None:
            out[t] = o
    return out


def b8d_core(x, g):
    """Build path before P0: bank8 and its row states on the fixed block, DBS one row at a time."""
    p, T = B8P.pad(x)
    base = np.asarray(B._BASE(p, g), np.float32)
    L = B8.post8(base, p, g)
    rows = np.nonzero(dbs_rows(base, p, g)[:T])[0]
    nn = gallery_nn(g)
    out = L[:T].copy()
    for t in rows:
        out = dbs_block(out, np.array([t]), nn)
    return out


def b8d(x, g):
    return T8V.compose(b8d_core(x, g), T8V.b_scores(x, g))


def b8d_ref_core(x, g):
    """Reference before P0: unpadded per-frame bank8, unpadded row states, DBS on one block of all rows."""
    import analyze_t1_bank5 as A5
    base = np.asarray(B._BASE(x, g), np.float32)
    L = np.asarray(B8.bank8_ref(x, g), np.float32)
    rows = np.nonzero(dbs_rows(base, x, g, owner_fn=A5.masks4))[0]
    return dbs_block(L, rows, gallery_nn(g))


def outputs_fn(x, g):
    return {"b8d": T8V.compose(b8d_ref_core(x, g), T8V.b_scores(x, g))}
