"""Offline proxy evaluator for ogata research (evaluator-only pseudo labels).

Pseudo targets are derived offline from whole-sequence mean raw cosine and are
used ONLY to score candidate outputs. They never enter a processor.
Outputs are aggregate metrics only; no embeddings or scores are written to Git.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("RCFA_DATA", ROOT / "challenge_data"))


def normalize(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def load_track(track):
    g = np.load(DATA / track / "gallery.npz")["feature"].astype(np.float32)
    paths = sorted(glob.glob(str(DATA / track / "probe_seq_embeddings" / "*.npz")))
    seqs = [np.load(p)["feature"].astype(np.float32) for p in paths]
    return g, seqs, [Path(p).stem for p in paths]


def pseudo_targets(gallery, seqs):
    gn = normalize(gallery)
    tg, margin = [], []
    for x in seqs:
        m = (normalize(x) @ gn.T).mean(0)
        o = np.argsort(m)[::-1]
        tg.append(int(o[0]))
        margin.append(float(m[o[0]] - m[o[1]]))
    return np.array(tg), np.array(margin)


def t1_metrics(scores, targets):
    """scores: list of (n_i,200) arrays; targets: pseudo target per sequence."""
    gen, imp, seq_id = [], [], []
    for i, (s, t) in enumerate(zip(scores, targets)):
        gen.append(s[:, t].astype(np.float64))
        mask = np.ones(s.shape[1], bool)
        mask[t] = False
        imp.append(s[:, mask].ravel())
        seq_id.append(np.full(s.shape[0], i))
    gen = np.concatenate(gen)
    seq_id = np.concatenate(seq_id)
    imp = np.sort(np.concatenate(imp).astype(np.float32))
    gs = np.sort(gen)
    ni, ng = imp.size, gs.size
    # EER: scan thresholds over genuine values
    cand = np.unique(np.quantile(gs, np.linspace(0, 0.2, 4001)))
    far = 1.0 - np.searchsorted(imp, cand, side="left") / ni
    frr = np.searchsorted(gs, cand, side="left") / ng
    k = np.argmin(np.abs(far - frr))
    eer = float((far[k] + frr[k]) / 2)
    # threshold at FAR=1e-5 (accept s >= thr)
    kth = int(np.ceil(ni * (1 - 1e-5)))
    thr = float(imp[min(kth, ni - 1)])
    frr_low = float((gen < thr).mean())
    macro = float(np.mean([(gen[seq_id == i] < thr).mean() for i in range(len(scores))]))
    return {"eer": eer, "frr_far1e5": frr_low, "macro_seq_frr": macro, "thr": thr}


def run_metrics(accept):
    """Max consecutive True run length."""
    best = cur = 0
    for a in accept:
        cur = cur + 1 if a else 0
        best = max(best, cur)
    return best


def splice_fixtures(seqs, targets, gen_len=600, att_len=600, offset=100):
    """Identity-transition proxy from Track1: genuine A prefix then B frames."""
    fx = []
    n = len(seqs)
    for i in range(n):
        j = (i + offset) % n
        while targets[j] == targets[i]:
            j = (j + 1) % n
        a, b = seqs[i], seqs[j]
        ga = min(gen_len, a.shape[0])
        fb = min(att_len, b.shape[0])
        x = np.concatenate([a[:ga], b[:fb]], 0)
        lab = np.r_[np.zeros(ga, bool), np.ones(fb, bool)]
        fx.append((x, int(targets[i]), lab))
    return fx


def t2_proxy(score_fn, fixtures, gallery, thr):
    frr, aar, runs, succ = [], [], [], []
    for x, t, lab in fixtures:
        s = score_fn(x, gallery)[:, t]
        acc = s >= thr
        frr.append(float((~acc[~lab]).mean()))
        aar.append(float(acc[lab].mean()))
        r = run_metrics(acc[lab])
        runs.append(r)
        succ.append(r >= 10)
    return {"frr": float(np.mean(frr)), "aar": float(np.mean(aar)),
            "max_consec": float(np.mean(runs)), "succ10": float(np.mean(succ))}


def evaluate(score_fn, track1, fixtures=None, name=""):
    g, seqs, targets = track1
    scores = [score_fn(x, g) for x in seqs]
    m = t1_metrics(scores, targets)
    if fixtures is not None:
        m.update({"t2_" + k: v for k, v in t2_proxy(score_fn, fixtures, g, m["thr"]).items()})
    return m
