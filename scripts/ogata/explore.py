"""Proxy exploration with shared causal components and many score combiners per job.

Usage: explore.py SPEC.json [workers]
SPEC: list of {"name", "comp": {...}, "comb": [[label, expr], ...]}.
comp keys: "short": {key: [beta, p]} quality-EMA cubic states through the 941755 readout;
           "cv": {key: [beta, p]} cosine of a quality-EMA raw unit-vector state;
           "ce": {key: [beta, p]} quality-EMA of the raw frame cosine.
Every component is a strictly causal per-sequence recursion; combiners are fixed
per-frame maps of components (no labels, no track IDs). Pseudo targets and fixtures
are evaluator-only. Prints aggregate metrics only.
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from proxy_lib import load_track, pseudo_targets, splice_fixtures, t1_metrics, run_metrics, normalize  # noqa
from cubic_fast import _prep, cumulative_cubic_raw, readout  # noqa

REF = 20.0
STATE = {}


def ema_cubic_raw_kg(u, a, K, G, al):
    """Same recursion as cubic_fast.ema_cubic_raw (gamma=1), with K and G precomputed."""
    n = len(u)
    al = np.array(al, np.float64)
    al[0] = 1.0
    w = a
    coef = np.zeros(n)
    out = np.empty((n, G.shape[1]))
    num = np.zeros(G.shape[1])
    q = 0.0
    for t in range(n):
        c = 1.0 - al[t]
        cross = w[t] * np.dot(coef[:t], K[:t, t])
        q = c * c * q + 2 * c * al[t] * cross + (al[t] * w[t]) ** 2 * K[t, t]
        coef[:t] *= c
        coef[t] = al[t] * w[t]
        num = c * num + al[t] * w[t] * G[t]
        out[t] = num / np.sqrt(max(q, 1e-300))
    return out


RHO0 = 0.9
_BASIS = {}


def span_basis(g):
    key = id(g)
    if key not in _BASIS:
        _BASIS[key] = np.linalg.qr(normalize(np.asarray(g, np.float64)).T)[0]
    return _BASIS[key]


def qrate(nrm, beta, p, r=0, rho=None):
    """Quality rate; optional factor (rho/RHO0)^r with rho = in-gallery-span energy of the unit frame."""
    a = beta * (nrm / REF) ** p
    if r:
        a = a * (rho / RHO0) ** r
    return np.minimum(1.0, a)


def lead(S):
    """Per identity: own score minus best other identity's score (same frame)."""
    top2 = np.partition(S, -2, axis=1)[:, -2:]
    best_other = np.where(S >= top2[:, 1:2], top2[:, 0:1], top2[:, 1:2])
    return S - best_other


def components(x, g, comp):
    x = np.asarray(x, np.float64)
    nrm = np.linalg.norm(x, axis=1)
    u, a, c = _prep(x, g)
    ux0 = x / np.maximum(nrm, 1e-12)[:, None]
    rho = np.linalg.norm(ux0 @ span_basis(g), axis=1)
    out = {"mem": readout(cumulative_cubic_raw(u, a, c), c).astype(np.float64), "nrm": nrm}
    if comp.get("short"):
        K = (u @ u.T) ** 3
        G = (u @ c.T) ** 3
        out["s"] = {k: readout(ema_cubic_raw_kg(u, a, K, G, qrate(nrm, *bp, rho=rho)), c).astype(np.float64)
                    for k, bp in comp["short"].items()}
    ux = x / np.maximum(nrm, 1e-12)[:, None]
    gn = normalize(np.asarray(g, np.float64))
    if comp.get("cv"):
        out["cv"] = {}
        for k, bp in comp["cv"].items():
            al = qrate(nrm, *bp); al[0] = 1.0
            m = np.empty_like(ux); s = np.zeros(ux.shape[1])
            for t in range(len(ux)):
                s = (1 - al[t]) * s + al[t] * ux[t]; m[t] = s
            out["cv"][k] = normalize(m) @ gn.T
    if comp.get("ce"):
        fc = ux @ gn.T
        out["ce"] = {}
        for k, bp in comp["ce"].items():
            al = qrate(nrm, *bp); al[0] = 1.0
            m = np.empty_like(fc); s = np.zeros(fc.shape[1])
            for t in range(len(fc)):
                s = (1 - al[t]) * s + al[t] * fc[t]; m[t] = s
            out["ce"][k] = m
    return out


def relu(v):
    return np.maximum(0.0, v)


def apply(expr, cm):
    env = {"np": np, "relu": relu, "mem": cm["mem"], "s": cm.get("s", {}), "cv": cm.get("cv", {}),
           "ce": cm.get("ce", {}), "lead": lead, "q": (cm["nrm"] / REF)[:, None]}
    return np.clip(eval(expr, env), -1, 1).astype(np.float32)


def diffusion_fixtures(seqs, targets, gallery, cos_level, gen_len=600, att_len=600, offset=100):
    """WEAK, UNVALIDATED synthetic proxy: genuine A prefix, then frames whose raw cosine
    to gallery A is exactly cos_level and whose remainder is B's off-gallery residual,
    with B's frame norms. Inside the gallery span such frames point only at A."""
    gn = normalize(np.asarray(gallery, np.float64))
    Q, _ = np.linalg.qr(gn.T)  # basis of gallery span
    fx = []
    n = len(seqs)
    for i in range(n):
        j = (i + offset) % n
        while targets[j] == targets[i]:
            j = (j + 1) % n
        a, b = seqs[i][:gen_len].astype(np.float64), seqs[j][:att_len].astype(np.float64)
        nb = np.linalg.norm(b, axis=1, keepdims=True)
        ub = b / nb
        r = ub - (ub @ Q) @ Q.T
        r = normalize(r)
        att = nb * (cos_level * gn[targets[i]][None] + np.sqrt(1 - cos_level ** 2) * r)
        x = np.concatenate([a, att], 0).astype(np.float32)
        lab = np.r_[np.zeros(len(a), bool), np.ones(len(att), bool)]
        fx.append((x, int(targets[i]), lab))
    return fx


def blend_fixtures(seqs, targets, gallery, cos_level, gen_len=600, att_len=600, offset=100):
    """WEAK, UNVALIDATED synthetic proxy: genuine A prefix, then B frames blended toward
    gallery A so that the raw frame cosine to A equals cos_level (B's other components kept)."""
    gn = normalize(np.asarray(gallery, np.float64))
    fx = []
    n = len(seqs)
    for i in range(n):
        j = (i + offset) % n
        while targets[j] == targets[i]:
            j = (j + 1) % n
        a, b = seqs[i][:gen_len].astype(np.float64), seqs[j][:att_len].astype(np.float64)
        nb = np.linalg.norm(b, axis=1, keepdims=True)
        ga = gn[targets[i]][None]
        v = normalize(b / nb - (b / nb @ ga.T) * ga)
        att = nb * (cos_level * ga + np.sqrt(1 - cos_level ** 2) * v)
        x = np.concatenate([a, att], 0).astype(np.float32)
        fx.append((x, int(targets[i]), np.r_[np.zeros(len(a), bool), np.ones(len(att), bool)]))
    return fx


def init():
    g, seqs, _ = load_track("track1")
    tg, _ = pseudo_targets(g, seqs)
    STATE.update(g=g, seqs=seqs, tg=tg, fx={"sp": splice_fixtures(seqs, tg),
                                            "d2": diffusion_fixtures(seqs, tg, g, 0.2),
                                            "db": blend_fixtures(seqs, tg, g, 0.25)})


def job(spec):
    t0 = time.time()
    g, seqs, tg = STATE["g"], STATE["seqs"], STATE["tg"]
    combs = spec["comb"]
    sc = {lab: [] for lab, _ in combs}
    for x in seqs:
        cm = components(x, g, spec["comp"])
        for lab, e in combs:
            sc[lab].append(apply(e, cm))
    res = {}
    for lab, _ in combs:
        res[lab] = t1_metrics(sc[lab], tg)
    del sc
    for fname, fx in STATE["fx"].items():
        acc = {lab: [] for lab, _ in combs}
        for x, t, lab_ in fx:
            cm = components(x, g, spec["comp"])
            for lab, e in combs:
                s = apply(e, cm)[:, t]
                a = s >= res[lab]["thr"]
                acc[lab].append((float((~a[~lab_]).mean()), float(a[lab_].mean()), run_metrics(a[lab_])))
        for lab, _ in combs:
            v = np.array(acc[lab])
            res[lab].update({f"{fname}_frr": v[:, 0].mean(), f"{fname}_aar": v[:, 1].mean(),
                             f"{fname}_run": v[:, 2].mean(), f"{fname}_s10": (v[:, 2] >= 10).mean()})
    sec = round(time.time() - t0)
    return [{"name": f"{spec['name']} | {lab}", **{k: round(float(v), 5) for k, v in res[lab].items()}, "sec": sec}
            for lab, _ in combs]


if __name__ == "__main__":
    specs = json.load(open(sys.argv[1]))
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else min(len(specs), 12)
    with ProcessPoolExecutor(max_workers=nw, initializer=init) as ex:
        for rows in ex.map(job, specs):
            for r in rows:
                print(json.dumps(r), flush=True)
