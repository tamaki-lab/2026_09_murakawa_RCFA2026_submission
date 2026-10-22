"""Shared proxy evaluation harness for ogata research experiments (aggregates only).

API
---
run_check(outputs_fn, settings, tag, workers=6) -> list[dict]

  outputs_fn(x, g) -> {setting_name: (T, J) array}
      x: (T, 512) raw probe embeddings of ONE sequence, g: (J=200, 512) raw gallery
      of the same track. The harness passes nothing else: no pseudo targets, labels,
      fixture flags or track names. outputs_fn must be a fixed, strictly causal,
      per-sequence processor (fresh state per call) with identical parameters for
      every input. Values must be finite and in [-1, 1], larger = stronger match.
  settings: names (keys of outputs_fn's dict) to evaluate, in print order. Extra
      keys are ignored. Reserved names: "955399", "955690".
  tag: free-form label printed in the header/footer lines.
  workers: process count, hard-capped at 6.

The harness always adds two anchors computed by itself with the exact processors
of build_candidate.CONFIGS: "955399" = dveto_g1_q005p4_d0005 (discrim_veto) and
"955690" = rg2s_s005_f02_p4_ds0005_df001_m0 (rank_gated_twospeed_veto).

Built-in checks (every setting, every evaluated sequence/fixture): shape (T, 200),
finite, within [-1, 1]; causal prefix invariance on 2 Track1 + 2 Track2 sequences
(first half computed alone vs. the full-sequence prefix over all columns, after the
float32 cast used for scoring; pass iff max abs diff < 1e-6). A setting failing any
check gets a "checks_pass": false line and no metrics.

Metrics per setting (one JSON line; pseudo targets are evaluator-only, mean-cosine
argmax over the whole sequence; threshold at pooled frame-level FAR=1e-5 on Track1):
  eer, frr_far1e5, thr       proxy_lib.t1_metrics, unchanged (eer searches only the
                             lowest 20% of genuine scores; returns ~.5 when many
                             genuine frames sit at -1)
  eer_full                   EER with thresholds over ALL unique genuine values
  t1_neg1_share              share of genuine T1 frames <= -0.999
  splice_*                   proxy_lib.splice_fixtures (600 genuine A + 600 other-
                             identity T1 frames); frr on A part, aar/run/s10 on B part
  t1_band_reject, t2_band_reject
                             frames where the pseudo target leads the raw cosine with
                             value in [.30, .41): share rejected at thr (T1 low, T2 high)
  t2_all_reject              share of all T2 target-column frames rejected at thr
  diffsynth_UNVALIDATED_*    explore.diffusion_fixtures at cos_level 0.2 (the level
                             explore.py's own init() uses for its "d2" fixture; fixed
                             before any candidate of this workflow). Never validated
                             against official diffusion AAR; mechanism signal only.

Writes nothing but stdout aggregates. The caller tees stdout to results/ogata/<tag>.log.
"""
import json
import multiprocessing as mp
import os
import sys
import time

# One BLAS thread per worker so `workers` bounds CPU use. Effective only when this
# module is imported before numpy; otherwise set these in the shell (header reports).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from proxy_lib import load_track, pseudo_targets, splice_fixtures, t1_metrics, run_metrics, normalize  # noqa: E402
from explore import diffusion_fixtures  # noqa: E402
import candidates  # noqa: E402
import build_candidate  # noqa: E402

ANCHORS = {"955399": "dveto_g1_q005p4_d0005", "955690": "rg2s_s005_f02_p4_ds0005_df001_m0"}
BAND = (0.30, 0.41)
DIFF_COS = 0.2
PREFIX_TOL = 1e-6
PREFIX_SEQS = ((("t1", 0), ("t1", 1), ("t2", 0), ("t2", 1)))
MAX_WORKERS = 6
D = {}
FN = {}


def _anchor_fns():
    return {k: eval(build_candidate.CONFIGS[v], vars(candidates)) for k, v in ANCHORS.items()}


def _load():
    if D:
        return
    g1, s1, _ = load_track("track1")
    t1, _ = pseudo_targets(g1, s1)
    g2, s2, _ = load_track("track2")
    t2, _ = pseudo_targets(g2, s2)
    D.update(g1=g1, s1=s1, t1=t1, g2=g2, s2=s2, t2=t2, sp=splice_fixtures(s1, t1),
             df=diffusion_fixtures(s1, t1, g1, DIFF_COS))


def _init_worker(fn, settings):
    FN.update(fn=fn, settings=list(settings), anchors=_anchor_fns())
    _load()


def _all_outputs(x, g):
    """User settings + anchors as float32, with per-setting check errors (None = ok)."""
    o = FN["fn"](x, g)
    if not isinstance(o, dict):
        raise TypeError("outputs_fn must return a dict")
    want = (x.shape[0], g.shape[0])
    out, err = {}, {}
    items = [(k, o.get(k)) for k in FN["settings"]] + [(k, f(x, g)) for k, f in FN["anchors"].items()]
    for k, v in items:
        if v is None:
            out[k], err[k] = None, "missing"
            continue
        v = np.asarray(v)
        if v.shape != want:
            out[k], err[k] = None, f"shape {v.shape} != {want}"
            continue
        if not np.isfinite(v).all():
            out[k], err[k] = None, "non-finite"
            continue
        v = v.astype(np.float32)
        if v.min() < -1 or v.max() > 1:
            out[k], err[k] = None, f"range [{float(v.min())}, {float(v.max())}]"
            continue
        out[k], err[k] = v, None
    return out, err


def _job(task):
    kind, i = task
    if kind == "t1":
        o, e = _all_outputs(D["s1"][i], D["g1"])
        return kind, i, o, e
    if kind == "t2":
        o, e = _all_outputs(D["s2"][i], D["g2"])
        t = D["t2"][i]
        return kind, i, {k: (None if v is None else v[:, t]) for k, v in o.items()}, e
    if kind in ("sp", "df"):
        x, t, _ = D[kind][i]
        o, e = _all_outputs(x, D["g1"])
        return kind, i, {k: (None if v is None else v[:, t]) for k, v in o.items()}, e
    # prefix invariance: ("pf", (track_kind, index))
    tk, j = i
    x, g = (D["s1"][j], D["g1"]) if tk == "t1" else (D["s2"][j], D["g2"])
    full, e = _all_outputs(x, g)
    h = max(1, x.shape[0] // 2)
    half, eh = _all_outputs(x[:h], g)
    diff = {}
    for k in full:
        if full[k] is None or half[k] is None:
            diff[k] = None
            e[k] = e[k] or eh[k]
        else:
            diff[k] = float(np.abs(half[k].astype(np.float64) - full[k][:h].astype(np.float64)).max())
    return kind, i, diff, e


def _band_mask(x, g, t):
    s = normalize(x) @ normalize(g).T
    return (s.argmax(1) == t) & (s[:, t] >= BAND[0]) & (s[:, t] < BAND[1])


def _eer_full(gen, imp_sorted):
    gs = np.sort(gen)
    cand = np.unique(gs)
    far = 1.0 - np.searchsorted(imp_sorted, cand, side="left") / imp_sorted.size
    frr = np.searchsorted(gs, cand, side="left") / gs.size
    k = np.argmin(np.abs(far - frr))
    return float((far[k] + frr[k]) / 2)


def _fixture_stats(outs, fixtures, thr):
    v = []
    for o, (_, _, lab) in zip(outs, fixtures):
        acc = o >= thr
        v.append(((~acc[~lab]).mean(), acc[lab].mean(), run_metrics(acc[lab])))
    return np.array(v, np.float64)


def run_check(outputs_fn, settings, tag, workers=6):
    t0 = time.time()
    settings = list(settings)
    clash = set(settings) & set(ANCHORS)
    if clash or len(set(settings)) != len(settings):
        raise ValueError(f"setting names must be unique and not reserved: {sorted(clash)}")
    nw = max(1, min(int(workers), MAX_WORKERS))
    names = settings + list(ANCHORS)
    print(json.dumps({"tag": tag, "settings": settings, "anchors": ANCHORS, "workers": nw,
                      "diffsynth_cos_level": DIFF_COS, "band": BAND,
                      "blas_threads_env": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}),
          flush=True)
    _load()
    n1, n2 = len(D["s1"]), len(D["s2"])
    tasks = [("pf", p) for p in PREFIX_SEQS] + [("t1", i) for i in range(n1)] + \
        [("sp", i) for i in range(len(D["sp"]))] + [("df", i) for i in range(len(D["df"]))] + \
        [("t2", i) for i in range(n2)]
    res = {"t1": {}, "sp": {}, "df": {}, "t2": {}, "pf": {}}
    errs = {k: [] for k in names}
    # fork: outputs_fn (possibly a closure defined in __main__ or -c) and the loaded
    # data are inherited by workers; the initializer re-loads only if absent.
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=nw, mp_context=ctx, initializer=_init_worker,
                             initargs=(outputs_fn, settings)) as ex:
        for kind, i, o, e in ex.map(_job, tasks, chunksize=1):
            res[kind][i] = o
            for k, msg in e.items():
                if msg:
                    errs[k].append(f"{kind}{i}: {msg}")
    pf = {k: [res["pf"][p][k] for p in PREFIX_SEQS] for k in names}
    b1 = [_band_mask(D["s1"][i], D["g1"], D["t1"][i]) for i in range(n1)]
    b2 = [_band_mask(D["s2"][i], D["g2"], D["t2"][i]) for i in range(n2)]
    rows = []
    for name in names:
        diffs = pf[name]
        pmax = None if any(d is None for d in diffs) else max(diffs)
        ok = not errs[name] and pmax is not None and pmax < PREFIX_TOL
        check = {"setting": name, "checks_pass": ok, "prefix_max_abs_diff": pmax,
                 "n_errors": len(errs[name]), "errors_head": errs[name][:5]}
        print(json.dumps({"check": check}), flush=True)
        if not ok:
            rows.append(check)
            continue
        t1o = [res["t1"][i][name] for i in range(n1)]
        m = t1_metrics(t1o, D["t1"])
        thr = m["thr"]
        gen = np.concatenate([t1o[i][:, D["t1"][i]].astype(np.float64) for i in range(n1)])
        imp = []
        for i in range(n1):
            mask = np.ones(t1o[i].shape[1], bool)
            mask[D["t1"][i]] = False
            imp.append(t1o[i][:, mask].ravel())
        imp = np.sort(np.concatenate(imp).astype(np.float32))
        eer_full = _eer_full(gen, imp)
        del imp
        sp = _fixture_stats([res["sp"][i][name] for i in range(len(D["sp"]))], D["sp"], thr)
        df = _fixture_stats([res["df"][i][name] for i in range(len(D["df"]))], D["df"], thr)
        rej1 = np.concatenate([t1o[i][:, D["t1"][i]][b1[i]] < thr for i in range(n1)])
        rej2 = np.concatenate([res["t2"][i][name][b2[i]] < thr for i in range(n2)])
        rej2_all = np.concatenate([res["t2"][i][name] < thr for i in range(n2)])
        row = {"setting": name, "eer": round(m["eer"], 5), "eer_full": round(eer_full, 6),
               "frr_far1e5": round(m["frr_far1e5"], 4), "thr": round(thr, 5),
               "t1_neg1_share": round(float((gen <= -0.999).mean()), 5),
               "splice_frr": round(sp[:, 0].mean(), 4), "splice_aar": round(sp[:, 1].mean(), 4),
               "splice_run": round(sp[:, 2].mean(), 1), "splice_s10": round((sp[:, 2] >= 10).mean(), 3),
               "t1_band_reject": round(float(rej1.mean()), 4), "t2_band_reject": round(float(rej2.mean()), 4),
               "t2_all_reject": round(float(rej2_all.mean()), 4),
               "diffsynth_UNVALIDATED_aar": round(df[:, 1].mean(), 4),
               "diffsynth_UNVALIDATED_run": round(df[:, 2].mean(), 1),
               "diffsynth_UNVALIDATED_s10": round((df[:, 2] >= 10).mean(), 3),
               "prefix_max_abs_diff": pmax}
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"tag": tag, "band_frames_t1": int(sum(b.sum() for b in b1)),
                      "band_frames_t2": int(sum(b.sum() for b in b2)), "seconds": round(time.time() - t0)}),
          flush=True)
    return rows
