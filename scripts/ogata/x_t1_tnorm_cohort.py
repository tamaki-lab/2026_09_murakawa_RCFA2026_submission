"""TN: classical per-frame cohort T-norm of the 941755 readout (Samuelson-bounded).

Preregistered in docs/experiments/20261005_ogata_t1_tnorm_cohort.md.
Frozen design (selector key t1_tnorm_cohort; sources readout_normalization#1,
cold_start_degraded#1, quality#1, fusion_wildcard#1).

outputs_fn(x, g) is a strictly causal, per-sequence processor with fresh state per
call and identical constants on both tracks. It receives only the raw probe sequence
and the raw gallery of the same track. The per-frame normalization uses only the 200
scores of the same frame (the cohort is the other enrolled identities).

  python scripts/ogata/x_t1_tnorm_cohort.py              # single preregistered run
  python scripts/ogata/x_t1_tnorm_cohort.py --gallery    # gallery-only beta_j spread (pre-run)
  python scripts/ogata/x_t1_tnorm_cohort.py --stats      # label-free sd==0 count, max|o|

--gallery reads only the two galleries. --stats reads probe sequences but uses no
labels or pseudo targets; both print aggregates only.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_check as cc  # noqa: E402  (before numpy: pins BLAS threads)
import numpy as np  # noqa: E402

from cubic_fast import _prep, cumulative_cubic_raw, discrim_scores  # noqa: E402
from rcfa.methods.gallery_discriminative_readout import discriminative_mapping  # noqa: E402

SETTINGS = ["tn_s", "p941755_ref"]


def _tnorm64(x, g):
    """Float64 T-normed scores o, plus the per-frame population sd (for stats)."""
    u, a, c = _prep(x, g)
    raw = cumulative_cubic_raw(u, a, c)
    M, beta, _ = discriminative_mapping(c)
    s = np.tanh(raw @ M.T + beta)             # float64, NOT cast to float32 first
    K = c.shape[0]                            # gallery size (200 on both tracks); no Track logic
    d = s - s.mean(1, keepdims=True)
    sd = np.sqrt((d * d).mean(1, keepdims=True))
    o = np.where(sd > 0, d / (np.where(sd > 0, sd, 1.0) * np.sqrt(K - 1)), 0.0)
    return o, sd


def outputs_fn(x, g):
    o, _ = _tnorm64(x, g)
    return {"tn_s": o.astype(np.float32),
            "p941755_ref": discrim_scores(x, g)}   # non-candidate reference, no gate


def gallery_check():
    import json
    from scripts.cubic_kernel_reference import prepare
    out = {"check": "t1_tnorm_cohort_gallery_beta"}
    for tr in ("track1", "track2"):
        g = load_track_gallery(tr)
        _, coords, _ = prepare(g, g)          # coords depend on the gallery only
        _, beta, _ = discriminative_mapping(np.asarray(coords, np.float64))
        beta = np.asarray(beta, np.float64)
        tb = np.tanh(beta)
        out[tr] = {"K": int(beta.size), "beta_sd": round(float(beta.std()), 6),
                   "beta_min": round(float(beta.min()), 6), "beta_max": round(float(beta.max()), 6),
                   "tanh_beta_sd": round(float(tb.std()), 6)}
    print(json.dumps(out), flush=True)


def load_track_gallery(track):
    from proxy_lib import DATA
    return np.load(DATA / track / "gallery.npz")["feature"].astype(np.float32)


def _stats_job(task):
    tr, i = task
    x, g = (cc.D["s1"][i], cc.D["g1"]) if tr == "t1" else (cc.D["s2"][i], cc.D["g2"])
    o, sd = _tnorm64(x, g)
    o32 = o.astype(np.float32)
    return tr, {"n": int(o.shape[0]), "sd0": int((sd[:, 0] <= 0).sum()),
                "maxabs64": float(np.abs(o).max()), "maxabs32": float(np.abs(o32.astype(np.float64)).max()),
                "n_ge1_32": int((np.abs(o32) >= 1).sum()), "sd_min": float(sd.min())}


def stats(workers=6):
    import json
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
    out = {"stats": "t1_tnorm_cohort"}
    for tr, rs in acc.items():
        out[tr] = {"frames": sum(r["n"] for r in rs), "sd_eq0_frames": sum(r["sd0"] for r in rs),
                   "sd_min": min(r["sd_min"] for r in rs),
                   "max_abs_o_float64": max(r["maxabs64"] for r in rs),
                   "max_abs_o_float32": max(r["maxabs32"] for r in rs),
                   "n_abs_ge1_float32": sum(r["n_ge1_32"] for r in rs)}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--gallery":
        gallery_check()
    elif len(sys.argv) > 1 and sys.argv[1] == "--stats":
        stats()
    else:
        cc.run_check(outputs_fn, SETTINGS, "t1_tnorm_cohort", workers=6)
