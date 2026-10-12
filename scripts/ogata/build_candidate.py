"""Build one ogata research candidate artifact (same method/params for both tracks).

Usage: build_candidate.py CONFIG_NAME OUTPUT_DIR
Writes OUTPUT_DIR/submission.zip (classic outer ZIP, never overwrites) and receipt.json.
"""
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from scripts.build_candidate_submission import fixed_zip_info, load_features, serialize_score  # noqa: E402
import candidates  # noqa: E402

CONFIGS = {
    "dveto_g1_q005p4_d0005": "discrim_veto(1.0, 0.05, 4, 0.005)",
    "dveto_g1_q005p4_d001": "discrim_veto(1.0, 0.05, 4, 0.01)",
    "dveto_g1_q005p4_d002": "discrim_veto(1.0, 0.05, 4, 0.02)",
    "rg2s_s005_f02_p4_ds0005_df001_m0": "rank_gated_twospeed_veto(0.05, 0.2, 4, 0.005, 0.01, 0.0)",
    "rg2s_s005_f02_p4_ds0005_df001_m0005": "rank_gated_twospeed_veto(0.05, 0.2, 4, 0.005, 0.01, 0.005)",
    "vtn_mem": "vtn_mem()",
    "ospc_vtn": "ospc_vtn()",
    "fr_hcp": "fr_hcp()",
    "esr_hcp": "esr_hcp()",
    "fco": "fco()",
    "esrn_fco": "esrn_fco()",
    "esr_n": "esr_n()",
    "blvs_n_fco": "blvs_n_fco()",
    "pez_blvs_n_fco": "pez_blvs_n_fco()",
    "peznh_blvs_n_fco": "peznh_blvs_n_fco()",
    "pxm_peznh": "pxm_peznh()",
    "fusion_mean_peznh_943955": "fusion_mean_peznh_943955()",
}
# Fusion family and T2 family stage 0 rules (20261009_ogata_fusion_family.md, 20261009_ogata_t2_family.md).
CONFIGS.update({f"ff_{r}": f"fusion_family('{r}')" for r in ('mean', 'atanh', 'bl_nh', 'bl_pez', 'flat_nh', 'def_nh', 'def_pez', 'tri_o', 'preq', 'tcf0')})
CONFIGS.update({f"tf_{v}": f"t2_family('{v}')" for v in ('pxm_hist', 'slowref_nh', 'ts_nh', 'med_nh', 'hold_nh', 'ver_nh', 'dframe_nh', 'damp_nh', 'dcoh_nh', 'lx_nh', 'owc_nh', 'hi_nh', 'novg_nh', 'novg_pez', 'rel_nh', 'rme_nh', 'prof_nh', 'cont_nh', 'rtpl_nh', 'disp_nh', 'norm_nh', 'span_nh', 'fplo_nh', 'fphi_nh')})
# Research module whose evaluated output an artifact must reproduce bit for bit.
REFERENCE = {"vtn_mem": ("x_combo_vtn_mem", "vtn_mem"), "ospc_vtn": ("x_cu_ospc", "ospc_vtn"),
             "fr_hcp": ("x_fr_hcp", "hcp"), "esr_hcp": ("x_r3", "esr_hcp"),
             "fco": ("x_r4", "fco"), "esrn_fco": ("x_r5", "esrn_fco"),
             "esr_n": ("x_r5", "esr_n"), "blvs_n_fco": ("analyze_blvs_stage0", "blvs_n_fco"),
             "pez_blvs_n_fco": ("analyze_pez_stage0", "pez_blvs_n_fco"),
             "peznh_blvs_n_fco": ("analyze_peznh_stage0", "peznh_blvs_n_fco"),
             "pxm_peznh": ("analyze_pxm_stage0", "pxm_peznh"),
             "fusion_mean_peznh_943955": ("analyze_fusion_943955", "fusion_mean_peznh_943955")}
REFERENCE.update({k: ("analyze_fusion_family", k[3:]) for k in CONFIGS if k.startswith("ff_")})
REFERENCE.update({k: ("analyze_t2_family", k[3:]) for k in CONFIGS if k.startswith("tf_")})
CONFIGS.update({f"tfbl_{k[3:]}": f"t2_family_bl('{k[3:]}')" for k in list(CONFIGS) if k.startswith("tf_")})
REFERENCE.update({k: ("analyze_t2_family", "bl_" + k[5:]) for k in CONFIGS if k.startswith("tfbl_")})
# Band-split composites (20261009_ogata_rtpl_cmp.md): H's high band, L's low band.
CONFIGS.update({f"tfcmp_{h}__{l_}": f"t2_cmp('{h}', '{l_}')" for h, l_ in (("rtpl_nh", "hold_nh"), ("hold_rtpl_nh", "hold_nh"))})
REFERENCE.update({k: ("analyze_rtpl_cmp", "cmp_" + k[6:]) for k in CONFIGS if k.startswith("tfcmp_")})
# Round-2 screen on rtpl_nh (20261009_ogata_round2_rtpl_screen.md): cmp(H | hold_nh).
CONFIGS.update({f"tfr2_{h}": f"round2_cmp('{h}')" for h in ("rtpl_pez", "rtpl_prof_nh", "rtpl_span_nh", "rtpl_mem", "rtpl_min")})
REFERENCE.update({k: ("analyze_round2", f"cmp_{k[5:]}__hold_nh") for k in CONFIGS if k.startswith("tfr2_")})
# T1 premise search stage 1b (20261009_ogata_t1_premise_search.md): cmp(rtpl_nh | hold_nh_c2).
CONFIGS["t1c2_cmp"] = "t1_cau2_cmp()"
REFERENCE["t1c2_cmp"] = ("analyze_t1_cau2_chain", "cmp_c2")
CONFIGS["t1c2_pez_cmp"] = "t1_cau2_pez_cmp()"
REFERENCE["t1c2_pez_cmp"] = ("analyze_t1_c2_pez", "cmp_pez_c2")
CONFIGS["t1geo_pez_cmp"] = "t1_geo_pez_cmp()"
REFERENCE["t1geo_pez_cmp"] = ("analyze_t1_stage1d", "geo")
# T1 stage 2: the teammate's bank low band under 973452's high band.
CONFIGS.update({f"t1s2_{r}_{l_}": f"t1_bank_pez_cmp('{r}_{l_}')" for l_ in ("b5", "b6") for r in ("or50", "or75", "cap50", "min50")})
REFERENCE.update({k: ("analyze_t1_stage2_bank", k[5:]) for k in CONFIGS if k.startswith("t1s2_")})
# T2 stage H: a new high band H' composed as 976810's recipe over bank8p.
CONFIGS.update({f"t2h_{c}": f"t2h_cmp('{c}')" for c in ("mat", "s4", "nst")})
REFERENCE.update({k: ("analyze_t2_stage_h", "cmp_" + k[4:]) for k in CONFIGS if k.startswith("t2h_")})
# T2 stage R: rme on rtpl_pez, built as a disclosed post-hoc exception (20261012_ogata_t2_stage_r.md).
CONFIGS["t2r_rme"] = "t2r_cmp('rme')"
REFERENCE["t2r_rme"] = ("analyze_t2_stage_r", "cmp_rme")
# T2 stage S: early frozen residual template beside rtpl's (20261012_ogata_t2_stage_s.md).
CONFIGS.update({f"t2s_{c}": f"t2s_cmp('{c}')" for c in ("etx", "ete", "etn")})
REFERENCE.update({k: ("analyze_t2_stage_s", "cmp_" + k[4:]) for k in CONFIGS if k.startswith("t2s_")})
# T2 stage T: rtpl's residual EMA restarted on the column's return (20261012_ogata_t2_stage_t.md).
CONFIGS["t2t_rst"] = "t2t_cmp('rst')"
REFERENCE["t2t_rst"] = ("analyze_t2_stage_t", "cmp_rst")
# Balanced: 979937's high band over the teammate's bank8 + DBS10 low band (20261012_ogata_bal_etx_b8d.md).
CONFIGS["bal_etx_b8d"] = "bal_etx_b8d()"
REFERENCE["bal_etx_b8d"] = ("bal_etx_b8d", "bal_etx_b8d")
CONFIGS["bal_ete_b8d"] = "bal_ete_b8d()"
REFERENCE["bal_ete_b8d"] = ("bal_ete_b8d", "bal_ete_b8d")
TRACKS = ("track1", "track2")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(name, out):
    fn = eval(CONFIGS[name], vars(candidates))
    data = Path(os.environ.get("RCFA_DATA", ROOT / "challenge_data"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    art = out / "submission.zip"
    if art.exists():
        raise FileExistsError(art)
    t0 = time.time()
    receipt = {"config": name, "expr": CONFIGS[name], "tracks": {}}
    inner = {}
    for track in TRACKS:
        g = load_features(data / track / "gallery.npz")
        paths = sorted((data / track / "probe_seq_embeddings").glob("*.npz"))
        tz = out / f".{track}.zip.part"
        lo, hi, frames, prefix_err, ref_checked = 1.0, -1.0, 0, 0.0, 0
        with zipfile.ZipFile(tz, "w") as z:
            for i, p in enumerate(paths):
                x = load_features(p)
                s = fn(x, g)
                if s.shape != (x.shape[0], g.shape[0]) or s.dtype != np.float32 or not np.isfinite(s).all():
                    raise ValueError(f"{track}/{p.name}: bad score")
                if s.min() < -1 or s.max() > 1:
                    raise ValueError(f"{track}/{p.name}: out of range")
                if i < 3:  # causal prefix invariance spot check
                    h = max(1, x.shape[0] // 2)
                    prefix_err = max(prefix_err, float(np.abs(fn(x[:h], g) - s[:h]).max()))
                    if name in REFERENCE:  # bit-identity with the evaluated research output
                        mod, key = REFERENCE[name]
                        ref = __import__(mod).outputs_fn(x, g)[key]
                        if not np.array_equal(ref, s):
                            raise ValueError(f"{track}/{p.name}: differs from {mod}.{key}")
                        ref_checked += 1
                lo, hi, frames = min(lo, float(s.min())), max(hi, float(s.max())), frames + x.shape[0]
                z.writestr(fixed_zip_info(f"score_{p.name}", compressed=False), serialize_score(s))
        inner[track] = tz
        receipt["tracks"][track] = {"files": len(paths), "frames": frames, "min": lo, "max": hi,
                                    "prefix_max_abs_diff": prefix_err, "reference_identical_files": ref_checked}
        print(track, receipt["tracks"][track], flush=True)
    part = out / ".submission.zip.part"
    with zipfile.ZipFile(part, "w", allowZip64=False) as z:
        for track in TRACKS:
            z.write(inner[track], f"{track}.zip", compress_type=zipfile.ZIP_STORED)
    os.link(part, art)
    part.unlink()
    for t in inner.values():
        t.unlink()
    os.chmod(art, 0o600)
    receipt.update(size=art.stat().st_size, sha256=sha256(art), seconds=round(time.time() - t0, 1))
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
