"""Research candidate processors (strictly causal, same params for both tracks)."""
import numpy as np


def normalize(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def quality_ema_states(x, beta, p, ref=20.0, amax=1.0):
    """m_t = (1-a_t) m_{t-1} + a_t u_t with a_t = min(amax, beta*(|x_t|/ref)^p).

    Low-quality (low-norm) frames neither add much evidence nor erase memory.
    """
    x = np.asarray(x, np.float64)
    nrm = np.linalg.norm(x, axis=1)
    u = x / np.maximum(nrm, 1e-12)[:, None]
    a = np.minimum(amax, beta * (nrm / ref) ** p)
    out = np.empty_like(u)
    m = u[0].copy()
    out[0] = m
    for t in range(1, len(u)):
        m = (1 - a[t]) * m + a[t] * u[t]
        out[t] = m
    return out


def quality_ema_scores(beta, p, ref=20.0):
    def f(x, g):
        return (normalize(quality_ema_states(x, beta, p, ref)) @ normalize(np.asarray(g, np.float64)).T).astype(np.float32)
    f.__name__ = f"qema_b{beta}_p{p}"
    return f


def weighted_cumulative_scores(p):
    """Cumulative mean of unit vectors weighted by norm^p (p=1 ~ official raw mean)."""
    def f(x, g):
        x = np.asarray(x, np.float64)
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        m = np.cumsum(x / np.maximum(nrm, 1e-12) * nrm ** p, axis=0)
        return (normalize(m) @ normalize(np.asarray(g, np.float64)).T).astype(np.float32)
    return f


def veto_scores(mem_fn, short_fn, delta, lam=1.0):
    """s = mem - lam*relu(mem - short - delta): long memory unless recent evidence collapses."""
    def f(x, g):
        m = mem_fn(x, g).astype(np.float64)
        r = short_fn(x, g).astype(np.float64)
        s = m - lam * np.maximum(0.0, m - r - delta)
        return np.clip(s, -1, 1).astype(np.float32)
    return f


from cubic_fast import _prep, cumulative_cubic_raw, ema_cubic_raw, readout, cubic_scores, discrim_scores  # noqa: E402


def discrim_gamma(gamma):
    return lambda x, g: discrim_scores(x, g, gamma)


def discrim_veto(gamma, beta, p, delta, ref=20.0):
    """941755-style memory, vetoed by a quality-EMA cubic state through the SAME readout."""
    def f(x, g):
        u, a, c = _prep(x, g)
        mem = readout(cumulative_cubic_raw(u, a, c, gamma), c).astype(np.float64)
        nrm = np.linalg.norm(np.asarray(x, np.float64), axis=1)
        al = np.minimum(1.0, beta * (nrm / ref) ** p)
        short = readout(ema_cubic_raw(u, a, c, al, 1.0), c).astype(np.float64)
        return np.clip(mem - np.maximum(0.0, mem - short - delta), -1, 1).astype(np.float32)
    return f


def _lead(S):
    """Per identity: own score minus the best other identity's score in the same frame."""
    top2 = np.partition(S, -2, axis=1)[:, -2:]
    best_other = np.where(S >= top2[:, 1:2], top2[:, 0:1], top2[:, 1:2])
    return S - best_other


def rank_gated_twospeed_veto(beta_s, beta_f, p, delta_s, delta_f, margin=0.0, ref=20.0):
    """941755 memory vetoed by min(slow + delta_s, fast + delta_f); both are quality-EMA
    cubic states through the SAME readout. The veto is skipped for identity k while k
    leads the slow state across the gallery by more than `margin`. Strictly causal, per sequence."""
    def f(x, g):
        u, a, c = _prep(x, g)
        mem = readout(cumulative_cubic_raw(u, a, c, 1.0), c).astype(np.float64)
        nrm = np.linalg.norm(np.asarray(x, np.float64), axis=1)
        q = (nrm / ref) ** p
        slow = readout(ema_cubic_raw(u, a, c, np.minimum(1.0, beta_s * q), 1.0), c).astype(np.float64)
        fast = readout(ema_cubic_raw(u, a, c, np.minimum(1.0, beta_f * q), 1.0), c).astype(np.float64)
        veto = np.minimum(mem, np.minimum(slow + delta_s, fast + delta_f))
        s = np.where(_lead(slow) > margin, mem, veto)
        return np.clip(s, -1, 1).astype(np.float32)
    return f


def vtn_mem():
    """VTN-M (docs/experiments/20261006_ogata_combo_vtn_mem.md, primary vtn_mem): the 955399
    veto in P units, then a per-column leave-one-out T-norm against the unvetoed memory row.
    Same float64 arithmetic as x_combo_vtn_mem._core; only the vtn_mem output is computed."""
    from rcfa.methods.gallery_discriminative_readout import discriminative_mapping

    def f(x, g):
        u, a, c = _prep(x, g)
        K = c.shape[0]
        M, beta, _ = discriminative_mapping(c)
        nrm = np.linalg.norm(np.asarray(x, np.float64), axis=1)
        al = np.minimum(1.0, 0.05 * (nrm / 20.0) ** 4)
        sm = np.tanh(cumulative_cubic_raw(u, a, c, 1.0) @ M.T + beta)
        ss = np.tanh(ema_cubic_raw(u, a, c, al, 1.0) @ M.T + beta)
        P = (sm - sm.mean(1, keepdims=True)) / 2
        S = (ss - ss.mean(1, keepdims=True)) / 2
        V = np.minimum(P, S + 0.005)
        mu = (P.sum(1, keepdims=True) - P) / (K - 1)
        var = ((P * P).sum(1, keepdims=True) - P * P) / (K - 1) - mu * mu
        sd = np.sqrt(np.maximum(var, 0.0))
        z = np.where(sd > 0, (V - mu) / np.where(sd > 0, sd, 1.0), 0.0)
        return (z / np.sqrt(K + z * z)).astype(np.float32)
    return f


def ospc_vtn():
    """OSP-C (docs/experiments/20261006_ogata_cu_ospc.md, primary ospc_vtn): the VTN-M front end
    up to its LOO z, then the open-set cohort posterior map ospc(zV, zP). Same float64
    arithmetic as x_cu_ospc._fe/ospc; only the ospc_vtn output is computed."""
    from rcfa.methods.gallery_discriminative_readout import discriminative_mapping

    def f(x, g):
        u, a, c = _prep(x, g)
        K = c.shape[0]
        M, beta, _ = discriminative_mapping(c)
        nrm = np.linalg.norm(np.asarray(x, np.float64), axis=1)
        al = np.minimum(1.0, 0.05 * (nrm / 20.0) ** 4)
        sm = np.tanh(cumulative_cubic_raw(u, a, c, 1.0) @ M.T + beta)
        ss = np.tanh(ema_cubic_raw(u, a, c, al, 1.0) @ M.T + beta)
        P = (sm - sm.mean(1, keepdims=True)) / 2
        S = (ss - ss.mean(1, keepdims=True)) / 2
        V = np.minimum(P, S + 0.005)
        mu = (P.sum(1, keepdims=True) - P) / (K - 1)
        var = ((P * P).sum(1, keepdims=True) - P * P) / (K - 1) - mu * mu
        sd = np.sqrt(np.maximum(var, 0.0))
        zP = np.where(sd > 0, (P - mu) / np.where(sd > 0, sd, 1.0), 0.0)
        zV = np.where(sd > 0, (V - mu) / np.where(sd > 0, sd, 1.0), 0.0)
        w = np.maximum(zP, 0.0) ** 2
        m = (w.sum(1, keepdims=True) - w) / (K - 1)
        A = 2.0 * (1.0 + np.maximum(zV, 0.0) ** 2)
        B = 2.0 + m
        return ((A - B) / (A + B)).astype(np.float32)
    return f


def fr_hcp():
    """HCP (docs/experiments/20261007_ogata_fr_hcp.md, primary hcp): the history/current switching
    posterior hcp(zP, zS). Uses the frozen x_fr_hcp front end and map unchanged; only the hcp
    output is cast and returned."""
    import x_fr_hcp

    def f(x, g):
        fe = x_fr_hcp._fe(x, g)
        return x_fr_hcp.hcp(fe["zP"], fe["zS"], fe["K"]).astype(np.float32)
    return f


def esr_hcp():
    """ESR-HCP (docs/experiments/20261007_ogata_r3_sei_esr_rsc.md, primary esr_hcp): HCP with the
    current look LS replaced by 1 + omega (LS - 1), omega the excursion-survival weight. Uses the
    frozen x_r3 implementation unchanged; only the output is cast and returned."""
    import x_r3

    def f(x, g):
        return x_r3.esr_hcp(x, g).astype(np.float32)
    return f


def fco():
    """FCO (docs/experiments/20261007_ogata_r4_mux_rvs_fco.md, primary fco): HCP with finite competitive
    ownership votes added to its numerator and denominator. Uses the frozen x_r4 implementation unchanged;
    only the output is cast and returned."""
    import x_r4

    def f(x, g):
        return x_r4.fco(x, g).astype(np.float32)
    return f


def esrn_fco():
    """ESR-N + FCO (docs/experiments/20261008_ogata_r5_esrn.md, primary esrn_fco): HCP with ESR's excursion-survival
    weight on the claimed column's own numerator only, plus FCO's finite ownership votes. Uses the frozen x_r5
    implementation unchanged; only the output is cast and returned."""
    import x_r5

    def f(x, g):
        return x_r5.esrn_fco(x, g).astype(np.float32)
    return f


def esr_n():
    """ESR-N (docs/experiments/20261008_ogata_r5_esrn.md, primary esr_n): HCP with ESR's excursion-survival weight
    applied to the claimed column's own numerator only; the denominator stays HCP's. Uses the frozen x_r5
    implementation unchanged; only the output is cast and returned."""
    import x_r5

    def f(x, g):
        return x_r5.esr_n(x, g).astype(np.float32)
    return f


def blvs_n_fco():
    """BLVS-N+FCO (docs/experiments/20261008_ogata_blvs_stage0.md): ESR-N + FCO with ESR's weight multiplied by the
    Beta-Laplace verification-survival weight omega_V. Uses analyze_blvs_stage0.blvs_n_fco unchanged; only the output
    is cast and returned."""
    import analyze_blvs_stage0

    def f(x, g):
        return analyze_blvs_stage0.blvs_n_fco(x, g).astype(np.float32)
    return f


def pez_blvs_n_fco():
    """PEZ-BLVS-N+FCO (docs/experiments/20261008_ogata_pez_stage0.md): BLVS-N+FCO with the teammate's prefix-RMS
    row-energy ratio as a third numerator-only factor on verified-history columns. Uses
    analyze_pez_stage0.pez_blvs_n_fco unchanged; only the output is cast and returned."""
    import analyze_pez_stage0

    def f(x, g):
        return analyze_pez_stage0.pez_blvs_n_fco(x, g).astype(np.float32)
    return f


def peznh_blvs_n_fco():
    """PEZ-NH (docs/experiments/20261009_ogata_peznh_stage0.md): PEZ-BLVS-N+FCO without the verified-history gate on
    the energy factor. Uses analyze_peznh_stage0.peznh_blvs_n_fco unchanged; only the output is cast and returned."""
    import analyze_peznh_stage0

    def f(x, g):
        return analyze_peznh_stage0.peznh_blvs_n_fco(x, g).astype(np.float32)
    return f


def pxm_peznh():
    """PXM (docs/experiments/20261009_ogata_pxm_stage0.md): PEZ-NH with the teammate's max-prefix energy reference.
    Uses analyze_pxm_stage0.pxm_peznh unchanged; only the output is cast and returned."""
    import analyze_pxm_stage0

    def f(x, g):
        return analyze_pxm_stage0.pxm_peznh(x, g).astype(np.float32)
    return f


def fusion_mean_peznh_943955():
    """F-mean (docs/experiments/20261009_ogata_fusion_943955.md): (PEZ-NH + 943955 row-L2 readout) / 2.
    Uses analyze_fusion_943955.fusion_mean_peznh_943955 unchanged."""
    import analyze_fusion_943955

    def f(x, g):
        return analyze_fusion_943955.fusion_mean_peznh_943955(x, g)
    return f


def fusion_family(rule):
    """One rule of the fusion family stage 0 (docs/experiments/20261009_ogata_fusion_family.md).
    Uses analyze_fusion_family.rules unchanged and returns only the named rule."""
    import analyze_fusion_family

    def f(x, g):
        return analyze_fusion_family.rules(x, g)[rule]
    return f


def t2_family(var):
    """One variant of the T2 family stage 0 (docs/experiments/20261009_ogata_t2_family.md).
    Uses analyze_t2_family.variant_one (the same table entry as variants) and casts the output."""
    import analyze_t2_family

    def f(x, g):
        return analyze_t2_family.variant_one(x, g, var).astype(np.float32)
    return f


def t2_family_bl(var):
    """Band-limited fusion (C0 = .5 with 943955) on top of one T2 family variant
    (docs/experiments/20261009_ogata_t2_family.md, build amendment). Uses analyze_t2_family.bl_variant."""
    import analyze_t2_family

    def f(x, g):
        return analyze_t2_family.bl_variant(x, g, var)
    return f


def t2_cmp(h, l_):
    """Band-split composite cmp(h | l_): h's high band (>= .5), the band-limited mean of min(l_, h) and 943955 below
    (docs/experiments/20261009_ogata_rtpl_cmp.md). Uses analyze_rtpl_cmp.composite."""
    import analyze_rtpl_cmp

    def f(x, g):
        return analyze_rtpl_cmp.composite(x, g, h, l_)
    return f


def round2_cmp(h):
    """cmp(h | hold_nh) for a round-2 variant h on rtpl_nh (docs/experiments/20261009_ogata_round2_rtpl_screen.md).
    Uses analyze_round2.composite."""
    import analyze_round2

    def f(x, g):
        return analyze_round2.composite(x, g, h)
    return f


def t1_cau2_cmp():
    """cmp(rtpl_nh | hold_nh_c2): rtpl_nh's high band, the low band of hold_nh with the two-sided Cauchy likelihood
    (docs/experiments/20261009_ogata_t1_premise_search.md, stage 1b). Uses analyze_t1_cau2_chain.composite_c2."""
    import analyze_t1_cau2_chain

    def f(x, g):
        return analyze_t1_cau2_chain.composite_c2(x, g)
    return f


def t1_cau2_pez_cmp():
    """cmp(rtpl_pez | hold_nh_c2): 973452's high band and the two-sided Cauchy hold_nh low band
    (docs/experiments/20261009_ogata_t1_premise_search.md, stage 1c). Uses analyze_t1_c2_pez.composite_pez_c2."""
    import analyze_t1_c2_pez

    def f(x, g):
        return analyze_t1_c2_pez.composite_pez_c2(x, g)
    return f


def t1_geo_pez_cmp():
    """Stage-1d BUILD cell geo: 973452's high band, low band sqrt((1 + min(hold_nh_c2, H))(1 + 943955)) - 1
    (docs/experiments/20261009_ogata_t1_premise_search.md, stage 1d). Uses analyze_t1_stage1d.composite_geo."""
    import analyze_t1_stage1d

    def f(x, g):
        return analyze_t1_stage1d.composite_geo(x, g)
    return f


def t1_bank_pez_cmp(cell):
    """Stage-2 cell: 973452's high band over the teammate's bank low band (docs/experiments/
    20261009_ogata_t1_premise_search.md, stage 2). Uses analyze_t1_stage2_bank.composite_bank."""
    import analyze_t1_stage2_bank

    def f(x, g):
        return analyze_t1_stage2_bank.composite_bank(x, g, cell)
    return f


def t2h_cmp(cell):
    """T2 stage-H cell composed as 976810's recipe: veto_band(H', bank8p, 943955) (docs/experiments/
    20261011_ogata_t2_stage_h.md). Uses analyze_t2_stage_h.composite_h."""
    import analyze_t2_stage_h

    def f(x, g):
        return analyze_t2_stage_h.composite_h(x, g, cell)
    return f


def t2r_cmp(cell):
    """T2 stage-R cell composed as 976810's recipe: veto_band(H', bank8p, 943955) (docs/experiments/
    20261012_ogata_t2_stage_r.md). Uses analyze_t2_stage_r.composite_h."""
    import analyze_t2_stage_r

    def f(x, g):
        return analyze_t2_stage_r.composite_h(x, g, cell)
    return f


def t2s_cmp(cell):
    """T2 stage-S cell composed as 976810's recipe: veto_band(H', bank8p, 943955) (docs/experiments/
    20261012_ogata_t2_stage_s.md). Uses analyze_t2_stage_s.composite_h."""
    import analyze_t2_stage_s

    def f(x, g):
        return analyze_t2_stage_s.composite_h(x, g, cell)
    return f


def t2t_cmp(cell):
    """T2 stage-T cell composed as 976810's recipe: veto_band(H', bank8p, 943955) (docs/experiments/
    20261012_ogata_t2_stage_t.md). Uses analyze_t2_stage_t.composite_h."""
    import analyze_t2_stage_t

    def f(x, g):
        return analyze_t2_stage_t.composite_h(x, g, cell)
    return f


def bal_etx_b8d():
    """etx high band over the teammate's bank8 + DBS10 low band (docs/experiments/20261012_ogata_bal_etx_b8d.md)."""
    import bal_etx_b8d as BE
    return BE.bal


def bal_ete_b8d():
    """ete high band over the teammate's bank8 + DBS10 low band (docs/experiments/20261012_ogata_bal_etx_b8d.md)."""
    import bal_ete_b8d as BE
    return BE.bal
