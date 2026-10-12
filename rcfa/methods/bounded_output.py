"""Shared causal diagnostic from submission 918071; not a promoted baseline."""
import numpy as np
from example import normalize
from rcfa.embedding import dual_ema_short_tail
from scripts.evaluate_relative_support import relative_evidence

CAP = .1
MAX_DISTANCE = float(np.sqrt(2-2*np.sqrt(1-(CAP/(1-CAP))**2)))


def validate_matrix(value):
    x = np.asarray(value)
    if (x.ndim != 2 or not all(x.shape) or
            not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all()):
        raise ValueError('Expected nonempty finite floating matrix')
    return x


def mix_output(base, current, q):
    b = validate_matrix(base).astype(np.float64)
    x = validate_matrix(current).astype(np.float64)
    q = np.asarray(q, dtype=np.float64)
    if b.shape != x.shape or q.shape != (len(b),) or not np.isfinite(q).all() or np.any((q<0)|(q>1)):
        raise ValueError('Invalid mixing shape or evidence')
    w = CAP*np.clip((q-.6)/.4,0,1)
    with np.errstate(over='ignore', invalid='ignore'):
        bn, xn = np.linalg.norm(b,axis=1), np.linalg.norm(x,axis=1)
    if not np.isfinite(bn).all() or not np.isfinite(xn).all():
        raise ValueError('Nonfinite embedding norm')
    active = (w>0)&(bn>1e-12)&(xn>1e-12)
    w = np.where(active,w,0.)
    out = b.copy()
    out[active] = ((1-w[active,None])*b[active] +
                   w[active,None]*bn[active,None]*normalize(x[active]))
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite mixed output')
    return out,w


def process_bounded(features, gallery_normalized):
    """Raw outputs, immutable baseline, evidence and effective weights.

    Caller supplies the same normalized gallery as the official pipeline.
    All state is recomputed from the sequence input; no state/track/boundary input.
    """
    x, g = validate_matrix(features), validate_matrix(gallery_normalized)
    if x.shape[1] != g.shape[1] or len(g)<2:
        raise ValueError('Gallery dimension/count mismatch')
    # Same dot-product path for contiguous and reversed gallery views.
    g = np.ascontiguousarray(g)
    base = dual_ema_short_tail(x)
    # Reduce each gallery dot product independently: matvec kernels can change
    # float32 rounding for SIMD remainder rows after a gallery permutation.
    current_scores = np.stack([np.einsum('ij,j->i',g,row,optimize=False) for row in normalize(x)])
    baseline_scores = np.stack([np.einsum('ij,j->i',g,row,optimize=False) for row in normalize(base)])
    previous = np.concatenate((current_scores[:1],baseline_scores[:-1]))
    q = relative_evidence(current_scores,previous)
    out,w = mix_output(base,x,q)
    return out,base,q,w


def raw_scores(outputs, gallery_normalized):
    """Row-local raw cosine; float32 conversion, no scale/shift/clipping."""
    x,g = validate_matrix(outputs),validate_matrix(gallery_normalized)
    if x.shape[1] != g.shape[1]:
        raise ValueError('Gallery dimension mismatch')
    g = np.ascontiguousarray(g)
    scores = np.stack([np.einsum('ij,j->i',g,row,optimize=False) for row in normalize(x)]).astype(np.float32)
    if not np.isfinite(scores).all() or np.any(np.abs(scores)>1):
        raise ValueError('Score contract violated')
    return scores
