"""Frozen relative-support evidence diagnostic; never changes submission scores."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_support_failures import score_matrices
from scripts.analyze_identity_support import load, select_panel
from scripts.evaluate_causal_methods import choose_transition_pairs

CONFIG = dict(beta=.1, threshold=.6, spread_floor=.05)


def relative_evidence(current, reference):
    """Independent decaying evidence per gallery entry, reset on incumbent change.

    reference[t] must contain scores from the prior short-tail state, never a
    state computed using future frames. No track or synthetic-boundary input.
    """
    c, r = np.asarray(current, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if (c.ndim != 2 or c.shape != r.shape or len(c) == 0 or c.shape[1] < 2 or
            not np.isfinite(c).all() or not np.isfinite(r).all()):
        raise ValueError('Expected matching nonempty finite score matrices')
    q = np.zeros(c.shape[1], dtype=np.float64)
    out = np.zeros(len(c), dtype=np.float64)
    previous = None
    for t, row in enumerate(c):
        incumbent, challenger = int(r[t].argmax()), int(row.argmax())
        if incumbent != previous:
            q.fill(0)
        q *= 1-CONFIG['beta']
        sigma = max(float(row.std()), CONFIG['spread_floor'])
        second = np.partition(row, -2)[-2]
        advantage = np.clip((row[challenger]-row[incumbent])/sigma, 0, 1)
        margin = np.clip((row[challenger]-second)/sigma, 0, 1)
        if challenger != incumbent:
            q[challenger] += CONFIG['beta']*advantage*margin
        out[t] = q.max()
        previous = incumbent
    return out


def controls():
    ref = np.tile([.8,.1,.1], (300,1))
    stable = ref.copy()
    positive = stable.copy(); positive[100:] = [.1,.8,.1]
    spike = stable.copy(); spike[100] = [.1,.8,.1]
    alternating = positive.copy(); alternating[101::2] = [.1,.1,.8]
    collapse = stable.copy(); collapse[100:] = [0,.1,.1]
    weak = stable.copy(); weak[100:] = [0,.001,0]
    out = {}
    for name,c in dict(stable=stable,positive=positive,spike=spike,
                       alternating=alternating,collapse=collapse,weak=weak).items():
        q = relative_evidence(c,ref)
        hits = np.flatnonzero(q[100:] >= CONFIG['threshold'])
        out[name] = dict(max_q=float(q.max()),first_post_frame=None if not len(hits) else int(hits[0])+1)
    return out


def evaluate(root, track, panel):
    folder = root/track
    paths = select_panel(sorted((folder/'probe_seq_embeddings').glob('*.npz')),panel)
    g = load(folder/'gallery.npz')
    from example import normalize
    g = normalize(g)
    seqs = [load(p) for p in paths]
    qs = []
    prefix_error = 0.
    for x in seqs:
        c,r = score_matrices(x,g)
        q = relative_evidence(c,r)
        qs.append(q)
        for end in (1,len(x)//3,len(x)//2):
            cp,rp = score_matrices(x[:end],g)
            qp = relative_evidence(cp,rp)
            prefix_error = max(prefix_error,float(np.max(np.abs(qp-q[:end]))))
    allq = np.concatenate(qs)
    result = dict(sequence_count=40,frame_count=len(allq),prefix_max_abs_error=prefix_error,
        q_quantiles_50_95_99_max=np.quantile(allq,[.5,.95,.99,1]).tolist(),
        event_frame_fraction=float(np.mean(allq >= CONFIG['threshold'])),
        event_sequence_fraction=float(np.mean([np.any(q >= CONFIG['threshold']) for q in qs])),synthetic={})
    pairs = dict(distant=choose_transition_pairs(seqs,20,100),adjacent=[(2*i,2*i+1) for i in range(20)])
    for name,panel_pairs in pairs.items():
        detected, latencies, prior = 0,[],0
        for a,b in panel_pairs:
            pre,post = seqs[a][-200:],seqs[b][:400]
            if len(pre) !=200 or len(post) !=400:
                raise ValueError('Insufficient fixture length')
            c,r = score_matrices(np.concatenate((pre,post)),g)
            q = relative_evidence(c,r)
            flags = q >= CONFIG['threshold']
            prior += int(flags[:200].any())
            # Count a NEW threshold crossing after the join, not a preexisting
            # sustained event. No boundary is passed to relative_evidence.
            rising = flags & ~np.concatenate(([False],flags[:-1]))
            hits = np.flatnonzero(rising[200:])
            detected += int(bool(len(hits)))
            latencies.append(400 if not len(hits) else int(hits[0])+1)
        result['synthetic'][name] = dict(pair_count=20,detected_count=detected,
            detected_fraction=detected/20,latency_censored_mean=float(np.mean(latencies)),
            pre_event_pair_fraction=prior/20)
    return result


def gates(report):
    t = report['tracks']['track1']
    checks = dict(normal_frames=t['event_frame_fraction']<=.01,
                  normal_sequences=t['event_sequence_fraction']<=.05,
                  controls=all(v['first_post_frame'] is None for k,v in report['controls'].items() if k!='positive')
                    and report['controls']['positive']['first_post_frame']==9)
    for track,r in report['tracks'].items():
        checks[track+'_prefix'] = r['prefix_max_abs_error']==0
        for name,s in r['synthetic'].items():
            checks[track+'_'+name] = s['detected_fraction']>=.8 and s['latency_censored_mean']<=40 and s['pre_event_pair_fraction']<=.05
    return checks


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data-root',type=Path,default=Path('challenge_data'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--panel',choices=('discovery','confirmation'),default='discovery')
    p.add_argument('--discovery-report',type=Path)
    args=p.parse_args()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite report')
    if args.panel=='confirmation':
        if not args.discovery_report:
            p.error('Requires passing discovery report from same commit/config')
        d=json.loads(args.discovery_report.read_text())
        if d['panel']!='discovery' or d['git_commit']!=commit or d['config']!=CONFIG or not all(gates(d).values()):
            p.error('Discovery did not pass same frozen configuration')
    os.umask(0o077)
    start=time.perf_counter()
    report=dict(config=CONFIG,git_commit=commit,panel=args.panel,controls=controls(),tracks={},official_metrics_estimated=False)
    for track in ('track1','track2'):
        report['tracks'][track]=evaluate(args.data_root,track,args.panel)
        print(track+': relative evidence complete',flush=True)
    report['gates']=gates(report)
    report['verdict']='PASS_DIAGNOSTIC_ONLY' if all(report['gates'].values()) else 'REJECT_NO_PROCESSOR'
    report['elapsed_seconds']=time.perf_counter()-start
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report,f,indent=2); f.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
