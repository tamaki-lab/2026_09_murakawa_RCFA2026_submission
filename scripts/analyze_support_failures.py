"""Aggregate-only decomposition of the frozen support diagnostic; not a method."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example import normalize
from rcfa.embedding import dual_ema_short_tail
from scripts.analyze_identity_support import CONFIG, load, select_panel, support_signals
from scripts.evaluate_causal_methods import choose_transition_pairs

ORDER = ('different', 'old_drop', 'new_gain', 'margin', 'support')


def components(current, reference):
    flags = support_signals(current, reference)
    rows = np.arange(len(current))
    old, new = reference.argmax(1), current.argmax(1)
    top = current[rows, new]
    masks = dict(different=old != new,
        old_drop=reference[rows, old]-current[rows, old] >= CONFIG['old_drop'],
        new_gain=top-reference[rows, new] >= CONFIG['new_gain'],
        margin=top-np.partition(current, -2, axis=1)[:, -2] >= CONFIG['margin'],
        support=top >= CONFIG['support'])
    combined = np.logical_and.reduce(list(masks.values()))
    np.testing.assert_array_equal(combined, flags['transfer'])
    return masks, flags, old, new


def score_matrices(features, gallery):
    current = np.stack([x @ gallery.T for x in normalize(features)])
    state = np.stack([x @ gallery.T for x in normalize(dual_ema_short_tail(features))])
    return current, np.concatenate((current[:1], state[:-1]))


def run_summary(transfer, old, new):
    """Count observed (not right-censored) run interruptions; reasons overlap."""
    count = maximum = short_ended = 0
    reasons = Counter()
    for t in range(len(transfer)):
        changed = t > 0 and (old[t] != old[t-1] or new[t] != new[t-1])
        if count and (not transfer[t] or changed):
            if count < CONFIG['persistence']:
                short_ended += 1
                reasons['incumbent_changed'] += int(old[t] != old[t-1])
                reasons['challenger_changed'] += int(new[t] != new[t-1])
                reasons['conditions_failed'] += int(not transfer[t])
                reasons['incumbent_adopted_previous_challenger'] += int(old[t] == new[t-1])
            count = 0
        if transfer[t]:
            count += 1
            maximum = max(maximum, count)
    return dict(max_run=maximum, short_runs_ended=short_ended,
                short_run_interruptions={k: reasons[k] for k in
                    ('incumbent_changed', 'challenger_changed', 'conditions_failed',
                     'incumbent_adopted_previous_challenger')},
                terminal_short_run_censored=int(0 < count < CONFIG['persistence']))


def window_summary(masks, flags, old, new, start, end):
    m = {k: v[start:end] for k, v in masks.items()}
    pending = np.ones(end-start, dtype=bool)
    first_fail = {}
    for k in ORDER:
        first_fail[k] = int((pending & ~m[k]).sum())
        pending &= m[k]
    first_fail['all_pass'] = int(pending.sum())
    return dict(frame_count=end-start,
        independent_failures={k: int((~v).sum()) for k, v in m.items()},
        first_failure_ordered=first_fail,
        persistent_frames=int(flags['persistent'][start:end].sum()),
        **run_summary(flags['transfer'][start:end], old[start:end], new[start:end]))


def aggregate(items):
    if not items:
        return dict(sample_count=0)
    n = sum(x['frame_count'] for x in items)
    return dict(sample_count=len(items), frame_count=n,
        independent_failure_fractions={k: sum(x['independent_failures'][k] for x in items)/n for k in ORDER},
        first_failure_ordered_fractions={k: sum(x['first_failure_ordered'][k] for x in items)/n
                                        for k in (*ORDER, 'all_pass')},
        persistent_frame_fraction=sum(x['persistent_frames'] for x in items)/n,
        max_run_quantiles_0_50_100=np.quantile([x['max_run'] for x in items], [0,.5,1]).tolist(),
        short_runs_ended=sum(x['short_runs_ended'] for x in items),
        short_run_interruptions={k: sum(x['short_run_interruptions'][k] for x in items)
                                 for k in items[0]['short_run_interruptions']},
        terminal_short_runs_censored=sum(x['terminal_short_run_censored'] for x in items))


def detection(flags, boundary=200):
    # Ten post-join frames must have elapsed. Keep missing separate from a
    # detection ON the last frame (legacy report used400 for both).
    hits = np.flatnonzero(flags[boundary+CONFIG['persistence']-1:])
    return None if not len(hits) else int(hits[0])+CONFIG['persistence']


def evaluate(root, track):
    folder = root/track
    paths = select_panel(sorted((folder/'probe_seq_embeddings').glob('*.npz')), 'discovery')
    gallery = normalize(load(folder/'gallery.npz'))
    sequences = [load(p) for p in paths]
    normal = []
    error = 0
    for x in sequences:
        c, r = score_matrices(x, gallery)
        m, f, old, new = components(c, r)
        normal.append(window_summary(m, f, old, new, 0, len(x)))
        for end in (1, len(x)//3, len(x)//2):
            cp, rp = score_matrices(x[:end], gallery)
            error = max(error, float(np.max(np.abs(cp-c[:end]))), float(np.max(np.abs(rp-r[:end]))))
    sets = dict(distant=choose_transition_pairs(sequences,20,100),
                adjacent=[(2*i,2*i+1) for i in range(20)])
    result = dict(normal=aggregate(normal), prefix_max_abs_score_error=error, synthetic={})
    for name, pairs in sets.items():
        windows = {k: [] for k in ('first40', 'full400', 'miss_first40', 'miss_full400')}
        classification = Counter()
        latency, prior = [], 0
        for a,b in pairs:
            before, after = sequences[a][-200:], sequences[b][:400]
            if len(before) != 200 or len(after) != 400:
                raise ValueError('Insufficient fixture length')
            c,r = score_matrices(np.concatenate((before,after)), gallery)
            m,f,old,new = components(c,r)
            hit = detection(f['persistent'])
            prior += int(f['persistent'][:200].any())
            latency.append(400 if hit is None else hit)
            category = 'early' if hit is not None and hit <=40 else 'late' if hit is not None else 'miss'
            classification[category] += 1
            if hit == 400:
                classification['last_frame_detection'] += 1
            if hit is None:
                if not m['different'][200:].any():
                    classification['miss_no_different_winner'] += 1
                elif not f['transfer'][200:].any():
                    classification['miss_no_instant_conjunction'] += 1
                else:
                    classification['miss_fragmented_conjunction'] += 1
            for key, end in [('first40',240),('full400',600)]:
                w = window_summary(m,f,old,new,200,end)
                windows[key].append(w)
                if hit is None:
                    windows['miss_'+key].append(w)
        result['synthetic'][name] = dict(pair_count=20, counts=dict(classification),
            latency_censored_mean=float(np.mean(latency)), pre_event_pair_fraction=prior/20,
            **{k: aggregate(v) for k,v in windows.items()})
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data-root', type=Path, default=Path('challenge_data'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite report')
    os.umask(0o077)
    start = time.perf_counter()
    report = dict(config=CONFIG, panel='discovery_only', official_metrics_estimated=False,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(), tracks={})
    for track in ('track1','track2'):
        report['tracks'][track] = evaluate(args.data_root, track)
        print(track+': decomposition complete', flush=True)
    report['elapsed_seconds'] = time.perf_counter()-start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report,f,indent=2)
        f.write('\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
