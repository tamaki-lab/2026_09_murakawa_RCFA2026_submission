"""Preregistered diagnostic only: no processor, labels, tuning, or submission."""
import argparse
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
from scripts.analyze_timescale_signals import select_evenly
from scripts.evaluate_causal_methods import choose_transition_pairs

CONFIG = dict(old_drop=0.10, new_gain=0.10, margin=0.05,
              support=0.40, persistence=10)


def support_signals(current, reference):
    """Inputs at t are current scores and state scores from t-1, never track ID.

    Identity here means gallery index, not a private ground-truth identity.
    Consecutive evidence requires the same (incumbent, challenger) pair.
    """
    current, reference = np.asarray(current), np.asarray(reference)
    if (current.ndim != 2 or current.shape != reference.shape or
            current.shape[1] < 2 or len(current) == 0 or
            not np.isfinite(current).all() or not np.isfinite(reference).all()):
        raise ValueError('Expected matching nonempty finite score matrices')
    rows = np.arange(len(current))
    incumbent, challenger = reference.argmax(1), current.argmax(1)
    top = current[rows, challenger]
    margin = top - np.partition(current, -2, axis=1)[:, -2]
    drop = reference[rows, incumbent] - current[rows, incumbent]
    gain = top - reference[rows, challenger]
    loss = drop >= CONFIG['old_drop']
    transfer = ((challenger != incumbent) & loss &
                (gain >= CONFIG['new_gain']) & (margin >= CONFIG['margin']) &
                (top >= CONFIG['support']))
    run = np.zeros(len(current), dtype=np.int32)
    previous = None
    count = 0
    for t in range(len(current)):
        pair = (int(incumbent[t]), int(challenger[t]))
        count = (count + 1 if pair == previous else 1) if transfer[t] else 0
        run[t] = count
        previous = pair if transfer[t] else None
    return dict(loss=loss, transfer=transfer, persistent=run >= CONFIG['persistence'])


def diagnose(features, gallery):
    # Row-local dot products ensure the arithmetic is independent of prefix size.
    current = np.stack([row @ gallery.T for row in normalize(features)])
    state = np.stack([row @ gallery.T for row in normalize(dual_ema_short_tail(features))])
    reference = np.concatenate((current[:1], state[:-1]), axis=0)
    return support_signals(current, reference)


def select_panel(paths, panel):
    if len(paths) != 200:
        raise ValueError('Expected exactly 200 sequences per track')
    discovery = select_evenly(paths, 40)
    if panel == 'discovery':
        return discovery
    if panel != 'confirmation':
        raise ValueError('Unknown panel')
    used = set(discovery)
    return select_evenly([p for p in paths if p not in used], 40)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        x = data['feature'].astype(np.float32)
    if x.ndim != 2 or len(x) < 1 or not np.isfinite(x).all():
        raise ValueError('Invalid feature matrix')
    return x


def controls():
    reference = np.tile([.8, .1, .1], (60, 1))
    stable = reference.copy()
    positive = stable.copy()
    positive[20:] = [.1, .8, .1]
    spike = stable.copy()
    spike[20] = [.1, .8, .1]
    alternating = positive.copy()
    alternating[21::2] = [.1, .1, .8]
    collapse = stable.copy()
    collapse[20:] = [.0, .1, .1]
    signals = {k: support_signals(v, reference) for k, v in
               dict(stable=stable, positive=positive, spike=spike,
                    alternating=alternating, collapse=collapse).items()}
    return {k: int(v['persistent'].sum()) for k, v in signals.items()}


def evaluate_track(root, track, panel):
    folder = root / track
    paths = select_panel(sorted((folder / 'probe_seq_embeddings').glob('*.npz')), panel)
    gallery = normalize(load(folder / 'gallery.npz'))
    sequences = [load(p) for p in paths]
    counts = {k: 0 for k in ('loss', 'transfer', 'persistent')}
    seq_counts = dict.fromkeys(counts, 0)
    prefix_error = 0
    for x in sequences:
        full = diagnose(x, gallery)
        for k, values in full.items():
            counts[k] += int(values.sum())
            seq_counts[k] += int(values.any())
        for end in (1, len(x)//3, len(x)//2):
            prefix = diagnose(x[:max(1, end)], gallery)
            for k in full:
                prefix_error = max(prefix_error, int(np.any(prefix[k] != full[k][:max(1, end)])))
    # Gallery indices/sequence names and individual traces are never serialized.
    # Pair selection is offline fixture construction, not online input/state.
    pairs = dict(distant=choose_transition_pairs(sequences, 20, 100),
                 adjacent=[(2*i, 2*i+1) for i in range(20)])
    synthetic = {}
    for name, selected in pairs.items():
        latencies, pre_events = [], 0
        for a, b in selected:
            before, after = sequences[a][-200:], sequences[b][:400]
            if len(before) != 200 or len(after) != 400:
                raise ValueError('Insufficient synthetic fixture length')
            flags = diagnose(np.concatenate((before, after)), gallery)['persistent']
            pre_events += int(flags[:200].any())
            # Require ten NEW qualifying post-boundary frames. A run already
            # active at the join must not count as immediate successful detection.
            hits = np.flatnonzero(flags[209:])
            latencies.append(400 if not len(hits) else int(hits[0]) + 10)
        synthetic[name] = dict(pair_count=20,
            detected_fraction=float(np.mean(np.asarray(latencies) < 400)),
            latency_censored_mean=float(np.mean(latencies)),
            latency_censor_frames=400, pre_event_pair_fraction=pre_events/20)
    n = sum(map(len, sequences))
    return dict(sequence_count=40, frame_count=n, prefix_max_abs_error=prefix_error,
                frame_fractions={k: v/n for k, v in counts.items()},
                sequence_fractions={k: v/40 for k, v in seq_counts.items()},
                synthetic=synthetic)


def gates(report):
    t = report['tracks']['track1']
    c = report['controls']
    checks = dict(normal_proxy_frames=t['frame_fractions']['persistent'] <= .01,
                  normal_proxy_sequences=t['sequence_fractions']['persistent'] <= .05,
                  controls=c == dict(stable=0, positive=31, spike=0, alternating=0, collapse=0))
    for track, result in report['tracks'].items():
        checks[track+'_causal'] = result['prefix_max_abs_error'] == 0
        for name, s in result['synthetic'].items():
            checks[track+'_'+name] = (s['detected_fraction'] >= .8 and
                s['latency_censored_mean'] <= 40 and s['pre_event_pair_fraction'] <= .05)
    return checks


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('challenge_data'))
    parser.add_argument('--panel', choices=('discovery', 'confirmation'), default='discovery')
    parser.add_argument('--discovery-report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite report')
    if args.panel == 'confirmation':
        if not args.discovery_report:
            parser.error('Confirmation requires a passing discovery report')
        previous = json.loads(args.discovery_report.read_text())
        if (previous['panel'] != 'discovery' or previous['config'] != CONFIG or
                previous['verdict'] != 'PASS_DIAGNOSTIC_ONLY' or
                not all(gates(previous).values())):
            parser.error('Discovery did not pass the frozen specification')
    os.umask(0o077)
    start = time.perf_counter()
    report = dict(panel=args.panel, config=CONFIG, controls=controls(), tracks={},
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  official_metrics_estimated=False)
    for track in ('track1', 'track2'):
        report['tracks'][track] = evaluate_track(args.data_root, track, args.panel)
        print(track + ': aggregate diagnostics complete', flush=True)
    report['gates'] = gates(report)
    report['verdict'] = 'PASS_DIAGNOSTIC_ONLY' if all(report['gates'].values()) else 'REJECT_NO_PROCESSOR'
    report['elapsed_seconds'] = time.perf_counter()-start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
