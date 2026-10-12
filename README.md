# RCFA 2026 — team TamakiLab — code for the final submission 980314

This repository contains the complete inference code that generated our selected final submission to the
RCFA 2026 Challenge (Codabench), together with the environment specification and the commands to reproduce it.

| Item | Value |
| --- | --- |
| Team | TamakiLab |
| CodaBench accounts of the team | `mositaku`, `yogata` |
| Account of the final submission | **`yogata`** |
| Submission ID | **980314** |
| Submitted file | `RCFA2026_ogata_bal_ete_b8d.zip` (262,355,334 bytes, SHA-256 `acb908f5ded59e62769d7f9c30e2860122d244a20ac5bbf493a4d4cc0e018e72`) |
| Official result | T1-EER 0.0019176081, T2 Balanced Error 0.08577983 |
| Configuration name in the code | `bal_ete_b8d` |

## Requirements

- Linux on x86-64 (tested: Ubuntu, Intel Xeon Gold 5418Y). CPU only; no GPU.
- Python 3.12 (tested 3.12.3) and NumPy 2.2.6 (the PyPI wheel with its bundled OpenBLAS). NumPy is the only
  third-party dependency (`requirements.txt`).
- No model weights, no pretrained networks and no external data: the method has no trained parameters. All
  constants are fixed in the code; everything else is computed at run time from the released embeddings.
- Runtime: about 13 minutes for both tracks with 6 workers on an Intel Xeon Gold 5418Y (775 s of scoring,
  13 min wall clock for `reproduce.sh`). Memory: at most about 0.6 GB per process (about 4 GB with 6 workers).
  Disk: about 0.5 GB for the output.

The scores are bit-reproducible on Linux x86-64 with the versions above and one BLAS thread per process (set
automatically). On macOS, NumPy's Accelerate backend can raise `FloatingPointError` in `matmul` or give different
floating-point rounding, so please use Linux.

## Data layout

The released challenge data, unchanged:

```text
DATA_DIR/
  track1/gallery.npz
  track1/probe_seq_embeddings/*.npz   # 200 sequences
  track2/gallery.npz
  track2/probe_seq_embeddings/*.npz   # 200 sequences
```

## Installation and execution

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHON=.venv/bin/python bash reproduce.sh /path/to/DATA_DIR outputs/980314 6
```

`reproduce.sh DATA_DIR OUT_DIR [WORKERS]` runs three steps:

1. `scripts/ogata/build_candidate_par.py bal_ete_b8d OUT_DIR WORKERS` scores all 400 sequences (one worker
   process per sequence, fresh state for every sequence) and writes `OUT_DIR/submission.zip` in the challenge format
   (`track1.zip` and `track2.zip`, one `score_<sequence>.npz` with a `(frames, 200)` float32 array `score` per
   sequence) plus `OUT_DIR/receipt.json`. It refuses to overwrite an existing `submission.zip`. While building it
   also checks, on the first three sequences of each track, that scoring the first half of the sequence gives the
   same rows as scoring the whole sequence, and that the result is bit-identical to the reference implementation
   (`bal_ete_b8d.outputs_fn`).
2. `scripts/validate_submission.py` checks the ZIP structure, shapes, dtype, finiteness and the range [-1, 1].
3. `scripts/compare_submission.py` compares the result with submission 980314 (see below).

The number of workers (1 to 6) does not change the result.

## Checking the result

The outer ZIP stores `track1.zip` and `track2.zip` together with their file timestamps, so the SHA-256 of the
outer file differs from run to run. The two inner archives are written with fixed metadata and are byte-identical
whenever the scores are identical:

| Inner archive of 980314 | SHA-256 |
| --- | --- |
| `track1.zip` (139,277,873 bytes) | `e95affb69bb6cb37499353fcbed39285594c1546d53ad423b3f680c8c556b2e9` |
| `track2.zip` (123,077,247 bytes) | `9061267752bb80ed5b3b27f291aa66da2b77c4cff361a963e8cf817fbbac6b94` |

```bash
.venv/bin/python scripts/compare_submission.py outputs/980314/submission.zip                 # against the table above
.venv/bin/python scripts/compare_submission.py outputs/980314/submission.zip RCFA2026_ogata_bal_ete_b8d.zip   # array by array
```

## Causality (strictly online processing)

- The score of frame t is computed from frames 0..t of the same sequence and the gallery only. Every statistic is a
  running quantity (cumulative sums, exponential moving averages, counters, templates built from earlier frames);
  no score is revised after later frames arrive.
- The state is created fresh for every sequence. Nothing is shared across sequences, and nothing uses track
  identity, attack types, labels or ground truth. Both tracks run the same code with the same constants.
- The code scores a whole sequence array in one call for speed, but each output row depends only on the rows up to
  it. `scripts/check_causality.py` verifies this directly: it scores a sequence once, then scores prefixes cut at
  1, 2, 5, 9, 10, 11, 15, 30, 59, 60, 61, 100, 300 frames and at 25 %, 50 % and 75 % of the length, and requires the
  overlapping rows to be bit-identical.

```bash
.venv/bin/python scripts/check_causality.py /path/to/DATA_DIR --sequences 2   # about 12 minutes, prints PASS
```

## Method overview

For frame t and gallery identity j the output is a composite of three causal scorers, H, L and b:

```text
out = H                      if H >= 0.75
      -1 + 0.025 (1 + b)     if H < 0.75 and L == -1
      min(L, 0.5)            otherwise
```

The Track-1 operating threshold (FAR = 1e-5, about 0.84) lies inside the band H >= 0.75, so every Track-2 decision
and the operating point come from H, while the Track-1 EER threshold (about -0.10) lies inside the band below 0.5,
which comes from L.

- **H — attack-aware score** (`scripts/ogata/analyze_t2_stage_s.py`, `variants_h(x, g)["ete"]`). A posterior-odds
  score from a long-term cubic-kernel memory of gallery evidence and a quality-weighted short-term moving average,
  with competitive ownership across identities and multiplicative discounts that lower an identity's score when its
  recent evidence stops looking like the verified frames of the same sequence: survival-type discounts on the
  current evidence and on verification, a row-energy discount relative to the sequence's own history, and a
  residual-template discount whose per-identity template is frozen after the first 100 verified clean frames.
- **L — identity-discriminative score** (`scripts/final_stretch/t1_b8d.py`, `b8d_core`, building on
  `t1_bank*.py`). Label-free causal ownership: once one identity leads the running count of verified clean frames,
  the other 199 columns of that frame are set to -1. Frames without an owner are mapped from causal z-statistics
  chosen by the state of the sequence (opening frames, middle, late, low-information rows); after the detected
  change at the start of the sequence, such frames are re-ranked by averaging standardized scores over each
  identity's 10 nearest gallery neighbours, keeping the frame's set of values unchanged.
- **b — tie-breaking order for -1 values** (`rcfa/methods/gallery_discriminative_row_l2.py`). A causal
  gallery-discriminative cubic-kernel readout with per-frame L2 normalization; it only orders the values that L set
  to -1 inside [-1, -0.95].

## Code layout

| Path | Content |
| --- | --- |
| `reproduce.sh` | One-command reproduction |
| `scripts/ogata/build_candidate_par.py` | Builder used for the submission (config `bal_ete_b8d`) |
| `scripts/ogata/bal_ete_b8d.py` | The composite scorer: `bal(x, g)` |
| `scripts/ogata/analyze_t2_stage_s.py` and imports | H |
| `scripts/final_stretch/t1_b8d.py` and imports | L |
| `rcfa/` | Shared embedding methods and b |
| `scripts/build_candidate_submission.py` | Deterministic ZIP / NPZ writing and data loading |
| `scripts/validate_submission.py` | Format check |
| `scripts/check_causality.py`, `scripts/compare_submission.py` | Verification helpers added for this package |

All other files are unmodified copies, from the team repository at the commit used to build 980314 (`72af240`),
of every module the scorer imports. Several of them are research modules that also contain analysis code; only the
functions reached from `bal_ete_b8d.bal` are used to produce the scores.
