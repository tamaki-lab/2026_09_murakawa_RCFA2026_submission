"""Check a reproduced submission ZIP against CodaBench submission 980314.

Usage:
  python scripts/compare_submission.py OUT.zip            # compare the inner archives with 980314's SHA-256
  python scripts/compare_submission.py OUT.zip REF.zip    # compare every score array with another submission ZIP

The outer ZIP stores track1.zip and track2.zip with their file timestamps, so the outer SHA-256 changes from run to
run. The inner archives are written with fixed metadata (scripts/build_candidate_submission.fixed_zip_info) and are
byte-identical whenever the scores are; they are what this script compares. Exit code 0 means identical.
"""
import hashlib
import io
import sys
import zipfile

import numpy as np

# SHA-256 of track1.zip / track2.zip inside the submitted 980314 ZIP
# (RCFA2026_ogata_bal_ete_b8d.zip, 262,355,334 bytes, outer SHA-256
# acb908f5ded59e62769d7f9c30e2860122d244a20ac5bbf493a4d4cc0e018e72).
EXPECTED = {
    "track1.zip": "e95affb69bb6cb37499353fcbed39285594c1546d53ad423b3f680c8c556b2e9",  # 139,277,873 bytes
    "track2.zip": "9061267752bb80ed5b3b27f291aa66da2b77c4cff361a963e8cf817fbbac6b94",  # 123,077,247 bytes
}


def inner(path):
    with zipfile.ZipFile(path) as z:
        return {n: z.read(n) for n in ("track1.zip", "track2.zip")}


def scores(blob):
    out = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for n in sorted(z.namelist()):
            with np.load(io.BytesIO(z.read(n)), allow_pickle=False) as f:
                out[n] = f["score"]
    return out


def main(argv):
    if len(argv) not in (1, 2):
        print(__doc__)
        return 2
    a = inner(argv[0])
    ok = True
    if len(argv) == 1:
        for n, blob in a.items():
            h = hashlib.sha256(blob).hexdigest()
            want = EXPECTED[n]
            same = want is not None and h == want
            ok &= same
            print(f"{n}: sha256 {h} {'== 980314' if same else '!= 980314 (expected ' + str(want) + ')'}")
        return 0 if ok else 1
    b = inner(argv[1])
    for n in ("track1.zip", "track2.zip"):
        if a[n] == b[n]:
            print(f"{n}: byte-identical ({hashlib.sha256(a[n]).hexdigest()})")
            continue
        sa, sb = scores(a[n]), scores(b[n])
        if sorted(sa) != sorted(sb):
            print(f"{n}: member sets differ")
            ok = False
            continue
        diff = [k for k in sa if sa[k].shape != sb[k].shape or not np.array_equal(sa[k], sb[k])]
        worst = max((float(np.abs(sa[k].astype(np.float64) - sb[k]).max()) for k in diff
                     if sa[k].shape == sb[k].shape), default=0.0)
        print(f"{n}: {len(sa)} members, {len(diff)} differ, max abs diff {worst:.3g}")
        ok &= not diff
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
