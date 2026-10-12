"""Analysis helpers on top of bank8 (aggregates only; pseudo targets are evaluator-only, never inside a processor).

`state8(base0, x, g)` is `t1_eval7.state7` plus bank8's output in "s8" (bank8 keeps bank7's / bank6's row states);
`report8(R, keys)` is `t1_eval7.report` with the job's entries built on bank8's output, its baseline key renamed
"bank8".
`load`, `entry`, `swap` and `run` are t1_eval7's, applied to bank8's output.
"""
import numpy as np

import t1_bank8 as B8
import t1_eval7 as E7

D = E7.D
GEQ = E7.GEQ
load, entry, swap, run = E7.load, E7.entry, E7.swap, E7.run


def state8(base0, x, g):
    st = E7.state7(base0, x, g)
    st["s8"] = B8.post8(np.asarray(base0, np.float32), x, g)
    return st


def report8(R, keys, name=str, top=5, subsets=True):
    res = E7.report(R, keys, name=name, top=top, subsets=subsets)
    res["bank8"] = res.pop("bank7")
    return res
