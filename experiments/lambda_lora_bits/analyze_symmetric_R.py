#!/usr/bin/env python3
"""R with a symmetric normaliser, as a robustness check of where the adapter changes direction.

R(lambda) = T(lambda) / B(lambda) divides the cross-teacher overlap T by the data-only overlap at lambda alone
(Table 3, with B in Table 16), so a collapsing B(lambda), as in A/Qwen, can hold R up after the switch. The
symmetric version divides by the geometric mean of the data-only overlaps at both ends,

    R_sym(lambda) = T(lambda) / sqrt(B(0) * B(lambda)) = R(lambda) * sqrt(B(lambda) / B(0)),

using the treatment and baseline columns of experiments/phi_resample_s012.json (seed means, as in Table 3).
Reported per cell: R_sym at every lambda, its drop over each interval, and the interval of the largest drop.

Run from the repo root; writes experiments/symmetric_R.json.
"""

import json
from pathlib import Path

X = Path("experiments")
LAM = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
SWITCH = {"A": "0.2->0.4", "B": "0.4->0.6", "D": "0.4->0.6"}   # first switch step (pair C is "D" in this file)


def main():
    res = json.loads((X / "phi_resample_s012.json").read_text())
    out = {}
    for cell, v in res.items():
        rows = v["rows"]
        b0 = rows["0.0"]["baseline"]
        rsym = {l: rows[l]["treatment"] / (b0 * rows[l]["baseline"]) ** 0.5 for l in LAM}
        drops = {f"{LAM[i]}->{LAM[i + 1]}": rsym[LAM[i]] - rsym[LAM[i + 1]] for i in range(5)}
        largest = max(drops, key=drops.get)
        out[cell] = {"R_sym": rsym, "drops": drops, "largest_drop": largest,
                     "largest_at_switch": largest == SWITCH[cell.split("/")[0]]}
        print(f"{cell:16s} " + " ".join(f"{d:.3f}" for d in drops.values()) + f" | largest {largest}")
    out["largest_at_switch_cells"] = sum(c["largest_at_switch"] for c in out.values() if isinstance(c, dict))
    print("largest drop at the switch in", out["largest_at_switch_cells"], "of 9 cells")
    (X / "symmetric_R.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
