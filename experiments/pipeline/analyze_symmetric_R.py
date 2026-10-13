#!/usr/bin/env python3
import json
from config import LAMBDAS as LAM, RESULTS as X

SWITCH = {"A": "0.2->0.4", "B": "0.4->0.6", "C": "0.4->0.6"}


def main():
    res = json.loads((X / "phi_resample_s012.json").read_text())
    out = {}
    for cell, v in res.items():
        rows = v["rows"]
        b0 = rows["0.0"]["baseline"]
        rsym = {l: rows[l]["treatment"] / (b0 * rows[l]["baseline"]) ** 0.5 for l in LAM}
        drops = {f"{LAM[i]}->{LAM[i + 1]}": rsym[LAM[i]] - rsym[LAM[i + 1]] for i in range(5)}
        largest = max(drops, key=drops.get)
        out[cell] = {
            "R_sym": rsym,
            "drops": drops,
            "largest_drop": largest,
            "largest_at_switch": largest == SWITCH[cell.split("/")[0]],
        }
        print(f"{cell:16s} " + " ".join((f"{d:.3f}" for d in drops.values())) + f" | largest {largest}")
    out["largest_at_switch_cells"] = sum(
        (c["largest_at_switch"] for c in out.values() if isinstance(c, dict))
    )
    print("largest drop at the switch in", out["largest_at_switch_cells"], "of 9 cells")
    (X / "symmetric_R.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
