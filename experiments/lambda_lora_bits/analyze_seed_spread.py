#!/usr/bin/env python3
"""Mean +- standard deviation over seeds 0-2 for every student table (Tables 1, 2 and 6-9).

Each seed is analysed on its own -- its own lambda0 / lambda1 bases and its own edited adapters, both
under the shared student setting -- by running analyze_main_density.py and analyze_main_edits.py with
--seeds <s>. The tables then report the mean of the three per-seed values with their sample standard
deviation, so the spread shown is the seed-to-seed variation of the quantity itself.

Run from the repo root; writes experiments/seed_spread.json.
"""

import json
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path

E = Path("experiments/lambda_lora_bits")
SEEDS = [0, 1, 2]
LAMBDAS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
EDITS = {"rm_pre": "lambda0.0_rm_pre", "rm_other": "lambda0.0_rm_decisive", "rm_scale": "lambda0.0_rm_scale",
         "rm_pre_l1": "lambda1.0_rm_pre", "graft1": "graft1", "graft_other1": "graft_orth1",
         "graft_whole1": "graft_full1", "graft2": "graft2", "graft_other2": "graft_orth2",
         "graft_whole2": "graft_full2", "ampn1": "ampn1", "ampn4": "ampn4", "ampV": "ampV"}


def per_seed(tmp):
    runs = {}
    for s in SEEDS:
        md, me = tmp / f"md{s}.json", tmp / f"me{s}.json"
        for script, extra in (("analyze_main_density.py", ["--output", md]),
                              ("analyze_main_edits.py", ["--base", md, "--output", me])):
            subprocess.run([sys.executable, E / script, "--seeds", str(s), *map(str, extra)],
                           check=True, stdout=subprocess.DEVNULL)
        runs[s] = (json.loads(md.read_text()), json.loads(me.read_text()))
    return runs


def summarise(values):
    return {"mean": st.mean(values), "sd": st.stdev(values), "seeds": values}


def main():
    with tempfile.TemporaryDirectory() as tmp:
        runs = per_seed(Path(tmp))
    out = {}
    for cell in runs[0][0]:
        dens = [runs[s][0][cell] for s in SEEDS]
        edit = [runs[s][1][cell] for s in SEEDS]
        row = {"table1": {lam: summarise([d[lam]["markers_per_1k"] for d in dens]) for lam in LAMBDAS}}
        row["table2"] = {
            "tokens0": summarise([d["0.0"]["avg_tokens"] for d in dens]),
            "tokens1": summarise([d["1.0"]["avg_tokens"] for d in dens]),
            "trunc0": summarise([d["0.0"]["truncated"] for d in dens]),
            "trunc1": summarise([d["1.0"]["truncated"] for d in dens]),
            "d_acc": summarise([100 * (d["1.0"]["accuracy"] - d["0.0"]["accuracy"]) for d in dens]),
            "d_acc_completed": summarise([100 * (d["1.0"]["accuracy_completed"] - d["0.0"]["accuracy_completed"])
                                          for d in dens])}
        base = {"base0": [e["base0"] for e in edit], "base1": [e["base1"] for e in edit]}
        row["base"] = {f"{b}_{k}": summarise([x[k] for x in v]) for b, v in base.items()
                       for k in ("markers_per_1k", "head_markers_per_1k", "avg_tokens", "accuracy")}
        row["edits"] = {name: {k: summarise([e[v][k] for e in edit])
                               for k in ("markers_per_1k", "head_markers_per_1k", "avg_tokens", "completed",
                                         "accuracy")}
                        for name, v in EDITS.items()}
        out[cell] = row
        print(f"{cell:8} lambda0 {row['table1']['0.0']['mean']:6.2f} +- {row['table1']['0.0']['sd']:5.2f}"
              f" | shared removed {row['edits']['rm_pre']['markers_per_1k']['mean']:5.2f}"
              f" +- {row['edits']['rm_pre']['markers_per_1k']['sd']:4.2f}")
    Path("experiments/seed_spread.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
