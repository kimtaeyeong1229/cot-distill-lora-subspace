#!/usr/bin/env python3
import argparse
import json
import math
import statistics as st
from itertools import permutations
from pathlib import Path
from config import EXP, LAMBDAS, PAIR_DIR, PAIR_KEY, RESULTS

INTERVALS = [f"{LAMBDAS[i - 1]}→{LAMBDAS[i]}" for i in range(1, len(LAMBDAS))]


def pearson(a, b):
    ma, mb = (st.mean(a), st.mean(b))
    num = sum(((x - ma) * (y - mb) for x, y in zip(a, b)))
    den = (sum(((x - ma) ** 2 for x in a)) * sum(((y - mb) ** 2 for y in b))) ** 0.5
    return num / den if den else float("nan")


def permutation_p(teacher, adapter):
    observed = pearson(teacher, adapter)
    perms = list(permutations(teacher))
    hits = sum((1 for p in perms if pearson(list(p), adapter) >= observed - 1e-12))
    return (observed, hits / len(perms))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--think", default=str(RESULTS / "think_rate.json"))
    ap.add_argument("--resample", default=str(RESULTS / "phi_resample_s012.json"))
    ap.add_argument(
        "--output",
        default=None,
        help="default: alignment_s012.json (think), alignment_markers_s012.json (markers) or alignment_length_s012.json (length)",
    )
    ap.add_argument("--teacher", choices=["think", "markers", "length"], default="think")
    ap.add_argument("--density", default=str(RESULTS / "teacher_density.json"))
    a = ap.parse_args()
    if a.output is None:
        a.output = str(
            RESULTS
            / {
                "think": "alignment_s012.json",
                "markers": "alignment_markers_s012.json",
                "length": "alignment_length_s012.json",
            }[a.teacher]
        )
    think = json.loads(Path(a.think).read_text())
    if a.teacher == "markers":
        dens = json.loads(Path(a.density).read_text())
        think = {p: {"rates": dens[q]["teacher"]} for p, q in (("A", "A"), ("B", "B"), ("C", "C"))}
    if a.teacher == "length":
        think = {}
        for p, d in ((PAIR_KEY[q], PAIR_DIR[q]) for q in PAIR_DIR):
            stats = json.loads((EXP / d / "splits" / "split_stats.json").read_text())
            think[p] = {
                "rates": {
                    l: math.log(stats["per_lambda"][l]["train_tokens"] / stats["n_train"]) for l in LAMBDAS
                }
            }
    res = json.loads(Path(a.resample).read_text())
    print("intervals: " + "  ".join((f"{s:>11s}" for s in INTERVALS)))
    print(f"\n{'cell':16s} {'adapter drop profile':>58s} {'r':>7s} {'p':>7s}")
    by_pair, out = ({}, {})
    for key, cell in sorted(res.items()):
        pair = key.split("/")[0]
        rates = think[pair]["rates"]
        teacher = [rates[LAMBDAS[i - 1]] - rates[LAMBDAS[i]] for i in range(1, len(LAMBDAS))]
        adapter = [
            cell["rows"][LAMBDAS[i - 1]]["retention"] - cell["rows"][LAMBDAS[i]]["retention"]
            for i in range(1, len(LAMBDAS))
        ]
        r, p = permutation_p(teacher, adapter)
        label = key.replace("2.5-", "").replace("-3.2", "")
        print(f"{label:16s} {'  '.join((f'{x:+10.2f}' for x in adapter)):>58s} {r:7.3f} {p:7.3f}")
        by_pair.setdefault(pair, []).append((key, r, p))
        out[key] = dict(teacher_profile=teacher, adapter_profile=adapter, r=r, p=p)
    print(f"\n{'pair':4s} {'teacher drop profile':>58s} {'mean r':>8s} {'cells':>5s}")
    for pair, cells in sorted(by_pair.items()):
        rates = think[pair]["rates"]
        teacher = [rates[LAMBDAS[i - 1]] - rates[LAMBDAS[i]] for i in range(1, len(LAMBDAS))]
        print(
            f"{pair:4s} {'  '.join((f'{x:+10.2f}' for x in teacher)):>58s} {st.mean((r for _, r, _ in cells)):8.3f} {len(cells):4d}"
        )
    every = [r for cells in by_pair.values() for _, r, _ in cells]
    if every:
        print(
            f"\n{len(every)} cells: mean r {st.mean(every):.3f} (min {min(every):.3f}, max {max(every):.3f}) | cells with p <= 0.05: {sum((1 for c in by_pair.values() for _, _, p in c if p <= 0.05))}/{len(every)}"
        )
    perms = [list(q) for q in permutations(range(len(INTERVALS)))]
    null, pooled_obs, summary = ([], 0.0, {})
    for pair, cells in sorted(by_pair.items()):
        rates = think[pair]["rates"]
        teacher = [rates[LAMBDAS[i - 1]] - rates[LAMBDAS[i]] for i in range(1, len(LAMBDAS))]
        profiles = [out[key]["adapter_profile"] for key, _, _ in cells]
        dist = [st.mean((pearson([teacher[j] for j in q], ad) for ad in profiles)) for q in perms]
        obs = st.mean((r for _, r, _ in cells))
        p_pair = sum((1 for x in dist if x >= obs - 1e-12)) / len(dist)
        summary[pair] = dict(mean_r=obs, p=p_pair, cells=len(cells))
        null.append(dist)
        pooled_obs += obs
        print(f"pair {pair}: mean r over students {obs:.3f}, exact p {p_pair:.4f} ({len(dist)} permutations)")
    if len(null) == 3:
        import numpy as np

        A, B, C = (np.array(d) for d in null)
        total = A[:, None, None] + B[None, :, None] + C[None, None, :]
        p_pool = float((total >= pooled_obs - 1e-09).mean())
        summary["pooled"] = dict(mean_r=pooled_obs / 3, p=p_pool, orderings=int(total.size))
        print(
            f"three pairs pooled: mean r {pooled_obs / 3:.3f}, exact p {p_pool:.2e} ({total.size:,} joint orderings)"
        )
    if a.teacher == "length":
        tr = json.loads((RESULTS / "think_rate.json").read_text())
        for pair in by_pair:
            ln = [
                think[pair]["rates"][LAMBDAS[i - 1]] - think[pair]["rates"][LAMBDAS[i]]
                for i in range(1, len(LAMBDAS))
            ]
            th = [
                tr[pair]["rates"][LAMBDAS[i - 1]] - tr[pair]["rates"][LAMBDAS[i]]
                for i in range(1, len(LAMBDAS))
            ]
            summary[pair]["r_length_vs_think_drop"] = pearson(ln, th)
    out["_summary"] = summary
    Path(a.output).write_text(json.dumps(out, indent=2))
    print(a.output)


if __name__ == "__main__":
    main()
