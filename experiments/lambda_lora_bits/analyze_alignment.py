#!/usr/bin/env python3
"""Does the adapter's drop profile line up with the teacher's, interval by interval?

Reading off the single lambda of largest drop on each side makes the verdict hinge on how the teacher's
transition is defined: "first lambda at 0%" and "largest drop" disagree for pairs B and D, and choosing
between them after seeing the adapter result would be post hoc. Correlating the whole profile of
per-interval drops avoids the choice, uses every interval, and is fixed before any cell is inspected.

  teacher profile   drop in the </think> rate across each of the five lambda intervals
  adapter profile   drop in retention R(lambda) = T(lambda)/B(lambda) across the same intervals

Significance is exact rather than asymptotic: with five intervals there are only 120 orderings, so the
teacher profile is permuted over all of them and p is the fraction reaching the observed correlation.
Cells are not independent -- three students share each teacher pair -- so the cell-level p values are
combined only within a pair, and the three pair-level results are reported separately rather than pooled
into one number that would pretend there are nine independent tests.

--teacher markers swaps the teacher profile for the drop in the teacher's reflective-marker density
(experiments/teacher_density.json) instead of the </think> rate, to test whether the adapter follows
the reasoning itself rather than the tag; the permutation test is unchanged.
"""

import argparse
import json
import statistics as st
from itertools import permutations
from pathlib import Path

from config import LAMBDAS, RESULTS

INTERVALS = [f"{LAMBDAS[i - 1]}→{LAMBDAS[i]}" for i in range(1, len(LAMBDAS))]


def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def permutation_p(teacher, adapter):
    """Exact one-sided p over all orderings of the teacher profile."""
    observed = pearson(teacher, adapter)
    perms = list(permutations(teacher))
    hits = sum(1 for p in perms if pearson(list(p), adapter) >= observed - 1e-12)
    return observed, hits / len(perms)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--think", default=str(RESULTS / "think_rate.json"))
    ap.add_argument("--resample", default=str(RESULTS / "phi_resample_s012.json"))
    ap.add_argument("--output", default=None,
                    help="default: alignment_s012.json (think) or alignment_markers_s012.json (markers)")
    ap.add_argument("--teacher", choices=["think", "markers"], default="think")
    ap.add_argument("--density", default=str(RESULTS / "teacher_density.json"))
    a = ap.parse_args()
    if a.output is None:
        a.output = str(RESULTS / ("alignment_markers_s012.json" if a.teacher == "markers" else "alignment_s012.json"))

    think = json.loads(Path(a.think).read_text())
    if a.teacher == "markers":
        # teacher_density.json names the paper's pair C "C"; the adapter results keep the directory's "D"
        dens = json.loads(Path(a.density).read_text())
        think = {p: {"rates": dens[q]["teacher"]} for p, q in (("A", "A"), ("B", "B"), ("D", "C"))}
    res = json.loads(Path(a.resample).read_text())

    print("intervals: " + "  ".join(f"{s:>11s}" for s in INTERVALS))
    print(f"\n{'cell':16s} {'adapter drop profile':>58s} {'r':>7s} {'p':>7s}")
    by_pair, out = {}, {}
    for key, cell in sorted(res.items()):
        pair = key.split("/")[0]
        rates = think[pair]["rates"]
        teacher = [rates[LAMBDAS[i - 1]] - rates[LAMBDAS[i]] for i in range(1, len(LAMBDAS))]
        adapter = [cell["rows"][LAMBDAS[i - 1]]["retention"] - cell["rows"][LAMBDAS[i]]["retention"]
                   for i in range(1, len(LAMBDAS))]
        r, p = permutation_p(teacher, adapter)
        label = key.replace("2.5-", "").replace("-3.2", "")
        print(f"{label:16s} {'  '.join(f'{x:+10.2f}' for x in adapter):>58s} {r:7.3f} {p:7.3f}")
        by_pair.setdefault(pair, []).append((key, r, p))
        out[key] = dict(teacher_profile=teacher, adapter_profile=adapter, r=r, p=p)

    print(f"\n{'pair':4s} {'teacher drop profile':>58s} {'mean r':>8s} {'cells':>5s}")
    for pair, cells in sorted(by_pair.items()):
        rates = think[pair]["rates"]
        teacher = [rates[LAMBDAS[i - 1]] - rates[LAMBDAS[i]] for i in range(1, len(LAMBDAS))]
        print(f"{pair:4s} {'  '.join(f'{x:+10.2f}' for x in teacher):>58s} "
              f"{st.mean(r for _, r, _ in cells):8.3f} {len(cells):4d}")
    every = [r for cells in by_pair.values() for _, r, _ in cells]
    if every:
        print(f"\n{len(every)} cells: mean r {st.mean(every):.3f} (min {min(every):.3f}, max {max(every):.3f})"
              f" | cells with p <= 0.05: {sum(1 for c in by_pair.values() for _, _, p in c if p <= 0.05)}/{len(every)}")
    # Pair level: the three students share one teacher profile, so the statistic is their mean r and
    # the same teacher ordering is applied to all three. Pooled: the three pairs have separate teachers,
    # so their orderings are permuted independently -- 120^3 joint orderings, enumerated exactly.
    perms = [list(q) for q in permutations(range(len(INTERVALS)))]
    null, pooled_obs, summary = [], 0.0, {}
    for pair, cells in sorted(by_pair.items()):
        rates = think[pair]["rates"]
        teacher = [rates[LAMBDAS[i - 1]] - rates[LAMBDAS[i]] for i in range(1, len(LAMBDAS))]
        profiles = [out[key]["adapter_profile"] for key, _, _ in cells]
        dist = [st.mean(pearson([teacher[j] for j in q], ad) for ad in profiles) for q in perms]
        obs = st.mean(r for _, r, _ in cells)
        p_pair = sum(1 for x in dist if x >= obs - 1e-12) / len(dist)
        summary[pair] = dict(mean_r=obs, p=p_pair, cells=len(cells))
        null.append(dist); pooled_obs += obs
        print(f"pair {pair}: mean r over students {obs:.3f}, exact p {p_pair:.4f} ({len(dist)} permutations)")
    if len(null) == 3:
        import numpy as np
        A, B, C = (np.array(d) for d in null)
        total = A[:, None, None] + B[None, :, None] + C[None, None, :]
        p_pool = float((total >= pooled_obs - 1e-9).mean())
        summary["pooled"] = dict(mean_r=pooled_obs / 3, p=p_pool, orderings=int(total.size))
        print(f"three pairs pooled: mean r {pooled_obs / 3:.3f}, exact p {p_pool:.2e} ({total.size:,} joint orderings)")
    out["_summary"] = summary
    Path(a.output).write_text(json.dumps(out, indent=2))
    print(a.output)


if __name__ == "__main__":
    main()
