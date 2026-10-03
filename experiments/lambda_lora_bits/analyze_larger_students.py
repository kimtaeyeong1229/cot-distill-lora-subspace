#!/usr/bin/env python3
"""Do larger students switch where their teacher stops thinking? A position-only check on seed 0.

The main analysis stops at 1B students. Qwen2.5-1.5B and -3B were trained across the same lambda grid,
but from seed 0 only and without data halves, so the ratio R cannot be formed. What can be read is the
position of the largest drop in phi(lambda0 adapter, lambda adapter) between adjacent lambdas: both
adapters share one LoRA initialisation, which lifts the whole curve but is constant along lambda, so it
cannot move the drop (the same reading as analyze_control_pairc.py). The 0.0 -> 0.2 step is excluded
because phi(lambda0, lambda0) = 1 is an adapter compared with itself. The 0.5B student of the same
teacher, read the same way, is printed alongside as the reference.

Writes experiments/larger_students.json.
"""

import json
from pathlib import Path

from analyze_control_pairc import load_basis, overlap

EXP = Path(__file__).resolve().parent
LAMBDAS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
# paper pair, directory of the larger student, directory of the 0.5B reference, the teacher's transition
CELLS = [
    ("A", "sweep_qwen1.5b", "sweep", "0.2→0.4"), ("A", "sweep_qwen3b", "sweep", "0.2→0.4"),
    ("B", "pairB/qwen1.5b", "pairB/qwen0.5b", "0.4→0.6"),
    ("C", "pairD/qwen1.5b", "pairD/qwen0.5b", "0.4→0.6, 0.6→0.8"),
    ("pairC control", "pairC/qwen1.5b", "pairC/qwen0.5b", "0.2→0.4 (markers)"),
    ("pairC control", "pairC/qwen3b", "pairC/qwen0.5b", "0.2→0.4 (markers)"),
]


def profile(tag):
    basis = {lam: load_basis(EXP / tag / "runs" / f"lambda{lam}_seed0" / "adapter" / "adapter_model.safetensors")
             for lam in LAMBDAS}
    phi = {lam: overlap(basis["0.0"], basis[lam]) for lam in LAMBDAS}
    drops = {f"{LAMBDAS[i - 1]}→{LAMBDAS[i]}": phi[LAMBDAS[i - 1]] / phi[LAMBDAS[i]] for i in range(2, len(LAMBDAS))}
    return phi, drops, max(drops, key=drops.get)


def main():
    out = {}
    for pair, tag, ref, teacher in CELLS:
        phi, drops, cliff = profile(tag)
        _, rdrops, rcliff = profile(ref)
        print(f"{pair:14s} {tag:18s} " + " ".join(f"{phi[l]:.3f}" for l in LAMBDAS)
              + f" | 절벽 {cliff} ({drops[cliff]:.1f}배) | 0.5B {rcliff} ({rdrops[rcliff]:.1f}배) | teacher {teacher}")
        out[tag] = dict(pair=pair, phi=phi, drops=drops, cliff=cliff, reference=ref, reference_cliff=rcliff,
                        teacher_transition=teacher)
    path = EXP.parent / "larger_students.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(path)


if __name__ == "__main__":
    main()
