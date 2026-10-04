#!/usr/bin/env python3
"""Turn the consensus region up, and graft it onto a student that never had it.

Removing the region collapses the reasoning style, but removal alone only shows the region is
necessary -- an engine stops when you pull a part out, and that does not make the part the throttle.
Two edits decide whether it behaves like a control:

  amplify   dW' = dW + alpha * V V^T dW on the lambda0 adapter. This first form grows the update's
            norm along with the region, and at alpha >= 1 that alone pushes generation to the length
            limit -- every sample truncates and the density measurement stops meaning anything. Kept
            for the record; the interpretable form is the next one.
  ampn      the same reweighting renormalised back to the original ||dW||. The update keeps its size
            and only its distribution moves toward the region, which separates "more of it sits in
            the region" from "the update is simply bigger".
  ampV      the limit of that: dW projected entirely onto the region and rescaled to the original
            norm. Same magnitude, everything inside the region.
  graft     dW' = dW(lambda1) + beta * V V^T dW(lambda0). The lambda1 student was distilled from a
            teacher that had already dropped the format, so it never wrote this region. Adding the
            lambda0 adapter's component inside the region -- and nothing else of it -- asks whether
            the style transfers with the subspace alone.

Amplification keeps rank 16 (only lora_B changes). Grafting cannot: the two adapters have different
lora_A, so their sum is genuinely rank 32, and the output is written as a rank-32 adapter with the
two A matrices stacked. vLLM serves it with --max_lora_rank 32.

Controls for the graft, both matched on the energy added:
  graft_rand  the lambda0 component inside a random subspace of the same dimension instead of V.
              This turned out to decide nothing: a random subspace of the output space barely
              intersects a rank-16 update, so it injects 0.3-0.7% of the energy the treatment does.
              Kept only to document that.
  graft_orth  the control that does decide it: the part of the lambda0 update *orthogonal* to the
              region, rescaled to inject the same energy. Same donor adapter, same amount added,
              only the directions differ.
  graft_full  the whole lambda0 update scaled to add the same energy -- asks whether the region is
              enough on its own or only works as part of the whole adapter
"""

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from config import EXP, LAMBDAS, REPO_ROOT, SEEDS

KEY_RE = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)")


def read_adapter(path):
    parts, raw = defaultdict(dict), {}
    with safe_open(path, "pt") as handle:
        for name in handle.keys():
            raw[name] = handle.get_tensor(name)
            m = KEY_RE.search(name)
            if m:
                parts[(int(m.group(1)), m.group(2))]["B" if "lora_B" in name else "A"] = name
    return raw, parts


def left_basis(B, A, scale):
    Qb, Rb = torch.linalg.qr(B.float() * scale)
    Qa, Ra = torch.linalg.qr(A.float().T)
    U, _, _ = torch.linalg.svd(Rb @ Ra.T)
    return Qb @ U


def consensus(bases, threshold=0.5):
    M = torch.cat(bases, dim=1)
    Q, S, _ = torch.linalg.svd(M, full_matrices=False)
    k = max(1, int((((S ** 2) / len(bases)) > threshold).sum()))
    return Q[:, :k], k


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cliff", required=True, choices=LAMBDAS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loo", action="store_true",
                    help="build the region without the lambda0 adapter that donates or gets amplified")
    ap.add_argument("--alphas", nargs="+", type=float, default=[1.0, 2.0])
    ap.add_argument("--norm-alphas", nargs="+", type=float, default=[1.0, 4.0],
                    help="norm-preserving amplification factors")
    ap.add_argument("--variants", nargs="+", default=["amp", "ampn", "ampV", "graft"])
    ap.add_argument("--betas", nargs="+", type=float, default=[1.0])
    ap.add_argument("--out", required=True, help="output directory inside the cell, e.g. runs_edit_s0")
    a = ap.parse_args()

    runs = EXP / a.tag / "runs"
    out_root = EXP / a.tag / a.out
    owner = REPO_ROOT.stat()
    generator = torch.Generator().manual_seed(4321)

    index = LAMBDAS.index(a.cliff)
    pre = [(l, s) for l in LAMBDAS[:index] for s in SEEDS]
    if a.loo:
        pre = [p for p in pre if p != ("0.0", a.seed)]

    bases = defaultdict(list)
    for lam, seed in pre:
        p = runs / f"lambda{lam}_seed{seed}" / "adapter" / "adapter_model.safetensors"
        if not p.exists():
            continue
        cfg = json.loads((p.parent / "adapter_config.json").read_text())
        scale = cfg["lora_alpha"] / cfg["r"]
        raw, parts = read_adapter(p)
        for key, names in parts.items():
            if len(names) == 2:
                bases[key].append(left_basis(raw[names["B"]], raw[names["A"]], scale))
    V = {key: consensus(vs) for key, vs in bases.items()}
    print(f"[{a.tag}] 절벽 λ{a.cliff} | loo={a.loo} adapter {len(pre)}개 | 합의 차원 평균 {sum(k for _, k in V.values())/len(V):.1f}", flush=True)

    src0 = runs / f"lambda0.0_seed{a.seed}" / "adapter"
    src1 = runs / f"lambda1.0_seed{a.seed}" / "adapter"
    cfg0 = json.loads((src0 / "adapter_config.json").read_text())
    raw0, parts0 = read_adapter(src0 / "adapter_model.safetensors")
    raw1, parts1 = read_adapter(src1 / "adapter_model.safetensors")

    def write(name, tensors, rank, note):
        dest = out_root / name / "adapter"
        dest.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(dest / "adapter_model.safetensors"))
        cfg = dict(cfg0)
        cfg["r"], cfg["lora_alpha"] = rank, cfg0["lora_alpha"] * rank // cfg0["r"]
        (dest / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
        for path in (out_root / name).rglob("*"):
            try:
                os.chown(path, owner.st_uid, owner.st_gid)
                path.chmod(0o755 if path.is_dir() else 0o644)
            except OSError:
                pass
        print(f"  {name}: rank {rank}, {note}", flush=True)

    def dw_norm(B, A, scale):
        return float(((B.float() * scale) @ A.float()).pow(2).sum()) ** 0.5

    # norm-preserving amplification, and its limit
    if "ampn" in a.variants or "ampV" in a.variants:
        jobs = [(f"ampn{x:g}", x) for x in a.norm_alphas if "ampn" in a.variants]
        if "ampV" in a.variants:
            jobs.append(("ampV", None))
        for name, alpha in jobs:
            tensors = {n: t_.clone() for n, t_ in raw0.items()}
            for key, names in parts0.items():
                if len(names) != 2 or key not in V:
                    continue
                B, A = raw0[names["B"]].float(), raw0[names["A"]].float()
                Vk = V[key][0]
                comp = Vk @ (Vk.T @ B)
                new = comp if alpha is None else B + alpha * comp
                before = dw_norm(B, A, 1.0)
                after = dw_norm(new, A, 1.0)
                if after > 1e-12:
                    new = new * (before / after)
                tensors[names["B"]] = new.to(raw0[names["B"]].dtype)
            note = "λ0 전부 V로 투영 후 원래 노름" if alpha is None else f"λ0 + {alpha:g}·V 성분, 노름 보존"
            write(name, tensors, cfg0["r"], note)

    # amplify (norm grows): only lora_B moves, rank unchanged
    for alpha in (a.alphas if "amp" in a.variants else []):
        tensors = {n: t.clone() for n, t in raw0.items()}
        for key, names in parts0.items():
            if len(names) != 2 or key not in V:
                continue
            B = raw0[names["B"]].float()
            Vk = V[key][0]
            tensors[names["B"]] = (B + alpha * (Vk @ (Vk.T @ B))).to(raw0[names["B"]].dtype)
        write(f"amp{alpha:g}", tensors, cfg0["r"], f"λ0 + {alpha:g}·V 성분")

    # graft: lambda1 adapter plus a piece of the lambda0 adapter, stacked to rank 2r
    for beta in (a.betas if "graft" in a.variants else []):
        for variant in ("graft", "graft_rand", "graft_full", "graft_orth"):
            tensors = {}
            for key, names in parts1.items():
                if len(names) != 2 or key not in V or key not in parts0:
                    continue
                n0 = parts0[key]
                B0, A0 = raw0[n0["B"]].float(), raw0[n0["A"]].float()
                B1, A1 = raw1[names["B"]].float(), raw1[names["A"]].float()
                Vk, k = V[key]
                if variant == "graft":
                    add = Vk @ (Vk.T @ B0)
                elif variant == "graft_rand":
                    G = torch.randn(B0.shape[0], k, generator=generator)
                    R, _ = torch.linalg.qr(G)
                    add = R @ (R.T @ B0)
                elif variant == "graft_orth":
                    comp = Vk @ (Vk.T @ B0)
                    rest = B0 - comp
                    # match the energy the region's component would have injected
                    target = float((comp @ A0).pow(2).sum())
                    have = float((rest @ A0).pow(2).sum())
                    add = rest * (target / max(have, 1e-12)) ** 0.5
                else:                                   # graft_full, scaled to the same energy
                    target = float((Vk.T @ B0).pow(2).sum())
                    add = B0 * (target / max(float(B0.pow(2).sum()), 1e-12)) ** 0.5
                tensors[names["B"]] = torch.cat([B1, beta * add], dim=1).to(raw1[names["B"]].dtype)
                tensors[names["A"]] = torch.cat([A1, A0], dim=0).to(raw1[names["A"]].dtype)
            write(f"{variant}{beta:g}", tensors, cfg0["r"] * 2, f"λ1 + {beta:g}·λ0의 {variant} 성분")


if __name__ == "__main__":
    main()
