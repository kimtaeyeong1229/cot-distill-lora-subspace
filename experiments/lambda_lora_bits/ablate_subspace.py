#!/usr/bin/env python3
"""Write adapters with a chosen subspace projected out of their update.

The claim so far is correlational: pre-threshold adapters share a 5-8 dimensional region and
post-threshold ones stop using it. Removing that region and watching what breaks is what turns it
into a causal statement -- but only if the controls rule out the obvious alternative, that damaging
any k directions of a rank-16 update would break things just as well.

The update is dW = (alpha/r) B A, so projecting a subspace V out of the output space is
dW' = (I - V V^T) dW = (alpha/r) [(I - V V^T) B] A: only lora_B changes, A is untouched and the
adapter stays a valid rank-r LoRA that vLLM can serve.

Variants written per source adapter:
  rm_pre    the consensus region of the pre-threshold adapters -- the treatment
  rm_rand   a random subspace of the same dimension, drawn per layer -- the control that decides
            whether the result is about this region or about losing k directions at all
  rm_post   the consensus region of the post-threshold adapters -- should matter much less if the
            pre-threshold region is what carries the reasoning format
  rm_least  the k directions of this adapter's own top-16 that are least aligned with the pre-threshold
            region -- same adapter, same k, deliberately the "wrong" k
  rm_energy the same idea taken until the *energy* removed matches rm_pre. rm_pre takes 30-40% of the
            update norm while a random subspace takes 0.1%, because the consensus region is where the
            energy sits, so an unmatched control would compare a large perturbation with a negligible
            one. This removes the same amount from the same adapter, choosing the directions least
            aligned with the region, and is the control the claim actually has to beat.
  rm_scale  the whole update shrunk uniformly to rm_pre's norm -- separates "how much update" from
            "which directions"
  rm_decisive the control the claim has to survive: a subspace of this adapter's own span that is
            *orthogonal* to the consensus region and removes the *same energy*. rm_energy had to take
            14 of 16 directions to match the energy, which leaves the adapter with almost nothing and
            therefore decides nothing; this one takes the highest-energy directions of the region's
            orthogonal complement, so it matches the energy with a comparable number of directions and
            differs from the treatment only in which subspace it is.
  gs_tau    Gold-Switch (Lee et al., 2025, arXiv 2510.06750) applied to this adapter: remove the top-r
            singular components of dW, with r per module the largest rank whose reversed cumulative
            energy e(r) = sum_{i>=r} s_i^2 / sum_i s_i^2 stays >= tau (0.6, the paper's value). The
            external training-free baseline: "the top of the update's own spectrum" instead of the region.
  gs_energy the same top singular components, taken until the dW energy removed matches rm_pre's --
            decides whether the consensus region is anything more than the top of the spectrum.

Applying all of them to the lambda1 adapter as well gives the third control: lambda1 barely uses the
pre-threshold region to begin with, so removing it there should do little.
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

EXP = Path(__file__).resolve().parent
LAMBDAS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
SEEDS = [0, 1, 2]
KEY_RE = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)")


def module_key(name):
    m = KEY_RE.search(name)
    return (int(m.group(1)), m.group(2)) if m else None


def read_adapter(path):
    """lora_A / lora_B tensor names and values, keyed by layer and module."""
    parts = defaultdict(dict)
    raw = {}
    with safe_open(path, "pt") as handle:
        for name in handle.keys():
            raw[name] = handle.get_tensor(name)
            key = module_key(name)
            if key:
                parts[key]["B" if "lora_B" in name else "A"] = name
    return raw, parts


def left_basis(B, A, scale):
    """Top-r left singular vectors of dW = scale * B A, without forming dW."""
    Qb, Rb = torch.linalg.qr(B.float() * scale)
    Qa, Ra = torch.linalg.qr(A.float().T)
    U, _, _ = torch.linalg.svd(Rb @ Ra.T)
    return Qb @ U


def spectrum(B, A, scale):
    """Left singular vectors (descending) and singular values of dW = scale * B A."""
    Qb, Rb = torch.linalg.qr(B.float() * scale)
    Qa, Ra = torch.linalg.qr(A.float().T)
    U, S, _ = torch.linalg.svd(Rb @ Ra.T)
    return Qb @ U, S


def consensus(bases, threshold=0.5, dims=None):
    """Eigenvectors of mean(U U^T), via the thin stacked SVD.

    `dims` overrides the eigenvalue cut and takes that many leading directions instead, which is how
    the post-threshold control is made dimension-matched to the treatment: removing 3.8 directions
    where the treatment removes 6.6 would confound "which subspace" with "how many directions".
    """
    M = torch.cat(bases, dim=1)
    Q, S, _ = torch.linalg.svd(M, full_matrices=False)
    eigenvalues = (S ** 2) / len(bases)
    k = dims if dims is not None else max(1, int((eigenvalues > threshold).sum()))
    k = min(k, Q.shape[1])
    return Q[:, :k], k


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="cell directory, e.g. sweep_llama or pairB/llama1b")
    ap.add_argument("--cliff", required=True, choices=LAMBDAS, help="cliff lambda from analyze_resample.py")
    ap.add_argument("--sources", nargs="+", default=["0.0", "1.0"], help="lambdas to ablate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loo", action="store_true",
                    help="build the regions without the adapters being edited, so no adapter is edited "
                         "along a region it helped define")
    ap.add_argument("--out", default="runs_ablation")
    ap.add_argument("--variants", nargs="+",
                    default=["rm_pre", "rm_rand", "rm_post", "rm_least", "rm_energy", "rm_scale",
                             "rm_decisive"])
    ap.add_argument("--tau", type=float, default=0.6, help="Gold-Switch energy threshold")
    ap.add_argument("--region", choices=["pre", "seed"], default="pre",
                    help="seed: build the region only from the other seeds' lambda0 adapters -- what a "
                         "practitioner who trained the same adapter a few times would have. Variants are "
                         "written as rmseed / rmseed_other instead of rm_pre / rm_decisive")
    ap.add_argument("--threshold", type=float, default=None,
                    help="eigenvalue cut for the region (default 0.5; 0.75 for --region seed, where two "
                         "members put almost every direction above 0.5: their paired directions have "
                         "eigenvalue (1 + cos theta) / 2, so 0.75 keeps those within 60 degrees)")
    a = ap.parse_args()

    runs = EXP / a.tag / "runs"
    out_root = EXP / a.tag / a.out
    stat = EXP.parent.parent.stat()
    owner_uid, owner_gid = stat.st_uid, stat.st_gid
    generator = torch.Generator().manual_seed(1234)

    index = LAMBDAS.index(a.cliff)
    pre = [(l, s) for l in LAMBDAS[:index] for s in SEEDS]
    if a.region == "seed":
        pre = [("0.0", s) for s in SEEDS if s != a.seed]
        a.sources = ["0.0"]
    threshold = a.threshold if a.threshold is not None else (0.75 if a.region == "seed" else 0.5)
    post = [(l, s) for l in LAMBDAS[index:] for s in SEEDS]
    if a.loo:
        edited = {(lam, a.seed) for lam in a.sources}
        pre = [p for p in pre if p not in edited]
        post = [p for p in post if p not in edited]

    def basis_of(lam, seed):
        path = runs / f"lambda{lam}_seed{seed}" / "adapter" / "adapter_model.safetensors"
        if not path.exists():
            return None
        config = json.loads((path.parent / "adapter_config.json").read_text())
        scale = config["lora_alpha"] / config["r"]
        raw, parts = read_adapter(path)
        return {key: left_basis(raw[names["B"]], raw[names["A"]], scale)
                for key, names in parts.items() if len(names) == 2}

    print(f"[{a.tag}] 절벽 λ{a.cliff} | 문턱 전 {len(pre)}개, 후 {len(post)}개 adapter 로 합의 구역 계산", flush=True)
    pre_bases, post_bases = defaultdict(list), defaultdict(list)
    for lam, seed in pre:
        b = basis_of(lam, seed)
        if b:
            for key, U in b.items():
                pre_bases[key].append(U)
    for lam, seed in post:
        b = basis_of(lam, seed)
        if b:
            for key, U in b.items():
                post_bases[key].append(U)

    V_pre, V_post, k_pre, k_post = {}, {}, {}, {}
    for key in pre_bases:
        V_pre[key], k_pre[key] = consensus(pre_bases[key], threshold=threshold)
        _, k_post[key] = consensus(post_bases[key])
        # dimension-matched to the treatment so the control differs only in which subspace it removes
        V_post[key], _ = consensus(post_bases[key], dims=k_pre[key])
    mean_k = sum(k_pre.values()) / len(k_pre)
    print(f"  합의 차원 평균: 문턱 전 {mean_k:.1f}, 후 {sum(k_post.values())/len(k_post):.1f}", flush=True)

    for lam in a.sources:
        src = runs / f"lambda{lam}_seed{a.seed}" / "adapter"
        if not (src / "adapter_model.safetensors").exists():
            print(f"  λ{lam}: 원본 없음 — 건너뜀", flush=True)
            continue
        config = json.loads((src / "adapter_config.json").read_text())
        scale = config["lora_alpha"] / config["r"]
        raw, parts = read_adapter(src / "adapter_model.safetensors")

        # energy rm_pre takes, per module, so the matched controls can aim at the same number
        pre_removed = {}
        for key, names in parts.items():
            if len(names) == 2 and key in V_pre:
                B = raw[names["B"]].float()
                V = V_pre[key]
                pre_removed[key] = float((V.T @ B).pow(2).sum())

        for variant in a.variants:
            tensors = {name: value.clone() for name, value in raw.items()}
            removed, energy_frac = [], []
            for key, names in parts.items():
                if len(names) != 2 or key not in V_pre:
                    continue
                B = raw[names["B"]].float()
                k = k_pre[key]
                if variant == "rm_pre":
                    V = V_pre[key]
                elif variant == "rm_post":
                    V = V_post[key]
                elif variant == "rm_rand":
                    G = torch.randn(B.shape[0], k, generator=generator)
                    V, _ = torch.linalg.qr(G)
                elif variant == "rm_least":
                    U = left_basis(B, raw[names["A"]], scale)
                    # rank this adapter's own directions by alignment with the pre-threshold region
                    align = (U.T @ V_pre[key]).pow(2).sum(dim=1)
                    V = U[:, torch.argsort(align)[:k]]
                elif variant == "rm_energy":
                    U = left_basis(B, raw[names["A"]], scale)
                    align = (U.T @ V_pre[key]).pow(2).sum(dim=1)
                    order = torch.argsort(align)          # least aligned with the region first
                    target, taken, cols = pre_removed[key], 0.0, []
                    for idx in order:
                        cols.append(idx)
                        taken += float((U[:, idx:idx + 1].T @ B).pow(2).sum())
                        if taken >= target:
                            break
                    V = U[:, torch.stack(cols)]
                elif variant == "rm_decisive":
                    U = left_basis(B, raw[names["A"]], scale)
                    # the part of this adapter's span that the consensus region does not touch
                    P = U - V_pre[key] @ (V_pre[key].T @ U)
                    Q, R = torch.linalg.qr(P)
                    keep = torch.linalg.norm(R, dim=0) > 1e-3      # drop directions swallowed by V
                    Q = Q[:, keep]
                    energy = (Q.T @ B).pow(2).sum(dim=1)
                    order = torch.argsort(energy, descending=True)
                    target, taken, cols = pre_removed[key], 0.0, []
                    for idx in order:
                        cols.append(idx)
                        taken += float(energy[idx])
                        if taken >= target:
                            break
                    V = Q[:, torch.stack(cols)]
                elif variant in ("gs_tau", "gs_energy"):
                    U, S = spectrum(B, raw[names["A"]], scale)
                    e2 = S.pow(2)
                    if variant == "gs_tau":
                        tail = e2.flip(0).cumsum(0).flip(0) / e2.sum()     # e(r), r = 1..rank
                        r = int((tail >= a.tau).sum())                     # largest r with e(r) >= tau
                    else:
                        # dW energy rm_pre takes: ||V^T dW||^2 = ||(V^T U) S||^2
                        target = float(((V_pre[key].T @ U) * S).pow(2).sum())
                        r = int((e2.cumsum(0) < target).sum()) + 1
                    r = max(1, min(r, U.shape[1]))
                    V = U[:, :r]
                    energy_frac.append(float(e2[:r].sum() / e2.sum()))
                elif variant == "rm_scale":
                    factor = (1 - pre_removed[key] / max(float(B.pow(2).sum()), 1e-12)) ** 0.5
                    tensors[names["B"]] = (B * factor).to(raw[names["B"]].dtype)
                    removed.append(0)
                    continue
                else:
                    raise SystemExit(f"unknown variant {variant}")
                tensors[names["B"]] = (B - V @ (V.T @ B)).to(raw[names["B"]].dtype)
                removed.append(V.shape[1])

            name = f"lambda{lam}_{variant}"
            if a.region == "seed":
                name = f"lambda{lam}_" + {"rm_pre": "rmseed", "rm_decisive": "rmseed_other"}.get(variant, f"seed_{variant}")
            dest = out_root / name / "adapter"
            dest.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(dest / "adapter_model.safetensors"))
            shutil.copy2(src / "adapter_config.json", dest / "adapter_config.json")
            (out_root / name / "ablation.json").write_text(json.dumps({
                "tag": a.tag, "source_lambda": lam, "seed": a.seed, "variant": variant,
                "cliff": a.cliff, "loo": a.loo, "pre_adapters": len(pre), "modules": len(removed),
                "dims_removed_mean": sum(removed) / max(len(removed), 1),
                "rank": config["r"], "region": a.region, "threshold": threshold,
                "energy_removed_mean": (sum(energy_frac) / len(energy_frac)) if energy_frac else None,
                "tau": a.tau if variant == "gs_tau" else None,
            }, indent=2))
            # the trainer's umask leaves container-written files root-owned and mode 600, which makes
            # them unreadable to the ssh user that copies them to the other machines -- rsync then
            # fails silently and vLLM dies loading an adapter that never arrived
            for path in (out_root / name).rglob("*"):
                try:
                    os.chown(path, owner_uid, owner_gid)
                    path.chmod(0o755 if path.is_dir() else 0o644)
                except OSError:
                    pass
            print(f"  {name}: {len(removed)}개 모듈, 평균 {sum(removed)/max(len(removed),1):.1f}차원 제거", flush=True)


if __name__ == "__main__":
    main()
