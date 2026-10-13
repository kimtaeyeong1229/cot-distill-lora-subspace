#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from config import EXP, REPO_ROOT


def load(path):
    config = json.loads((path / "adapter_config.json").read_text())
    with safe_open(path / "adapter_model.safetensors", "pt") as h:
        tensors = {k: h.get_tensor(k) for k in h.keys()}
    return (config, tensors)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 0.7])
    a = ap.parse_args()
    runs = EXP / a.tag / "runs"
    c0, t0 = load(runs / f"lambda0.0_seed{a.seed}" / "adapter")
    c1, t1 = load(runs / f"lambda1.0_seed{a.seed}" / "adapter")
    assert set(t0) == set(t1), "the two adapters target different modules"
    s0, s1 = (c0["lora_alpha"] / c0["r"], c1["lora_alpha"] / c1["r"])
    stat = REPO_ROOT.stat()
    for alpha in a.alphas:
        merged = {}
        for name in t0:
            if "lora_A" in name:
                merged[name] = torch.cat([t0[name], t1[name]], dim=0).contiguous()
            elif "lora_B" in name:
                merged[name] = (
                    torch.cat([alpha * s0 * t0[name].float(), (1 - alpha) * s1 * t1[name].float()], dim=1)
                    .to(t0[name].dtype)
                    .contiguous()
                )
            else:
                merged[name] = t0[name]
        config = dict(c0, r=c0["r"] + c1["r"], lora_alpha=c0["r"] + c1["r"])
        dest = EXP / a.tag / f"runs_edit_s{a.seed}" / f"merge{alpha}" / "adapter"
        dest.mkdir(parents=True, exist_ok=True)
        save_file(merged, str(dest / "adapter_model.safetensors"))
        (dest / "adapter_config.json").write_text(json.dumps(config, indent=2))
        (dest.parent / "ablation.json").write_text(
            json.dumps(
                {
                    "tag": a.tag,
                    "seed": a.seed,
                    "variant": f"merge{alpha}",
                    "alpha": alpha,
                    "sources": ["lambda0.0", "lambda1.0"],
                    "rank": config["r"],
                },
                indent=2,
            )
        )
        for p in dest.parent.rglob("*"):
            try:
                os.chown(p, stat.st_uid, stat.st_gid)
                p.chmod(493 if p.is_dir() else 420)
            except OSError:
                pass
        print(f"[{a.tag} s{a.seed}] merge{alpha}: rank {config['r']}", flush=True)


if __name__ == "__main__":
    main()
