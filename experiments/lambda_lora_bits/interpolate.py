#!/usr/bin/env python3
"""Stream-interpolate two same-architecture checkpoints, tensor by tensor.

    theta(lambda) = lambda * theta_instruct + (1 - lambda) * theta_reasoning

Unlike code/MI-Distillation/src/interpolate_models.py this never instantiates
either model, so peak RAM stays near one output checkpoint. The reasoning model
is the carrier: its config (untied lm_head, context length) is kept. When the
instruct model ties its embeddings, its lm_head is taken to be embed_tokens.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]


def tensor_files(model_dir: Path) -> dict:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return {name: model_dir / file for name, file in weight_map.items()}
    single = model_dir / "model.safetensors"
    with safe_open(single, "pt") as handle:
        return {name: single for name in handle.keys()}


def read(files: dict, name: str) -> torch.Tensor:
    with safe_open(files[name], "pt") as handle:
        return handle.get_tensor(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasoning", required=True)
    parser.add_argument("--instruct", required=True)
    parser.add_argument("--lam", type=float, required=True, help="Weight on the instruct endpoint.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer_from", choices=["reasoning", "instruct"], default="instruct")
    args = parser.parse_args()
    assert 0.0 <= args.lam <= 1.0

    reasoning_dir, instruct_dir, out_dir = Path(args.reasoning), Path(args.instruct), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    reasoning_files, instruct_files = tensor_files(reasoning_dir), tensor_files(instruct_dir)
    instruct_config = json.loads((instruct_dir / "config.json").read_text())

    extra = sorted(set(instruct_files) - set(reasoning_files) - {"lm_head.weight"})
    if extra:
        print(f"[warn] instruct-only tensors ignored: {extra}")

    mixed = {}
    for name in sorted(reasoning_files):
        r = read(reasoning_files, name)
        if name in instruct_files:
            i = read(instruct_files, name)
        elif name == "lm_head.weight" and instruct_config.get("tie_word_embeddings"):
            i = read(instruct_files, "model.embed_tokens.weight")
        else:
            raise KeyError(f"{name} missing from the instruct model")
        if r.shape != i.shape:
            raise ValueError(f"shape mismatch for {name}: {tuple(r.shape)} vs {tuple(i.shape)}")
        # Mix in fp32 so bf16 rounding does not bias intermediate points.
        mixed[name] = (args.lam * i.float() + (1.0 - args.lam) * r.float()).to(r.dtype).contiguous()

    save_file(mixed, out_dir / "model.safetensors", metadata={"format": "pt"})

    config = json.loads((reasoning_dir / "config.json").read_text())
    config["tie_word_embeddings"] = False
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    tokenizer_dir = reasoning_dir if args.tokenizer_from == "reasoning" else instruct_dir
    for file in TOKENIZER_FILES:
        if (tokenizer_dir / file).exists():
            shutil.copy(tokenizer_dir / file, out_dir / file)

    # Both endpoints' end-of-turn ids stop generation, whichever template is used:
    # 151643 ends R1-style turns, 151645 (<|im_end|>) ends ChatML turns.
    generation_config = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "eos_token_id": [151643, 151645]}
    (out_dir / "generation_config.json").write_text(json.dumps(generation_config, indent=2))
    (out_dir / "interpolation.json").write_text(json.dumps(vars(args), indent=2))
    print(f"lambda={args.lam}: {len(mixed)} tensors -> {out_dir} (tokenizer from {args.tokenizer_from})")


if __name__ == "__main__":
    main()
