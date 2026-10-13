#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
]


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


def nbytes(files: dict, name: str) -> int:
    with safe_open(files[name], "pt") as handle:
        view = handle.get_slice(name)
        shape, dtype = (view.get_shape(), view.get_dtype())
    size = {"F64": 8, "F32": 4, "I64": 8, "I32": 4, "F16": 2, "BF16": 2, "I8": 1, "U8": 1, "BOOL": 1}[dtype]
    count = 1
    for dim in shape:
        count *= dim
    return count * size


def eos_ids(model_dir: Path) -> list:
    config = model_dir / "generation_config.json"
    if not config.exists():
        return []
    eos = json.loads(config.read_text()).get("eos_token_id", [])
    return eos if isinstance(eos, list) else [eos]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasoning", required=True)
    parser.add_argument("--instruct", required=True)
    parser.add_argument("--lam", type=float, required=True, help="Weight on the instruct endpoint.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer_from", choices=["reasoning", "instruct"], default="instruct")
    parser.add_argument("--shard_gb", type=float, default=4.0)
    args = parser.parse_args()
    assert 0.0 <= args.lam <= 1.0
    reasoning_dir, instruct_dir, out_dir = (Path(args.reasoning), Path(args.instruct), Path(args.output))
    out_dir.mkdir(parents=True, exist_ok=True)
    reasoning_files, instruct_files = (tensor_files(reasoning_dir), tensor_files(instruct_dir))
    instruct_config = json.loads((instruct_dir / "config.json").read_text())
    config = json.loads((reasoning_dir / "config.json").read_text())
    both_tied = bool(config.get("tie_word_embeddings")) and bool(instruct_config.get("tie_word_embeddings"))
    extra = sorted(set(instruct_files) - set(reasoning_files) - {"lm_head.weight"})
    if extra:
        print(f"[warn] instruct-only tensors ignored: {extra}")
    names = sorted(reasoning_files)
    if both_tied:
        names = [n for n in names if n != "lm_head.weight"]
    elif "lm_head.weight" not in reasoning_files:
        names.append("lm_head.weight")

    def source(files: dict, tied: bool, name: str) -> torch.Tensor:
        if name in files:
            return read(files, name)
        if name == "lm_head.weight" and tied:
            return read(files, "model.embed_tokens.weight")
        raise KeyError(f"{name} missing from the checkpoint")

    total = sum(
        (nbytes(reasoning_files, n if n in reasoning_files else "model.embed_tokens.weight") for n in names)
    )
    limit = int(args.shard_gb * 1024**3)
    n_shards = max(1, -(-total // limit))
    shards, shard, shard_bytes, weight_map = ([], {}, 0, {})

    def flush() -> None:
        nonlocal shard, shard_bytes
        if not shard:
            return
        file = "model.safetensors" if n_shards == 1 else f"model-{len(shards) + 1:05d}.safetensors"
        save_file(shard, out_dir / file, metadata={"format": "pt"})
        weight_map.update({name: file for name in shard})
        shards.append(file)
        shard, shard_bytes = ({}, 0)

    for name in names:
        r = source(reasoning_files, bool(config.get("tie_word_embeddings")), name)
        i = source(instruct_files, bool(instruct_config.get("tie_word_embeddings")), name)
        if r.shape != i.shape:
            raise ValueError(f"shape mismatch for {name}: {tuple(r.shape)} vs {tuple(i.shape)}")
        mixed = (args.lam * i.float() + (1.0 - args.lam) * r.float()).to(r.dtype).contiguous()
        del r, i
        if shard and shard_bytes + mixed.numel() * mixed.element_size() > limit:
            flush()
        shard[name] = mixed
        shard_bytes += mixed.numel() * mixed.element_size()
    flush()
    if len(shards) == 1 and shards[0] != "model.safetensors":
        (out_dir / shards[0]).rename(out_dir / "model.safetensors")
        weight_map = {name: "model.safetensors" for name in weight_map}
    elif len(shards) > 1:
        renamed = {}
        for k, file in enumerate(shards, start=1):
            final = f"model-{k:05d}-of-{len(shards):05d}.safetensors"
            (out_dir / file).rename(out_dir / final)
            renamed[file] = final
        weight_map = {name: renamed[file] for name, file in weight_map.items()}
        index = {"metadata": {"total_size": total}, "weight_map": weight_map}
        (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    config["tie_word_embeddings"] = both_tied
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    tokenizer_dir = reasoning_dir if args.tokenizer_from == "reasoning" else instruct_dir
    for file in TOKENIZER_FILES:
        if (tokenizer_dir / file).exists():
            shutil.copy(tokenizer_dir / file, out_dir / file)
    stop_ids = sorted(set(eos_ids(reasoning_dir)) | set(eos_ids(instruct_dir)))
    generation_config = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "eos_token_id": stop_ids}
    (out_dir / "generation_config.json").write_text(json.dumps(generation_config, indent=2))
    (out_dir / "interpolation.json").write_text(json.dumps(vars(args), indent=2))
    print(
        f"lambda={args.lam}: {len(weight_map)} tensors in {len(shards)} file(s) -> {out_dir} (tokenizer from {args.tokenizer_from}, tied={both_tied}, eos={stop_ids})"
    )


if __name__ == "__main__":
    main()
