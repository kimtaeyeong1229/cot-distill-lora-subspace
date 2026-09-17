#!/usr/bin/env python3
"""Download a student or teacher checkpoint into models/<name> using the token in hf.key."""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPOS = {
    "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "OLMo-2-1B-Instruct": "allenai/OLMo-2-0425-1B-Instruct",
}

target = Path(sys.argv[1])
token = (Path(__file__).resolve().parents[2] / "hf.key").read_text().strip()
snapshot_download(REPOS[target.name], local_dir=str(target), token=token,
                  allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "tokenizer*", "*.model"])
print(f"downloaded {REPOS[target.name]} -> {target}")
