#!/usr/bin/env python3
"""Download a student or pair A endpoint checkpoint into models/<name> (token from hf.key if present)."""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPOS = {
    # students
    "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "OLMo-2-1B-Instruct": "allenai/OLMo-2-0425-1B-Instruct",
    # pair A endpoints (pairs B and C are downloaded by run_teachers_pair_bc.sh)
    "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen2.5-Math-1.5B-Instruct": "Qwen/Qwen2.5-Math-1.5B-Instruct",
}

target = Path(sys.argv[1])
key = Path(__file__).resolve().parents[2] / "hf.key"   # optional; otherwise HF_TOKEN or the cached login is used
token = key.read_text().strip() if key.exists() else None
snapshot_download(REPOS[target.name], local_dir=str(target), token=token,
                  allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "tokenizer*", "*.model"])
print(f"downloaded {REPOS[target.name]} -> {target}")
