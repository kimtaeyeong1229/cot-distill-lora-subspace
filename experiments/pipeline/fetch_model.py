#!/usr/bin/env python3
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

REPOS = {
    "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "OLMo-2-1B-Instruct": "allenai/OLMo-2-0425-1B-Instruct",
    "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen2.5-Math-1.5B-Instruct": "Qwen/Qwen2.5-Math-1.5B-Instruct",
}
target = Path(sys.argv[1])
snapshot_download(
    REPOS[target.name],
    local_dir=str(target),
    allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "tokenizer*", "*.model"],
)
print(f"downloaded {REPOS[target.name]} -> {target}")
