#!/usr/bin/env python3
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from config import MODELS, REPO_ROOT as ROOT, RESULTS

WEIGHT_SUFFIXES = (".safetensors", ".bin")
PAPER_MODELS = [
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "Llama-3.1-8B-Instruct",
    "Llama-3.2-1B-Instruct",
    "OLMo-2-1B-Instruct",
    "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-Math-1.5B-Instruct",
    "Qwen3-4B-Instruct-2507",
    "Qwen3-4B-Thinking-2507",
]


def read_metadata(model_dir):
    cache = model_dir / ".cache" / "huggingface" / "download"
    commits, files = (set(), {})
    if not cache.is_dir():
        return (None, files)
    for meta in sorted(cache.glob("*.metadata")):
        lines = meta.read_text().split("\n")
        if len(lines) >= 2 and lines[0].strip():
            commits.add(lines[0].strip())
            files[meta.name[: -len(".metadata")]] = lines[1].strip()
    return (commits.pop() if len(commits) == 1 else sorted(commits), files)


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("torch", "transformers", "peft", "safetensors", "vllm", "datasets", "accelerate"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    try:
        out["nvidia_smi"] = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            .stdout.strip()
            .split("\n")
        )
    except Exception:
        out["nvidia_smi"] = None
    try:
        out["git_head"] = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30
        ).stdout.strip()
        out["git_dirty"] = bool(
            subprocess.run(
                ["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True, timeout=60
            ).stdout.strip()
        )
    except Exception:
        pass
    return out


def main():
    lock = {"models": {}, "environment": versions()}
    for model_dir in (MODELS / name for name in PAPER_MODELS if (MODELS / name).is_dir()):
        commit, recorded = read_metadata(model_dir)
        weights = {}
        for f in sorted(model_dir.iterdir()):
            if f.is_file() and f.name.endswith(WEIGHT_SUFFIXES) or f.name == "config.json":
                weights[f.name] = dict(bytes=f.stat().st_size, sha256_at_download=recorded.get(f.name))
        lock["models"][model_dir.name] = dict(hf_commit=commit, files=weights)
        print(f"{model_dir.name:34s} commit {(commit if isinstance(commit, str) else commit)}")
    path = RESULTS / "reproducibility.lock.json"
    path.write_text(json.dumps(lock, indent=2))
    env = lock["environment"]
    print(
        f"\ntorch {env['torch']} / transformers {env['transformers']} / peft {env['peft']} / vllm {env['vllm']}"
    )
    print(f"git {env.get('git_head', '?')[:12]} (dirty={env.get('git_dirty')})")
    print(path)


if __name__ == "__main__":
    main()
