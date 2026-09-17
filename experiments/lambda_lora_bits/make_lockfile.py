#!/usr/bin/env python3
"""Pin everything the experiment depends on but does not own.

The interpolated teachers are a weighted average of two published checkpoints, so they are only
reproducible while those checkpoints stay byte-identical. interpolation.json records local paths,
which say nothing about which upload was used -- if an endpoint is re-uploaded, the teachers cannot
be rebuilt and every downstream number becomes unverifiable.

huggingface_hub leaves the commit hash and each file's sha256 in <model>/.cache/huggingface/download/
*.metadata (first line commit, second line sha256), so the pin can be recovered after the fact.
This writes experiments/reproducibility.lock.json with, for every model in models/:

  the HF commit hash the snapshot came from, the sha256 of config.json and the weight shards, and
  the installed versions of the libraries that decide numerics.
"""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
WEIGHT_SUFFIXES = (".safetensors", ".bin")


def read_metadata(model_dir):
    """Commit hash and per-file sha256 as recorded by huggingface_hub at download time."""
    cache = model_dir / ".cache" / "huggingface" / "download"
    commits, files = set(), {}
    if not cache.is_dir():
        return None, files
    for meta in sorted(cache.glob("*.metadata")):
        lines = meta.read_text().split("\n")
        if len(lines) >= 2 and lines[0].strip():
            commits.add(lines[0].strip())
            files[meta.name[: -len(".metadata")]] = lines[1].strip()
    # a snapshot is one commit; more than one means the directory was assembled from several
    return (commits.pop() if len(commits) == 1 else sorted(commits)), files


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("torch", "transformers", "peft", "safetensors", "vllm", "datasets", "accelerate"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    try:
        out["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip().split("\n")
    except Exception:
        out["nvidia_smi"] = None
    try:
        out["git_head"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                         capture_output=True, text=True, timeout=30).stdout.strip()
        out["git_dirty"] = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                               capture_output=True, text=True, timeout=60).stdout.strip())
    except Exception:
        pass
    return out


def main():
    lock = {"models": {}, "environment": versions()}
    for model_dir in sorted(p for p in MODELS.iterdir() if p.is_dir()):
        commit, recorded = read_metadata(model_dir)
        weights = {}
        for f in sorted(model_dir.iterdir()):
            if f.is_file() and f.name.endswith(WEIGHT_SUFFIXES) or f.name == "config.json":
                weights[f.name] = dict(bytes=f.stat().st_size, sha256_at_download=recorded.get(f.name))
        lock["models"][model_dir.name] = dict(hf_commit=commit, files=weights)
        print(f"{model_dir.name:34s} commit {commit if isinstance(commit, str) else commit}")
    path = ROOT / "experiments" / "reproducibility.lock.json"
    path.write_text(json.dumps(lock, indent=2))
    env = lock["environment"]
    print(f"\ntorch {env['torch']} / transformers {env['transformers']} / peft {env['peft']} / vllm {env['vllm']}")
    print(f"git {env.get('git_head', '?')[:12]} (dirty={env.get('git_dirty')})")
    print(path)


if __name__ == "__main__":
    main()
