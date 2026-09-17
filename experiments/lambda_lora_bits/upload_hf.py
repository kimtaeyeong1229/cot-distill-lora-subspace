#!/usr/bin/env python3
"""Stage and upload the artifacts that cannot be regenerated, to two private Hub repos.

Split by what a reader would do with them:

  model repo    the 332 LoRA adapters, plus the revision lock and the analysis outputs. Every number
                in the paper is linear algebra over these weights, so this repo alone lets someone
                re-derive the subspace results with no GPU and no retraining.
  dataset repo  the teacher CoT generations, the train/held splits, and the student generations kept
                in the eval json. These cost GPU-days to produce and cannot be reproduced exactly
                (sampling at temperature 0.6), so they are the irreplaceable half.

Left out: the interpolated teacher checkpoints (228 GB across the fleet). They are an element-wise
average of two published checkpoints, and reproducibility.lock.json pins the exact commit of each,
so interpolate.py rebuilds them byte-for-byte. Uploading them would cost 228 GB to save a few minutes.

Staging uses hard links, so nothing is copied twice on disk.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "lambda_lora_bits"
STAGE = ROOT / "experiments" / "hf_upload"
ACCOUNT = "kimtaeyeong1229"
MODEL_REPO = f"{ACCOUNT}/cot-distill-lora-adapters"
DATA_REPO = f"{ACCOUNT}/cot-distill-teacher-cot"

# directory name -> (teacher pair, student) for the README mapping
CELLS = {
    "sweep": ("A", "Qwen2.5-0.5B-Instruct"), "sweep_llama": ("A", "Llama-3.2-1B-Instruct"),
    "sweep_olmo2": ("A", "OLMo-2-1B-Instruct"), "sweep_qwen1.5b": ("A", "Qwen2.5-1.5B-Instruct"),
    "sweep_qwen3b": ("A", "Qwen2.5-3B-Instruct"), "sweep_smollm2": ("A", "SmolLM2-1.7B-Instruct"),
    "pairB/qwen0.5b": ("B", "Qwen2.5-0.5B-Instruct"), "pairB/llama1b": ("B", "Llama-3.2-1B-Instruct"),
    "pairB/olmo1b": ("B", "OLMo-2-1B-Instruct"), "pairB/qwen1.5b": ("B", "Qwen2.5-1.5B-Instruct"),
    "pairC/qwen0.5b": ("C", "Qwen2.5-0.5B-Instruct"), "pairC/llama1b": ("C", "Llama-3.2-1B-Instruct"),
    "pairC/olmo1b": ("C", "OLMo-2-1B-Instruct"), "pairC/qwen1.5b": ("C", "Qwen2.5-1.5B-Instruct"),
    "pairC/qwen3b": ("C", "Qwen2.5-3B-Instruct"),
    "pairD/qwen0.5b": ("D", "Qwen2.5-0.5B-Instruct"), "pairD/llama1b": ("D", "Llama-3.2-1B-Instruct"),
    "pairD/olmo1b": ("D", "OLMo-2-1B-Instruct"), "pairD/qwen1.5b": ("D", "Qwen2.5-1.5B-Instruct"),
    # side experiments kept for the record rather than used in the main tables
    "control_tokens": ("-", "토큰 통제 대조군"), "canary": ("-", "카나리 점검"),
}
PAIR_DIR = {"A": "sweep", "B": "pairB", "C": "pairC", "D": "pairD"}
ENDPOINTS = {
    "A": ("DeepSeek-R1-Distill-Qwen-1.5B", "Qwen2.5-Math-1.5B-Instruct"),
    "B": ("Qwen3-4B-Thinking-2507", "Qwen3-4B-Instruct-2507"),
    "C": ("DeepSeek-R1-Distill-Qwen-7B", "Qwen2.5-Math-7B-Instruct"),
    "D": ("DeepSeek-R1-Distill-Llama-8B", "Llama-3.1-8B-Instruct"),
}


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage_adapters():
    out = STAGE / "adapters"
    if out.exists():
        shutil.rmtree(out)
    rows, total = [], 0
    for tag in sorted(CELLS):
        runs = EXP / tag / "runs"
        if not runs.is_dir():
            continue
        found = 0
        for run in sorted(runs.iterdir()):
            weights = run / "adapter" / "adapter_model.safetensors"
            if not weights.exists():
                continue
            dest = out / tag / run.name
            link(weights, dest / "adapter_model.safetensors")
            link(run / "adapter" / "adapter_config.json", dest / "adapter_config.json")
            found += 1
            total += weights.stat().st_size
        if found:
            pair, student = CELLS[tag]
            rows.append((tag, pair, student, found))

    for name in ("reproducibility.lock.json", "think_rate.json", "phi_table.json", "phi_xseed.json",
                 "phi_resample.json", "alignment.json", "control_pairc.json",
                 "accuracy_eval_math500.json", "accuracy_eval_val.json"):
        p = ROOT / "experiments" / name
        if p.exists():
            link(p, out / "analysis" / name)

    lines = [
        "# CoT distillation: LoRA adapters across an interpolated teacher spectrum", "",
        "Private research artifact. Each adapter is a rank-16 LoRA (alpha 32) trained for 3 epochs",
        "at lr 2e-4 on chain-of-thought written by one interpolated teacher.", "",
        "## Teacher construction", "",
        "theta(lambda) = lambda * theta_instruct + (1 - lambda) * theta_reasoning, element-wise over",
        "every tensor, mixed in fp32 and cast to bf16. lambda weights the short-answer endpoint.", "",
        "| pair | lambda=0 (reasoning) | lambda=1 (instruct) |", "|---|---|---|",
    ]
    lines += [f"| {p} | {a} | {b} |" for p, (a, b) in sorted(ENDPOINTS.items())]
    lines += [
        "", "Endpoint commits are pinned in `analysis/reproducibility.lock.json`; `interpolate.py`",
        "in the source repo rebuilds any teacher byte-for-byte from them, which is why the",
        "interpolated checkpoints are not uploaded.", "",
        "## Layout", "",
        "`<cell>/lambda<value>_seed<0|1|2>/` are the main runs (three seeds per lambda).",
        "`<cell>/lambda<value>_half<A|B>/` are the data-resample baseline: same seed, each trained on",
        "one disjoint half of that lambda's training set, so comparing the two isolates the effect of",
        "changing the data while the initialisation is held fixed.", "",
        "| cell | teacher pair | student | adapters |", "|---|---|---|---|",
    ]
    lines += [f"| `{t}` | {p} | {s} | {n} |" for t, p, s, n in rows]
    lines += ["", f"Total {sum(n for *_, n in rows)} adapters, {total / 2**30:.1f} GiB.", "",
              "## Generation settings used to write the training CoT", "",
              "temperature 0.6, top-p 0.95, max 8192 new tokens, stop tokens the union of both",
              "endpoints' eos ids; 4000 MATH problems per teacher.", "",
              "## Licensing", "",
              "Derived from the endpoint checkpoints above and inherits their terms. Pair D's",
              "reasoning endpoint is a Llama derivative, so anything downstream of pair D carries the",
              "Llama community licence conditions. Keep this repo private until that is settled."]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    return out, sum(n for *_, n in rows), total


def stage_data():
    out = STAGE / "cot"
    if out.exists():
        shutil.rmtree(out)
    counts, total = {}, 0
    for pair, directory in PAIR_DIR.items():
        cot = EXP / directory / "cot"
        if cot.is_dir():
            for f in sorted(cot.glob("lambda*.jsonl")):
                link(f, out / "teacher_cot" / f"pair{pair}" / f.name)
                total += f.stat().st_size
                counts.setdefault(f"pair{pair} teacher_cot", 0)
                counts[f"pair{pair} teacher_cot"] += 1
        splits = EXP / directory / "splits"
        if splits.is_dir():
            for f in sorted(splits.rglob("*.json*")):
                link(f, out / "splits" / f"pair{pair}" / f.relative_to(splits))
                total += f.stat().st_size
    for tag in sorted(CELLS):
        for bench in ("eval_val", "eval_math500"):
            d = EXP / tag / bench
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.json")):
                link(f, out / "student_generations" / tag / bench / f.name)
                total += f.stat().st_size
                counts.setdefault(f"{tag} {bench}", 0)
                counts[f"{tag} {bench}"] += 1

    (out / "README.md").write_text("\n".join([
        "# CoT distillation: teacher generations, splits and student outputs", "",
        "Private research artifact. Sampled at temperature 0.6 / top-p 0.95, so it cannot be",
        "reproduced exactly even from the same checkpoints -- this is the irreplaceable half of the",
        "experiment.", "",
        "## Contents", "",
        "- `teacher_cot/pair{A,B,C,D}/lambda*.jsonl` -- 4000 MATH problems per interpolated teacher,",
        "  field `cot_response` holds the raw generation. Count `</think>` here, not in the splits:",
        "  the training files have the tags stripped and read 0% at every lambda.",
        "- `splits/pair*/lambda*/` -- the train/held split actually fed to the students, and",
        "  `split_stats.json` with per-lambda token counts and teacher accuracy.",
        "- `student_generations/<cell>/eval_{val,math500}/` -- per-adapter evaluation records with",
        "  `correct`, `tokens` and `truncated` per sample. The `truncated` flag is what separates a",
        "  wrong answer from one that ran out of generation budget.", "",
        "## Licensing", "",
        "Generated by the endpoint checkpoints listed in the companion model repo. Pair D descends",
        "from a Llama derivative and carries the Llama community licence conditions.",
    ]) + "\n")
    return out, counts, total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", choices=["adapters", "data", "both"], default="both")
    ap.add_argument("--stage-only", action="store_true")
    a = ap.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    if a.what in ("adapters", "both"):
        path, n, size = stage_adapters()
        print(f"[stage] adapters {n}개 {size / 2**30:.1f} GiB -> {path}", flush=True)
        if not a.stage_only:
            api.create_repo(MODEL_REPO, repo_type="model", private=True, exist_ok=True)
            api.upload_large_folder(repo_id=MODEL_REPO, repo_type="model", folder_path=str(path),
                                    private=True)
            print(f"[done] https://huggingface.co/{MODEL_REPO}", flush=True)
    if a.what in ("data", "both"):
        path, counts, size = stage_data()
        print(f"[stage] data {sum(counts.values())}개 파일 {size / 2**30:.2f} GiB -> {path}", flush=True)
        if not a.stage_only:
            api.create_repo(DATA_REPO, repo_type="dataset", private=True, exist_ok=True)
            api.upload_large_folder(repo_id=DATA_REPO, repo_type="dataset", folder_path=str(path),
                                    private=True)
            print(f"[done] https://huggingface.co/datasets/{DATA_REPO}", flush=True)


if __name__ == "__main__":
    main()
