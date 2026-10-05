#!/usr/bin/env python3
"""Stage and upload the artifacts that cannot be regenerated, to two public Hub repos.

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
MODEL_REPO = f"{ACCOUNT}/cot-distill-lora-subspace"
DATA_REPO = f"{ACCOUNT}/cot-distill-lora-subspace"

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
    "control_tokens": ("-", "token-count control"), "canary": ("-", "canary check"),
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


# directories that hold no research adapter: hardware smoke tests, copies kept from the retired
# servers, and pairD/olmo1b/runs_stale_server2 (six seed-0 runs that were retrained and replaced)
SKIP_TOP = {"_server_records", "_server5_root", "smoke"}
RUN_FILES = ("ablation.json", "train_summary.json", "train_args.json")
BASES = {
    "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct", "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "OLMo-2-1B-Instruct": "allenai/OLMo-2-0425-1B-Instruct", "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct", "SmolLM2-1.7B-Instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}


def stage_adapters():
    """Every adapter on disk. Main runs keep the old `<cell>/<run>` path; the rest keep their
    subdirectory, e.g. `<cell>/runs_edit_s0/lambda0.0_rm_pre`."""
    out = STAGE / "adapters"
    if out.exists():
        shutil.rmtree(out)
    counts, total = {}, 0
    for weights in sorted(EXP.rglob("adapter/adapter_model.safetensors")):
        run = weights.parent.parent
        parts = run.relative_to(EXP).parts
        if parts[0] in SKIP_TOP or any(x.startswith("runs_stale") for x in parts):
            continue
        rel = Path(*[x for x in parts if x != "runs"])
        link(weights, out / rel / "adapter_model.safetensors")
        link(weights.parent / "adapter_config.json", out / rel / "adapter_config.json")
        for name in RUN_FILES:
            if (run / name).exists():
                link(run / name, out / rel / name)
        group = str(rel.parent)
        counts[group] = counts.get(group, 0) + 1
        total += weights.stat().st_size

    for p in sorted((ROOT / "experiments").glob("*.json")):
        link(p, out / "analysis" / p.name)

    n = sum(counts.values())
    lines = [
        "---", "license: other", "library_name: peft", "tags:", "- lora", "- chain-of-thought",
        "- distillation", "- interpretability", "base_model:",
    ] + [f"- {b}" for b in BASES.values()] + [
        "---", "",
        "# CoT distillation: LoRA adapters across an interpolated teacher spectrum", "",
        "Every adapter behind the paper's results. Each is a rank-16 LoRA (alpha 32, all seven linear",
        "modules) trained for 3 epochs at lr 2e-4 on chain-of-thought written by one interpolated teacher,",
        "or an edit of such an adapter. The subspace analysis is linear algebra over these weights, so",
        "this repo alone re-derives it without a GPU.", "",
        "## Teacher construction", "",
        "theta(lambda) = lambda * theta_instruct + (1 - lambda) * theta_reasoning, element-wise over",
        "every tensor, mixed in fp32 and cast to bf16. lambda weights the short-answer endpoint.", "",
        "| pair (directory) | lambda=0 (reasoning) | lambda=1 (instruct) |", "|---|---|---|",
    ]
    lines += [f"| {p} (`{PAIR_DIR[p]}`) | {a} | {b} |" for p, (a, b) in sorted(ENDPOINTS.items())]
    lines += [
        "", "The paper's pairs A, B and C are the directories `sweep*`, `pairB` and `pairD`; `pairC` is an",
        "earlier pair kept for the record. Endpoint commits are pinned in",
        "`analysis/reproducibility.lock.json`, and `interpolate.py` in the source repo rebuilds any",
        "teacher byte-for-byte from them, so the interpolated checkpoints are not uploaded.", "",
        "## Layout", "",
        "Cells: `sweep` / `sweep_llama` / `sweep_olmo2` are pair A with Qwen2.5-0.5B / Llama-3.2-1B /",
        "OLMo-2-1B students; `pairB/<student>` and `pairD/<student>` likewise.", "",
        "- `<cell>/lambda<l>_seed<s>` -- main runs, three seeds per lambda.",
        "- `<cell>/lambda<l>_half<A|B>[_seed<s>]` -- data-resample baseline: the same seed trained on each",
        "  disjoint half of that lambda's training set (seed 0 has no suffix).",
        "- `<cell>/runs_edit_s<s>/` -- edits of the seed-s adapters: `*_rm_pre` removes the shared region,",
        "  `*_rm_decisive` removes the same energy outside it, `*_rm_scale` shrinks the whole update,",
        "  `*_gs_tau` / `*_gs_energy` remove top singular components, `*_rmseed` removes the region found",
        "  from two other seeds (`*_rmseed_other` its control), `graft*` add the region to lambda=1,",
        "  `amp*` amplify it at fixed norm, `merge0.5` / `merge0.7` merge lambda=0 with lambda=1.",
        "- `<cell>/runs_ablation*`, `runs_steer*`, `runs_graft_beta` -- the same edits from the exploratory",
        "  phase (`_loo`: region built leaving the edited adapter out).",
        "- `analysis/` -- the analysis outputs the tables are read from.",
        "- Each edit directory carries `ablation.json` describing the edit; each trained run carries",
        "  `train_summary.json` and `train_args.json`.", "",
        "Load with `peft`: `PeftModel.from_pretrained(base, repo, subfolder=\"sweep/lambda0.0_seed0\")`.", "",
        "| directory | adapters |", "|---|---|",
    ]
    lines += [f"| `{g}` | {c} |" for g, c in sorted(counts.items())]
    lines += ["", f"Total {n} adapters, {total / 2**30:.1f} GiB.", "",
              "## Generation settings used to write the training CoT", "",
              "temperature 0.6, top-p 0.95, max 8192 new tokens; 4000 MATH problems per teacher.", "",
              "## Licensing", "",
              "Each adapter inherits the licence of its base student and of the teacher whose outputs it",
              "was trained on. Qwen2.5-0.5B/1.5B, OLMo-2 and SmolLM2 are Apache-2.0; Qwen2.5-3B uses the",
              "Qwen research licence. Adapters on Llama-3.2-1B are subject to the Llama 3.2 Community",
              "License, and adapters trained on pair D (`pairD`) outputs descend from",
              "DeepSeek-R1-Distill-Llama-8B and Llama-3.1-8B-Instruct under the Llama 3.1 Community License.",
              "Built with Llama."]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    return out, n, total


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
        "Sampled at temperature 0.6 / top-p 0.95, so it cannot be",
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
        "  wrong answer from one that ran out of generation budget.",
        "- `student_generations/<cell>/eval_main_s{0,1,2}/` -- the paper's evaluation (first 200 MATH-500",
        "  problems x 2 samples) of each seed's adapters and their edits: a `.json` summary and the responses",
        "  in `.samples.jsonl.gz`.",
        "- `student_generations/<cell>/eval_gsm8k_s*/` -- the same adapters on GSM8K.",
        "- `student_generations/<cell>/eval_gpqa_diamond_s*/` -- GPQA-Diamond summaries and per-item",
        "  correctness, token count and finish reason only. Question and response text are omitted because",
        "  the GPQA authors ask that its examples not be posted online.",
        "- `student_generations/untrained/<student>/eval_main_s*/` -- the untrained students under the same",
        "  evaluation, with the sampling seed varied instead of the training seed.", "",
        "## Licensing", "",
        "Generated by the endpoint checkpoints listed in the companion model repo. Pair D descends",
        "from a Llama derivative and carries the Llama community licence conditions.",
    ]) + "\n")
    return out, counts, total


PAPER_CELLS = ["sweep", "sweep_llama", "sweep_olmo2", "pairB/qwen0.5b", "pairB/llama1b", "pairB/olmo1b",
               "pairD/qwen0.5b", "pairD/llama1b", "pairD/olmo1b"]


# analysis/ JSONs the paper reads (model repo). Others in experiments/ are past-design or retracted.
PAPER_ANALYSIS = (
    "alignment_markers_s012", "alignment_s012", "base_students", "baselines", "consensus_post", "consensus_pre",
    "graft_beta", "main_density", "main_edits", "ood", "phi_resample_s012", "phi_xseed", "reproducibility.lock",
    "sae_llama", "seed_region_overlap", "seed_spread", "teacher_density", "think_rate",
    # added 2026-10-04 (appendix tables 16-22 and the coverage figure)
    "coverage_by_lambda", "degeneration", "edit_stats", "length_confound", "threshold_sensitivity", "top_overlap",
    # added 2026-10-06: the analyses behind the current manuscript's Tables 1-17 and Figure 2
    # (bottom), and the numbers cited in Sections 4-5 and Appendices A-C (README.md, "Paper artifacts")
    "agreement_normalised", "alignment_length_s012", "c13_additions", "energy_match", "examples",
    "lambda0_retention", "pairA_crossseed", "pairA_seed_ratio", "paired_bootstrap", "paired_bootstrap_grades",
    "region_anatomy", "region_logit_lens", "removal_vs_untrained", "review_stats", "student_structure",
    "supplementary_stats", "switch_regression", "symmetric_R", "teacher_structure", "think_rate_completed",
    "two_seed_floor",
)
FRESH_SECONDS = 600  # a file written this recently may still be in progress; leave it for the next run


def extra_sources():
    """(source, path in the dataset repo) for the paper's student generations beyond the first upload.

    eval_math500_7680 is the 7,680-token re-run the paper reads for A/Qwen and A/Llama.
    eval_graft_beta holds the seed-0 grafts at beta = 2, 4, 8 behind analysis/graft_beta.json and
    degeneration.json. merge0.7 was removed from the Hub in the 2026-10-03 cleanup and is not re-added.
    """
    for tag in PAPER_CELLS:
        for d in sorted((EXP / tag).glob("eval_*")):
            if not (d.name[:-1] in ("eval_main_s", "eval_gsm8k_s", "eval_gpqa_diamond_s")
                    or d.name in ("eval_graft_beta", "eval_math500_7680")):
                continue
            for f in sorted(d.iterdir()):
                if f.name.startswith("merge0.7."):
                    continue
                yield f, f"student_generations/{tag}/{d.name}/{f.name}"
    for f in sorted((EXP / "base_eval").rglob("*")):
        if f.is_file() and f.name != "eval.log":
            yield f, f"student_generations/untrained/{f.relative_to(EXP / 'base_eval').as_posix()}"


def put(src: Path, dest: Path):
    """Hard-link src to dest, except GPQA-Diamond samples: its authors ask that the examples not be posted
    online in plain text, so those records keep only item index, sample index, finish reason, token count
    and correctness; question and response text are dropped. MATH-500 and GSM8K are copied as written."""
    import gzip
    if "eval_gpqa_diamond" in src.parent.name and src.name.endswith(".samples.jsonl.gz"):
        rows = [json.loads(line) for line in gzip.open(src, "rt")]
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(dest, "wt") as g:
            for i, r in enumerate(rows):
                g.write(json.dumps({"item": i // 2, "sample": i % 2, "finish_reason": r["finish_reason"],
                                    "tokens": r["tokens"], "correct": r["correct"]}) + "\n")
    else:
        link(src, dest)


def stage_extra():
    """Student generations behind the paper's tables, added to the dataset repo after its first upload."""
    out = STAGE / "cot_extra"
    if out.exists():
        shutil.rmtree(out)
    n = 0
    for src, rel in extra_sources():
        put(src, out / rel)
        n += 1
    return out, n


def upload_missing(api, stage_only):
    """Upload only the paper files the Hub lacks, into a new staging folder (nothing existing is removed).
    Files modified in the last FRESH_SECONDS are skipped and reported."""
    import time
    out = STAGE / f"missing_{time.strftime('%Y%m%d_%H%M%S')}"
    now = time.time()
    plan = {"dataset": [], "model": []}
    hub = {"dataset": set(api.list_repo_files(DATA_REPO, repo_type="dataset")),
           "model": set(api.list_repo_files(MODEL_REPO, repo_type="model"))}
    candidates = [("dataset", src, rel) for src, rel in extra_sources()]
    candidates += [("model", ROOT / "experiments" / f"{n}.json", f"analysis/{n}.json") for n in PAPER_ANALYSIS]
    for kind, src, rel in candidates:
        if rel in hub[kind]:
            continue
        if not src.exists():
            print(f"[skip] {kind} {rel}: no local file", flush=True)
        elif now - src.stat().st_mtime < FRESH_SECONDS:
            print(f"[skip] {kind} {rel}: modified {now - src.stat().st_mtime:.0f} s ago", flush=True)
        else:
            put(src, out / kind / rel)
            plan[kind].append(rel)
    for kind, repo in (("dataset", DATA_REPO), ("model", MODEL_REPO)):
        size = sum((out / kind / r).stat().st_size for r in plan[kind])
        print(f"[stage] {kind} {len(plan[kind])} files {size / 1e6:.1f} MB -> {out / kind}", flush=True)
        for r in plan[kind]:
            print(f"  {r}", flush=True)
        if plan[kind] and not stage_only:
            api.upload_folder(repo_id=repo, repo_type=kind, folder_path=str(out / kind), allow_patterns=plan[kind],
                              commit_message=f"Add {len(plan[kind])} paper files missing from the Hub")
            print(f"[done] {kind}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", choices=["adapters", "data", "both", "extra", "missing"], default="both")
    ap.add_argument("--stage-only", action="store_true")
    a = ap.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    if a.what == "missing":
        upload_missing(api, a.stage_only)
        return
    if a.what == "extra":
        path, n = stage_extra()
        print(f"[stage] extra {n} files -> {path}", flush=True)
        if not a.stage_only:
            api.upload_large_folder(repo_id=DATA_REPO, repo_type="dataset", folder_path=str(path))
            for name in PAPER_ANALYSIS:
                api.upload_file(path_or_fileobj=str(ROOT / "experiments" / f"{name}.json"),
                                path_in_repo=f"analysis/{name}.json", repo_id=MODEL_REPO, repo_type="model")
            print("[done] extra", flush=True)
        return
    if a.what in ("adapters", "both"):
        path, n, size = stage_adapters()
        print(f"[stage] {n} adapters {size / 2**30:.1f} GiB -> {path}", flush=True)
        if not a.stage_only:
            api.create_repo(MODEL_REPO, repo_type="model", private=False, exist_ok=True)
            api.upload_large_folder(repo_id=MODEL_REPO, repo_type="model", folder_path=str(path),
                                    private=False)
            api.update_repo_settings(MODEL_REPO, repo_type="model", private=False)
            print(f"[done] https://huggingface.co/{MODEL_REPO}", flush=True)
    if a.what in ("data", "both"):
        path, counts, size = stage_data()
        print(f"[stage] data {sum(counts.values())} files {size / 2**30:.2f} GiB -> {path}", flush=True)
        if not a.stage_only:
            api.create_repo(DATA_REPO, repo_type="dataset", private=False, exist_ok=True)
            api.upload_large_folder(repo_id=DATA_REPO, repo_type="dataset", folder_path=str(path),
                                    private=False)
            api.update_repo_settings(DATA_REPO, repo_type="dataset", private=False)
            print(f"[done] https://huggingface.co/datasets/{DATA_REPO}", flush=True)


if __name__ == "__main__":
    main()
