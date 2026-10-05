---
license: other
language:
- en
pretty_name: "CoT distillation across interpolated teachers: teacher CoT, splits and student generations"
task_categories:
- text-generation
tags:
- chain-of-thought
- reasoning
- distillation
- math
- model-interpolation
size_categories:
- 10K<n<100K
---

# CoT distillation across interpolated teachers: teacher CoT, splits and student generations

This repository holds the data half of the paper on what a student's weights keep when its teacher stops thinking. The paper is [citation pending]. The LoRA adapters and analysis outputs are in the companion model repository, [`kimtaeyeong1229/cot-distill-lora-subspace`](https://huggingface.co/kimtaeyeong1229/cot-distill-lora-subspace).

**Question.** A teacher is built by mixing a reasoning model (λ = 0) and a short-answer instruct model (λ = 1) with ratio λ, using weight interpolation. At some λ this teacher abruptly stops thinking: it drops `</think>` and reflective phrases. Small students are distilled from each teacher with LoRA, and the adapters serve as a measuring instrument for what changes in the student's weights at that point. The paper finds that adapters trained before the switch share a small region of directions. That region acts like an on/off switch for thinking: removing it turns thinking off, while grafting it into a short-answer adapter starts thinking but does not finish the solution.

All generations were sampled at temperature 0.6 and top-p 0.95, so they cannot be reproduced exactly even from the same checkpoints. This repository keeps them.

## Pairs and directory names

| paper pair | directory here | λ = 0 (reasoning) | λ = 1 (instruct) | student cells |
|---|---|---|---|---|
| A | `pairA` (teacher), `sweep*` (students) | DeepSeek-R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B-Instruct | `sweep`, `sweep_llama`, `sweep_olmo2` |
| B | `pairB` | Qwen3-4B-Thinking-2507 | Qwen3-4B-Instruct-2507 | `pairB/qwen0.5b`, `pairB/llama1b`, `pairB/olmo1b` |
| C | **`pairD`** | DeepSeek-R1-Distill-Llama-8B | Llama-3.1-8B-Instruct | `pairD/qwen0.5b`, `pairD/llama1b`, `pairD/olmo1b` |

**The paper's pair C is stored under the directory `pairD`.** This is a historical name. In student directory names, `qwen0.5b` / `sweep` is Qwen2.5-0.5B-Instruct, `llama1b` / `sweep_llama` is Llama-3.2-1B-Instruct, and `olmo1b` / `sweep_olmo2` is OLMo-2-0425-1B-Instruct.

λ ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}. Seeds 0, 1 and 2 are the student training seeds.

## Contents

```
teacher_cot/pair{A,B,D}/lambda{λ}.jsonl
splits/pair{A,B,D}/lambda{λ}/{train,held,train_halfA,train_halfB}.jsonl
splits/pair{A,B,D}/{split_ids,split_stats}.json
student_generations/<cell>/eval_math500/lambda{λ}_seed{s}.json          (see note below)
student_generations/<cell>/eval_main_s{s}/<condition>.json
student_generations/<cell>/eval_main_s{s}/<condition>.samples.jsonl.gz
student_generations/<cell>/eval_gsm8k_s{s}/<condition>{.json,.samples.jsonl.gz}
student_generations/<cell>/eval_gpqa_diamond_s{s}/<condition>{.json,.samples.jsonl.gz}
student_generations/untrained/<student>/eval_main_s{s}/base{.json,.samples.jsonl.gz}
```

### Teacher generations: `teacher_cot/`

Each interpolated teacher wrote solutions to the same 4,000 MATH training problems. These problems were drawn from the MATH train split after removing problems that overlap MATH-500. There is one JSON line per problem, with these fields:
- `unique_id`, `problem`, `answer`, `level`, `type`
- `cot_response`: the raw generation, including `</think>` where the teacher wrote it
- `generated_tokens`, `finish_reason`
- `teacher`: a local path that names the λ

The teacher's think rate (the share of responses containing `</think>`) and reflective-marker density are counted on these files. Do not count them on the splits, where the tags are stripped.

### Training splits: `splits/`

A solution is kept only if it is complete, correct and fits in 8,192 tokens. Only problems that pass at all six λ of a pair are used. 300 of them are held out (`held.jsonl`). The rest form the training set (`train.jsonl`): 2,113 problems for pair A, 2,261 for pair B and 1,015 for pair C. Every λ of a pair sees the same problems.

`train_halfA.jsonl` and `train_halfB.jsonl` split the training set into two disjoint halves. The data-half adapters, which form the baseline of the paper's ratio R(λ), are trained on these. The fields are `unique_id`, `problem`, `answer`, `level`, `response` (with the `</think>` tag stripped), `n_tokens` and `teacher_tokens`.

`split_ids.json` lists the train and held ids. `split_stats.json` gives per-λ counts, teacher accuracy, truncation and token totals. For pair A it also lists λ = 0.1 and 0.3, which are not part of the paper.

### Student generations: `student_generations/`

Students generate at temperature 0.6 and top-p 0.95, with at most 7,680 new tokens (OLMo 3,840).

- **`eval_math500/lambda{λ}_seed{s}.json`**: the unedited adapters on MATH-500. The file holds 500 problems × 4 samples in `records`, with per-sample fields `correct`, `tokens` and `truncated` and the first response's text in `response_0`. The paper uses the first 200 problems and the first 2 samples. `truncated` separates an answer that ran out of generation budget from a wrong one.
  These files cover all nine cells, seeds 0–2. For `sweep` and `sweep_llama` the paper reads the re-run at the 7,680-token limit in `eval_math500_7680/` (same format). For `pairD/olmo1b`, the seed-0 files are the 200 problems × 2 samples evaluation of the retrained seed-0 adapters.
- **`eval_main_s{s}/`**: the paper's evaluation, covering the first 200 MATH-500 problems × 2 samples for each edit of seed s's adapters.
  - `<condition>.json` holds the summary: accuracy, average tokens, truncation rate, reflective markers per 1,000 words and `</think>` rate.
  - `<condition>.samples.jsonl.gz` holds the responses, two consecutive lines per problem, with fields `problem`, `text`, `finish_reason` and `tokens`.
- **`eval_gsm8k_s{s}/`** and **`eval_gpqa_diamond_s{s}/`**: the MATH-trained adapters, unchanged, on GSM8K (first 200 problems) and GPQA-Diamond (all 198), each with 2 samples. The conditions are `lambda0.0`, `lambda0.0_rm_pre`, `lambda0.0_rm_decisive` and `lambda1.0`. GSM8K samples carry `question`, `text`, `finish_reason`, `tokens` and `correct`.
- **GPQA-Diamond text is stripped.** The GPQA authors ask that its examples not be posted online. Its `.samples.jsonl.gz` files keep only `item`, `sample`, `finish_reason`, `tokens` and `correct`, and the `.json` summaries contain no question text.
- **`eval_graft_beta/graft{,_orth,_full}{2,4,8}`**: the seed-0 grafts at β = 2, 4, 8 (same format as `eval_main_s*`), behind the model repo's `analysis/graft_beta.json` and `degeneration.json`; `pairB/llama1b` lacks `graft_full{4,8}`.
- **`untrained/<student>/eval_main_s{s}/`**: the untrained students under the same MATH-500 setting. Here `s` is the sampling seed, because there is no adapter seed.

### Condition names and the paper's terms

| condition | paper term |
|---|---|
| `lambda0.0_rm_pre` | Region removed (the shared region V projected out of the λ = 0 adapter, V built leave-one-out) |
| `lambda0.0_rm_decisive` | Other removed (the same energy removed outside V) |
| `lambda0.0_rm_scale` | Scaled down |
| `lambda0.0_gs_energy` | Top removed (matched) (Tables 2, 11 and 12, energy-matched) |
| `lambda0.0_gs_tau` | Gold-Switch-style removal, τ = 0.6 (Table 3) |
| `lambda1.0_rm_pre` | λ = 1 control: region removed (Appendix C) |
| `lambda0.0_budget` | Length-matched cut: the unedited λ = 0 adapter, cut at the region-removed mean length and made to answer |
| `lambda0.0_nothink` | Think-skip prompt: a "finished thinking" sentence placed before the answer |
| `lambda0.0_rmseed` / `lambda0.0_rmseed_other` | Two-seed region removed / Other removed (two-seed region) (Table 3) |
| `graft{1,2}`, `graft_orth{1,2}`, `graft_full{1,2}` | Graft into λ = 1 at β = 1, 2: region / other / whole |
| `ampn1`, `ampn4`, `ampV` | Amplify γ = 1, γ = 4, All in region (norm held fixed) |
| `merge0.5` | Merged with λ = 1 (half-half) |
| `lambda0.0`, `lambda1.0` (GSM8K / GPQA) | The unedited λ = 0 and λ = 1 adapters |


The edited adapters themselves are in the model repository under `<cell>/runs_edit_s{s}/<condition>/`.

**Reflective markers.** The density is the number of matches per 1,000 words of the case-insensitive pattern below, counted on each problem's first sample:

`\b(wait|hmm|actually|but let me|let me check|recheck|verify|reconsider|alternatively|on second thought|let me re|double-check|double check|is that right|hold on)\b`

## Limitations

- The teacher data covers MATH problems only, from three teacher pairs whose family and scale change together.
- "Thinking" on the student side is a reflective-marker proxy. Read it next to token counts and truncation, which are included.
- The GPQA-Diamond records cannot be re-scored or re-read, because their text is withheld.
- Some `teacher` and `adapter` fields hold paths from the original machine. They identify the run and do not point to anything downloadable.

## Licensing

The problems come from MATH (Hendrycks et al., 2021), GSM8K (Cobbe et al., 2021) and GPQA (Rein et al., 2024), and their own terms apply. The generations were produced by the checkpoints listed above:
- Pair C (`pairD`) descends from Llama 3.1 and carries the Llama 3.1 Community License conditions.
- Llama-3.2-1B-Instruct student outputs carry the Llama 3.2 Community License conditions.

Built with Llama.

## Citation

The citation will be added when the paper is public. Code: [github.com/kimtaeyeong1229/cot-distill-lora-subspace](https://github.com/kimtaeyeong1229/cot-distill-lora-subspace).
