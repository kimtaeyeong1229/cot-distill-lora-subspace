---
license: other
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- peft
- chain-of-thought
- reasoning
- distillation
- model-interpolation
- interpretability
base_model:
- Qwen/Qwen2.5-0.5B-Instruct
- meta-llama/Llama-3.2-1B-Instruct
- allenai/OLMo-2-0425-1B-Instruct
datasets:
- kimtaeyeong1229/cot-distill-lora-subspace
language:
- en
---

# CoT distillation across interpolated teachers: LoRA adapters

This repository holds the LoRA adapters and analysis outputs for the paper on what a student's weights keep when its teacher stops thinking. The paper is [citation pending].

**Question.** Small students often get worse when distilled on a reasoning model's long solutions. A recent remedy mixes a reasoning model (λ = 0) with a short-answer instruct model (λ = 1) by weight interpolation. The ratio λ then tunes the teacher's solution length. At some λ such a mixed teacher abruptly stops thinking: it drops `</think>` and the reflective phrases ("wait", "let me check", ...). This repository asks two things about that point. What changes in the student's weights? And does the changed part carry the way of thinking? We train students with LoRA and use the adapters as a measuring instrument, not as a proposed distillation method. The update is confined to a pair of small matrices, so adapters trained under different teachers can be compared on equal terms.

We use 3 teacher pairs × 3 students × 6 λ values × 3 seeds. Across them, the adapter changes its update direction at the same λ where its own teacher stops thinking. Before that point, the adapters keep sharing a small region of 5–8 of their 16 directions. The edits stored here show that this **shared region** acts like an on/off switch for thinking. Removing it turns thinking off, but removing the same energy elsewhere in the adapter does not. Amplifying it at a fixed norm does not add thinking. Grafting it into a short-answer adapter starts thinking but does not finish the solution.

## Teachers, students, cells

Each teacher is θ(λ) = λ·θ_instruct + (1 − λ)·θ_reasoning. The mix is element-wise over every tensor, done in fp32 and cast to bf16, for λ ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}. The two endpoints of each pair branch from the same base model.

| paper pair | directory | λ = 0 (reasoning) | λ = 1 (instruct) |
|---|---|---|---|
| A | `sweep`, `sweep_llama`, `sweep_olmo2` | DeepSeek-R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B-Instruct |
| B | `pairB/` | Qwen3-4B-Thinking-2507 | Qwen3-4B-Instruct-2507 |
| C | `pairD/` | DeepSeek-R1-Distill-Llama-8B | Llama-3.1-8B-Instruct |

**The paper's pair C is stored under the directory `pairD`.** This is a historical name. No `pairC` directory remains in this repository.

The paper calls each teacher-pair × student combination a "cell".

| | Qwen2.5-0.5B-Instruct | Llama-3.2-1B-Instruct | OLMo-2-0425-1B-Instruct |
|---|---|---|---|
| pair A | `sweep` | `sweep_llama` | `sweep_olmo2` |
| pair B | `pairB/qwen0.5b` | `pairB/llama1b` | `pairB/olmo1b` |
| pair C | `pairD/qwen0.5b` | `pairD/llama1b` | `pairD/olmo1b` |

The interpolated teacher checkpoints are not uploaded. Each one is an element-wise average of two public checkpoints and can be rebuilt with `interpolate.py` in the code repository.

## Training

- **Data.** The problems come from the MATH train split, with problems that overlap MATH-500 removed; 4,000 problems are drawn. Each of the six teachers of a pair solves all 4,000. A solution is kept only if it is complete, correct and fits in 8,192 tokens, and only problems that pass at all six λ are used. 300 of these are held out. The rest form the training set: 2,113 problems for pair A, 2,261 for pair B and 1,015 for pair C. The training texts have the `</think>` tag stripped.
- **LoRA.** r = 16, α = 32, on all seven linear modules (`q,k,v,o,gate,up,down_proj`). Training uses batch 32, learning rate 2e-4 and 3 epochs. The maximum length is 8,192 (OLMo 4,096).
- **Prompt.** The student's chat template is applied to a single user turn: `{problem}\nPlease reason step by step, and put your final answer within \boxed{}.`
- Each cell × λ is trained on the full training set with seeds 0, 1 and 2. For each seed, the training set is also split into two disjoint halves and one adapter is trained per half. These data-half adapters are the baseline (denominator) of the paper's direction-change ratio R(λ). Adapters that share a seed also share the LoRA initialisation, so seed-only differences are not used as a baseline.

## Layout

The nine cell directories have the same structure:

```
<cell>/
  lambda{0.0,0.2,0.4,0.6,0.8,1.0}_seed{0,1,2}/          main runs                       (18 per cell)
  lambda{...}_half{A,B}/                                 data halves, seed 0             (12 per cell)
  lambda{...}_half{A,B}_seed{1,2}/                       data halves, seeds 1 and 2      (24 per cell)
  runs_edit_s{0,1,2}/<variant>/                          edits of that seed's adapters   (18 variants x 3 seeds)
  runs_graft_beta/graft{,_orth,_full}{4,8}/              graft at beta = 4, 8, seed 0    (6 per cell)
analysis/                                                analysis outputs behind the tables
```


The files in each run are:
- trained runs: `adapter_config.json`, `adapter_model.safetensors`, `train_args.json` and `train_summary.json`
- removal and merge edits: `adapter_config.json`, `adapter_model.safetensors` and `ablation.json`, which records the variant, λ, seed, transition λ, leave-one-out flag, dimensions removed and so on
- graft and amplify edits: `adapter_config.json` and `adapter_model.safetensors` only

In total there are 1,026 adapters: 162 main, 324 data-half, 486 edits and 54 β = 4/8 grafts.

### Edit variants and the paper's terms

V is the **shared region**. It is built per layer and module from the pre-transition adapters (those before the teacher's `</think>` rate first falls by more than half): their rank-16 projectors U Uᵀ are averaged, and the eigenvectors with eigenvalue > 0.5 are kept. Each edit uses a V built **leave-one-out**, without the adapter being edited. Removal acts on `lora_B` only, so the rank stays 16. Grafts and merges stack two adapters and are therefore rank 32.

| directory (`runs_edit_s<seed>/…`) | paper term | what it is |
|---|---|---|
| `lambda0.0_rm_pre` | Region removed | ΔW′ = (I − VVᵀ)ΔW on the λ = 0 adapter |
| `lambda0.0_rm_decisive` | Other removed | The same energy removed from the highest-energy directions orthogonal to V |
| `lambda0.0_rm_scale` | Scaled down | The whole update shrunk uniformly to `rm_pre`'s norm |
| `lambda0.0_gs_energy` | Top removed (matched) (Tables 2, 11 and 15) | The top singular components of ΔW removed until the energy matches `rm_pre` |
| `lambda0.0_gs_tau` | Gold-Switch-style removal, τ = 0.6 (Table 3) | The Gold-Switch criterion at τ = 0.6 (Lee et al., 2025) |
| `lambda0.0_rm_spanrand` | Random directions / Random removed (Tables 2 and 11) | Randomly rotated directions of the adapter's own span, taken until the ΔW energy matches `rm_pre` |
| `lambda0.0_rm_post` | Post-switch removed (Table 11) | The post-switch region at `rm_pre`'s dimension (energy not matched) |
| `lambda1.0_rm_pre` | λ = 1: region removed | The same removal applied to the λ = 1 adapter (side-effect control) |
| `lambda0.0_rmseed` | Two-seed region removed | The region found from the other two seeds' λ = 0 adapters only (no interpolated teacher), with eigenvalue > 0.75 |
| `lambda0.0_rmseed_other` | Other removed (two-seed region) (Table 3) | The same energy removed outside that seed region |
| `graft1`, `graft2` | Region graft, β = 1, 2 | ΔW(λ=1) + β·VVᵀΔW(λ=0) |
| `graft_orth1`, `graft_orth2` | Other graft | The part of ΔW(λ=0) orthogonal to V, scaled to add the same energy |
| `graft_full1`, `graft_full2` | Whole graft | All of ΔW(λ=0), scaled to add the same energy |
| `ampn1`, `ampn4` | Amplify, γ = 1, 4 | ΔW + γ·VVᵀΔW, renormalised to the original ‖ΔW‖_F |
| `ampV` | All in region | ΔW projected entirely onto V, renormalised to the original norm |
| `merge0.5` | Merged with λ = 1 | 0.5·ΔW(λ=0) + 0.5·ΔW(λ=1) |

`runs_graft_beta/graft{,_orth,_full}{4,8}` are the region, other and whole grafts at β = 4 and 8, for seed 0 only. The paper leaves them out of its comparison because at these β every condition repeats the same sentence. They are kept as the record behind that statement, summarised in `analysis/graft_beta.json`. The β = 1 and 2 grafts used in the tables are in `runs_edit_s*`.

Two conditions in the paper are generation-time and have no adapter of their own. They are evaluated with the unedited λ = 0 adapter, and their outputs are in the dataset repository:
- `lambda0.0_budget`, the length-matched cut: the response is cut at the cell's mean length after region removal, and the model is then made to answer.
- `lambda0.0_nothink`, the think-skip prompt: a "finished thinking" sentence is put before the answer (Ma et al., 2025b).

### `analysis/`

| file | used for |
|---|---|
| `think_rate.json`, `teacher_density.json` | Teacher `</think>` rate and reflective-marker density per λ (Figure 2, top; Tables 1 and 10) |
| `main_density.json`, `seed_spread.json` | Unedited students on MATH-500, and seed means and SDs for Tables 2, 5, 10, 11 and 17 and Figure 3 |
| `phi_resample_s012.json` | R(λ) with numerator and denominator, seeds 0–2 (Figure 2, middle; Appendix B) |
| `alignment_s012.json`, `alignment_markers_s012.json` | Interval-wise correlation of R with the teacher's `</think>` rate and its marker density, with exact permutation p (Table 12) |
| `phi_xseed.json` | Same-seed and cross-seed overlap (why seed-only pairs are not a baseline) |
| `consensus_pre.json`, `consensus_post.json` | Shared-region dimensions and coverage, pre and post transition (Table 16) |
| `main_edits.json` | Removal, graft and amplify edits on MATH-500 (Tables 2, 11 and 17; Appendix C) |
| `graft_beta.json` | Grafts at β = 2, 4, 8, seed 0 (why β = 4, 8 are excluded; Appendix C) |
| `sae_llama.json` | Thinking-feature activations from the public Llama-3.2-1B SAE (Table 13, bottom) |
| `ood.json` | GSM8K and GPQA-Diamond (Table 3, top; Figure 4) |
| `baselines.json` | Ways to turn thinking off, compared (Tables 2, 3 and 11) |
| `base_students.json` | Untrained students (Table 3, "Untrained") |
| `seed_region_overlap.json` | Share of the shared region spanned by the two-seed region, 54–60% (Section 6.2) |
| `lambda0_retention.json` | Share of the λ = 0 region kept at each λ, against a same-teacher half-data reference (Figure 2, bottom) |
| `pairA_crossseed.json`, `pairA_seed_ratio.json`, `two_seed_floor.json` | Pair A held vs switch contrast without a shared LoRA initialization, per seed pairing, and floor-corrected (Table 1 caption, Appendix B) |
| `agreement_normalised.json` | Region kept relative to the whole update's cross-seed overlap and to same-λ self-agreement (Section 4.3) |
| `alignment_length_s012.json`, `think_rate_completed.json` | Interval correlation of R with the drop in training length, and the think rate on completed and training solutions (Table 12) |
| `symmetric_R.json`, `switch_regression.json`, `c13_additions.json` | Symmetric-normalisation R, switch-indicator regression of region loss, pair B loss per unit length and λ = 1 vs untrained (Appendices B and C) |
| `region_anatomy.json` | Region dimension and share of update energy by module type and depth (Table 4, Appendix A) |
| `energy_match.json` | Energy match of the removal controls and amplitude gain of the amplification edits, read from the edited adapters (Tables 2, 5 and 8) |
| `paired_bootstrap.json`, `paired_bootstrap_grades.json` | Problem-level paired bootstrap of the removal against its controls, and the per-problem grades it reads (Table 15; Section 5.1; Appendix C) |
| `review_stats.json` | Seed-level graft recovery, amplification, interval and SAE-subset statistics (Tables 2, 5 and 15; Section 5) |
| `removal_vs_untrained.json` | Region removal against the untrained student, per problem (Table 7) |
| `teacher_structure.json`, `student_structure.json` | Structure of the teacher's training solutions and of the student responses per edit (Tables 6 and 13) |
| `region_logit_lens.json` | Region directions read through the unembedding (Appendix C) |
| `supplementary_stats.json` | Teacher think rate on the training subset, long OLMo training examples, two-seed region energy share (Appendices A and C) |
| `examples.json` | Example outputs chosen by a fixed rule (Table 14) |
| `coverage_by_lambda.json`, `length_confound.json` | Coverage of the λ = 0 region at every λ, and every interval classed by the teacher with length, region kept and coverage changes (Tables 1 and 9; Appendix B) |
| `threshold_sensitivity.json`, `top_overlap.json` | The region rebuilt at other eigenvalue cuts and the two-seed overlap (Table 16; Appendix A), and the top singular directions taken by top removal (Appendix A) |
| `edit_stats.json`, `degeneration.json` | Per-cell edit effects with seed spread (Table 15) and responses ending in a loop (Table 17) |
| `reproducibility.lock.json` | Hub commits and file hashes of the downloaded checkpoints, plus the software environment |


## Usage

`adapter_config.json` records a local path in `base_model_name_or_path`, so load the base model by its Hub id and attach the adapter:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "meta-llama/Llama-3.2-1B-Instruct"   # must match the cell's student
tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(
    base, "kimtaeyeong1229/cot-distill-lora-subspace",
    subfolder="sweep_llama/lambda0.0_seed0",          # or e.g. "sweep_llama/runs_edit_s0/lambda0.0_rm_pre"
)

problem = "What is the smallest positive integer n such that 3n is a perfect square?"
msgs = [{"role": "user", "content": f"{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
out = model.generate(ids, max_new_tokens=2048, do_sample=True, temperature=0.6, top_p=0.95)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

The subspace analysis needs only the weights, not a GPU. Per module, ΔW = (α/r)·B·A, where `lora_A` and `lora_B` are stored in `adapter_model.safetensors`. To serve the rank-32 graft and merge adapters with vLLM, set `--max_lora_rank 32`.

## Evaluation settings (as in the paper)

- **Sampling.** Generation uses temperature 0.6 and top-p 0.95. The maximum is 8,192 new tokens for teachers and 7,680 for students (OLMo 3,840).
- **MATH-500.** Students are tested on the first 200 problems, which match the full set's difficulty and subject mix. Each seed's adapter answers every problem twice. Tables report the mean ± SD of the per-seed values.
- **Accuracy.** Accuracy is pass@1. It is also reported over completed (non-truncated) responses only, so that truncation is not counted as a wrong answer.
- **Thinking.** For teachers, thinking is the share of responses containing `</think>`. For students, it is the density of reflective markers per 1,000 words in the first response. Token count and truncation rate are reported alongside. Under repetition-prone conditions, the density is taken over the first 300 words.
- **Out of distribution.** The MATH-trained adapters are tested unchanged on GSM8K (first 200 problems) and GPQA-Diamond (all 198) with the same settings.
- **SAE.** For the Llama student only, thinking features are read with the public `EleutherAI/sae-Llama-3.2-1B-131k`.
- **Untrained baseline.** Untrained students are evaluated the same way, varying the sampling seed instead of the training seed.

The generations behind these numbers are in the [dataset repository](https://huggingface.co/datasets/kimtaeyeong1229/cot-distill-lora-subspace). GPQA-Diamond records there keep only per-item correctness, token count and finish reason. Question and response text are omitted at the request of the GPQA authors.

## Limitations

- LoRA is the measuring instrument here, not the usual way reasoning is distilled. The results cover off-policy distillation with rank-16 LoRA only. Whether full fine-tuning, on-policy distillation or larger students form the same region is not known.
- The students are 1B parameters or smaller, and training is on MATH only. The only evidence outside mathematics is GPQA-Diamond, and it is read through markers and length, not accuracy.
- Teacher family and scale change together across pairs A, B and C, so no claim is made about scale.
- "Thinking" is measured by a reflective-marker proxy. The internal re-check uses an SAE and was done only for Llama. Those features were selected by a text contrast of teacher solutions, so they may capture style rather than reasoning.
- The finding that amplifying does not add thinking cannot fully rule out a ceiling effect, because 51–87% of λ = 0 responses already hit the generation limit.
- Thinking and length are separated only at generation time, by the length-matched cut. There is no training-time control that uses short solutions of matched length.
- The edits turn thinking off, but the edited students are not better than the untrained students. They should not be read as a recipe for better models.

## Licensing

Each adapter is subject to the licence of its base student and to the terms of the teacher whose outputs it was trained on:
- Qwen2.5-0.5B-Instruct and OLMo-2-0425-1B-Instruct are released under Apache-2.0.
- Adapters on Llama-3.2-1B-Instruct are subject to the Llama 3.2 Community License.
- The pair C teachers (`pairD`: DeepSeek-R1-Distill-Llama-8B and Llama-3.1-8B-Instruct) descend from Llama 3.1, so adapters trained on their outputs also carry the Llama 3.1 Community License conditions.

Built with Llama.

## Citation

The citation will be added when the paper is public. Code: [github.com/kimtaeyeong1229/cot-distill-lora-subspace](https://github.com/kimtaeyeong1229/cot-distill-lora-subspace).
