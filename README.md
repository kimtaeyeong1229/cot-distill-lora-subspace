# What a student's weights keep when its teacher stops thinking

Code and result files for the paper's experiments.

A teacher is built by interpolating the weights of a reasoning model (λ = 0) and a short-answer instruct
model (λ = 1). Past some λ the teacher stops thinking: it no longer writes `</think>` or reflective phrases.
For three teacher pairs and three small students we distil each teacher's chain-of-thought into a LoRA
adapter. LoRA serves here as a measuring instrument that gathers the update in a few comparable directions,
not as a recommended way to distil. Adapters that differ only in λ are compared against adapters that
differ only in their training data, so that only the teacher's effect remains. At the λ where the teacher
stops thinking the adapters change direction, and the adapters trained before that point share 5-8 of
their 16 update directions. Editing that shared region in weight space (removing it, grafting it into the
λ = 1 adapter, or raising its weight at a fixed norm) shows that thinking needs the region to turn on but
that the region does not carry a solution to its end. The same holds on GSM8K and GPQA-Diamond.

## Contents

```
README.md
reproduce/                    one script per pipeline stage (00-07), each lists the paper artifacts it produces
experiments/
  *.json                      result files; every number in the paper's tables and figures comes from these
  lambda_lora_bits/           code, and the per-cell runs and evaluations it reads and writes
    config.py, config.sh      shared constants: cells, lambda grid, seeds, switch lambdas, generation limits,
                              reflective-marker pattern, REPO_ROOT / RESULTS_DIR
    common.py                 prompt format, MATH grader (via code/MI-Distillation), data I/O, training loss
    drivers/                  shell drivers for the edits and evaluations, called by reproduce/
    data/                     problem pool (4,000 MATH train problems) and MATH-500
    sweep*/, pairB/, pairD/   pair A, B and C (see "Names" below): teacher splits, per-cell runs and evaluations
    sae_llama/                SAE scores of the Llama students (Table 12)
    hf_cards/                 cards of the Hugging Face releases
code/MI-Distillation/         git submodule; its answer extraction and grader score MATH, GSM8K and GPQA
materials/main_figure_pptx/   Figure 1 (fig1_cde.pptx, written by paper_fig1_cde.py)
overleaf/, papers/            manuscript drafts and reading notes (not needed to run anything)
```

### Names

| Paper | Directory | λ = 0 (reasoning) | λ = 1 (instruct) |
|---|---|---|---|
| pair A | `sweep/` (teacher, splits) | DeepSeek-R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B-Instruct |
| pair B | `pairB/` | Qwen3-4B-Thinking-2507 | Qwen3-4B-Instruct-2507 |
| pair C | **`pairD/`** | DeepSeek-R1-Distill-Llama-8B | Llama-3.1-8B-Instruct |

A cell is one (pair, student). The cell directories are `sweep`, `sweep_llama`, `sweep_olmo2` (pair A) and
`pair{B,D}/{qwen0.5b,llama1b,olmo1b}`, for the students Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct and
OLMo-2-0425-1B-Instruct. Result files name cells `A/Qwen` ... `C/OLMo`; some key pair C as `D` (its
directory). `config.py` (`CELL_DIR`, `PAIR_DIR`, `PAIR_KEY`) holds these maps.

Inside a cell:

| Path | What |
|---|---|
| `runs/lambda<λ>_seed<s>/` | LoRA adapter trained on the full training set (seeds 0-2) |
| `runs/lambda<λ>_half{A,B}[_seed<s>]/` | adapter trained on one disjoint half (data-only baseline of R(λ)) |
| `eval_math500/`, `eval_math500_7680/` | MATH-500 generations of the unedited adapters (`_7680`: A/Qwen and A/Llama) |
| `runs_edit_s<s>/<edit>/` | edited adapters built from seed `s` (region built leave-one-out) |
| `eval_main_s<s>/<edit>.json`, `.samples.jsonl.gz` | their MATH-500 summary and generations |
| `runs_graft_beta/`, `eval_graft_beta/` | grafts at β = 2, 4, 8 (seed 0) |
| `eval_gsm8k_s<s>/`, `eval_gpqa_diamond_s<s>/` | GSM8K and GPQA-Diamond generations |
| `eval_ablation_loo/lambda0.0_base.json` | seed-0 λ = 0 reference used by Table 21 |
| `logs/` | training and evaluation logs |

Git tracks the code, the result JSONs, the evaluation summaries and generations of the edited adapters
(`eval_main_s*`, `eval_graft_beta`, `eval_gsm8k_s*`, `eval_gpqa_diamond_s*`), the SAE scores, and the
split ids and statistics. Adapter weights, teacher checkpoints, teacher generations (`cot/`), training splits
(`splits/lambda*/`), the unedited MATH-500 generations (`eval_math500*`) and the untrained-student
generations (`base_eval/`) are not in git. The adapters, teacher generations, splits and student generations
are released on the Hugging Face Hub (a model and a dataset repository, both named
`cot-distill-lora-subspace`; their cards are in `experiments/lambda_lora_bits/hf_cards/`). In the dataset
repository, `teacher_cot/pair{A,B,D}/` and `splits/pair{A,B,D}/` go to `sweep/`, `pairB/` and `pairD/`
(`cot/` and `splits/`), and `student_generations/<cell>/` goes to the cell directory.

## Environment

Every stage runs in one Docker container on a single GPU (the runs used 16-32 GB consumer GPUs), with the
repository mounted at the same path inside the container:

```bash
docker run -dit --name cot-distill-lora-bit --gpus all --ipc=host --net=host \
  -v "$PWD":"$PWD" -w "$PWD" pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel /bin/bash
docker exec cot-distill-lora-bit bash reproduce/00_setup.sh
```

| Package | Version | | Package | Version |
|---|---|---|---|---|
| Python | 3.12.3 | | datasets | 5.0.1 |
| torch | 2.13.0+cu130 | | safetensors | 0.8.0 |
| transformers | 5.17.0 | | numpy | 2.3.5 |
| peft | 0.20.0 | | matplotlib | 3.11.1 |
| vllm | 0.29.0 | | python-pptx | 1.0.2 |
| accelerate | 1.15.0 | | huggingface_hub | 1.31.0 |

`experiments/lambda_lora_bits/requirements.txt` pins the packages installed on top of the image, and
`experiments/reproducibility.lock.json` (`make_lockfile.py`) records the commit hash and weight checksums
of every model used. The CPU analyses and the figures need only the Python standard library, numpy, torch
and safetensors (adapter analyses), and matplotlib (figures).

## Pipeline

```
data/pool.parquet (4,000 MATH train problems)
  -> teachers: interpolate the two endpoints at λ = 0, 0.2, ..., 1 and generate one CoT per problem    01
  -> splits: problems every λ teacher solves completely and correctly; 300 held out; two data halves  01
  -> students: rank-16 LoRA per cell, λ and seed (0-2), and per data half                               02
  -> evaluation: MATH-500 (first 200 problems x 2 samples, 7,680 new tokens, OLMo 3,840)                03
  -> edits: remove / graft / amplify the shared region and controls; evaluate on MATH-500, GSM8K, GPQA  04
  -> SAE features of the Llama students                                                                05
  -> analysis: experiments/*.json                                                                      06
  -> figures                                                                                           07
```

Run the stages in order inside the container (`docker exec cot-distill-lora-bit bash reproduce/0N_*.sh`).
Stages 01-05 need a GPU; 06 and 07 read only files on disk. Generation samples at temperature 0.6 and top-p
0.95, so regenerated text differs from the released generations; the analyses in stage 06 run on the
released generations and adapters as well as on regenerated ones.

Every analysis and figure script resolves its paths from `config.py`: data from
`$REPO_ROOT/experiments/lambda_lora_bits` (default: this repository) and result JSONs from `$RESULTS_DIR`
(default: `experiments/`). They can be run from any directory, and setting `RESULTS_DIR` writes the results
elsewhere without touching the committed ones.

## Paper artifacts

Figure and table numbers are as printed in the paper. "Stage" is the reproduce script that produces the
inputs; every analysis command below is also run, in dependency order, by `reproduce/06_analysis.sh`.
Commands run from the repository root; `E=experiments/lambda_lora_bits`.

| Artifact | Content | Script(s) | Command | Output | Stage |
|---|---|---|---|---|---|
| Figure 1 | schematic (no data) | `paper_fig1_cde.py` | `python $E/paper_fig1_cde.py` | `materials/main_figure_pptx/cde/fig1_cde.pptx` | 07 |
| Figure 2 | teacher think rate, R(λ), coverage of the shared region | `paper_figures.py` | `FIG_OUT=figures python $E/paper_figures.py` | `fig2_switch.pdf` (from `think_rate`, `phi_resample_s012`, `coverage_by_lambda`) | 07 |
| Figure 3 | edits per cell (removal, amplification, graft) | `paper_figures.py` | same | `fig3_states.pdf` (from `seed_spread`, `baselines`) | 07 |
| Table 1 | nine-cell means: GSM8K / GPQA-Diamond, and ways to turn thinking off | `analyze_ood.py`, `analyze_baselines.py`, `analyze_base_students.py` | `python $E/analyze_ood.py`, ... | `ood.json`, `baselines.json`, `base_students.json` | 03, 04 |
| Table 2 | teacher think rate, teacher and student markers across λ, r | `measure_think_rate.py`, `analyze_seed_spread.py` (runs `analyze_main_density.py`), `analyze_teacher_density.py` | `python $E/analyze_seed_spread.py`, then `python $E/analyze_teacher_density.py` | `think_rate.json`, `seed_spread.json` (`table1`), `teacher_density.json` | 01, 03 |
| Table 3 | think rate and R(λ) | `measure_think_rate.py`, `analyze_resample.py` | `python $E/analyze_resample.py` | `think_rate.json`, `phi_resample_s012.json` | 02 |
| Table 4 | interval-drop correlation of R with think rate / teacher markers | `analyze_alignment.py` | `python $E/analyze_alignment.py`; `... --teacher markers` | `alignment_s012.json`, `alignment_markers_s012.json` | 02 |
| Table 5 | shared dims and coverage before / after the switch | `analyze_consensus.py` | `python $E/analyze_consensus.py --side pre` (and `--side post`) | `consensus_pre.json`, `consensus_post.json` | 02 |
| Table 6 | MATH-500 length, truncation, accuracy change, λ = 0 vs 1 | `analyze_seed_spread.py` | `python $E/analyze_seed_spread.py` | `seed_spread.json` (`table2`) | 03 |
| Table 7 | markers and length after removal and controls | `analyze_seed_spread.py` (runs `analyze_main_edits.py`), `analyze_baselines.py` | `python $E/analyze_seed_spread.py`; `python $E/analyze_baselines.py` | `seed_spread.json` (`edits`), `baselines.json` (`gs_energy`, `budget`) | 04 |
| Table 8 | accuracy of the edited λ = 0 adapter | same as Table 7 | same | same | 04 |
| Table 9 | graft at β = 1, recovery | `analyze_seed_spread.py` | same | `seed_spread.json` (`graft1`, `graft_other1`, `graft_whole1`) | 04 |
| Table 10 | graft at β = 2: early markers, completion, accuracy | `analyze_seed_spread.py` | same | `seed_spread.json` (`graft2`, ...) | 04 |
| Table 11 | amplification at a fixed norm | `analyze_seed_spread.py` | same | `seed_spread.json` (`ampn1`, `ampn4`, `ampV`) | 04 |
| Table 12 | SAE thinking features, Llama students | `sae_llama.py`, `analyze_sae_llama.py` | `python $E/analyze_sae_llama.py` | `sae_llama.json` | 05 |
| Table 13 | markers on GSM8K and GPQA-Diamond | `analyze_ood.py` | `python $E/analyze_ood.py` | `ood.json` | 04 |
| Table 14 | GSM8K accuracy and truncation | `analyze_ood.py` | same | `ood.json` | 04 |
| Table 15 | ways to turn thinking off, per cell | `analyze_baselines.py`, `analyze_base_students.py` | `python $E/analyze_baselines.py`; `python $E/analyze_base_students.py` | `baselines.json`, `base_students.json` | 03, 04 |
| Table 16 | data-only baseline B(λ), cross-seed overlaps | `analyze_resample.py`, `analyze_phi_xseed.py`, `analyze_threshold_sensitivity.py` | `python $E/analyze_phi_xseed.py`; `python $E/analyze_threshold_sensitivity.py` | `phi_resample_s012.json` (`baseline`), `phi_xseed.json`, `threshold_sensitivity.json` | 02 |
| Table 17 | eigenvalue-cut sensitivity of the region | `analyze_threshold_sensitivity.py` | same | `threshold_sensitivity.json` | 02 |
| Table 18 | training length vs think rate as predictors of R | `analyze_length_confound.py` | `python $E/analyze_length_confound.py` | `length_confound.json` | 01, 02 |
| Table 19 | energy-matched top removal vs the shared region | `analyze_top_overlap.py`, `analyze_baselines.py` | `python $E/analyze_top_overlap.py` | `top_overlap.json`, `baselines.json` | 02, 04 |
| Table 20 | responses ending in a loop | `analyze_degeneration.py` | `python $E/analyze_degeneration.py` | `degeneration.json`, `seed_spread.json` | 04 |
| Table 21 | grafts at β = 2, 4, 8 (seed 0) | `analyze_graft_beta.py`, `analyze_degeneration.py` | `python $E/analyze_graft_beta.py` | `graft_beta.json`, `degeneration.json` | 03, 04 |
| Table 22 | seed-level ranges and bootstrap intervals of the main edits | `analyze_edit_stats.py` | `python $E/analyze_edit_stats.py` | `edit_stats.json` | 04 |
| Text (Sec. 6, App.) | the two-seed region covers 54-60% of the shared region | `analyze_seed_region_overlap.py` | `python $E/analyze_seed_region_overlap.py` | `seed_region_overlap.json` | 02 |
| App. (environment) | model commits, library versions | `make_lockfile.py` | `python $E/make_lockfile.py` | `reproducibility.lock.json` | 00 |

Outputs without a directory are in `experiments/`. The tables were typeset by hand from these JSONs.

### What each stage runs

| Stage | Script | Main code |
|---|---|---|
| 00 | `reproduce/00_setup.sh` | `setup_container.sh`, `prepare_data.py`, `fetch_model.py` |
| 01 | `reproduce/01_teachers.sh` | `run_teachers_pair_a.sh`, `run_teachers_pair_bc.sh` (`interpolate.py`, `generate.py`, `build_splits.py`), `pilot_template_pair_a.sh` (template pilot), `make_halves.py` |
| 02 | `reproduce/02_train_students.sh` | `run_student.sh` (STAGES=train), `run_halves.sh` (`train_lora.py`) |
| 03 | `reproduce/03_eval_students.sh` | `run_student.sh` (STAGES=eval500, `eval_student.py`), `drivers/run_math500.sh`, `drivers/run_ablation.sh`, `drivers/run_base_eval.sh` (`eval_ablation.py`) |
| 04 | `reproduce/04_edits.sh` | `drivers/make_main_edits.sh`, `make_seed_edits.sh`, `make_baseline_edits.sh` (`ablate_subspace.py`, `steer_subspace.py`, `merge_adapters.py`); `drivers/run_steer.sh` (`eval_ablation.py`), `drivers/run_baselines.sh` (`eval_baselines.py`), `drivers/run_bench_edits.sh` (`eval_bench_edits.py`) |
| 05 | `reproduce/05_sae.sh` | `sae_llama.py` (`check`, `select --layer 11`, `score --cell A/B/C`) |
| 06 | `reproduce/06_analysis.sh` | `measure_think_rate.py`, `analyze_*.py`, `make_lockfile.py` |
| 07 | `reproduce/07_figures.sh` | `paper_figures.py`, `paper_fig1_cde.py` |

### Edit names

| Name in files | Paper |
|---|---|
| `lambda0.0_rm_pre` | shared region removed from the λ = 0 adapter |
| `lambda0.0_rm_decisive` | other directions removed (same energy, orthogonal to the region) |
| `lambda0.0_rm_scale` | whole update scaled down by the same energy |
| `lambda1.0_rm_pre` | shared region removed from the λ = 1 adapter |
| `lambda0.0_gs_energy`, `lambda0.0_gs_tau` | top singular components removed (energy-matched; Gold-Switch τ = 0.6) |
| `lambda0.0_rmseed`, `lambda0.0_rmseed_other` | two-seed λ = 0 region removed; same energy outside it |
| `lambda0.0_budget`, `lambda0.0_nothink` | length-matched cut; think-skip prompt |
| `graft<β>`, `graft_orth<β>`, `graft_full<β>` | graft of the shared region / other directions / whole update into the λ = 1 adapter |
| `ampn1`, `ampn4`, `ampV` | region weight raised by γ = 1, 4 at a fixed norm; whole update moved into the region |
| `merge0.5` | merged with the λ = 1 adapter |

## Checks

- `python -m py_compile` passes on every Python file and `bash -n` on every shell script.
- With `RESULTS_DIR` pointed at a scratch directory, the analyses of stage 06 regenerate their committed JSONs
  from the released adapters and generations byte for byte. `consensus_*.json` and `coverage_by_lambda.json`
  match to 1e-15 (summation order over a set of module names; identical with a fixed `PYTHONHASHSEED`), and
  the committed `consensus_post.json` predates the per-seed fields (`in_loo_by_seed`, `out_by_seed`) that the
  script now also writes. `make_lockfile.py` was not rerun.
- `paper_figures.py` renders Figures 2 and 3 pixel-identical to the version that made the manuscript figures,
  and `paper_fig1_cde.py` rewrites `fig1_cde.pptx` with identical slide contents.

## Other files

`analyze_diagnostic.py` (with `<cell>/eval_halves/`, `experiments/diagnostic.json`) and
`analyze_output_vs_weights.py` (`experiments/output_vs_weights.json`) are follow-up analyses that the paper
does not report. `paper_fig1_pptx.py` draws an earlier layout of Figure 1. `upload_hf.py` stages and uploads
the Hugging Face releases.
