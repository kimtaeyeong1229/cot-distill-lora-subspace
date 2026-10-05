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
    sae_llama/                SAE scores of the Llama students (Table 15, bottom)
    hf_cards/                 cards of the Hugging Face releases
  manuscript_audit/           checks every table cell and in-text number of the manuscript against *.json
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
| `eval_ablation_loo/lambda0.0_base.json` | seed-0 λ = 0 reference of the β = 2, 4, 8 grafts (App. C text) |
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

Figure and table numbers are as printed in the current manuscript. LaTeX numbers floats continuously, body
(Figures 1-3, Tables 1-3) then appendix (Tables 4-17, Figures 4-5). "Stage" is the reproduce script that
produces the inputs; every analysis command below is also run, in dependency order, by
`reproduce/06_analysis.sh`, and every figure command by `reproduce/07_figures.sh`. Commands run from the
repository root (any directory works); `E=experiments/lambda_lora_bits`. JSON files are in `experiments/`;
`<cell>` is `A/Qwen` ... `C/OLMo`, and `<P'>/<full>` is the key style that names pair C `D` and the student
in full (`D/OLMo-2-1B`), used by `think_rate`, `phi_resample_s012`, `alignment_*`, `consensus_*`,
`symmetric_R`, `phi_xseed` and `think_rate_completed`.

### Body

| Artifact | Content | Script(s) | Command | Output (keys) | Stage |
|---|---|---|---|---|---|
| Figure 1 | overview schematic (no data) | `paper_fig1_cde.py` | `python $E/paper_fig1_cde.py` | `materials/main_figure_pptx/cde/fig1_cde.pptx`, exported to PDF; caption numbers from `consensus_pre` (`dims`), `energy_match` (`<cell>.mean.gain_*`), `seed_spread` (`edits.ampV.head_markers_per_1k`) | 07 |
| Figure 2 | teacher think rate, R(λ), share of the λ = 0 region kept | `paper_figures.py` (`fig2`); inputs from `measure_think_rate.py`, `analyze_resample.py`, `analyze_lambda0_retention.py` | `FIG_OUT=figures python $E/paper_figures.py` | `fig2_switch.pdf` from `think_rate` (`<A,B,D>.rates`), `phi_resample_s012` (`<P'>/<full>.rows.<λ>.retention`), `lambda0_retention` (`<cell>.kept`, `.chance_kept`) | 07 |
| Table 1 | held and switch intervals: log training-length drop, think-rate change, marker ratio, region kept and loss per unit length (student means) | `analyze_length_confound.py`, then `paper_tables.py`; caption `analyze_pairA_crossseed.py`, `analyze_two_seed_floor.py` | `python $E/analyze_length_confound.py`; `python $E/paper_tables.py` (writes the manuscript's `tables/`); `python $E/analyze_pairA_crossseed.py`; `python $E/analyze_two_seed_floor.py` | `length_confound` (`pairs.<P>.intervals[i].{class,log_len_drop,think,markers,kept_vs_half_reference}`); `pairA_crossseed` (`<A/stu>.switch_over_held_by_seed`); `two_seed_floor` (`floors.{chance,lambda1}`) | 01, 02 |
| Table 2 | edit effects as pair means, one quantity per panel | `analyze_baselines.py`, `analyze_edit_stats.py`, `analyze_review_stats.py`, `analyze_main_edits.py`, `analyze_seed_spread.py`, `analyze_energy_match.py`, then `paper_tables.py` | `python $E/<script>.py` (no arguments); `python $E/paper_tables.py` (writes the manuscript's `tables/`) | `baselines` (`<cell>.{base0,rm_pre,rm_other,gs_energy}.markers`); `edit_stats` (`cells.<cell>.rm_minus_cut`); `review_stats` (`graft_recovery_early`); `main_edits` (`<cell>.{base0,base1,graft2,graft_full2,ampV}`); `seed_spread` (`table2.trunc1`); `energy_match` (`<cell>.mean.gain_ampV`) | 04 |
| Figure 3 | removal, amplification, early markers and completion after the graft, per cell | `paper_figures.py` (`fig3`) | same as Figure 2 | `fig3_states.pdf` from `baselines` (`<cell>.{base0,rm_other,rm_pre}.markers`), `seed_spread` (`table1.0.0`, `table2.trunc1`, `edits.{ampn1,ampn4,graft2,graft_other2}`, `base.base1_head_markers_per_1k`) | 07 |
| Table 3 | nine-cell means: GSM8K / GPQA-Diamond (top), ways to turn thinking off (bottom) | `analyze_ood.py`, `analyze_baselines.py`, `analyze_base_students.py` | `python $E/analyze_ood.py`; `python $E/analyze_baselines.py`; `python $E/analyze_base_students.py` | `ood` (`<gsm8k,gpqa_diamond>.<cell>.{base0,rm_pre,rm_other,base1}`); `baselines` (`<cell>.{base0,rm_pre,rmseed,gs_tau,budget,nothink,rmseed_other,merge0.5,base1}`); `base_students` (`<Model>-Instruct`) | 03, 04 |

### Appendix

| Artifact | Content | Script(s) | Command | Output (keys) | Stage |
|---|---|---|---|---|---|
| Table 4 | region anatomy: size, energy share at λ = 0 and 1 against chance, λ = 0 energy by depth | `analyze_region_anatomy.py` | `python $E/analyze_region_anatomy.py` | `region_anatomy` (`<cell>.overall`, `.by_depth`) | 02 |
| Table 5 | energy audit of region removal: region share of ‖B‖², ratio reached by the other-directions control, shortfall, directions removed | `analyze_energy_match.py` | `python $E/analyze_energy_match.py` | `energy_match` (`<cell>.mean.{region_B,ratio_B,ratio_dW,unreachable,dims_region,dims_other}`) | 04 |
| Table 6 | MATH-500 tokens, truncation and accuracy change from the λ = 0 to the λ = 1 student, teacher-student marker r (top); accuracy at every λ (bottom) | `analyze_main_density.py`, `analyze_main_edits.py`, `analyze_seed_spread.py` (reruns the first two per seed), `analyze_teacher_density.py` | `python $E/analyze_seed_spread.py`; `python $E/analyze_teacher_density.py` | `seed_spread` (`<cell>.table2.{tokens0,tokens1,trunc0,trunc1,d_acc,d_acc_completed}`); `main_density` (`<cell>.<λ>.accuracy`); `teacher_density` (`<P>.corr`, `pooled_corr`) | 01, 03 |
| Table 7 | interval-drop correlation of R with the teacher's think rate, markers and training length; completed and training subsets | `analyze_alignment.py`, `analyze_think_rate_completed.py` | `python $E/analyze_alignment.py`; `... --teacher markers`; `... --teacher length`; `python $E/analyze_think_rate_completed.py` | `alignment_s012`, `alignment_markers_s012`, `alignment_length_s012` (`<P'>/<full>.r`, `_summary.<A,B,D>.{mean_r,p}`, `_summary.pooled`); `think_rate_completed` (`alignment.{completed,train}`) | 01, 02 |
| Table 8 | switch λ, shared dims before / after, coverage, two-seed overlap at cut 0.75 (other cuts in the caption; cuts 0.4 and 0.6 in App. A text) | `analyze_consensus.py`, `analyze_threshold_sensitivity.py` | `python $E/analyze_consensus.py --side pre`; `... --side post`; `python $E/analyze_threshold_sensitivity.py` | `consensus_pre` (`<P'>/<full>.{transition,dims,in_loo,out,chance}`); `consensus_post` (`dims`); `threshold_sensitivity` (`<cell>.shared_region`, `<cell>.two_seed_region`) | 02 |
| Table 9 | every interval classed Held / Switch / After, with length, end think rate, region kept per student, and coverage changes (student means) | `analyze_length_confound.py`, then `paper_tables.py` | `python $E/analyze_length_confound.py` (needs `think_rate`, `teacher_density`, `phi_resample_s012`, `coverage_by_lambda`); `python $E/paper_tables.py` (writes the manuscript's `tables/`) | `length_confound` (`pairs.<P>.intervals[i]`) | 01, 02 |
| Table 10 | region removal against the untrained student | `analyze_removal_vs_untrained.py` | `python $E/analyze_removal_vs_untrained.py` | `removal_vs_untrained` (`cells.<cell>.{rm_pre,lambda1}`, `untrained.<stu>`) | 03, 04 |
| Table 11 | structure of the teacher's training solutions across the held intervals | `analyze_teacher_structure.py` | `python $E/analyze_teacher_structure.py` (needs `teacher_density`, `length_confound`) | `teacher_structure` (`<P>.per_lambda.<λ>`) | 01 |
| Table 12 | region raised by γ, or the whole update moved into it, at a fixed norm; region weight gain | `analyze_seed_spread.py`, `analyze_energy_match.py`, `analyze_review_stats.py` | `python $E/analyze_seed_spread.py`; `python $E/analyze_energy_match.py`; `python $E/analyze_review_stats.py` | `seed_spread` (`edits.{ampn1,ampn4,ampV}`); `energy_match` (`<cell>.mean.gain_*`); `review_stats` (`amplification.nine_cell`, caption) | 04 |
| Table 13 | removal, length-matched cut, other directions, top removal, scale-down: whole-response and early markers, accuracy | `analyze_seed_spread.py`, `analyze_baselines.py`, `analyze_main_edits.py` | `python $E/analyze_seed_spread.py`; `python $E/analyze_baselines.py` | `seed_spread` (`table1.0.0`, `edits.{rm_pre,rm_other,rm_scale}`, `base.base0_accuracy`); `baselines` (`{budget,gs_energy}`); `main_edits` (`<cell>.{base0,lambda0.0_*}.head_markers_per_1k`) | 04 |
| Table 14 | region removal and the β = 2 graft per cell as value ± SD over seeds; nine-cell intervals | `analyze_edit_stats.py`, `analyze_paired_bootstrap.py`, `analyze_review_stats.py`, then `paper_tables.py` | `python $E/analyze_edit_stats.py`; `python $E/analyze_paired_bootstrap.py`; `python $E/analyze_review_stats.py`; `python $E/paper_tables.py` (writes the manuscript's `tables/`) | `edit_stats` (`cells.<cell>.*`, `nine_cell`); `paired_bootstrap` (`nine_cell.{top,cut}.ci95_pp`; per-problem grades cached in `paired_bootstrap_grades.json`); `review_stats` (`cell_interval`, `graft_recovery_early`, `graft_seed_recovery`) | 04 |
| Table 15 | student response structure per edit (top); SAE thinking features of the Llama students (bottom) | `analyze_student_structure.py`; `sae_llama.py`, `analyze_sae_llama.py` | `python $E/analyze_student_structure.py`; `bash reproduce/05_sae.sh`, then `python $E/analyze_sae_llama.py` | `student_structure` (`summary.all.level`); `sae_llama` (`<A,B,C>.<adapter>.{sae,sae_masked}`, `<P>.corr_keyword_sae`); the FVU in App. A is in `$E/sae_llama/check.json` | 03, 04, 05 |
| Table 16 | grafts of the region, other directions and the whole update at β = 1 and 2: markers, completion, accuracy, tokens, recovered gap, loops | `analyze_seed_spread.py`, `analyze_degeneration.py` | `python $E/analyze_seed_spread.py`; `python $E/analyze_degeneration.py` | `seed_spread` (`edits.{graft1,graft_other1,graft_whole1,graft2,graft_other2,graft_whole2}`, `table1`, `base.base1_head_markers_per_1k`); `degeneration` (`<cell>.{base0,base1,rm_pre,graft1,graft2,graft_other2,graft_whole2}`) | 03, 04 |
| Table 17 | example outputs chosen by a fixed rule (A/Qwen, seed 0) | `make_examples_table.py` | `python $E/make_examples_table.py` | `examples` (`rule`, `problem_index`, `students[]`, `teachers[]`) and `$E/examples_table.tex` (the manuscript's `tables/tab25.tex`) | 03, 04 |

### Numbers cited in the text only

| Where | Script | Command | Output |
|---|---|---|---|
| Sec. 4.3, App. B | `analyze_phi_xseed.py` | `python $E/analyze_phi_xseed.py` | `phi_xseed.json` (λ = 0 cross-seed overlap and same-λ self-agreement) |
| Sec. 4.3 | `analyze_agreement_normalised.py` | `python $E/analyze_agreement_normalised.py` (needs `length_confound`, `phi_xseed`) | `agreement_normalised.json` (region kept relative to the whole update's cross-seed overlap and to self-agreement) |
| App. A, App. C | `analyze_supplementary_stats.py` | `python $E/analyze_supplementary_stats.py` (container; needs `length_confound`) | `supplementary_stats.json` (teacher think rate on the training subset, OLMo examples over 4,096 tokens, two-seed region energy share, per-unit loss) |
| App. B | `analyze_symmetric_R.py` | `python $E/analyze_symmetric_R.py` | `symmetric_R.json` (R_sym = T / sqrt(B(0) B(λ))) |
| App. B | `analyze_pairA_seed_ratio.py` | `python $E/analyze_pairA_seed_ratio.py` | `pairA_seed_ratio.json` (pair A held / switch contrast per seed pairing) |
| App. B | `analyze_switch_regression.py` | `python $E/analyze_switch_regression.py` | `switch_regression.json` (switch-indicator regression of region loss) |
| App. B, App. C | `analyze_c13_additions.py` | `python $E/analyze_c13_additions.py` (needs `lambda0_retention`, `length_confound`, `main_edits`) | `c13_additions.json` (pair B loss per unit length; λ = 1 vs untrained) |
| App. C | `analyze_region_logit_lens.py` | `python $E/analyze_region_logit_lens.py` (GPU if available) | `region_logit_lens.json` (region directions read through the unembedding) |
| Sec. 5, Limitations | `analyze_review_stats.py` | `python $E/analyze_review_stats.py` (needs `main_edits`, `sae_llama`, `paired_bootstrap`, `length_confound`, `energy_match`, `seed_spread`, `degeneration`) | `review_stats.json` (`amplification_ratios`, `sae_subsets`, `split_difficulty`, `thresholds`, ...) |
| App. (environment) | `make_lockfile.py` | `python $E/make_lockfile.py` | `reproducibility.lock.json` (model commits, library versions) |

`coverage_by_lambda.json` (`analyze_coverage_by_lambda.py`) is an input of `length_confound.json`, and
`seed_region_overlap.json` (`analyze_seed_region_overlap.py`) equals `threshold_sensitivity.json`
`two_seed_region.0.75.overlap`; neither is printed directly. Tables 1, 2, 9 and 14 are written by
`paper_tables.py` and Table 17 by `make_examples_table.py`; the other tables copy JSON fields and were typeset by
hand. App. A also states the cut-0.4/0.6 rebuild (`threshold_sensitivity`) and the top directions of energy-matched
top removal (`top_overlap`, `<cell>.{dims_top,cover_region,cover_top,chance}`) in its text.

### Manuscript audit

`experiments/manuscript_audit/` checks every table cell, caption number and in-text range of the manuscript
against the JSONs above (`t_tables.py` for every table and its caption numbers, `t01_05.py`, `t02b.py`, `t11_15.py`
and `t22_length.py` for the remaining caption and appendix numbers, `text_checks.py` for the text; `common.py` holds
the table parser and the paths):

```bash
MANUSCRIPT_DIR=<manuscript checkout> bash experiments/manuscript_audit/run_all.sh   # writes audit.log there
```

`MANUSCRIPT_DIR` defaults to `NAACL_cot_distill_lora_subspace/` next to this repository and `RESULTS_DIR` to
`experiments/`. The last line gives the totals; on the current manuscript it reports 1,594 table and caption
values and 191 in-text values or ranges, 0 mismatches.

### What each stage runs

| Stage | Script | Main code |
|---|---|---|
| 00 | `reproduce/00_setup.sh` | `setup_container.sh`, `prepare_data.py`, `fetch_model.py` |
| 01 | `reproduce/01_teachers.sh` | `run_teachers_pair_a.sh`, `run_teachers_pair_bc.sh` (`interpolate.py`, `generate.py`, `build_splits.py`), `pilot_template_pair_a.sh` (template pilot), `make_halves.py` |
| 02 | `reproduce/02_train_students.sh` | `run_student.sh` (STAGES=train), `run_halves.sh` (`train_lora.py`) |
| 03 | `reproduce/03_eval_students.sh` | `run_student.sh` (STAGES=eval500, `eval_student.py`), `drivers/run_math500.sh`, `drivers/run_ablation.sh`, `drivers/run_base_eval.sh` (`eval_ablation.py`) |
| 04 | `reproduce/04_edits.sh` | `drivers/make_main_edits.sh`, `make_seed_edits.sh`, `make_baseline_edits.sh` (`ablate_subspace.py`, `steer_subspace.py`, `merge_adapters.py`); `drivers/run_steer.sh` (`eval_ablation.py`), `drivers/run_baselines.sh` (`eval_baselines.py`), `drivers/run_bench_edits.sh` (`eval_bench_edits.py`) |
| 05 | `reproduce/05_sae.sh` | `sae_llama.py` (`check`, `select --layer 11`, `score --cell A/B/C`) |
| 06 | `reproduce/06_analysis.sh` | `measure_think_rate.py`, `analyze_*.py`, `make_examples_table.py`, `make_lockfile.py` |
| 07 | `reproduce/07_figures.sh` | `paper_figures.py` (Figures 2, 3), `paper_appendix_figures.py` (Figures 4, 5), `paper_fig1_cde.py` (Figure 1) |

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

Run on the merged code (2026-10-05), with `RESULTS_DIR` pointed at a scratch copy of `experiments/*.json` and the
scripts run from another directory:

- `python -m py_compile` passes on every tracked Python file (74) and `bash -n` on every shell script (25).
- Byte-identical to the committed JSONs: `think_rate`, `main_density`, `main_edits`, `seed_spread`,
  `teacher_density`, `baselines`, `base_students`, `graft_beta`, `ood`, `sae_llama`, `degeneration`,
  `edit_stats`, `phi_resample_s012`, `alignment_s012`, `alignment_markers_s012`, `alignment_length_s012`,
  `phi_xseed`, `seed_region_overlap`, `top_overlap`, `threshold_sensitivity`, `region_anatomy`, `energy_match`,
  `symmetric_R`, `think_rate_completed`, `teacher_structure`, `student_structure`, `removal_vs_untrained`,
  `paired_bootstrap` (from the cached `paired_bootstrap_grades.json`), `review_stats`, `region_logit_lens`,
  `examples` (and `examples_table.tex`), and, from the committed inputs, `two_seed_floor`,
  `agreement_normalised` and `c13_additions`.
- Equal up to float summation order (largest difference 6e-15 absolute; 5e-14 for `supplementary_stats`):
  `consensus_pre`, `consensus_post`, `coverage_by_lambda`, `length_confound`, `lambda0_retention`,
  `pairA_seed_ratio`, `pairA_crossseed`, `supplementary_stats`. These sum over a Python set of module names,
  whose order follows `PYTHONHASHSEED`; with `PYTHONHASHSEED=0` the code before the cleanup and the merged code
  write byte-identical files for all eight. The committed `consensus_post.json` predates the per-seed fields
  (`in_loo_by_seed`, `out_by_seed`) that the script also writes. `switch_regression` differs by 3e-15 under
  numpy 2.3 in the container and is byte-identical under the host's numpy 1.26. `make_lockfile.py` was not rerun.
- `paper_figures.py` and `paper_appendix_figures.py` render Figures 2-5 pixel-identical (300 dpi) to the
  manuscript's PDFs; the PDF bytes differ only in the creation date.
- `experiments/manuscript_audit/run_all.sh` reports 2,249 table and caption values and 191 in-text values, 0
  mismatches, against the current manuscript.

## Other files

`analyze_diagnostic.py` (with `<cell>/eval_halves/`, `experiments/diagnostic.json`) and
`analyze_output_vs_weights.py` (`experiments/output_vs_weights.json`) are follow-up analyses that the paper
does not report; `analyze_seed_region_overlap.py` and `analyze_coverage_by_lambda.py` are kept as inputs and
cross-checks (see above). `paper_fig1_pptx.py` draws an earlier layout of Figure 1. `upload_hf.py` stages and uploads
the Hugging Face releases.
