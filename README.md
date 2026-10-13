# Where Do Distilled Students Store Thinking? (code)

This repository reproduces every table and figure of the paper. It contains

- `reproduce/`: the full pipeline in eight stages (`00_setup.sh` to `07_figures.sh`);
- `experiments/pipeline/`: the scripts the stages call (teacher interpolation and generation, LoRA training, evaluation, subspace analysis, edits, SAE read-out, tables and figures);
- `experiments/pipeline/data/`: the 4,000-problem MATH training pool and MATH-500;
- `experiments/*.json`: the analysis results behind every number in the paper.

Trained adapters, edited adapters and the analysis outputs are on the Hugging Face Hub at [kimtaeyeong1229/cot-distill-lora-subspace](https://huggingface.co/kimtaeyeong1229/cot-distill-lora-subspace). Teacher solutions, training splits and student generations are in the dataset [kimtaeyeong1229/cot-distill-lora-subspace](https://huggingface.co/datasets/kimtaeyeong1229/cot-distill-lora-subspace). Both use the same layout as this repository: pairs `pairA`, `pairB` and `pairC`, and students `qwen0.5b`, `llama1b` and `olmo1b`.

## Quick check (CPU, about one minute)

The tables and figures that are computed rather than copied are regenerated from the included results.

```bash
pip install numpy pandas matplotlib
python experiments/pipeline/paper_tables.py      # Tables 1, 2, 9 and 15 -> tables/
python experiments/pipeline/paper_figures.py     # Figures 2 and 3 -> figures/
python experiments/pipeline/paper_appendix_figures.py   # Figure 4 -> figures/
```

The tables match the paper byte for byte. The figures match pixel for pixel when Arial (or the metric-compatible Liberation Sans) is installed.

## Full reproduction (GPU)

### Setup

Stages run inside `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel` with the repository mounted at the same path.

```bash
git clone https://github.com/yslanprime/MI-Distillation code/MI-Distillation
git -C code/MI-Distillation checkout 0bc95cf87108c974f62f461cd5a1252b8bef1132
docker run -dit --name cdl --gpus all --ipc=host -v "$PWD":"$PWD" -w "$PWD" pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel bash
docker exec cdl bash reproduce/00_setup.sh
```

`00_setup.sh` installs `experiments/pipeline/requirements.txt` (vLLM 0.29.0, Transformers 5.17.0, PEFT 0.20.0) and downloads the students, the pair-A endpoints and the Llama-3.2-1B SAE from the Hugging Face Hub. Gated models (Llama) need `HF_TOKEN` in the environment. MI-Distillation provides the MATH answer grader.

**Benchmarks.** `00_setup.sh` also downloads the GSM8K test split (`openai/gsm8k`) and GPQA-Diamond with the answer choices written into each question (`fingertap/GPQA-Diamond`, built from `Idavidrein/gpqa`) into `experiments/pipeline/benchmarks/`. To fetch them without the rest of the setup:

```bash
python -c "from huggingface_hub import hf_hub_download as d; B='experiments/pipeline/benchmarks'; d('openai/gsm8k', 'main/test-00000-of-00001.parquet', repo_type='dataset', local_dir=B+'/gsm8k'); d('fingertap/GPQA-Diamond', 'test/gpqa_diamond.parquet', repo_type='dataset', local_dir=B+'/GPQA-Diamond')"
```

### Stages

| Stage | Command | Produces |
|---|---|---|
| 1 | `bash reproduce/01_teachers.sh` | interpolated teachers (pairs A, B, C at λ = 0, 0.2, ..., 1), their solutions, training splits and data halves |
| 2 | `bash reproduce/02_train_students.sh` | LoRA adapters (3 students × 6 λ × 3 seeds per pair, plus data-half adapters) |
| 3 | `bash reproduce/03_eval_students.sh` | MATH-500 evaluation of every adapter and of the untrained students |
| 4 | `bash reproduce/04_edits.sh` | removed, grafted and amplified adapters and all controls, evaluated on MATH-500, GSM8K and GPQA-Diamond |
| 5 | `bash reproduce/05_sae.sh` | SAE read-out of the Llama students |
| 6 | `bash reproduce/06_analysis.sh` | every `experiments/*.json` and the generated tables |
| 7 | `bash reproduce/07_figures.sh` | Figures 1 to 4 |

Each pair's teachers, solutions and training splits are written to `experiments/pipeline/pair{A,B,C}/`, and each student's adapters and evaluations to `experiments/pipeline/pair{A,B,C}/{qwen0.5b,llama1b,olmo1b}/`. Result files key pairs as `A`, `B` and `C` and cells as `<pair>/<student>` (for example `C/Qwen` or `C/Qwen2.5-0.5B`).

Run stages 0 to 6 with `docker exec cdl bash reproduce/<stage>.sh`. Stage 7 needs only the result JSONs, `python-pptx` and Arial (or Liberation Sans), so it can run on the host. Stages 1 to 4 need a GPU; stage 6 needs the outputs of stages 1 to 5. Training one λ = 0 adapter of a 1B student takes about one hour on a 16 GB GPU.

### Where each result comes from

| Paper | Script | Result file |
|---|---|---|
| Figure 2 | `paper_figures.py` | `think_rate`, `phi_resample_s012`, `lambda0_retention` |
| Figure 3 | `paper_figures.py` | `baselines`, `seed_spread`, `main_edits` |
| Figure 4 | `paper_appendix_figures.py` | `ood` |
| Tables 1, 9 | `analyze_length_confound.py`, `paper_tables.py` | `length_confound`, `pairA_crossseed`, `two_seed_floor` |
| Table 2 | `analyze_baselines.py`, `analyze_edit_stats.py`, `analyze_summary_stats.py`, `paper_tables.py` | `baselines`, `edit_stats`, `summary_stats`, `main_edits` |
| Table 3 | `analyze_ood.py`, `analyze_baselines.py`, `analyze_base_students.py` | `ood`, `baselines`, `base_students` |
| Tables 4, 8 | `analyze_region_anatomy.py`, `analyze_energy_match.py` | `region_anatomy`, `energy_match` |
| Tables 5, 11, 17 | `analyze_seed_spread.py`, `analyze_baselines.py`, `analyze_degeneration.py` | `seed_spread`, `baselines`, `degeneration`, `energy_match` |
| Tables 6, 13 | `analyze_teacher_structure.py`, `analyze_student_structure.py`, `analyze_sae_llama.py` | `teacher_structure`, `student_structure`, `sae_llama` |
| Table 7 | `analyze_removal_vs_untrained.py` | `removal_vs_untrained` |
| Table 10 | `analyze_main_density.py`, `analyze_seed_spread.py` | `main_density`, `seed_spread`, `teacher_density` |
| Table 12 | `analyze_alignment.py`, `analyze_think_rate_completed.py` | `alignment_*_s012`, `think_rate_completed` |
| Table 14 | `make_examples_table.py` | `examples` |
| Table 15 | `analyze_edit_stats.py`, `analyze_paired_bootstrap.py`, `paper_tables.py` | `edit_stats`, `paired_bootstrap`, `summary_stats` |
| Table 16 | `analyze_consensus.py`, `analyze_threshold_sensitivity.py` | `consensus_pre`, `consensus_post`, `threshold_sensitivity` |

Scripts are in `experiments/pipeline/`; result files are `experiments/<name>.json`.

## License

MIT
