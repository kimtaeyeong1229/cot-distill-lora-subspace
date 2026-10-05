#!/usr/bin/env bash
# Stage 0: environment, problem pool, models and benchmarks.
#
# Paper artifacts: none directly; every later stage needs these inputs.
#
# Run on the host to start the container (the repository is mounted at the same path inside it), then
# run this script inside the container:
#   docker run -dit --name cot-distill-lora-bit --gpus all --ipc=host --net=host \
#     -v "$PWD":"$PWD" -w "$PWD" pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel /bin/bash
#   docker exec cot-distill-lora-bit bash reproduce/00_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
E=experiments/lambda_lora_bits

# Python dependencies (pinned in requirements.txt; versions in README.md).
bash "${E}/setup_container.sh"

# MATH answer grader (git submodule code/MI-Distillation, imported by common.py and eval_bench_edits.py).
git submodule update --init code/MI-Distillation 2>/dev/null || true
test -f code/MI-Distillation/evaluation/evaluate.py

# Problem pool (4,000 MATH train problems, disjoint from MATH-500) and MATH-500.
# Writes experiments/lambda_lora_bits/data/{pool,math500}.parquet (both are tracked in git already).
[[ -f "${E}/data/pool.parquet" ]] || python "${E}/prepare_data.py"

# Students and the pair A endpoints (pairs B and C are downloaded by run_teachers_pair_bc.sh).
for m in Qwen2.5-0.5B-Instruct Llama-3.2-1B-Instruct OLMo-2-1B-Instruct \
         DeepSeek-R1-Distill-Qwen-1.5B Qwen2.5-Math-1.5B-Instruct; do
  [[ -f "models/${m}/config.json" ]] || python "${E}/fetch_model.py" "models/${m}"
done

# Sparse autoencoder for Table 15 (EleutherAI, one TopK SAE per layer on the Llama-3.2-1B MLP output).
[[ -d models/saes/llama-3.2-1b-131k/layers.11.mlp ]] || \
  python -c "from huggingface_hub import snapshot_download as d; d('EleutherAI/sae-Llama-3.2-1B-131k', local_dir='models/saes/llama-3.2-1b-131k', allow_patterns=['layers.5.mlp/*', 'layers.8.mlp/*', 'layers.11.mlp/*'])"

# GSM8K and GPQA-Diamond for Table 3 and Figure 4, at MI-Distillation's DEFAULT_PATHS relative to
# experiments/lambda_lora_bits/benchmarks/:
#   gsm8k/main/test-00000-of-00001.parquet          (openai/gsm8k, config main, test split)
#   GPQA-Diamond/test/gpqa_diamond.parquet          (GPQA diamond subset)
# code/MI-Distillation/scripts/download_assets.py --group benchmarks fetches both into data/raw/.
for f in gsm8k/main/test-00000-of-00001.parquet GPQA-Diamond/test/gpqa_diamond.parquet; do
  [[ -f "${E}/benchmarks/${f}" ]] || echo "missing ${E}/benchmarks/${f} (see the comment above)"
done

chown -R "$(stat -c '%u:%g' .)" models "${E}" 2>/dev/null || true
