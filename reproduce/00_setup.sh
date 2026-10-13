#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
E=experiments/pipeline

bash "${E}/setup_container.sh"

[[ -d code/MI-Distillation ]] || command -v git >/dev/null || { echo "clone the grader first (README, Setup)"; exit 1; }
[[ -d code/MI-Distillation ]] || { git clone -q https://github.com/yslanprime/MI-Distillation code/MI-Distillation && git -C code/MI-Distillation checkout -q 0bc95cf87108c974f62f461cd5a1252b8bef1132; }
test -f code/MI-Distillation/evaluation/evaluate.py

[[ -f "${E}/data/pool.parquet" ]] || python "${E}/prepare_data.py"

for m in Qwen2.5-0.5B-Instruct Llama-3.2-1B-Instruct OLMo-2-1B-Instruct \
         DeepSeek-R1-Distill-Qwen-1.5B Qwen2.5-Math-1.5B-Instruct; do
  [[ -f "models/${m}/config.json" ]] || python "${E}/fetch_model.py" "models/${m}"
done

[[ -d models/saes/llama-3.2-1b-131k/layers.11.mlp ]] || \
  python -c "from huggingface_hub import snapshot_download as d; d('EleutherAI/sae-Llama-3.2-1B-131k', local_dir='models/saes/llama-3.2-1b-131k', allow_patterns=['layers.5.mlp/*', 'layers.8.mlp/*', 'layers.11.mlp/*'])"

python - <<'PY'
from huggingface_hub import hf_hub_download
B = "experiments/pipeline/benchmarks"
hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset", local_dir=f"{B}/gsm8k")
hf_hub_download("fingertap/GPQA-Diamond", "test/gpqa_diamond.parquet", repo_type="dataset", local_dir=f"{B}/GPQA-Diamond")
PY

chown -R "$(stat -c '%u:%g' .)" models "${E}" 2>/dev/null || true
