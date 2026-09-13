#!/usr/bin/env bash
# Smoke test of train_lora.py, measure_bits.py and eval_student.py on a tiny dataset
# built from MATH reference solutions, so no teacher generation is needed.
#   docker exec cot-distill-lora-bit bash experiments/lambda_lora_bits/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP=experiments/lambda_lora_bits
SMOKE="${EXP}/smoke"
STUDENT=models/Qwen2.5-0.5B-Instruct
rm -rf "${SMOKE}"
mkdir -p "${SMOKE}"

python - "${EXP}" "${SMOKE}" <<'PY'
import sys
import pandas as pd
sys.path.insert(0, sys.argv[1])
from common import write_jsonl
pool = pd.read_parquet(f"{sys.argv[1]}/data/pool.parquet")
rows = [{"unique_id": int(r.unique_id), "problem": r.problem, "answer": r.answer, "level": r.level, "response": r.solution}
        for r in pool.itertuples()]
write_jsonl(f"{sys.argv[2]}/train.jsonl", rows[:64])
write_jsonl(f"{sys.argv[2]}/held.jsonl", rows[64:80])
PY

python "${EXP}/train_lora.py" --base "${STUDENT}" --train "${SMOKE}/train.jsonl" --output "${SMOKE}/run" \
  --epochs 2 --batch_size 16 --seed 0
python "${EXP}/measure_bits.py" --base "${STUDENT}" --run_dir "${SMOKE}/run" --train "${SMOKE}/train.jsonl" \
  --held "${SMOKE}/held.jsonl" --cross "self=${SMOKE}/held.jsonl" --step_multipliers "0.5 4 32" --fidelity_samples 16 \
  --cache_dir "${SMOKE}/cache" --output "${SMOKE}/bits.json"
python "${EXP}/eval_student.py" --base "${STUDENT}" --include_base --adapters "smoke=${SMOKE}/run/adapter" \
  --data "${SMOKE}/held.jsonl" --out_dir "${SMOKE}/eval" --n_samples 2 --max_new_tokens 1024 --max_model_len 2048
echo SMOKE_OK
