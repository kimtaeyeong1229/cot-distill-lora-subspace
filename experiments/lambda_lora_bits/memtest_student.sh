#!/usr/bin/env bash
# Peak-memory check for a new student before a full sweep: train, bits, and vLLM eval
# on the longest lambda=0.0 CoTs, with the production settings.
#   docker exec cot-distill-lora-bit bash experiments/lambda_lora_bits/memtest_student.sh models/Qwen2.5-3B-Instruct qwen3b
set -uo pipefail
cd "$(dirname "$0")/../.."

EXP=experiments/lambda_lora_bits
STUDENT="$1"; TAG="$2"
TOKEN_BUDGET="${TOKEN_BUDGET:-16384}"   # padded tokens per micro-batch; the optimizer batch stays 32 samples
OUT="${EXP}/smoke/memtest_${TAG}"
rm -rf "${OUT}" && mkdir -p "${OUT}"

python - "${EXP}" "${OUT}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from common import read_jsonl, write_jsonl
rows = read_jsonl(f"{sys.argv[1]}/sweep/splits/lambda0.0/train.jsonl")
rows.sort(key=lambda r: -r["n_tokens"])
write_jsonl(f"{sys.argv[2]}/train.jsonl", rows[:64])
write_jsonl(f"{sys.argv[2]}/held.jsonl", read_jsonl(f"{sys.argv[1]}/sweep/splits/lambda0.0/held.jsonl")[:16])
print(f"stress set: 64 longest CoTs, student-label tokens {rows[63]['n_tokens']}..{rows[0]['n_tokens']}")
PY

phase () {  # phase <name> <command...>
  local name="$1"; shift
  local log="${OUT}/${name}.log" mem="${OUT}/${name}.mem"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "${mem}"; sleep 2; done ) &
  local sampler=$!
  "$@" > "${log}" 2>&1; local rc=$?
  kill "${sampler}" 2>/dev/null
  local peak; peak=$(sort -n "${mem}" | tail -1)
  local oom; oom=$(grep -ciE "out of memory|OutOfMemoryError" "${log}")
  echo "PHASE ${name}: exit=${rc} peak_gpu=${peak}MiB oom_lines=${oom}"
  return ${rc}
}

phase train python "${EXP}/train_lora.py" --base "${STUDENT}" --train "${OUT}/train.jsonl" \
  --output "${OUT}/run" --epochs 1 --batch_size 32 --seed 0 --token_budget "${TOKEN_BUDGET}"
python -c "import json; s=json.load(open('${OUT}/run/train_summary.json')); print(f\"  torch peak allocated {s['peak_mem_gb']:.2f} GB, {s['steps']} steps in {s['minutes']:.1f} min\")" 2>/dev/null

phase bits python "${EXP}/measure_bits.py" --base "${STUDENT}" --run_dir "${OUT}/run" \
  --train "${OUT}/train.jsonl" --held "${OUT}/held.jsonl" --step_multipliers "1.5 2" \
  --fidelity_samples 8 --cache_dir "${OUT}/cache" --output "${OUT}/bits.json"

phase eval python "${EXP}/eval_student.py" --base "${STUDENT}" --adapters "stress=${OUT}/run/adapter" \
  --data "${OUT}/held.jsonl" --limit 8 --n_samples 1 --out_dir "${OUT}/eval" \
  --max_new_tokens 8192 --max_model_len 9216
python -c "import json; s=json.load(open('${OUT}/eval/stress.json'))['summary']; print(f\"  eval ok: {s['problems']} problems, avg {s['avg_tokens']:.0f} tokens\")" 2>/dev/null
echo "MEMTEST_DONE ${TAG}"
