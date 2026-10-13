#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/pipeline
tag="${CELL:?set CELL (pairA/qwen0.5b or pairA/llama1b)}"; student="${STUDENT:?set STUDENT}"
mkdir -p "${E}/${tag}/logs"
for seed in ${SEEDS:-0 1 2}; do
  adapters=()
  for lam in 0.0 1.0 0.2 0.4 0.6 0.8; do
    d="${E}/${tag}/runs/lambda${lam}_seed${seed}/adapter"
    [[ -f "${d}/adapter_model.safetensors" ]] && adapters+=("lambda${lam}_seed${seed}=${d}")
  done
  echo "[$(date '+%F %T')] ${tag} seed ${seed}: evaluating ${#adapters[@]} adapters"
  python3 "${E}/eval_student.py" --base "models/${student}" --adapters "${adapters[@]}" \
    --data "${E}/data/math500.parquet" --out_dir "${E}/${tag}/eval_math500_7680" \
    --n_samples "${SAMPLES:-4}" ${LIMIT:+--limit "${LIMIT}"} --max_new_tokens 7680 --max_model_len 8192 \
    >> "${E}/${tag}/logs/eval_math500_7680.log" 2>&1 || exit 1
done
chown -R "$(stat -c '%u:%g' .)" "${E}/${tag}" 2>/dev/null || true
echo "[$(date '+%F %T')] done"
