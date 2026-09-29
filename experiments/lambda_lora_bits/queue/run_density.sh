#!/usr/bin/env bash
# Student reflective-marker density across the whole lambda grid (seed 0), at the edit experiments'
# scale (first 200 MATH-500 problems x 2 samples). The main evaluation keeps only the tail of each
# response, so the density has to come from fresh generations.
#   CELLS="sweep:Qwen2.5-0.5B-Instruct" bash run_density.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits
LIMIT="${LIMIT:-200}"; SAMPLES="${SAMPLES:-2}"; EVAL="${EVAL:-eval_density}"
for entry in ${CELLS}; do
  tag="${entry%%:*}"; student="${entry#*:}"
  mkdir -p "${E}/${tag}/logs"
  adapters=()
  for lam in 0.0 0.2 0.4 0.6 0.8 1.0; do
    d="${E}/${tag}/runs/lambda${lam}_seed0/adapter"
    [[ -f "${d}/adapter_model.safetensors" ]] && adapters+=("lambda${lam}=${d}")
  done
  echo "[$(date '+%F %T')] ${tag}: ${#adapters[@]}개 평가"
  maxlen=8192; newtok=7680
  [[ "${student}" == *OLMo* ]] && maxlen=4096 && newtok=3840
  python3 "${E}/eval_ablation.py" --base "models/${student}" --adapters "${adapters[@]}" \
    --data "${E}/data/math500.parquet" --out_dir "${E}/${tag}/${EVAL}" \
    --limit "${LIMIT}" --n_samples "${SAMPLES}" \
    --max_model_len "${maxlen}" --max_new_tokens "${newtok}" \
    >> "${E}/${tag}/logs/${EVAL}.log" 2>&1 || exit 1
  chown -R "$(stat -c '%u:%g' .)" "${E}/${tag}" 2>/dev/null || true
done
echo "[$(date '+%F %T')] 완료"
