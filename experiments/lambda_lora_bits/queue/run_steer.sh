#!/usr/bin/env bash
# One machine's share of the steering sweep: amplify the consensus region, and graft it into the
# student that never had it. Only these two edits decide whether the region behaves like a control
# rather than merely a part whose removal breaks things.
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits
LIMIT="${LIMIT:-200}"; SAMPLES="${SAMPLES:-2}"
# RUNS/EVAL/LOG pick the directories, so a re-run (e.g. the leave-one-out regions) never collides
# with finished results -- eval_ablation.py skips any condition whose output already exists
RUNS="${RUNS:-runs_steer}"; EVAL="${EVAL:-eval_steer}"; LOG="${LOG:-eval_steer}"
for entry in ${CELLS}; do
  tag="${entry%%:*}"; student="${entry#*:}"
  mkdir -p "${E}/${tag}/logs"
  adapters=()
  for v in ${VARIANTS:-amp1 amp2 graft1 graft_rand1 graft_full1}; do
    d="${E}/${tag}/${RUNS}/${v}/adapter"
    [[ -f "${d}/adapter_model.safetensors" ]] && adapters+=("${v}=${d}")
  done
  (( ${#adapters[@]} == 0 )) && { echo "[$(date '+%F %T')] ${tag}: 없음"; continue; }
  echo "[$(date '+%F %T')] ${tag}: ${#adapters[@]}개 평가"
  maxlen=8192; newtok=7680
  [[ "${student}" == *OLMo* ]] && maxlen=4096 && newtok=3840
  python3 "${E}/eval_ablation.py" --base "models/${student}" --adapters "${adapters[@]}" \
    --data "${E}/data/math500.parquet" --out_dir "${E}/${tag}/${EVAL}" \
    --limit "${LIMIT}" --n_samples "${SAMPLES}" \
    --max_model_len "${maxlen}" --max_new_tokens "${newtok}" \
    >> "${E}/${tag}/logs/${LOG}.log" 2>&1
  chown -R "$(stat -c '%u:%g' .)" "${E}/${tag}" 2>/dev/null || true
done
echo "[$(date '+%F %T')] 완료"
