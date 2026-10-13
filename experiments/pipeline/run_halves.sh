#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP=experiments/pipeline
STUDENT="${STUDENT:?set STUDENT}"
TAG="${TAG:?set TAG}"
SPLITS="${SPLITS:?set SPLITS}"
LAMBDAS="${LAMBDAS:-0.0 0.2 0.4 0.6 0.8 1.0}"
HALVES="${HALVES:-halfA halfB}"
TRAIN_MAX_LEN="${TRAIN_MAX_LEN:-8192}"
TRAIN_TOKEN_BUDGET="${TRAIN_TOKEN_BUDGET:-8192}"
SEED="${SEED:-0}"
SUFFIX=""; [[ "${SEED}" != 0 ]] && SUFFIX="_seed${SEED}"

RUNS="${EXP}/${TAG}/runs"; LOGS="${EXP}/${TAG}/logs"
mkdir -p "${RUNS}" "${LOGS}"
for half in ${HALVES}; do
  for lam in ${LAMBDAS}; do
    name="lambda${lam}_${half}${SUFFIX}"
    [[ -f "${RUNS}/${name}/adapter/adapter_model.safetensors" ]] && continue
    echo "[$(date '+%F %T')] train ${TAG} ${name}"
    python "${EXP}/train_lora.py" --base "${STUDENT}" --train "${SPLITS}/lambda${lam}/train_${half}.jsonl" \
      --output "${RUNS}/${name}" --seed "${SEED}" --max_len "${TRAIN_MAX_LEN}" \
      --token_budget "${TRAIN_TOKEN_BUDGET}" > "${LOGS}/train_${name}.log" 2>&1
    chown -R "$(stat -c '%u:%g' .)" "${RUNS}/${name}" 2>/dev/null || true
    chmod -R u+rwX,go+rX "${RUNS}/${name}" 2>/dev/null || true
  done
done
chown -R "$(stat -c '%u:%g' .)" "${EXP}/${TAG}" 2>/dev/null || true
echo "[$(date '+%F %T')] done ${TAG}"
