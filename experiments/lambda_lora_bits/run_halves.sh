#!/usr/bin/env bash
# Data-resample baseline: one cell's twelve half-data adapters.
#
# For every lambda the student is trained twice from the same seed, once on each disjoint half of
# that lambda's training set. Two halves at one lambda differ only in the data, so their subspace
# overlap is the baseline the seed-to-seed overlap could not give: the seed baseline also changed
# the LoRA initialisation, which turned out to carry most of the pre-threshold agreement.
# The treatment is read off the same runs (halfA at lambda0 against halfA at lambda), so both
# quantities come from the same data budget and the same initialisation.
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP=experiments/lambda_lora_bits
STUDENT="${STUDENT:?set STUDENT}"
TAG="${TAG:?set TAG}"
SPLITS="${SPLITS:?set SPLITS}"
LAMBDAS="${LAMBDAS:-0.0 0.2 0.4 0.6 0.8 1.0}"
HALVES="${HALVES:-halfA halfB}"
TRAIN_MAX_LEN="${TRAIN_MAX_LEN:-8192}"
TRAIN_TOKEN_BUDGET="${TRAIN_TOKEN_BUDGET:-8192}"

RUNS="${EXP}/${TAG}/runs"; LOGS="${EXP}/${TAG}/logs"
mkdir -p "${RUNS}" "${LOGS}"
for half in ${HALVES}; do
  for lam in ${LAMBDAS}; do
    name="lambda${lam}_${half}"
    [[ -f "${RUNS}/${name}/adapter/adapter_model.safetensors" ]] && continue
    echo "[$(date '+%F %T')] train ${TAG} ${name}"
    python "${EXP}/train_lora.py" --base "${STUDENT}" --train "${SPLITS}/lambda${lam}/train_${half}.jsonl" \
      --output "${RUNS}/${name}" --seed 0 --max_len "${TRAIN_MAX_LEN}" \
      --token_budget "${TRAIN_TOKEN_BUDGET}" > "${LOGS}/train_${name}.log" 2>&1
    # hand each adapter over as soon as it is written: the trainer leaves it root-owned and mode 600,
    # which makes it invisible to the ssh user that collects results, and waiting for the whole cell
    # to finish would hide every completed run behind the one still training
    chown -R "$(stat -c '%u:%g' .)" "${RUNS}/${name}" 2>/dev/null || true
    chmod -R u+rwX,go+rX "${RUNS}/${name}" 2>/dev/null || true
  done
done
chown -R "$(stat -c '%u:%g' .)" "${EXP}/${TAG}" 2>/dev/null || true
echo "[$(date '+%F %T')] done ${TAG}"
