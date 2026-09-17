#!/usr/bin/env bash
# Calibration: does the bits split call pure random strings memorized?
#
# Random canaries (only memorizable, no transferable content beyond format) are mixed into the
# pair-A training set at the two ends of the spectrum and trained with the main recipe. Bits are
# measured on the real train/held sets (to check the main split is unchanged) and on training vs
# held-out canaries (the ground-truth memorized set).
set -euo pipefail
cd "$(dirname "$0")/../.."
EXP=experiments/lambda_lora_bits
STUDENT="${STUDENT:-models/Qwen2.5-0.5B-Instruct}"
SPLITS="${SPLITS:-${EXP}/sweep/splits}"
OUT="${EXP}/canary"
LAMBDAS="${LAMBDAS:-1.0 0.0}"
N="${N:-100}"
stage() { echo "[$(date '+%F %T')] $*"; }
mkdir -p "${OUT}/logs" "${OUT}/bits" "${OUT}/runs"
for lam in ${LAMBDAS}; do
  data="${OUT}/data/lambda${lam}"
  [[ -f "${data}/meta.json" ]] || python "${EXP}/canary_data.py" --splits "${SPLITS}/lambda${lam}" --out "${data}" --n "${N}"
  run="${OUT}/runs/lambda${lam}_canary"
  stage "train lambda=${lam} + ${N} canaries"
  python "${EXP}/train_lora.py" --base "${STUDENT}" --train "${data}/train_mixed.jsonl" --output "${run}" --seed 0 \
    > "${OUT}/logs/train_lambda${lam}.log" 2>&1
  if [[ ! -f "${OUT}/bits/lambda${lam}_canary.json" ]]; then
    stage "bits lambda=${lam}"
    python "${EXP}/measure_bits.py" --base "${STUDENT}" --run_dir "${run}" \
      --train "${data}/train_real.jsonl" --held "${data}/held.jsonl" \
      --cross "canary_train=${data}/canary_train.jsonl" "canary_held=${data}/canary_held.jsonl" \
      --cache_dir "${EXP}/sweep/cache/base_bits" --output "${OUT}/bits/lambda${lam}_canary.json" \
      > "${OUT}/logs/bits_lambda${lam}.log" 2>&1
  fi
done
stage "canary test done"
