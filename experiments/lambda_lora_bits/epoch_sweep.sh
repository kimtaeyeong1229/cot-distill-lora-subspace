#!/usr/bin/env bash
# How repetition moves the memorisation share.
#
# The control run showed that at a fixed 3 epochs the student-side information
# density explains the split, but repeating the same high-density data 17 times
# pushes it to 0.83. This fills in the epoch axis for both ends of the spectrum
# on the same 375-problem subsets, so the density x repetition surface has more
# than two points.
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP=experiments/lambda_lora_bits
STUDENT="${STUDENT:-models/Qwen2.5-0.5B-Instruct}"
SPLITS="${SPLITS:-${EXP}/control_tokens/splits}"
SWEEP_SPLITS="${SWEEP_SPLITS:-${EXP}/sweep/splits}"
OUT="${EXP}/control_tokens"
EPOCHS="${EPOCHS:-1 6 12}"
EXTRA_EPOCHS_LOW="${EXTRA_EPOCHS_LOW:-17}"   # lambda=1.0 also needs the 17-epoch point
mkdir -p "${OUT}/runs" "${OUT}/bits" "${OUT}/logs"

stage() { echo "[$(date '+%F %T')] $*"; }

run () {  # run <lambda> <epochs>
  local lam="$1" epochs="$2"
  local name="lambda${lam}_n375_e${epochs}"
  if [[ -f "${OUT}/bits/${name}.json" ]]; then stage "skip ${name} (already measured)"; return; fi
  stage "train ${name}"
  python "${EXP}/train_lora.py" --base "${STUDENT}" --train "${SPLITS}/lambda${lam}/train.jsonl" \
    --output "${OUT}/runs/${name}" --epochs "${epochs}" --seed 0 > "${OUT}/logs/train_${name}.log" 2>&1
  stage "bits ${name}"
  python "${EXP}/measure_bits.py" --base "${STUDENT}" --run_dir "${OUT}/runs/${name}" \
    --train "${SPLITS}/lambda${lam}/train.jsonl" --held "${SPLITS}/lambda${lam}/held.jsonl" \
    --cross "lambda0.0=${SWEEP_SPLITS}/lambda0.0/held.jsonl" "lambda1.0=${SWEEP_SPLITS}/lambda1.0/held.jsonl" \
    --cache_dir "${OUT}/cache/base_bits" --output "${OUT}/bits/${name}.json" > "${OUT}/logs/bits_${name}.log" 2>&1
}

for ep in ${EPOCHS}; do run 1.0 "${ep}"; done          # cheap end first
for ep in ${EXTRA_EPOCHS_LOW}; do run 1.0 "${ep}"; done
for ep in ${EPOCHS}; do run 0.0 "${ep}"; done

OWNER="$(stat -c '%u:%g' .)"
chown -R "${OWNER}" "${OUT}"
stage "done"
