#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
EXP=experiments/lambda_lora_bits
for lam in 1.0 0.0; do
  python "${EXP}/teacher_floor.py" --teacher "${EXP}/teachers_r1tok/lambda${lam}" --cot "${EXP}/sweep/cot/lambda${lam}.jsonl" \
    --splits "${EXP}/sweep/splits/lambda${lam}" --output "${EXP}/pilot_validity/floor_lambda${lam}.json" \
    > "${EXP}/pilot_validity/logs/floor_lambda${lam}.log" 2>&1
done
chown -R "$(stat -c '%u:%g' .)" "${EXP}/pilot_validity"
