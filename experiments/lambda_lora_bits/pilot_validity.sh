#!/usr/bin/env bash
# Validity pilot on pair A (overleaf/method_defensible.md): generator floor, token categories,
# mismatched-problem control, teacher resample control, repeated canaries.
set -euo pipefail
cd "$(dirname "$0")/../.."
EXP=experiments/lambda_lora_bits
OUT="${EXP}/pilot_validity"
LAMBDAS="${LAMBDAS:-1.0 0.0}"
N_RESAMPLE="${N_RESAMPLE:-200}"
stage() { echo "[$(date '+%F %T')] $*"; }
mkdir -p "${OUT}/logs"

for lam in ${LAMBDAS}; do
  SPL="${EXP}/sweep/splits/lambda${lam}"
  TEACHER="${EXP}/teachers_r1tok/lambda${lam}"
  RES_RAW="${OUT}/resample/lambda${lam}_raw.jsonl"
  RES="${OUT}/resample/lambda${lam}.jsonl"
  if [[ ! -f "${RES}" ]]; then
    stage "resample lambda=${lam}"
    python - "${SPL}/train.jsonl" "${EXP}/data/pool.parquet" "${OUT}/resample/lambda${lam}_problems.parquet" "${N_RESAMPLE}" <<'PY'
import sys, json, random
import pandas as pd
rows = [json.loads(l) for l in open(sys.argv[1])]
ids = [r["unique_id"] for r in random.Random(0).sample(rows, min(600, len(rows)))][: int(sys.argv[4])]
pool = pd.read_parquet(sys.argv[2])
from pathlib import Path; Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
pool[pool["unique_id"].isin(ids)].to_parquet(sys.argv[3])
PY
    python "${EXP}/generate.py" --model "${TEACHER}" --problems "${OUT}/resample/lambda${lam}_problems.parquet" \
      --output "${RES_RAW}" --seed 1 --gpu_memory_utilization 0.80 > "${OUT}/logs/resample_lambda${lam}.log" 2>&1
    python - "${RES_RAW}" "${RES}" <<'PY'
import sys, json
sys.path.insert(0, "experiments/lambda_lora_bits")
from common import clean_response, is_correct
out = []
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("finish_reason") != "stop":
        continue
    resp = clean_response(r["cot_response"])
    if is_correct(r["problem"], resp, r["answer"]):
        out.append({"unique_id": r["unique_id"], "problem": r["problem"], "answer": r["answer"], "response": resp})
with open(sys.argv[2], "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print("resample kept", len(out))
PY
  fi
  if [[ ! -f "${OUT}/teacher_lambda${lam}.json" ]]; then
    stage "teacher floor lambda=${lam}"
    python "${EXP}/pilot_validity.py" teacher --teacher "${TEACHER}" --cot "${EXP}/sweep/cot/lambda${lam}.jsonl" \
      --splits "${SPL}" --resample "${RES}" --resample_raw "${RES_RAW}" --output "${OUT}/teacher_lambda${lam}.json" \
      > "${OUT}/logs/teacher_lambda${lam}.log" 2>&1
  fi
  for pair in "qwen0.5b models/Qwen2.5-0.5B-Instruct sweep" "llama1b models/Llama-3.2-1B-Instruct sweep_llama"; do
    set -- ${pair}
    if [[ ! -f "${OUT}/student_${1}_lambda${lam}.json" ]]; then
      stage "student ${1} lambda=${lam}"
      python "${EXP}/pilot_validity.py" student --student "${2}" --adapter "${EXP}/${3}/runs/lambda${lam}_seed0/adapter" \
        --splits "${SPL}" --resample "${RES}" --output "${OUT}/student_${1}_lambda${lam}.json" \
        > "${OUT}/logs/student_${1}_lambda${lam}.log" 2>&1
    fi
  done
done

# repeated canaries, short rationales, Qwen2.5-0.5B
for rep in 1 4 16; do
  data="${OUT}/canary/rep${rep}"
  [[ -f "${data}/meta.json" ]] || python "${EXP}/canary_data.py" --splits "${EXP}/sweep/splits/lambda1.0" --out "${data}" --n 100 --repeat "${rep}"
  run="${OUT}/canary/run_rep${rep}"
  stage "canary train rep=${rep}"
  python "${EXP}/train_lora.py" --base models/Qwen2.5-0.5B-Instruct --train "${data}/train_mixed.jsonl" --output "${run}" --seed 0 \
    > "${OUT}/logs/canary_train_rep${rep}.log" 2>&1
  [[ -f "${OUT}/canary_rep${rep}.json" ]] || python "${EXP}/pilot_validity.py" canary --student models/Qwen2.5-0.5B-Instruct \
    --adapter "${run}/adapter" --splits "${data}" --output "${OUT}/canary_rep${rep}.json" > "${OUT}/logs/canary_bits_rep${rep}.log" 2>&1
done
chown -R "$(stat -c '%u:%g' .)" "${OUT}"
stage "pilot done"
