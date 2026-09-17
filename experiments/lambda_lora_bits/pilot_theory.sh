#!/usr/bin/env bash
# Theory-grounded analysis on pair A: exact generator floor for every lambda, per-token student codelength
# for every student x lambda, and filtering correction pi(x).
set -uo pipefail
cd "$(dirname "$0")/../.."
EXP=experiments/lambda_lora_bits
OUT="${EXP}/pilot_validity"
stage() { echo "[$(date '+%F %T')] $*"; }
LAMS="0.0 0.2 0.4 0.6 0.8 1.0"
for lam in 0.2 0.6 0.8; do
  d="${EXP}/teachers_r1tok/lambda${lam}"
  if [[ ! -f "${d}/model.safetensors" && ! -f "${d}/model.safetensors.index.json" ]]; then
    stage "interpolate ${lam}"
    python "${EXP}/interpolate.py" --reasoning models/DeepSeek-R1-Distill-Qwen-1.5B --instruct models/Qwen2.5-Math-1.5B-Instruct \
      --lam "${lam}" --output "${d}" --tokenizer_from reasoning | tail -1
  fi
done
for lam in ${LAMS}; do
  [[ -f "${OUT}/floor_lambda${lam}.json" ]] && python -c "import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if {'held','train'}<=set(d['sets']) else 1)" "${OUT}/floor_lambda${lam}.json" && continue
  stage "floor ${lam}"
  python "${EXP}/teacher_floor.py" --teacher "${EXP}/teachers_r1tok/lambda${lam}" --cot "${EXP}/sweep/cot/lambda${lam}.jsonl" \
    --splits "${EXP}/sweep/splits/lambda${lam}" --output "${OUT}/floor_lambda${lam}.json" > "${OUT}/logs/floor_lambda${lam}.log" 2>&1
done
for lam in ${LAMS}; do
  for spec in "qwen0.5b models/Qwen2.5-0.5B-Instruct sweep 8192" "llama1b models/Llama-3.2-1B-Instruct sweep_llama 8192" \
              "qwen1.5b models/Qwen2.5-1.5B-Instruct sweep_qwen1.5b 8192" "qwen3b models/Qwen2.5-3B-Instruct sweep_qwen3b 8192" \
              "olmo1b models/OLMo-2-1B-Instruct sweep_olmo2 4096"; do
    set -- ${spec}
    f="${OUT}/student_${1}_lambda${lam}.json"
    [[ -f "${f}" ]] && continue
    stage "student ${1} ${lam}"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python "${EXP}/pilot_validity.py" student --student "${2}" \
      --adapter "${EXP}/${3}/runs/lambda${lam}_seed0/adapter" --splits "${EXP}/sweep/splits/lambda${lam}" --max_len "${4}" \
      --resample "${OUT}/resample/lambda${lam}.jsonl" --output "${f}" > "${OUT}/logs/student_${1}_lambda${lam}.log" 2>&1 || stage "FAILED student ${1} ${lam}"
  done
done
for lam in ${LAMS}; do
  [[ -f "${OUT}/pi_lambda${lam}.json" ]] && continue
  sub="${OUT}/pi/lambda${lam}_problems.parquet"
  mkdir -p "${OUT}/pi"
  python - "${EXP}/sweep/splits/lambda${lam}" "${EXP}/data/pool.parquet" "${sub}" <<'PY'
import sys, json, random
import pandas as pd
d = sys.argv[1]
tr = [json.loads(l) for l in open(f"{d}/train.jsonl")]
ids = {r["unique_id"] for r in random.Random(0).sample(tr, 600)} | {json.loads(l)["unique_id"] for l in open(f"{d}/held.jsonl")}
pool = pd.read_parquet(sys.argv[2]); pool[pool["unique_id"].isin(ids)].to_parquet(sys.argv[3])
PY
  files=()
  for seed in 11 12 13 14; do
    stage "pi ${lam} seed ${seed}"
    python "${EXP}/generate.py" --model "${EXP}/teachers_r1tok/lambda${lam}" --problems "${sub}" \
      --output "${OUT}/pi/lambda${lam}_seed${seed}.jsonl" --seed "${seed}" --gpu_memory_utilization 0.85 > "${OUT}/logs/pi_${lam}_${seed}.log" 2>&1
    files+=("${OUT}/pi/lambda${lam}_seed${seed}.jsonl")
  done
  python "${EXP}/estimate_pi.py" --samples "${files[@]}" --output "${OUT}/pi_lambda${lam}.json"
done

# cross-style: adapter trained on lambda_i scored on the held rationales of lambda_j (KL to a different generator)
for pair in "0.0 1.0" "1.0 0.0"; do
  set -- ${pair}
  for spec in "qwen0.5b models/Qwen2.5-0.5B-Instruct sweep 8192" "llama1b models/Llama-3.2-1B-Instruct sweep_llama 8192" \
              "qwen1.5b models/Qwen2.5-1.5B-Instruct sweep_qwen1.5b 8192" "qwen3b models/Qwen2.5-3B-Instruct sweep_qwen3b 8192" \
              "olmo1b models/OLMo-2-1B-Instruct sweep_olmo2 4096"; do
    tr_lam=$1; ev_lam=$2
    set -- ${spec}
    f="${OUT}/cross_${1}_train${tr_lam}_eval${ev_lam}.json"
    [[ -f "${f}" ]] && continue
    stage "cross ${1} train ${tr_lam} eval ${ev_lam}"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python "${EXP}/pilot_validity.py" student --student "${2}" \
      --adapter "${EXP}/${3}/runs/lambda${tr_lam}_seed0/adapter" --splits "${EXP}/sweep/splits/lambda${ev_lam}" --max_len "${4}" \
      --n_train 200 --output "${f}" > "${OUT}/logs/cross_${1}_${tr_lam}_${ev_lam}.log" 2>&1 || stage "FAILED cross ${1} ${tr_lam} ${ev_lam}"
  done
done
chown -R "$(stat -c '%u:%g' .)" "${OUT}" "${EXP}/teachers_r1tok"
stage "theory analysis done"
