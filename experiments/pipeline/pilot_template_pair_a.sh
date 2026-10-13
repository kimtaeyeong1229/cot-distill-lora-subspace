#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP=experiments/pipeline
REASONING=models/DeepSeek-R1-Distill-Qwen-1.5B
INSTRUCT=models/Qwen2.5-Math-1.5B-Instruct
STUDENT=models/Qwen2.5-0.5B-Instruct
LAMBDAS="${LAMBDAS:-0.0 0.4 1.0}"
LIMIT="${LIMIT:-200}"
P="${EXP}/pairA/pilot"
mkdir -p "${P}"

for lam in ${LAMBDAS}; do
  if [[ ! -f "${P}/teachers/lambda${lam}/model.safetensors" ]]; then
    python "${EXP}/interpolate.py" --reasoning "${REASONING}" --instruct "${INSTRUCT}" \
      --lam "${lam}" --output "${P}/teachers/lambda${lam}" --tokenizer_from instruct
  fi
  alt="${P}/teachers_r1tok/lambda${lam}"
  mkdir -p "${alt}"
  ln -sfn "../../teachers/lambda${lam}/model.safetensors" "${alt}/model.safetensors"
  cp "${P}/teachers/lambda${lam}/config.json" "${P}/teachers/lambda${lam}/generation_config.json" "${alt}/"
  cp "${REASONING}/tokenizer.json" "${REASONING}/tokenizer_config.json" "${alt}/"
done

for template in teachers teachers_r1tok; do
  for lam in ${LAMBDAS}; do
    python "${EXP}/generate.py" --model "${P}/${template}/lambda${lam}" \
      --problems "${EXP}/data/pool.parquet" --limit "${LIMIT}" \
      --output "${P}/${template}_lambda${lam}.jsonl" \
      2>&1 | tee "${P}/${template}_lambda${lam}.log" | grep -E "^\[|pending|Error|error" || true
  done
done

for template in teachers teachers_r1tok; do
  echo "================ template: ${template}"
  python "${EXP}/build_splits.py" --gen_dir "${P}" --pattern "${template}_lambda{lam}.jsonl" \
    --lambdas "${LAMBDAS}" --student "${STUDENT}" --out_dir "${P}/splits_${template}" --report_only
done
