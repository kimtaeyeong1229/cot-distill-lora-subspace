#!/usr/bin/env bash
# Pilot: which chat template should every interpolated teacher use?
#
# The endpoints disagree: R1-Distill was trained on <｜User｜>/<｜Assistant｜><think>,
# Qwen2.5-Math-Instruct on ChatML, and the two tokenizers map ids 151644-151649 to
# different special tokens. Interpolated weights must use one template for all
# lambdas, so we sample LIMIT problems at lambda in {0.0, 0.4, 1.0} under each
# template and compare accuracy, CoT length and truncation.
#
#   docker exec cot-distill-lora-bit bash experiments/lambda_lora_bits/pilot_template_pair_a.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP=experiments/lambda_lora_bits
REASONING=models/DeepSeek-R1-Distill-Qwen-1.5B
INSTRUCT=models/Qwen2.5-Math-1.5B-Instruct
STUDENT=models/Qwen2.5-0.5B-Instruct
LAMBDAS="${LAMBDAS:-0.0 0.4 1.0}"
LIMIT="${LIMIT:-200}"

for lam in ${LAMBDAS}; do
  if [[ ! -f "${EXP}/teachers/lambda${lam}/model.safetensors" ]]; then
    python "${EXP}/interpolate.py" --reasoning "${REASONING}" --instruct "${INSTRUCT}" \
      --lam "${lam}" --output "${EXP}/teachers/lambda${lam}" --tokenizer_from instruct
  fi
  # Same weights under the R1 template: symlink the checkpoint, swap the tokenizer.
  alt="${EXP}/teachers_r1tok/lambda${lam}"
  mkdir -p "${alt}"
  ln -sfn "../../teachers/lambda${lam}/model.safetensors" "${alt}/model.safetensors"
  cp "${EXP}/teachers/lambda${lam}/config.json" "${EXP}/teachers/lambda${lam}/generation_config.json" "${alt}/"
  cp "${REASONING}/tokenizer.json" "${REASONING}/tokenizer_config.json" "${alt}/"
done

for template in teachers teachers_r1tok; do
  for lam in ${LAMBDAS}; do
    python "${EXP}/generate.py" --model "${EXP}/${template}/lambda${lam}" \
      --problems "${EXP}/data/pool.parquet" --limit "${LIMIT}" \
      --output "${EXP}/pilot/${template}_lambda${lam}.jsonl" \
      2>&1 | tee "${EXP}/pilot/${template}_lambda${lam}.log" | grep -E "^\[|pending|Error|error" || true
  done
done

for template in teachers teachers_r1tok; do
  echo "================ template: ${template}"
  python "${EXP}/build_splits.py" --gen_dir "${EXP}/pilot" --pattern "${template}_lambda{lam}.jsonl" \
    --lambdas "${LAMBDAS}" --student "${STUDENT}" --out_dir "${EXP}/pilot/splits_${template}" --report_only
done
