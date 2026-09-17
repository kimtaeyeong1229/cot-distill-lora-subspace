#!/usr/bin/env bash
# Pair A: MI-Distillation's 7 benchmarks for the finished students, then the Qwen2.5-3B sweep and its benchmarks.
#
#   docker exec -d cot-distill-lora-bit bash -c "bash experiments/lambda_lora_bits/bench_chain.sh > experiments/lambda_lora_bits/logs_bench_chain.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/../.."
EXP=experiments/lambda_lora_bits
stage() { echo "[$(date '+%F %T')] $*"; }

bench() {  # student_dir base max_model_len
  stage "generate ${1} (${2}, max_model_len=${3})"
  python "${EXP}/bench_generate.py" --student_dir "${EXP}/${1}" --base "${2}" --max_model_len "${3}" \
    2>&1 | grep --line-buffered "^\[gen\]\|^\[skip\]\|Error\|error:\|Traceback"
  stage "score ${1}"
  python "${EXP}/bench_score.py" --student_dir "${EXP}/${1}" 2>&1 | grep --line-buffered "^\[score\]\|Error\|Traceback"
  chown -R "$(stat -c '%u:%g' .)" "${EXP}/${1}"
}

bench sweep          models/Qwen2.5-0.5B-Instruct 16384
bench sweep_llama    models/Llama-3.2-1B-Instruct 16384
bench sweep_olmo2    models/OLMo-2-1B-Instruct    4096
bench sweep_qwen1.5b models/Qwen2.5-1.5B-Instruct 16384

stage "Qwen2.5-3B pair A sweep"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True STUDENT=models/Qwen2.5-3B-Instruct TAG=sweep_qwen3b \
  bash "${EXP}/run_student.sh" || stage "Qwen2.5-3B sweep FAILED"
bench sweep_qwen3b   models/Qwen2.5-3B-Instruct   16384
chown -R "$(stat -c '%u:%g' .)" "${EXP}"
stage "chain done"
