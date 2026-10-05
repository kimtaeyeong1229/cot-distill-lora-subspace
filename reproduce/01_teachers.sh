#!/usr/bin/env bash
# Stage 1: interpolated teachers, their chain-of-thought pools and the shared training splits.
#
# Paper artifacts: the training data of every student; the teacher think rate and teacher markers
# (Figure 2 top; Tables 1, 6, 10, 13 and 14) are counted from these generations in reproduce/06_analysis.sh.
# Outputs: experiments/lambda_lora_bits/{sweep,pairB,pairD}/{teachers,cot,splits}/
#          (pair A = sweep/, pair B = pairB/, pair C = pairD/)
#
# Run inside the container: docker exec cot-distill-lora-bit bash reproduce/01_teachers.sh
# PILOT=1 also re-runs the chat-template pilots that chose TEMPLATE=reasoning for every pair.
set -euo pipefail
cd "$(dirname "$0")/.."
E=experiments/lambda_lora_bits

if [[ "${PILOT:-0}" == 1 ]]; then
  bash "${E}/pilot_template_pair_a.sh"
  PAIR=B STAGE=pilot GPU_UTIL=0.90 bash "${E}/run_teachers_pair_bc.sh" > "${E}/pairB/pilot.log" 2>&1
  PAIR=D STAGE=pilot GPU_UTIL=0.90 bash "${E}/run_teachers_pair_bc.sh" > "${E}/pairD/pilot.log" 2>&1
fi

# Pair A: DeepSeek-R1-Distill-Qwen-1.5B -> Qwen2.5-Math-1.5B-Instruct. The shared split was built over
# eight lambdas (0.1 and 0.3 are generated but not used by the paper), so the same list is passed here.
TEMPLATE=reasoning LAMBDAS="0.0 0.1 0.2 0.3 0.4 0.6 0.8 1.0" bash "${E}/run_teachers_pair_a.sh"

# Pair B: Qwen3-4B-Thinking-2507 -> Qwen3-4B-Instruct-2507.
PAIR=B STAGE=full TEMPLATE=reasoning GPU_UTIL=0.90 bash "${E}/run_teachers_pair_bc.sh" > "${E}/pairB/full.log" 2>&1
# Pair C (directory pairD): DeepSeek-R1-Distill-Llama-8B -> Llama-3.1-8B-Instruct.
PAIR=D STAGE=full TEMPLATE=reasoning GPU_UTIL=0.90 bash "${E}/run_teachers_pair_bc.sh" > "${E}/pairD/full.log" 2>&1

# Disjoint data halves of every training set, for the data-only baseline B(lambda) of R(lambda).
for pair in sweep pairB pairD; do
  python "${E}/make_halves.py" "${E}/${pair}/splits"
done
