#!/usr/bin/env bash
# One machine's share of the ablation sweep: every cell of one teacher pair, in sequence.
#
# Each cell evaluates the lambda0 adapter with the pre-threshold consensus region removed, against
# five controls that remove something else of comparable size, plus the untouched lambda0 and lambda1
# adapters for reference. The controls matter more than the treatment here: the consensus region holds
# 30-40% of the update norm, so a control that removes a negligible amount decides nothing.
#
#   CELLS="sweep:Qwen2.5-0.5B-Instruct sweep_llama:Llama-3.2-1B-Instruct" bash run_ablation.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits
source "${E}/config.sh"   # gen_limits, PAPER_SWITCH, PAPER_SEEDS
LIMIT="${LIMIT:-200}"; SAMPLES="${SAMPLES:-2}"
# RUNS/EVAL/LOG pick the directories; eval_ablation.py skips any condition whose output already exists.
# The paper uses only the unedited seed-0 adapters of this script (EVAL=eval_ablation_loo,
# lambda0.0_base.json, read by analyze_graft_beta.py); edited variants are evaluated only if RUNS holds them.
RUNS="${RUNS:-runs_ablation_loo}"; EVAL="${EVAL:-eval_ablation_loo}"; LOG="${LOG:-${EVAL}}"
for entry in ${CELLS}; do
  tag="${entry%%:*}"; student="${entry#*:}"
  mkdir -p "${E}/${tag}/logs"
  adapters=()
  for lam in 0.0 1.0; do
    adapters+=("lambda${lam}_base=${E}/${tag}/runs/lambda${lam}_seed0/adapter")
    for v in rm_pre rm_rand rm_post rm_least rm_energy rm_scale rm_decisive; do
      d="${E}/${tag}/${RUNS}/lambda${lam}_${v}/adapter"
        [[ -f "${d}/adapter_model.safetensors" ]] && adapters+=("lambda${lam}_${v}=${d}")
    done
  done
  echo "[$(date '+%F %T')] ${tag}: ${#adapters[@]}개 평가"
  gen_limits "${student}"; maxlen="${MAXLEN}"; newtok="${NEWTOK}"
  python3 "${E}/eval_ablation.py" --base "models/${student}" --adapters "${adapters[@]}" \
    --data "${E}/data/math500.parquet" --out_dir "${E}/${tag}/${EVAL}" \
    --limit "${LIMIT}" --n_samples "${SAMPLES}" \
    --max_model_len "${maxlen}" --max_new_tokens "${newtok}" \
    >> "${E}/${tag}/logs/${LOG}.log" 2>&1
  chown -R "$(stat -c '%u:%g' .)" "${E}/${tag}" 2>/dev/null || true
done
echo "[$(date '+%F %T')] 완료"
