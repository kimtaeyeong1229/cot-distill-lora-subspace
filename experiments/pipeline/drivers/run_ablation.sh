#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/pipeline
source "${E}/config.sh"
LIMIT="${LIMIT:-200}"; SAMPLES="${SAMPLES:-2}"
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
  echo "[$(date '+%F %T')] ${tag}: evaluating ${#adapters[@]} adapters"
  gen_limits "${student}"; maxlen="${MAXLEN}"; newtok="${NEWTOK}"
  python3 "${E}/eval_ablation.py" --base "models/${student}" --adapters "${adapters[@]}" \
    --data "${E}/data/math500.parquet" --out_dir "${E}/${tag}/${EVAL}" \
    --limit "${LIMIT}" --n_samples "${SAMPLES}" \
    --max_model_len "${maxlen}" --max_new_tokens "${newtok}" \
    >> "${E}/${tag}/logs/${LOG}.log" 2>&1
  chown -R "$(stat -c '%u:%g' .)" "${E}/${tag}" 2>/dev/null || true
done
echo "[$(date '+%F %T')] done"
