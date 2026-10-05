#!/usr/bin/env bash
# One machine's share of the steering sweep: amplify the consensus region, and graft it into the
# student that never had it. Only these two edits decide whether the region behaves like a control
# rather than merely a part whose removal breaks things.
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=experiments/lambda_lora_bits
source "${E}/config.sh"   # gen_limits, PAPER_SWITCH, PAPER_SEEDS
LIMIT="${LIMIT:-200}"; SAMPLES="${SAMPLES:-2}"
# RUNS/EVAL/LOG pick the directories (e.g. RUNS=runs_edit_s0 EVAL=eval_main_s0 LOG=eval_main_s0);
# eval_ablation.py skips any condition whose output already exists
RUNS="${RUNS:?set RUNS (e.g. runs_edit_s0)}"; EVAL="${EVAL:?set EVAL (e.g. eval_main_s0)}"; LOG="${LOG:-${EVAL}}"
for entry in ${CELLS}; do
  tag="${entry%%:*}"; student="${entry#*:}"
  mkdir -p "${E}/${tag}/logs"
  adapters=()
  for v in ${VARIANTS:?set VARIANTS (adapter names inside RUNS)}; do
    d="${E}/${tag}/${RUNS}/${v}/adapter"
    [[ -f "${d}/adapter_model.safetensors" ]] && adapters+=("${v}=${d}")
  done
  (( ${#adapters[@]} == 0 )) && { echo "[$(date '+%F %T')] ${tag}: no adapters found"; continue; }
  echo "[$(date '+%F %T')] ${tag}: evaluating ${#adapters[@]} adapters"
  gen_limits "${student}"; maxlen="${MAXLEN}"; newtok="${NEWTOK}"
  python3 "${E}/eval_ablation.py" --base "models/${student}" --adapters "${adapters[@]}" \
    --data "${E}/data/math500.parquet" --out_dir "${E}/${tag}/${EVAL}" \
    --limit "${LIMIT}" --n_samples "${SAMPLES}" \
    --max_model_len "${maxlen}" --max_new_tokens "${newtok}" \
    >> "${E}/${tag}/logs/${LOG}.log" 2>&1 || exit 1
  chown -R "$(stat -c '%u:%g' .)" "${E}/${tag}" 2>/dev/null || true
done
echo "[$(date '+%F %T')] done"
