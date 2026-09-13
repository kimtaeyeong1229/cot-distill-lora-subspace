#!/usr/bin/env bash
# Control: separate information density from token volume.
#
# The main sweep varies both at once — a lambda with denser CoT also has ~5x more
# training tokens. Here the lambda=0.0 data is cut to ~375 problems so its total
# training tokens match lambda=1.0 on the full set, while density stays high.
# A step-matched variant (17 epochs) additionally holds the update count.
set -euo pipefail
cd "$(dirname "$0")/../.."

EXP=experiments/lambda_lora_bits
STUDENT="${STUDENT:-models/Qwen2.5-0.5B-Instruct}"
SPLITS="${SPLITS:-${EXP}/sweep/splits}"
OUT="${EXP}/control_tokens"
N_SUBSET="${N_SUBSET:-375}"
mkdir -p "${OUT}/runs" "${OUT}/bits" "${OUT}/logs" "${OUT}/splits"

stage() { echo "[$(date '+%F %T')] $*"; }

stage "build ${N_SUBSET}-problem subsets (same problems for every lambda)"
python - "${SPLITS}" "${OUT}/splits" "${N_SUBSET}" "${EXP}" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[4])
from common import read_jsonl, write_jsonl

splits, out, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
keep = [r["unique_id"] for r in read_jsonl(splits / "lambda0.0" / "train.jsonl")][:n]
for lam in ("0.0", "1.0"):
    rows = {r["unique_id"]: r for r in read_jsonl(splits / f"lambda{lam}" / "train.jsonl")}
    subset = [rows[i] for i in keep]
    write_jsonl(out / f"lambda{lam}" / "train.jsonl", subset)
    tokens = sum(r["n_tokens"] for r in subset)
    full = sum(r["n_tokens"] for r in rows.values())
    print(f"lambda={lam}: subset {len(subset)} problems, {tokens:,} tokens/epoch "
          f"(full set: {len(rows)} problems, {full:,} tokens/epoch)", flush=True)
PY

for lam in 0.0 1.0; do ln -sfn "$(cd "${SPLITS}/lambda${lam}" && pwd)/held.jsonl" "${OUT}/splits/lambda${lam}/held.jsonl"; done

run () {  # run <lambda> <epochs> <tag>
  local lam="$1" epochs="$2" name="$3"
  stage "train ${name} (lambda=${lam}, ${epochs} epochs)"
  python "${EXP}/train_lora.py" --base "${STUDENT}" --train "${OUT}/splits/lambda${lam}/train.jsonl" \
    --output "${OUT}/runs/${name}" --epochs "${epochs}" --seed 0 > "${OUT}/logs/train_${name}.log" 2>&1
  if [[ ! -f "${OUT}/bits/${name}.json" ]]; then
    stage "bits ${name}"
    python "${EXP}/measure_bits.py" --base "${STUDENT}" --run_dir "${OUT}/runs/${name}" \
      --train "${OUT}/splits/lambda${lam}/train.jsonl" --held "${OUT}/splits/lambda${lam}/held.jsonl" \
      --cross "lambda0.0=${SPLITS}/lambda0.0/held.jsonl" "lambda1.0=${SPLITS}/lambda1.0/held.jsonl" \
      --cache_dir "${OUT}/cache/base_bits" --output "${OUT}/bits/${name}.json" > "${OUT}/logs/bits_${name}.log" 2>&1
  fi
}

run 0.0 3  "lambda0.0_n${N_SUBSET}_e3"    # tokens matched to lambda=1.0 on the full set
run 1.0 3  "lambda1.0_n${N_SUBSET}_e3"    # same problem count at low density
run 0.0 17 "lambda0.0_n${N_SUBSET}_e17"   # tokens and steps matched to the full lambda=0.0 run

OWNER="$(stat -c '%u:%g' .)"
chown -R "${OWNER}" "${OUT}"
stage "done"
