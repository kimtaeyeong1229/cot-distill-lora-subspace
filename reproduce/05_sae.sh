#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
E=experiments/pipeline
REST="lambda0.0_rm_decisive lambda0.0_rm_scale lambda1.0_rm_pre graft1 graft_orth1 graft_full1 graft_full2 ampn1 ampn4 ampV"

python3 "${E}/sae_llama.py" check
python3 "${E}/sae_llama.py" select --layer 11
for cell in A B C; do
  python3 "${E}/sae_llama.py" score --layer 11 --cell "${cell}"
  python3 "${E}/sae_llama.py" score --layer 11 --cell "${cell}" --conds ${REST}
done
