#!/usr/bin/env bash
# Stage 5: thinking features of the Llama-3.2-1B students, read with EleutherAI's public SAE (layer 11).
#
# Paper artifacts: Table 13, bottom (via reproduce/06_analysis.sh -> experiments/sae_llama.json) and the SAE
# reconstruction check in the appendix.
# Outputs: experiments/lambda_lora_bits/sae_llama/{check*,select_L11*,A,B,C}/
#
# Run inside the container after stage 4: docker exec cot-distill-lora-bit bash reproduce/05_sae.sh
set -euo pipefail
cd "$(dirname "$0")/.."
E=experiments/lambda_lora_bits
REST="lambda0.0_rm_decisive lambda0.0_rm_scale lambda1.0_rm_pre graft1 graft_orth1 graft_full1 graft_full2 ampn1 ampn4 ampV"

python3 "${E}/sae_llama.py" check                    # reconstruction quality at layers 5, 8 and 11
python3 "${E}/sae_llama.py" select --layer 11        # feature selection on the teachers' own traces
for cell in A B C; do                                # the three Llama cells; C is directory pairD/llama1b
  python3 "${E}/sae_llama.py" score --layer 11 --cell "${cell}"                  # every lambda + rm_pre, graft2, graft_orth2
  python3 "${E}/sae_llama.py" score --layer 11 --cell "${cell}" --conds ${REST}  # the remaining section-5 edits
done
