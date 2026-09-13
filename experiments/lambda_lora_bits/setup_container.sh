#!/usr/bin/env bash
# Install experiment dependencies inside the cot-distill-lora-bit container.
#   docker exec cot-distill-lora-bit bash experiments/lambda_lora_bits/setup_container.sh
set -euo pipefail
cd "$(dirname "$0")"

# The image's system Python is marked externally managed (PEP 668) and its torch is itself
# pip-installed into /usr/local/lib/python3.12/dist-packages; this container is dedicated
# to the experiment, so install alongside it rather than in a venv.
pip install --no-cache-dir --break-system-packages -r requirements.txt

python - <<'PY'
import torch, transformers, peft, vllm
x = torch.randn(1024, 1024, device="cuda")
print(f"torch {torch.__version__} (cuda {torch.version.cuda}), capability {torch.cuda.get_device_capability()}, matmul ok={bool(torch.isfinite((x @ x).sum()))}")
print(f"vllm {vllm.__version__} | transformers {transformers.__version__} | peft {peft.__version__}")
PY
