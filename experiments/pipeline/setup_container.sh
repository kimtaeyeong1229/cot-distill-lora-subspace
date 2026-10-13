#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

pip install --no-cache-dir --break-system-packages -r requirements.txt

python - <<'PY'
import torch, transformers, peft, vllm
x = torch.randn(1024, 1024, device="cuda")
print(f"torch {torch.__version__} (cuda {torch.version.cuda}), capability {torch.cuda.get_device_capability()}, matmul ok={bool(torch.isfinite((x @ x).sum()))}")
print(f"vllm {vllm.__version__} | transformers {transformers.__version__} | peft {peft.__version__}")
PY
