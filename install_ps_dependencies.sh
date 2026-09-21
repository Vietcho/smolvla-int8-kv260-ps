#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: activate the Conda environment first: conda activate smolvla_ps" >&2
    exit 1
fi

echo "Python: $(python --version 2>&1)"
python -m pip install --upgrade pip
# Install a prebuilt CPU wheel that is compatible with Cortex-A53. The binary
# requirement prevents an accidental, impractical source build on the board.
python -m pip install --only-binary=:all: \
    torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cpu
# Keep the pinned PyTorch pair and install only missing dependencies.
python -m pip install -r requirements-ps.txt
python -m pip check

python - <<'PY'
import platform
import torch

print("Machine:", platform.machine())
print("PyTorch:", torch.__version__)
print("INT8 engines:", torch.backends.quantized.supported_engines)
if "qnnpack" not in torch.backends.quantized.supported_engines:
    raise SystemExit("ERROR: QNNPACK is not available in this PyTorch build")
PY

PYTHONPATH="$PROJECT_DIR/package/source" python -c \
    "from lerobot.policies.factory import make_pre_post_processors; print('LeRobot/SmolVLA import: OK')"

echo "PS dependencies: OK"
