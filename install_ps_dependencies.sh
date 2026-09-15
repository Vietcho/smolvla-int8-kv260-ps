#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: activate the smolvla-ps virtual/Conda environment first." >&2
    exit 1
fi

echo "Python: $(python --version 2>&1)"
python -m pip install --upgrade pip
# Do not pass --upgrade here: keep large ARM PyTorch wheels when they already
# satisfy the constraints and install only missing/incompatible dependencies.
python -m pip install -r requirements-ps.txt
python -m pip check

PYTHONPATH="$PROJECT_DIR/package/source" python -c \
    "from lerobot.policies.factory import make_pre_post_processors; print('LeRobot/SmolVLA import: OK')"

echo "PS dependencies: OK"
