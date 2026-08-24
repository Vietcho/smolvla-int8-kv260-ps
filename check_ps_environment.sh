#!/usr/bin/env bash
set -u

echo "===== IDENTITY ====="
hostnamectl 2>/dev/null || true
uname -a
cat /etc/os-release

echo "===== CPU ====="
lscpu
grep -m1 -i '^Features' /proc/cpuinfo || true

echo "===== MEMORY / STORAGE ====="
free -h
grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo
df -h /

echo "===== PYTHON / BUILD ====="
python3 --version || true
python3 -m pip --version || true
gcc --version 2>/dev/null | head -1 || true
g++ --version 2>/dev/null | head -1 || true
cmake --version 2>/dev/null | head -1 || true

echo "===== PYTHON PACKAGES ====="
python3 - <<'PY'
import importlib
for name in ("torch", "transformers", "safetensors", "numpy", "psutil", "PIL", "einops"):
    try:
        module = importlib.import_module(name)
        print(name, "OK", getattr(module, "__version__", "unknown"))
    except Exception as error:
        print(name, "MISSING", repr(error))
try:
    import torch
    print("torch.backends.quantized.supported_engines", torch.backends.quantized.supported_engines)
    print("torch.backends.quantized.engine", torch.backends.quantized.engine)
    print("torch.get_num_threads", torch.get_num_threads())
except Exception:
    pass
PY
