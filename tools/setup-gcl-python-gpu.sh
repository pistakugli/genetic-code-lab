#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${GCL_GPU_VENV:-$ROOT/.gcl-python-gpu}"
PYTHON="${PYTHON:-python3}"

if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$ROOT/requirements/gcl-python-gpu.txt"

"$VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import sys

import torch

print("python", sys.executable)
for package in ("torch", "transformers", "numpy", "torchao"):
    try:
        print(package, metadata.version(package))
    except metadata.PackageNotFoundError:
        print(package, "NOT_INSTALLED")
print("torch_cuda_version", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the GCL GPU Python environment")
device = torch.cuda.get_device_name(0)
probe = (torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda")).sum().item()
print("cuda_device_0", device)
print("cuda_probe_sum", probe)
PY
