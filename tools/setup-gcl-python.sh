#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.gcl-python"
PYTHON="${PYTHON:-python3}"

if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$ROOT/requirements/gcl-python.txt"

"$VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import sys

print("python", sys.executable)
for package in ("torch", "transformers", "numpy", "torchao"):
    try:
        print(package, metadata.version(package))
    except metadata.PackageNotFoundError:
        print(package, "NOT_INSTALLED")
PY
