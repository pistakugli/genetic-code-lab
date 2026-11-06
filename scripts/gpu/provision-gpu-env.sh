#!/usr/bin/env bash
# Idempotent GPU environment provisioner for GCL live lanes.
#
# Safe to re-run on any Linux host. Creates the repo-local Python virtualenv
# (.gcl-llm-reader-venv, gitignored), installs the pinned torch stack, warms
# the Hugging Face cache with the lane model, and validates the host with
# scripts/gpu/gpu-smoke.py.
#
# Usage:
#   bash scripts/gpu/provision-gpu-env.sh            # GPU stack (torch cu128)
#   bash scripts/gpu/provision-gpu-env.sh --cpu      # CPU stack (torch cpu)
#
# Environment overrides:
#   GCL_MODEL          model id to warm/download (default: gpt2)
#   GCL_ADAPTER        adapter id for the smoke check (default: gpt2)
#   GCL_DEVICE         device request for the smoke check (default: auto)
#   GCL_SKIP_CACHE_WARM=1   skip the model download/warm step
#   HF_HOME            custom Hugging Face cache location

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The release orchestrator (scripts/benchmark-nn-live-ordered-path-stability.ts)
# auto-detects this exact venv path, and recorded release runs set
# GCL_PYTHON=.gcl-python-gpu/bin/python. Keep the name in sync with it.
VENV_DIR="${REPO_ROOT}/.gcl-python-gpu"
STACK="gpu"
if [[ "${1:-}" == "--cpu" ]]; then
  STACK="cpu"
fi

GCL_MODEL="${GCL_MODEL:-gpt2}"
GCL_ADAPTER="${GCL_ADAPTER:-gpt2}"
GCL_DEVICE="${GCL_DEVICE:-auto}"

echo "[provision] repo root : ${REPO_ROOT}"
echo "[provision] stack     : ${STACK} (requirements/gcl-python${STACK/+gpu/-gpu}.txt)"

# --- 1. GPU preflight (informational; the smoke check makes the final call) ---
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[provision] nvidia-smi:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
  echo "[provision] WARNING: nvidia-smi not found; no NVIDIA GPU is visible."
  if [[ "${STACK}" == "gpu" ]]; then
    echo "[provision]          The cu128 torch wheel still installs and runs, but"
    echo "[provision]          lanes will execute on CPU only. Use --cpu for the smaller wheel."
  fi
fi

# --- 2. Python virtualenv ----------------------------------------------------
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[provision] creating virtualenv at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi
PIP="${VENV_DIR}/bin/pip"
PYTHON="${VENV_DIR}/bin/python"

# --- 3. Pinned requirements --------------------------------------------------
echo "[provision] installing pinned requirements"
if [[ "${STACK}" == "cpu" ]]; then
  "${PIP}" install --quiet -r "${REPO_ROOT}/requirements/gcl-python.txt"
else
  "${PIP}" install --quiet -r "${REPO_ROOT}/requirements/gcl-python-gpu.txt"
fi

# --- 4. Warm the model cache -------------------------------------------------
if [[ "${GCL_SKIP_CACHE_WARM:-0}" == "1" ]]; then
  echo "[provision] skipping cache warm (GCL_SKIP_CACHE_WARM=1)"
else
  echo "[provision] warming model cache for ${GCL_MODEL}"
  "${PYTHON}" -c "
import sys
sys.path.insert(0, '${REPO_ROOT}/tools')
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('${GCL_MODEL}')
AutoModelForCausalLM.from_pretrained('${GCL_MODEL}')
print('[provision] cache warm: ${GCL_MODEL} ready')
"
fi

# --- 5. Environment smoke check ----------------------------------------------
echo "[provision] running gpu-smoke validation (device=${GCL_DEVICE})"
GCL_DEVICE="${GCL_DEVICE}" "${PYTHON}" "${REPO_ROOT}/scripts/gpu/gpu-smoke.py" \
  --device "${GCL_DEVICE}" \
  --adapter "${GCL_ADAPTER}" \
  --model "${GCL_MODEL}" \
  --local-files-only \
  --output-json "${REPO_ROOT}/benchmarks/scratch/gpu-smoke/report.json"

echo "[provision] PROVISIONED: ${VENV_DIR}"
echo "[provision] next: run a live lane, e.g."
echo "[provision]   cd ${REPO_ROOT}/tools && GCL_DEVICE=${GCL_DEVICE} ../${VENV_DIR#${REPO_ROOT}/}/bin/python \\"
echo "[provision]     llm-reader-ordered-path-stability.py --adapter gpt2 --model gpt2 ..."
