#!/usr/bin/env bash
# Headless Kaggle GPU lane driver for the Mistral-7B ordered-path OOD
# replacement-transfer lane.
#
# Modeled on scripts/gpu/kaggle/run-kaggle-gpu-lane.sh, parameterized for
# this lane's kernel, patches dataset, suite path, and longer poll timeout
# (a 7B 4-bit replay with six rescue searches needs hours, not minutes).
#
# One command runs the whole cycle on a free Kaggle GPU session:
#   1. regenerate the patch bundle (origin/main..HEAD),
#   2. upload/refresh the private patches dataset,
#   3. push the GPU kernel (scripts/gpu/kaggle/ordered-path-ood-mistral-lane/kernel.py),
#   4. poll kernel status until it completes,
#   5. download the kernel output (artifacts + log),
#   6. copy the suite back into the repo layout and re-run the independent
#      verifier locally on the downloaded suite.
#
# Authentication (server 2.x): set KAGGLE_API_TOKEN to a token generated at
# https://www.kaggle.com/settings/api (Generate New Token under API), or save
# it to ~/.kaggle/access_token.
#
# Usage:
#   export KAGGLE_API_TOKEN=...        # required
#   bash scripts/gpu/kaggle/ordered-path-ood-mistral-lane/run-kaggle-ordered-path-ood-mistral-lane.sh
#
# Environment overrides:
#   KAGGLE_SLUG          kernel/dataset slug (default gcl-ordered-path-ood-mistral-lane)
#   KAGGLE_USER          account username (default kuglipista)
#   GCL_KAGGLE_TIMEOUT   poll timeout in seconds (default 14400 = 4h)
#   GCL_KAGGLE_OUT       download directory (default /tmp/gcl-kaggle-out)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
KAGGLE_DIR="${REPO_ROOT}/scripts/gpu/kaggle/ordered-path-ood-mistral-lane"
SLUG="${KAGGLE_SLUG:-gcl-ordered-path-ood-mistral-lane}"
USER_NAME="${KAGGLE_USER:-kuglipista}"
TIMEOUT_S="${GCL_KAGGLE_TIMEOUT:-14400}"
OUT_DIR="${GCL_KAGGLE_OUT:-/tmp/gcl-kaggle-out}"

PRIMARY_REL="benchmarks/release/nn-live-ordered-path-stability/ood-replacement-transfer-mistral-7b-2026-11-14"

if [[ -z "${KAGGLE_API_TOKEN:-}" && ! -f "${HOME}/.kaggle/access_token" ]]; then
  echo "ERROR: KAGGLE_API_TOKEN is not set and ~/.kaggle/access_token is missing." >&2
  echo "Generate a token at https://www.kaggle.com/settings/api (Generate New Token under API)." >&2
  exit 2
fi

# Resolve a kaggle CLI that speaks the new auth (2.x). Prefer repo venv, then
# any kaggle on PATH.
if [[ -x "${REPO_ROOT}/.gcl-python-gpu/bin/kaggle" ]]; then
  KAGGLE="${REPO_ROOT}/.gcl-python-gpu/bin/kaggle"
elif command -v kaggle >/dev/null 2>&1; then
  KAGGLE="kaggle"
else
  echo "ERROR: kaggle CLI not found. Install it into .gcl-python-gpu (see below):" >&2
  echo "  ${REPO_ROOT}/.gcl-python-gpu/bin/pip install kaggle>=2.2.5 six requests" >&2
  exit 2
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

echo "[mistral-lane] 1/6 patch bundle -> ${STAGE}"
mkdir -p "${STAGE}/bundle"
git -C "${REPO_ROOT}" format-patch origin/main..HEAD -o "${STAGE}/bundle/" >/dev/null
if ! compgen -G "${STAGE}/bundle/*.patch" >/dev/null; then
  echo "[mistral-lane] ERROR: no commits ahead of origin/main; push or commit the lane first." >&2
  exit 1
fi
cp "${KAGGLE_DIR}/dataset-metadata.json" "${STAGE}/bundle/"
ls "${STAGE}/bundle"

echo "[mistral-lane] 2/6 dataset upload (${USER_NAME}/${SLUG}-patches)"
DATASET_DIR="${STAGE}/bundle"
DATASET_ID="${USER_NAME}/${SLUG}-patches"
DATASET_META="${DATASET_DIR}/dataset-metadata.json"
python3 - "${DATASET_META}" "${DATASET_ID}" <<'PY'
import json, sys
path, dataset_id = sys.argv[1], sys.argv[2]
meta = json.load(open(path))
meta["id"] = dataset_id
meta["title"] = dataset_id.split("/", 1)[1]
json.dump(meta, open(path, "w"), indent=2)
PY
if "${KAGGLE}" datasets status "${DATASET_ID}" >/dev/null 2>&1; then
  echo "[mistral-lane] dataset exists -> new version"
  "${KAGGLE}" datasets version -p "${DATASET_DIR}" -m "refresh mistral OOD lane patches" --dir-mode zip
else
  echo "[mistral-lane] dataset missing -> create"
  "${KAGGLE}" datasets create -p "${DATASET_DIR}" --dir-mode zip
fi

echo "[mistral-lane] 3/6 kernel push (${USER_NAME}/${SLUG})"
KERNEL_DIR="${STAGE}/kernel"
mkdir -p "${KERNEL_DIR}"
cp "${KAGGLE_DIR}/kernel.py" "${KERNEL_DIR}/"
python3 - "${KAGGLE_DIR}/kernel-metadata.json" "${KERNEL_DIR}/kernel-metadata.json" "${USER_NAME}" "${SLUG}" <<'PY'
import json, sys
src, dst, user, slug = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
meta = json.load(open(src))
meta["id"] = f"{user}/{slug}"
meta["title"] = slug
meta["dataset_sources"] = [f"{user}/{slug}-patches"]
json.dump(meta, open(dst, "w"), indent=2)
PY
"${KAGGLE}" kernels push -p "${KERNEL_DIR}"

echo "[mistral-lane] 4/6 polling kernel status (timeout ${TIMEOUT_S}s)"
STARTED=$(date +%s)
while true; do
  STATUS_OUTPUT="$("${KAGGLE}" kernels status "${USER_NAME}/${SLUG}" 2>&1 || true)"
  echo "[mistral-lane]   $(date +%H:%M:%S) ${STATUS_OUTPUT}"
  if echo "${STATUS_OUTPUT}" | grep -qi 'has status "complete"'; then
    break
  fi
  if echo "${STATUS_OUTPUT}" | grep -qi 'has status "error"\|has status "cancel"'; then
    echo "[mistral-lane] kernel ended in error/cancelled; downloading log for diagnosis"
    break
  fi
  NOW=$(date +%s)
  if (( NOW - STARTED > TIMEOUT_S )); then
    echo "[mistral-lane] ERROR: timed out waiting for the kernel" >&2
    exit 1
  fi
  sleep 60
done

echo "[mistral-lane] 5/6 download output -> ${OUT_DIR}"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
"${KAGGLE}" kernels output "${USER_NAME}/${SLUG}" -p "${OUT_DIR}" || true
ls -la "${OUT_DIR}"

echo "[mistral-lane] 6/6 local re-verification of the downloaded suite"
# The kernel writes artifacts under repo-relative paths; kernels output
# flattens them into ${OUT_DIR}. Copy the suite back into repo layout and
# verify from the repo root.
SUITE_NAME="$(basename "${PRIMARY_REL}")"
SRC_DIR="$(find "${OUT_DIR}" -type d -name "${SUITE_NAME}" | head -1 || true)"
if [[ -n "${SRC_DIR}" && -f "${SRC_DIR}/report.json" ]]; then
  DEST_DIR="${REPO_ROOT}/${PRIMARY_REL}"
  rm -rf "${DEST_DIR}"
  mkdir -p "$(dirname "${DEST_DIR}")"
  cp -r "${SRC_DIR}" "${DEST_DIR}"
  echo "[mistral-lane] verifying ${PRIMARY_REL}"
  (cd "${REPO_ROOT}/tools" && python3 -m llm_reader.verify_ordered_path_ood_stress \
    --report "../${PRIMARY_REL}/report.json")
else
  echo "[mistral-lane] ERROR: suite ${SUITE_NAME} not found in kernel output" >&2
  find "${OUT_DIR}" -maxdepth 3 -type d | head -20
  exit 3
fi

echo "[mistral-lane] DONE. Artifacts: ${OUT_DIR}; suite copied into ${REPO_ROOT}/${PRIMARY_REL}."
echo "[mistral-lane] Check ${OUT_DIR}/*.log for the kernel execution log."
