#!/usr/bin/env bash
# Headless Kaggle GPU lane driver for GCL.
#
# One command runs the whole cycle on a free Kaggle GPU session:
#   1. regenerate the patch bundle (origin/main..HEAD),
#   2. upload/refresh the private patches dataset,
#   3. push the GPU kernel (scripts/gpu/kaggle/kernel.py),
#   4. poll kernel status until it completes,
#   5. download the kernel output (artifacts + log),
#   6. re-run the independent verifier locally on the downloaded suite.
#
# Authentication (server 2.x): set KAGGLE_API_TOKEN to a token generated at
# https://www.kaggle.com/settings/api (Generate New Token under API), or save
# it to ~/.kaggle/access_token. Legacy username/key pairs from kaggle.json
# are NOT accepted by the current server on write routes.
#
# Usage:
#   export KAGGLE_API_TOKEN=...        # required
#   bash scripts/gpu/kaggle/run-kaggle-gpu-lane.sh
#
# Environment overrides:
#   KAGGLE_SLUG          kernel/dataset slug (default gcl-gpu-lane)
#   KAGGLE_USER          account username (required; derived from token if
#                        the CLI can report it, otherwise must be set)
#   GCL_KAGGLE_TIMEOUT   poll timeout in seconds (default 3600)
#   GCL_KAGGLE_OUT       download directory (default /tmp/gcl-kaggle-out)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
KAGGLE_DIR="${REPO_ROOT}/scripts/gpu/kaggle"
SLUG="${KAGGLE_SLUG:-gcl-gpu-lane}"
USER_NAME="${KAGGLE_USER:-kuglipista}"
TIMEOUT_S="${GCL_KAGGLE_TIMEOUT:-3600}"
OUT_DIR="${GCL_KAGGLE_OUT:-/tmp/gcl-kaggle-out}"

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

echo "[kaggle-lane] 1/6 patch bundle -> ${STAGE}"
mkdir -p "${STAGE}/bundle"
git -C "${REPO_ROOT}" format-patch origin/main..HEAD -o "${STAGE}/bundle/" >/dev/null
if ! compgen -G "${STAGE}/bundle/*.patch" >/dev/null; then
  echo "[kaggle-lane] WARNING: no commits ahead of origin/main; using an empty marker patch is not supported - push commits first." >&2
  exit 1
fi
cp "${KAGGLE_DIR}/dataset-metadata.json" "${STAGE}/bundle/"
ls "${STAGE}/bundle"

echo "[kaggle-lane] 2/6 dataset upload (kuglipista/${SLUG}-patches)"
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
  echo "[kaggle-lane] dataset exists -> new version"
  "${KAGGLE}" datasets version -p "${DATASET_DIR}" -m "refresh patches" --dir-mode zip
else
  echo "[kaggle-lane] dataset missing -> create"
  "${KAGGLE}" datasets create -p "${DATASET_DIR}" --dir-mode zip
fi

echo "[kaggle-lane] 3/6 kernel push (${USER_NAME}/${SLUG})"
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

echo "[kaggle-lane] 4/6 polling kernel status (timeout ${TIMEOUT_S}s)"
STARTED=$(date +%s)
while true; do
  STATUS_OUTPUT="$("${KAGGLE}" kernels status "${USER_NAME}/${SLUG}" 2>&1 || true)"
  echo "[kaggle-lane]   $(date +%H:%M:%S) ${STATUS_OUTPUT}"
  if echo "${STATUS_OUTPUT}" | grep -qi 'has status "complete"'; then
    break
  fi
  if echo "${STATUS_OUTPUT}" | grep -qi 'has status "error"\|has status "cancel"'; then
    echo "[kaggle-lane] kernel ended in error/cancelled; downloading log for diagnosis"
    break
  fi
  NOW=$(date +%s)
  if (( NOW - STARTED > TIMEOUT_S )); then
    echo "[kaggle-lane] ERROR: timed out waiting for the kernel" >&2
    exit 1
  fi
  sleep 30
done

echo "[kaggle-lane] 5/6 download output -> ${OUT_DIR}"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
"${KAGGLE}" kernels output "${USER_NAME}/${SLUG}" -p "${OUT_DIR}" || true
ls -la "${OUT_DIR}"

echo "[kaggle-lane] 6/6 local re-verification of downloaded suites"
PRIMARY_REL="benchmarks/release/nn-live-ordered-path-stability/kaggle-t4-ood-2026-11-06"
EXTENDED_REL="benchmarks/release/nn-live-ordered-path-stability/kaggle-t4-ood-2026-11-06-extended"
PY="${REPO_ROOT}/.gcl-python-gpu/bin/python"
# The kernel writes artifacts under repo-relative paths; kernels output
# flattens them into ${OUT_DIR}. Copy the suites back into repo layout and
# verify from the repo root.
for REL in "${PRIMARY_REL}" "${EXTENDED_REL}"; do
  SUITE_NAME="$(basename "${REL}")"
  SRC_DIR="$(find "${OUT_DIR}" -type d -name "${SUITE_NAME}" | head -1 || true)"
  if [[ -n "${SRC_DIR}" && -f "${SRC_DIR}/report.json" ]]; then
    DEST_DIR="${REPO_ROOT}/${REL}"
    rm -rf "${DEST_DIR}"
    mkdir -p "$(dirname "${DEST_DIR}")"
    cp -r "${SRC_DIR}" "${DEST_DIR}"
    echo "[kaggle-lane] verifying ${REL}"
    (cd "${REPO_ROOT}/tools" && "${PY}" -m llm_reader.verify_ordered_path_ood_stress \
      --report "../${REL}/report.json") || true
  else
    echo "[kaggle-lane] NOTE: suite ${SUITE_NAME} not found in kernel output (extended run may have been skipped)"
  fi
done

echo "[kaggle-lane] DONE. Artifacts: ${OUT_DIR}; suites copied into benchmarks/release."
echo "[kaggle-lane] Check ${OUT_DIR}/*.log for the kernel execution log."
