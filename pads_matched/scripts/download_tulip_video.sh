#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PADS_MATCHED_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${PADS_MATCHED_DIR}/configs/tulip_ds.json"

if ! command -v gdown >/dev/null 2>&1; then
  echo "[ERROR] gdown not found. Install first: pip install gdown"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found."
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] config not found: ${CONFIG_PATH}"
  exit 1
fi

# Resolve metadata.tulip_pads_matched_base_dir robustly.
# Expected config value example: "PD-Dashboard/pads_matched/by_tulip"
BASE_DIR_RAW="$(python3 - <<'PY' "${CONFIG_PATH}"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)
print(cfg["metadata"]["tulip_pads_matched_base_dir"])
PY
)"

resolve_base_dir() {
  local raw="$1"
  if [[ "${raw}" = /* ]]; then
    echo "${raw}"
    return 0
  fi

  local c1="/workspace/${raw}"
  local c2="/workspace/source/${raw}"
  local c3="${PADS_MATCHED_DIR}/by_tulip"

  if [[ -d "${c1}" ]]; then
    echo "${c1}"
  elif [[ -d "${c2}" ]]; then
    echo "${c2}"
  else
    # Fallback to pads_matched/by_tulip (safe default in this repo)
    echo "${c3}"
  fi
}

BASE_DIR="$(resolve_base_dir "${BASE_DIR_RAW}")"
mkdir -p "${BASE_DIR}"

echo "[INFO] config: ${CONFIG_PATH}"
echo "[INFO] base_dir: ${BASE_DIR}"

# Emit TSV rows: subject \t task \t url
python3 - <<'PY' "${CONFIG_PATH}" | while IFS=$'\t' read -r subject task url; do
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)
for subject, tasks in cfg["tulip_video_folder_link"].items():
    for task_name, folder_url in tasks.items():
        print(f"{subject}\t{task_name}\t{folder_url}")
PY
  target_dir="${BASE_DIR}/${subject}/videos/${task}"
  mkdir -p "${target_dir}"
  echo "[DOWNLOAD] ${subject} / ${task}"
  echo "           -> ${target_dir}"
  gdown --folder "${url}" -O "${target_dir}"
done

echo "[DONE] All configured TULIP folders downloaded."
