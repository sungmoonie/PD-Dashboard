#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TULIP_SOURCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTRACT_SCRIPT="${SCRIPT_DIR}/extract_metrabs_joint_features.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_DIR="${BASE_DIR:-/workspace/source/PD-Dashboard/pads_matched/by_tulip}"

SUBJECTS=("TULIP_001" "TULIP_008")
SKIP_EXISTING="${SKIP_EXISTING:-1}"

echo "[INFO] python=${PYTHON_BIN}"
echo "[INFO] base_dir=${BASE_DIR}"
echo "[INFO] tulip_source_dir=${TULIP_SOURCE_DIR}"
echo "[INFO] skip_existing=${SKIP_EXISTING}"

if [[ ! -f "${EXTRACT_SCRIPT}" ]]; then
  echo "[ERROR] script not found: ${EXTRACT_SCRIPT}"
  exit 1
fi

for subject in "${SUBJECTS[@]}"; do
  videos_dir="${BASE_DIR}/${subject}/videos"
  save_dir="${BASE_DIR}/${subject}/videos_feature"

  if [[ ! -d "${videos_dir}" ]]; then
    echo "[WARN] skip ${subject}: videos dir not found (${videos_dir})"
    continue
  fi

  while IFS= read -r -d '' task_dir; do
    task_name="$(basename "${task_dir}")"
    task_save_dir="${save_dir}/${task_name}"

    if [[ "${SKIP_EXISTING}" == "1" ]]; then
      mp4_count=$(find "${task_dir}" -maxdepth 1 -type f -name "Camera*.mp4" | wc -l | tr -d ' ')
      feature_count=0
      if [[ -d "${task_save_dir}" ]]; then
        feature_count=$(find "${task_save_dir}" -maxdepth 1 -type f -name "Camera*_features.csv" | wc -l | tr -d ' ')
      fi
      if [[ "${mp4_count}" -gt 0 && "${feature_count}" -ge "${mp4_count}" ]]; then
        echo "[SKIP] ${subject} / ${task_name} (features already exist: ${feature_count}/${mp4_count})"
        continue
      fi
    fi

    echo "[RUN] ${subject} / ${task_name}"
    "${PYTHON_BIN}" "${EXTRACT_SCRIPT}" \
      --video_dir "${task_dir}" \
      --save_dir "${save_dir}"
  done < <(find "${videos_dir}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
done

echo "[DONE] batch extraction complete"
