#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CAPTURE_PREVIEW_CONFIG:-configs/curriculum_design/navigation_geom_v1/ppo_stage2_lane_safety_geom_capture_preview.yaml}"
DEVICE="${CAPTURE_PREVIEW_DEVICE:-cuda:1}"
RUN_ID="${CAPTURE_PREVIEW_RUN_ID:-$(date +%m_%d_%H_%M)_capture_preview}"
RUNS_ROOT="${CAPTURE_PREVIEW_RUNS_ROOT:-runs}"
DEFAULT_RESUME="runs/checkpoints_stage2_transition_lane_safety_geom_from_stage1_3m/ppo_stage2_transition_lane_safety_geom_from_stage1_3m_566848_steps.zip"

EXTRA_ARGS=()
if [[ "${CAPTURE_PREVIEW_FORCE_FRESH:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--fresh)
elif [[ ! -f "${DEFAULT_RESUME}" ]]; then
  echo "[capture-preview] checkpoint missing (${DEFAULT_RESUME}); starting fresh."
  EXTRA_ARGS+=(--fresh)
else
  echo "[capture-preview] using resume checkpoint: ${DEFAULT_RESUME}"
fi

echo "[capture-preview] config=${CONFIG}"
echo "[capture-preview] device=${DEVICE}"
echo "[capture-preview] run_id=${RUN_ID}"
echo "[capture-preview] runs_root=${RUNS_ROOT}"
echo "[capture-preview] press Ctrl+C to stop; frames are kept under runs/capture_stage2_lane_safety_geom_capture_preview/*/frames"

python -u gpudrive_chocolate/baselines/ppo/ppo_sb3.py \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --run-id "${RUN_ID}" \
  --runs-root "${RUNS_ROOT}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
