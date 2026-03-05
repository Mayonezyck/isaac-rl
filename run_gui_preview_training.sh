#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

CONFIG="${GUI_PREVIEW_CONFIG:-configs/curriculum_design/navigation_geom_v1/ppo_stage2_lane_safety_geom_gui_preview.yaml}"
DEVICE="${GUI_PREVIEW_DEVICE:-cuda:1}"
RUN_ID="${GUI_PREVIEW_RUN_ID:-$(date +%m_%d_%H_%M)_gui_preview}"
RUNS_ROOT="${GUI_PREVIEW_RUNS_ROOT:-runs}"
DEFAULT_RESUME="runs/checkpoints_stage2_transition_lane_safety_geom_from_stage1_3m/ppo_stage2_transition_lane_safety_geom_from_stage1_3m_566848_steps.zip"

EXTRA_ARGS=()
if [[ "${GUI_PREVIEW_FORCE_FRESH:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--fresh)
elif [[ ! -f "${DEFAULT_RESUME}" ]]; then
  echo "[gui-preview] checkpoint missing (${DEFAULT_RESUME}); starting fresh."
  EXTRA_ARGS+=(--fresh)
else
  echo "[gui-preview] using resume checkpoint: ${DEFAULT_RESUME}"
fi

echo "[gui-preview] config=${CONFIG}"
echo "[gui-preview] device=${DEVICE}"
echo "[gui-preview] run_id=${RUN_ID}"
echo "[gui-preview] runs_root=${RUNS_ROOT}"

python -u gpudrive_chocolate/baselines/ppo/ppo_sb3.py \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  --run-id "${RUN_ID}" \
  --runs-root "${RUNS_ROOT}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
