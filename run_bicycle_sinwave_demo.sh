#!/usr/bin/env bash
# ============================================================
# run_bicycle_sinwave_demo.sh
#
# Records a short video of vehicles driven by a hand-authored
# sin-wave action schedule using the kinematic bicycle dynamics
# model (dynamics_mode=bicycle -- PhysX articulation bypassed).
#
# Usage:
#   bash run_bicycle_sinwave_demo.sh [--gui]   (default: headless)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HEADLESS=true
if [[ "${1:-}" == "--gui" ]]; then
    HEADLESS=false
fi

DEVICE="${DEVICE:-cuda:0}"
CONFIG="configs/scene_factory/generated/bicycle_sinwave_demo.yaml"
LOG="/tmp/bicycle_sinwave_demo.log"

VIDEO_ARGS=(
    --video
    --video_fps 30
    --video_length 360
    --video_step_stride 1
    --video_camera_pose_mode top_down
    --video_vehicle_proxy_markers
    --enable_cameras
)

HEADLESS_FLAG=()
if $HEADLESS; then
    HEADLESS_FLAG=(--headless)
fi

echo "=========================================================="
echo " Bicycle sinwave demo"
echo "   dynamics_mode : bicycle (PhysX articulation bypassed)"
echo "   action        : sin-wave steer + constant throttle"
echo "   config        : $CONFIG"
echo "   device        : $DEVICE"
echo "   headless      : $HEADLESS"
if $HEADLESS; then
    echo "   video output  : logs/rsl_rl/bicycle_sinwave_demo/*/videos/"
fi
echo "=========================================================="

PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
    --config "$CONFIG" \
    --test_mode bicycle_sinwave_demo \
    --device "$DEVICE" \
    --no-use_fabric \
    --num_envs 1 \
    --num_agents_per_env 4 \
    "${HEADLESS_FLAG[@]}" \
    "${VIDEO_ARGS[@]}" \
    2>&1 | tee "$LOG"

echo ""
echo "Log → $LOG"
