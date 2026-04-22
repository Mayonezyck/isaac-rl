#!/usr/bin/env bash
# Multi-world cinematic demo: rise to overview → drift left → pan right + pitch up.
# A lateral-motion variant of the zoomout flyover.
# Usage: ./scripts/run_world_demo_drift.sh [NUM_ENVS] [START_ENV_INDEX]
set -euo pipefail

NUM_ENVS="${1:-256}"
START_ENV="${2:-0}"
CHECKPOINT="logs/rsl_rl/scene_factory_goal_reaching_roads/2026-04-10_13-48-01_scene_factory_256scene_curated_0326_train_fastgoal_v6_patient/model_975.pt"
CONFIG="configs/scene_factory/generated/eval_v6_patient_model_975_test64_video_per_env.yaml"

cd /home/yz8733/Github/isaac-rl

PYTHONPATH=/home/yz8733/Github/isaac-rl \
/home/yz8733/miniforge3/envs/isaac-pytorch/bin/python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  --config "$CONFIG" \
  --test_mode scene_factory_policy_eval \
  --checkpoint_path "$CHECKPOINT" \
  --invincible \
  --video \
  --video_view_mode whole_grid \
  --video_width 1920 \
  --video_height 1080 \
  --video_fps 30 \
  --video_step_stride 1 \
  --video_vehicle_proxy_markers \
  --video_vehicle_proxy_z_offset_m -0.5 \
  --video_camera_pose_mode flyover_drift \
  --video_flyover_start_env_index "$START_ENV" \
  --video_drift_rise_frames 300 \
  --video_drift_lateral_frames 900 \
  --video_drift_pan_frames 450 \
  --video_drift_start_height_m 8.0 \
  --video_drift_rise_height_m 400.0 \
  --video_drift_lateral_distance_m 3000.0 \
  --video_drift_pan_yaw_deg 90.0 \
  --video_drift_pitch_up_deg 45.0 \
  --video_drift_start_tilt_deg 25.0 \
  --video_drift_rise_tilt_deg 70.0 \
  --video_drift_azimuth_deg 0.0 \
  --hide_goal_markers \
  --no-use_fabric \
  --random_od \
  --num_agents_per_env 16 \
  --eval_max_steps 2800 \
  --episode_length_s 100.0 \
  --headless \
  --device cuda:1 \
  --num_envs "$NUM_ENVS" \
  2>&1 | tee /tmp/world_demo_drift_${NUM_ENVS}envs.log
