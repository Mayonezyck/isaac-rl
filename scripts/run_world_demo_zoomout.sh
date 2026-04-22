#!/usr/bin/env bash
# Multi-world cinematic demo: surveillance cam on one world → tilt to reveal
# neighbors → gentle zoom-out showing the full grid. Camera always stays
# centered on the starting world.
# Usage: ./scripts/run_world_demo_zoomout.sh [NUM_ENVS] [START_ENV_INDEX]
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
  --video_camera_pose_mode flyover \
  --video_flyover_start_height_m 8.0 \
  --video_flyover_end_height_m 800.0 \
  --video_flyover_surveillance_frames 120 \
  --video_flyover_tilt_frames 180 \
  --video_flyover_zoomout_frames 300 \
  --video_flyover_start_env_index "$START_ENV" \
  --video_flyover_start_tilt_deg 25.0 \
  --video_flyover_end_tilt_deg 75.0 \
  --video_flyover_start_distance_m 25.0 \
  --video_flyover_azimuth_deg 0.0 \
  --video_flyover_lookaway_frames 180 \
  --video_flyover_lookaway_pitch_deg 45.0 \
  --video_flyover_lookaway_yaw_deg 45.0 \
  --road_hidden_types "1,2,3" \
  --hide_goal_markers \
  --no-use_fabric \
  --random_od \
  --num_agents_per_env 16 \
  --headless \
  --device cuda:0 \
  --num_envs "$NUM_ENVS" \
  2>&1 | tee /tmp/world_demo_zoomout_${NUM_ENVS}envs.log
