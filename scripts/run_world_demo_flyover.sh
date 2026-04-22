#!/usr/bin/env bash
# Cinematic flyover demo: starts on one world, rises to reveal the full multi-world grid.
# Showcases SceneFactory's multi-world capability.
# Usage: ./scripts/run_world_demo_flyover.sh [NUM_ENVS] [START_ENV_INDEX]
#   NUM_ENVS       defaults to 64
#   START_ENV_INDEX defaults to 0 (which env the camera starts centered on)
set -euo pipefail

NUM_ENVS="${1:-64}"
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
  --video_flyover_start_height_m 10.0 \
  --video_flyover_end_height_m 500.0 \
  --video_flyover_rise_frames 300 \
  --video_flyover_orbit_deg_per_frame 0.2 \
  --video_flyover_start_env_index "$START_ENV" \
  --video_flyover_tilt_deg 65.0 \
  --road_hidden_types "1,2,3" \
  --hide_goal_markers \
  --no-use_fabric \
  --random_od \
  --headless \
  --device cuda:0 \
  --num_envs "$NUM_ENVS" \
  2>&1 | tee /tmp/world_demo_flyover_${NUM_ENVS}envs.log
