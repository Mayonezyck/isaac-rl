#!/usr/bin/env bash
# Capture a single-env driving video for one test world.
# Usage: ./scripts/run_cosmos_video_capture_single.sh [WORLD_INDEX]
#   WORLD_INDEX defaults to 0 (first test scene). Range: 0-63.
set -euo pipefail

WORLD_INDEX="${1:-2}"  # default=2 → scene_000002 in the train-64 config
CHECKPOINT="logs/rsl_rl/scene_factory_goal_reaching_roads/2026-04-10_13-48-01_scene_factory_256scene_curated_0326_train_fastgoal_v6_patient/model_975.pt"
CONFIG="configs/scene_factory/generated/eval_v6_patient_model_975_test64_video_per_env.yaml"
SCENE_CONFIG="configs/scene_factory/generated/scene_factory_64scene_curated_0326_train.yaml"

cd /home/yz8733/Github/isaac-rl

PYTHONPATH=/home/yz8733/Github/isaac-rl \
/home/yz8733/miniforge3/envs/isaac-pytorch/bin/python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  --config "$CONFIG" \
  --test_mode scene_factory_policy_eval \
  --checkpoint_path "$CHECKPOINT" \
  --invincible \
  --video \
  --video_view_mode single_env \
  --video_env_index 0 \
  --video_width 1920 \
  --video_height 1080 \
  --video_fps 30 \
  --video_step_stride 1 \
  --video_vehicle_proxy_markers \
  --video_vehicle_proxy_z_offset_m -0.5 \
  --video_camera_pose_mode traffic_cam \
  --video_traffic_cam_height_m 12.0 \
  --video_traffic_cam_distance_m 35.0 \
  --video_traffic_cam_look_height_m 0.5 \
  --video_traffic_cam_azimuth_deg 0.0 \
  --video_traffic_cam_lateral_offset_m 10.0 \
  --road_hidden_types "1,2,3" \
  --hide_goal_markers \
  --no-use_fabric \
  --random_od \
  --num_agents_per_env 40 \
  --max_distance_from_origin_m 40.0 \
  --goal_radius_min_m 5.0 \
  --goal_radius_max_m 100.0 \
  --headless \
  --device cuda:1 \
  --num_envs 1 \
  --scene_factory_config "$SCENE_CONFIG" \
  --scene_factory_world_selection_mode fixed \
  --scene_factory_world_index "$WORLD_INDEX" \
  2>&1 | tee /tmp/cosmos_video_single_world${WORLD_INDEX}.log
