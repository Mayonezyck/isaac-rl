#!/usr/bin/env bash
# Capture per-env driving videos from all 64 test worlds for Cosmos video-to-video demo.
# Uses model_975 checkpoint with invincible mode.
set -euo pipefail

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
  --video_view_mode per_env \
  --video_width 1920 \
  --video_height 1080 \
  --video_fps 30 \
  --video_step_stride 1 \
  --video_vehicle_proxy_markers \
  --no-use_fabric \
  --headless \
  --device cuda:0 \
  2>&1 | tee /tmp/cosmos_video_capture.log
