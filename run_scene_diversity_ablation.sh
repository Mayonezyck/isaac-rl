#!/bin/bash
# Scene diversity ablation: v8 with 8/16/32/64/128 unique scenes, 256 total envs
# All runs on cuda:1, 300 iterations each, sequential

set -e

DEVICE="cuda:1"
BASE_CMD="PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py --headless --device $DEVICE"

for N in 8 16 32 64 128; do
  CONFIG="configs/scene_factory/generated/scene_factory_${N}unique_256total_random_train_fastgoal_v8_sysid4_noweather.yaml"
  echo "========================================"
  echo "Starting: ${N} unique scenes"
  echo "Config: ${CONFIG}"
  echo "========================================"
  PYTHONPATH=. python -u src/train_student_vehicle_goal_multiagent_rsl_rl.py \
    --config "$CONFIG" \
    --headless \
    --device "$DEVICE"
  echo "Done: ${N} unique scenes"
done

echo "All scene diversity ablation runs complete."
