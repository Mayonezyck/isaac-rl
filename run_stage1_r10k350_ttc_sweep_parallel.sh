#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

mkdir -p runs/parallel_logs

run_one() {
  local config_path="$1"
  local run_id="$2"
  local log_path="runs/parallel_logs/${run_id}.log"
  echo "[launch] run_id=${run_id} config=${config_path} log=${log_path}"
  python -u gpudrive_chocolate/baselines/ppo/ppo_sb3.py \
    --config "${config_path}" \
    --run-id "${run_id}" > "${log_path}" 2>&1 &
  echo $!
}

P1=$(run_one \
  "configs/curriculum_design/navigation_geom_v1/ppo_stage1_goal_route_geom_best_roadedge_ttc_hardreset_r10_k350_knn_sweep_s1_scratch.yaml" \
  "03_10_stage1_r10k350_ttc_sweep_s1")
P2=$(run_one \
  "configs/curriculum_design/navigation_geom_v1/ppo_stage1_goal_route_geom_best_roadedge_ttc_hardreset_r10_k350_knn_sweep_s2_scratch.yaml" \
  "03_10_stage1_r10k350_ttc_sweep_s2")
P3=$(run_one \
  "configs/curriculum_design/navigation_geom_v1/ppo_stage1_goal_route_geom_best_roadedge_ttc_hardreset_r10_k350_knn_sweep_s3_scratch.yaml" \
  "03_10_stage1_r10k350_ttc_sweep_s3")
P4=$(run_one \
  "configs/curriculum_design/navigation_geom_v1/ppo_stage1_goal_route_geom_best_roadedge_ttc_hardreset_r10_k350_knn_sweep_s4_scratch.yaml" \
  "03_10_stage1_r10k350_ttc_sweep_s4")

echo "[launch] started pids: ${P1} ${P2} ${P3} ${P4}"
echo "[launch] monitor logs: tail -f runs/parallel_logs/03_10_stage1_r10k350_ttc_sweep_s*.log"
