#!/usr/bin/env bash
# export_public_repo.sh
# Copies the SceneFactory public release files into a clean directory.
# Usage: bash export_public_repo.sh [destination]
# Default destination: ~/Github/scenefactory-public
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${1:-$HOME/Github/scenefactory-public}"

echo "==> Exporting SceneFactory public repo"
echo "    Source : $SRC"
echo "    Dest   : $DST"
echo ""

# ── helpers ──────────────────────────────────────────────────────────────────
copy() {
  local src="$SRC/$1" dst="$DST/$1"
  mkdir -p "$(dirname "$dst")"
  cp -r "$src" "$dst"
  echo "  [+] $1"
}

# ── destination ──────────────────────────────────────────────────────────────
if [[ -d "$DST" ]]; then
  echo "WARNING: $DST already exists. Remove it first or pass a different path."
  exit 1
fi
mkdir -p "$DST"

# ════════════════════════════════════════════════════════════════════════════
# 1. Core source — SceneFactory engine
# ════════════════════════════════════════════════════════════════════════════
echo "── src/ (engine) ──"
for f in \
  src/__init__.py \
  src/isaaclab_bootstrap.py \
  src/isaaclab_vehicle_action.py \
  src/procedural_student_vehicle.py \
  src/procedural_student_vehicle_import.py \
  src/student_vehicle_multiagent_goal_env.py \
  src/student_vehicle_goal_env.py \
  src/student_vehicle_sysid.py \
  src/student_vehicle_sysid_visualizer.py \
  src/chocolate_waymo_builder.py \
  src/scene_factory_multiworld_scene.py \
  src/scene_factory_obs_contract.py \
  src/scene_factory_late_fusion_actor_critic.py \
  src/scene_factory_config_wizard.py \
  src/physx_teacher_dataset_builder.py \
  src/physx_teacher_patch_track.py \
  src/physx_teacher_command_program_generator.py \
  src/physx_teacher_rollout_visualizer.py \
  src/eval_student_vehicle_goal_ppo.py \
  src/train_student_vehicle_goal_multiagent_rsl_rl.py \
  src/run_student_vehicle_goal_multiagent_random.py \
; do copy "$f"; done

# trfc sub-package (friction API + observation helpers)
echo "── src/trfc/ ──"
copy src/trfc

# ════════════════════════════════════════════════════════════════════════════
# 2. Configs — canonical configs only
# ════════════════════════════════════════════════════════════════════════════
echo "── configs/scene_factory/ (canonical) ──"
for f in \
  configs/scene_factory/goal_reaching_roads_train.yaml \
  configs/scene_factory/goal_reaching_smoke.yaml \
  configs/scene_factory/multiworld_scene.yaml \
  configs/scene_factory/visualize_scene.yaml \
; do copy "$f"; done

# Generated configs referenced by the paper's eval/train shell scripts
echo "── configs/scene_factory/generated/ (paper configs only) ──"
GENERATED_CONFIGS=(
  # --- main training (base scene pools) ---
  scene_factory_256scene_random_0414_train.yaml
  scene_factory_256scene_random_0414_train_friction.yaml
  # --- main training (fastgoal wrappers) ---
  scene_factory_256scene_random_0414_train_fastgoal_v8_sysid4_noweather.yaml
  scene_factory_256scene_random_0414_train_fastgoal_v7_sysid4_weather.yaml
  scene_factory_256scene_random_train_fastgoal_bicycle_v1.yaml
  # --- scene diversity ablation training (base scene pools) ---
  scene_factory_8unique_256total_random_0414_train.yaml
  scene_factory_16unique_256total_random_0414_train.yaml
  scene_factory_32unique_256total_random_0414_train.yaml
  scene_factory_64unique_256total_random_0414_train.yaml
  scene_factory_128unique_256total_random_0414_train.yaml
  # --- scene diversity ablation training (fastgoal wrappers) ---
  scene_factory_8unique_256total_random_train_fastgoal_v8_sysid4_noweather.yaml
  scene_factory_16unique_256total_random_train_fastgoal_v8_sysid4_noweather.yaml
  scene_factory_32unique_256total_random_train_fastgoal_v8_sysid4_noweather.yaml
  scene_factory_64unique_256total_random_train_fastgoal_v8_sysid4_noweather.yaml
  scene_factory_128unique_256total_random_train_fastgoal_v8_sysid4_noweather.yaml
  # --- throughput stress tests ---
  scene_factory_512worlds_stress_1agent.yaml
  scene_factory_1024worlds_stress_1agent.yaml
  scene_factory_4096worlds_stress_1agent.yaml
  scene_factory_8192worlds_stress_1agent.yaml
  scene_factory_single_scene000077_512worlds_stress.yaml
  scene_factory_single_scene000077_1024worlds_stress.yaml
  scene_factory_single_scene000077_2048worlds_stress.yaml
  scene_factory_single_scene000077_4096worlds_stress.yaml
  scene_factory_single_scene000077_8192worlds_stress.yaml
  # --- evaluation: test scenes (base pools) ---
  scene_factory_64scene_curated_0326_test.yaml
  scene_factory_64scene_curated_0326_selection_summary.json
  scene_factory_64scene_curated_0326_test_weather_dry.yaml
  scene_factory_64scene_curated_0326_test_friction_hard_sma2mm.yaml
  scene_factory_64scene_curated_0326_test_friction_heavy_ac0p8mm.yaml
  scene_factory_64scene_curated_0326_test_friction_moderate_ac0p5mm.yaml
  # --- evaluation: v8 (friction-blind) ---
  eval_v8_sysid4_noweather_model_300_test64_dry.yaml
  eval_v8_sysid4_noweather_model_300_test64_dry_random_od.yaml
  eval_v8_sysid4_noweather_model_300_test64_moderate_ac0p5mm.yaml
  eval_v8_sysid4_noweather_model_300_test64_heavy_ac0p8mm.yaml
  eval_v8_sysid4_noweather_model_200_test64_hard_sma2mm.yaml
  # --- evaluation: v7 (friction-aware) ---
  eval_v7_sysid4_weather_model_600_test64_dry.yaml
  eval_v7_sysid4_weather_model_600_test64_dry_random_od.yaml
  eval_v7_sysid4_weather_model_600_test64_moderate_ac0p5mm.yaml
  eval_v7_sysid4_weather_model_600_test64_heavy_ac0p8mm.yaml
  eval_v7_sysid4_weather_model_600_test64_hard_sma2mm.yaml
  # --- evaluation: bicycle / transfer ---
  eval_bicycle_v1_model_350_test64_dry.yaml
  eval_bicycle_v1_physx_transfer_model_375_test64_dry.yaml
  eval_v8_physx_to_bicycle_transfer_model_300_test64_dry.yaml
  # --- evaluation: scene diversity ---
  eval_scenediv_8unique_256total_model_299_test64_dry.yaml
  eval_scenediv_16unique_256total_model_299_test64_dry.yaml
  eval_scenediv_32unique_256total_model_300_test64_dry.yaml
  eval_scenediv_64unique_256total_model_299_test64_dry.yaml
  eval_scenediv_128unique_256total_model_300_test64_dry.yaml
  eval_scenediv_256unique_256total_model_300_test64_dry.yaml
  # --- throughput stress tests ---
  scene_factory_512worlds_stress_1agent.yaml
  scene_factory_1024worlds_stress_1agent.yaml
  scene_factory_4096worlds_stress_1agent.yaml
)
for f in "${GENERATED_CONFIGS[@]}"; do
  copy "configs/scene_factory/generated/$f"
done

# ════════════════════════════════════════════════════════════════════════════
# 3. Artifacts — vehicle asset + final sysid result only
# ════════════════════════════════════════════════════════════════════════════
echo "── artifacts/ (vehicle asset + final sysid) ──"
copy artifacts/student_vehicle_assets/vehicle_student
copy artifacts/student_vehicle_sysid/comprehensive_fwd_v1_cem_v4/best_config.json
copy artifacts/student_vehicle_sysid/comprehensive_fwd_v1_cem_v4/best_result.json

# ════════════════════════════════════════════════════════════════════════════
# 4. Shell scripts — paper eval + training launchers
# ════════════════════════════════════════════════════════════════════════════
echo "── shell scripts ──"
for f in \
  quickstart.sh \
  run_bicycle_train.sh \
  run_bicycle_eval.sh \
  run_bicycle_physx_transfer_eval.sh \
  run_v8_vs_v7_physics_blind_eval.sh \
  run_v8_vs_v7_moderate_wet_eval.sh \
  run_v8_vs_v7_heavy_wet_eval.sh \
  run_v8_physx_to_bicycle_transfer_eval.sh \
  run_scene_diversity_eval.sh \
  run_scene_diversity_ablation.sh \
  run_visualize_scene.sh \
; do copy "$f"; done

# ════════════════════════════════════════════════════════════════════════════
# 5. Helper scripts
# ════════════════════════════════════════════════════════════════════════════
echo "── scripts/ ──"
for f in \
  scripts/summarize_2x2_eval.py \
  scripts/generate_weather_eval_configs.py \
; do copy "$f"; done

# ════════════════════════════════════════════════════════════════════════════
# 6. Top-level metadata
# ════════════════════════════════════════════════════════════════════════════
echo "── top-level ──"
copy README.md

# ════════════════════════════════════════════════════════════════════════════
# 7. .gitignore for the public repo
# ════════════════════════════════════════════════════════════════════════════
cat > "$DST/.gitignore" << 'GITIGNORE'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.env
.venv/
.conda/

# Training outputs
runs/
logs/
out/
*.log

# Isaac Sim / USD caches
_isaac_sim/
omni.*/
kit/

# Data (too large — download from Waymo)
data/

# Eval output directories (timestamped)
20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]_*/

# Misc
*.mp4
*.pyc
.DS_Store
GITIGNORE

echo ""
echo "==> Done. Files written to: $DST"
echo ""
echo "Next steps:"
echo "  cd $DST"
echo "  git init && git add . && git commit -m 'Initial public release'"
