# Stage 1 Evolution Search

This folder contains an evolutionary hyperparameter search setup for GPUDrive chocolate PPO.

The default study uses the stronger stage-1 baseline:

- [ppo_stage1_currentmaps_old_strategy.yaml](/home/yz8733/Github/isaac-rl/configs/curriculum_design/stage1_controlled_experiments/ppo_stage1_currentmaps_old_strategy.yaml)

It searches a curated mix of PPO and curriculum knobs that are likely to matter for fast goal-reaching:

- PPO optimizer and rollout shape
- success and progress reward scale
- throttle cap
- max episode horizon

## Files

- `search_stage1_currentmaps_old_strategy.yaml`
  Full search preset intended for real tuning runs.
- `search_stage1_currentmaps_old_strategy_smoke.yaml`
  Very short study intended to validate the search loop end to end.
- `search_stage1_currentmaps_old_strategy_smoke_cpu.yaml`
  Same short study, but with PPO tensors on CPU and IsaacSim still on GPU. Useful on hosts where the shell-side PyTorch CUDA path is unavailable.
- `search_stage1_navigation_safety.yaml`
  Real search preset for the 4-world safety-aware stage-1 task with road-edge and vehicle-contact termination enabled.
- `search_stage1_navigation_safety_smoke.yaml`
  Short smoke study for the safety-aware stage-1 task.
- `search_stage1_navigation_geom_v1.yaml`
  Real search preset for the new geometric stage-1 task with route-progress and lane-geometry rewards.
- `search_stage1_navigation_geom_v1_smoke.yaml`
  Short smoke study for the geometric stage-1 task.

## Usage

Smoke test:

```bash
python -u gpudrive_chocolate/baselines/ppo/evolve_hparams.py \
  --study configs/hparam_search/stage1_evolution/search_stage1_currentmaps_old_strategy_smoke.yaml
```

Real study:

```bash
python -u gpudrive_chocolate/baselines/ppo/evolve_hparams.py \
  --study configs/hparam_search/stage1_evolution/search_stage1_currentmaps_old_strategy.yaml
```

Safety-aware smoke study:

```bash
python -u gpudrive_chocolate/baselines/ppo/evolve_hparams.py \
  --study configs/hparam_search/stage1_evolution/search_stage1_navigation_safety_smoke.yaml
```

Safety-aware real study:

```bash
python -u gpudrive_chocolate/baselines/ppo/evolve_hparams.py \
  --study configs/hparam_search/stage1_evolution/search_stage1_navigation_safety.yaml
```

Geometric stage-1 smoke study:

```bash
python -u gpudrive_chocolate/baselines/ppo/evolve_hparams.py \
  --study configs/hparam_search/stage1_evolution/search_stage1_navigation_geom_v1_smoke.yaml
```

Geometric stage-1 real study:

```bash
python -u gpudrive_chocolate/baselines/ppo/evolve_hparams.py \
  --study configs/hparam_search/stage1_evolution/search_stage1_navigation_geom_v1.yaml
```

The search writes each generated candidate's PPO and curriculum YAMLs under `runs/hparam_search/...`, launches short trials, ranks them by rollout metrics, and breeds the next generation from the best candidates.

Run the search from an already-activated environment that contains the full training stack and `tensorboard`. The generated child trials inherit that same environment.

`max_parallel` is dynamic: if you ask for more workers than explicit slots, the runner round-robins the configured slots up to the requested count.
