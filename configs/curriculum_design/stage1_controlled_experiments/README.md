# Stage 1 Controlled Experiments

This folder contains two controlled stage-1 ablations intended to answer a specific question:

- is the difference mostly coming from the map set?
- or is it mostly coming from the old handcrafted training logic?

## Experiment A: Old Maps, Current Simplified Strategy

Files:

- `curriculum_stage1_oldmaps_current_strategy.yaml`
- `ppo_stage1_oldmaps_current_strategy.yaml`

This keeps the current simplified stage-1 training logic, but swaps in the exact old handcrafted scene list:

- `scene_000000.json` through `scene_000008.json`

Everything else stays aligned with the current simplified stage-1 design:

- dry / clear `AC`
- no lane reward
- no vehicle-contact termination
- no collision penalty shaping
- road-edge termination still enabled as a guard rail

## Experiment B: Current Maps, Old Handcrafted Strategy

Files:

- `curriculum_stage1_currentmaps_old_strategy.yaml`
- `ppo_stage1_currentmaps_old_strategy.yaml`

This keeps the current simplified stage-1 map selection:

- `scene_000098.json`
- `scene_000038.json`
- `scene_000076.json`
- `scene_000085.json`

but restores the old handcrafted stage-1 reward / termination strategy:

- no road-contact termination
- no vehicle-contact termination
- no lane reward
- no idle penalty
- old PPO shaping weights and minibatch setup

Everything else stays aligned with the current simplified stage-1 setup, so this experiment isolates training strategy rather than observation geometry.

The old handcrafted PPO preset resumed from an already-trained checkpoint by default. These ablations do **not** resume; both start from scratch so the comparison is fair.

## Parallel Run Note

The two PPO presets default to different devices so you can launch them in parallel:

- Experiment A: `cuda:1`
- Experiment B: `cuda:0`

Experiment B also points IsaacSim physics/render to GPU 0. If `cuda:0` on this host is not usable for training, edit both the curriculum `app.active_gpu` / `app.physics_gpu` fields and the PPO `device` field before launching.

## Usage

Experiment A:

```bash
python gpudrive_chocolate/baselines/ppo/ppo_sb3.py \
  --config configs/curriculum_design/stage1_controlled_experiments/ppo_stage1_oldmaps_current_strategy.yaml
```

Experiment B:

```bash
python gpudrive_chocolate/baselines/ppo/ppo_sb3.py \
  --config configs/curriculum_design/stage1_controlled_experiments/ppo_stage1_currentmaps_old_strategy.yaml
```
