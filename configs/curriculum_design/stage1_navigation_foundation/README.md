# Stage 1: Goal-Only Foundation

This package is now a narrow stage-1 warm start. The intent is fast acquisition of one skill only:

- move toward the goal reliably

Stage 1 deliberately does not try to teach the full driving task. Lane discipline, vehicle interaction, and wet-weather variation are pushed to stage 2.

## Design

- `4` worlds in a `2 x 2` layout
- `8` max controllable agents per world
- dry / clear `AC` pavement only
- easy-to-moderate scenes with short-to-moderate route lengths
- keep the observation shape the same as later stages:
  - road points stay enabled
  - vehicle observations stay enabled
- simplify the reward / termination surface:
  - keep base goal-progress reward and success bonus
  - disable lane reward
  - disable PPO-side collision/off-road shaping
  - disable direct collision penalties
  - keep road-edge termination only as a hard guard rail, with zero penalty
  - disable vehicle-contact termination in stage 1

The purpose is to get a policy that can point itself at the destination and make forward progress before adding the harder constraints.

## Scene Selection

The current stage-1 set uses:

- `scene_000098`
- `scene_000038`
- `scene_000076`
- `scene_000085`

These were chosen from the earlier 16-world pool because they are among the easier / shorter-route worlds under the real `200 m x 200 m` crop.

## PPO Preset

- `n_steps: 128`
- `num_minibatches: 8`
- `total_timesteps: 4_000_000`

This is intended to be a short warm start, not a long final run.

## Files

- `curriculum_stage1_navigation_foundation.yaml`
- `ppo_stage1_navigation_foundation.yaml`

## Usage

Train with:

```bash
python gpudrive_chocolate/baselines/ppo/ppo_sb3.py \
  --config configs/curriculum_design/stage1_navigation_foundation/ppo_stage1_navigation_foundation.yaml
```

Generate a world report with:

```bash
python -m src.trfc.curriculum_report \
  --config configs/curriculum_design/stage1_navigation_foundation/curriculum_stage1_navigation_foundation.yaml
```

## Promotion

After stage 1 reaches stable goal-seeking behavior, move to:

- `configs/curriculum_design/stage2_navigation_discipline`

Stage 2 reintroduces:

- lane occupancy reward
- road-edge / median penalties
- vehicle-contact termination
- moderate wet-weather variation
- higher scene and traffic diversity
