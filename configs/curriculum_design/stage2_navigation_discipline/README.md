# Stage 2: Navigation Discipline

This package is the follow-on stage after the simplified stage-1 warm start.

Stage 2 restores the parts of the task that stage 1 intentionally deferred:

- lane occupancy shaping
- road-edge / median termination
- solid-line penalties
- vehicle-contact termination
- moderate wet-road variation
- broader scene diversity
- higher traffic density

The observation shape stays aligned with stage 1 so the policy can be promoted by resuming from a stage-1 checkpoint.

## Design

- `16` worlds in a `4 x 4` layout
- `20` max controllable agents per world
- same observation layout as stage 1
- moderate wet `AC` worlds only, using water film in `[0.025, 0.10] mm`
- lane reward re-enabled
- road-edge and vehicle-contact terminations re-enabled

This stage is meant to turn a goal-seeking policy into a usable navigation policy.

## Files

- `curriculum_stage2_navigation_discipline.yaml`
- `ppo_stage2_navigation_discipline.yaml`

## Usage

Train from scratch with:

```bash
python gpudrive_chocolate/baselines/ppo/ppo_sb3.py \
  --config configs/curriculum_design/stage2_navigation_discipline/ppo_stage2_navigation_discipline.yaml
```

Or resume from a stage-1 checkpoint by setting `resume_from` inside:

- `ppo_stage2_navigation_discipline.yaml`

Generate a world report with:

```bash
python -m src.trfc.curriculum_report \
  --config configs/curriculum_design/stage2_navigation_discipline/curriculum_stage2_navigation_discipline.yaml
```
