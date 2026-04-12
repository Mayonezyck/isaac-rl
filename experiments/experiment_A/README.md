# Experiment A: Throughput Scaling Comparison

## Goal
Compare **CASPS** (controlled agent steps per second) between the gpudrive_chocolate
baseline pipeline and SceneFactory, as a function of world count.
Both pipelines use the same Waymo scene (`scene_000077.json`, 28 controllable agents),
the same PhysX rigid-body dynamics, and the same GPU (RTX PRO 6000 Blackwell).

## Scene
`scene_000077.json` — selected because it has **28 controllable agents** (above the
16-agent-per-world cap), ensuring all 16 slots are filled in every world.

## Configurations

| Pipeline          | Worlds | Agent slots | Config |
|-------------------|--------|-------------|--------|
| gpudrive_choco    | 32     | 512         | `baseline_config/ppo_expA_32w.yaml` |
| gpudrive_choco    | 64     | 1,024       | `baseline_config/ppo_expA_64w.yaml` |
| gpudrive_choco    | 128    | 2,048       | `baseline_config/ppo_expA_128w.yaml` |
| gpudrive_choco    | 256    | 4,096       | `baseline_config/ppo_expA_256w.yaml` |
| SceneFactory      | 32     | 512         | `sceneFactory_config/train_expA_32w.yaml` |
| SceneFactory      | 64     | 1,024       | `sceneFactory_config/train_expA_64w.yaml` |
| SceneFactory      | 128    | 2,048       | `sceneFactory_config/train_expA_128w.yaml` |
| SceneFactory      | 256    | 4,096       | `sceneFactory_config/train_expA_256w.yaml` |

## Training duration
- **Baseline (gpudrive_choco):** 10M `total_timesteps` (~150-300 PPO iterations
  depending on world count). Enough for ~100+ steady-state perf/ data points.
- **SceneFactory:** 300 PPO iterations (`max_iterations: 300`). Skip first ~50
  for warmup, use remaining ~250 for throughput measurement.

## Commands

All commands should be run from the repository root:
```bash
cd /home/yz8733/Github/isaac-rl
```

### Baseline (gpudrive_chocolate) — runs on `cuda:1`

```bash
# 32 worlds x 16 agents = 512 agent slots
python -u gpudrive_chocolate/baselines/ppo/ppo_sb3.py \
  --config experiments/experiment_A/baseline_config/ppo_expA_32w.yaml \
  --run-id expA_baseline_32w

# 64 worlds x 16 agents = 1,024 agent slots
python -u gpudrive_chocolate/baselines/ppo/ppo_sb3.py \
  --config experiments/experiment_A/baseline_config/ppo_expA_64w.yaml \
  --run-id expA_baseline_64w

# 128 worlds x 16 agents = 2,048 agent slots
python -u gpudrive_chocolate/baselines/ppo/ppo_sb3.py \
  --config experiments/experiment_A/baseline_config/ppo_expA_128w.yaml \
  --run-id expA_baseline_128w

# 256 worlds x 16 agents = 4,096 agent slots  (may OOM — that is a valid result)
python -u gpudrive_chocolate/baselines/ppo/ppo_sb3.py \
  --config experiments/experiment_A/baseline_config/ppo_expA_256w.yaml \
  --run-id expA_baseline_256w
```

### SceneFactory — runs on `cuda:0`

```bash
# 32 worlds x 16 agents = 512 agent slots
python -u -m src.train_student_vehicle_goal_multiagent_rsl_rl \
  --config experiments/experiment_A/sceneFactory_config/train_expA_32w.yaml \
  --headless

# 64 worlds x 16 agents = 1,024 agent slots
python -u -m src.train_student_vehicle_goal_multiagent_rsl_rl \
  --config experiments/experiment_A/sceneFactory_config/train_expA_64w.yaml \
  --headless

# 128 worlds x 16 agents = 2,048 agent slots
python -u -m src.train_student_vehicle_goal_multiagent_rsl_rl \
  --config experiments/experiment_A/sceneFactory_config/train_expA_128w.yaml \
  --headless

# 256 worlds x 16 agents = 4,096 agent slots
python -u -m src.train_student_vehicle_goal_multiagent_rsl_rl \
  --config experiments/experiment_A/sceneFactory_config/train_expA_256w.yaml \
  --headless
```

## Collecting results

After each run, extract CASPS from TensorBoard:

```bash
python3 -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np, glob, os

run_dirs = sorted(glob.glob('runs/expA_*'))
for rd in run_dirs:
    ef = glob.glob(os.path.join(rd, '**', 'events.out.tfevents.*'), recursive=True)
    if not ef:
        continue
    ea = EventAccumulator(ef[0], size_guidance={'scalars': 0})
    ea.Reload()
    for tag in ['perf/controlled_agent_steps_per_sec', 'Perf/CASPS']:
        try:
            evts = ea.Scalars(tag)
            vals = [e.value for e in evts[50:]]  # skip warmup
            print(f'{rd:60s}  {tag:45s}  mean={np.mean(vals):.1f}  std={np.std(vals):.1f}  n={len(vals)}')
        except:
            pass
"
```

For SceneFactory runs, logs go to `logs/rsl_rl/expA_scenefactory/expA_sf_{N}w/`.

## Results

| Pipeline       | 32w           | 64w           | 128w             | 256w              |
|----------------|---------------|---------------|------------------|-------------------|
| gpudrive_choco | 174 ± 27      | 159 ± 29      | 164 ± 1          | 152 ± 0           |
| SceneFactory   | 3,870 ± 66    | 7,173 ± 114   | 12,225 ± 1,520   | 19,250 ± 2,984    |

**Key findings:**
- The baseline CASPS is flat at ~152–174 across all world counts, confirming that
  per-agent Python loops are the bottleneck (not the PhysX solver).
- SceneFactory CASPS scales near-linearly: **22× faster at 32 worlds, 127× faster
  at 256 worlds** compared to the baseline.
- Neither pipeline OOM'd at 256 worlds on the RTX PRO 6000 Blackwell (96 GB VRAM).
- SceneFactory achieves ~19,250 controlled agent steps/sec at 4,096 agent slots,
  demonstrating that full rigid-body PPO training is practical on a single GPU.

## Notes
- gpudrive_choco uses `cuda:1`, SceneFactory uses `cuda:0`. Do NOT run both
  on the same GPU simultaneously — that would contaminate throughput numbers.
- Run each config alone, one at a time.
- If baseline 128w or 256w OOMs or hangs, record that as a result.
- The 64w baseline number should roughly match the existing `03_20` run (~167 CASPS)
  as a sanity check.
