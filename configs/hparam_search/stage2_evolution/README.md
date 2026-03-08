# Stage 2 Evolution Search

This folder contains search presets for the geometric stage-2 safety task:

- [ppo_stage2_lane_safety_geom.yaml](/home/yz8733/Github/isaac-rl/configs/curriculum_design/navigation_geom_v1/ppo_stage2_lane_safety_geom.yaml)
- [ppo_stage2_lane_safety_geom_warmstart.yaml](/home/yz8733/Github/isaac-rl/configs/curriculum_design/navigation_geom_v1/ppo_stage2_lane_safety_geom_warmstart.yaml)

This stage is the right surface for crash avoidance tuning because it already has:

- road-edge termination
- vehicle-contact termination
- TTC shaping
- geometric lane reward and route-progress reward

Smoke test:

```bash
python -u gpudrive_chocolate/baselines/ppo/evolve_hparams.py \
  --study configs/hparam_search/stage2_evolution/search_stage2_lane_safety_geom_smoke.yaml
```

Real study:

```bash
python -u gpudrive_chocolate/baselines/ppo/evolve_hparams.py \
  --study configs/hparam_search/stage2_evolution/search_stage2_lane_safety_geom.yaml
```

Warm-start smoke study from the stage-1 checkpoint:

```bash
python -u gpudrive_chocolate/baselines/ppo/evolve_hparams.py \
  --study configs/hparam_search/stage2_evolution/search_stage2_lane_safety_geom_warmstart_smoke.yaml
```

Warm-start real study from the stage-1 checkpoint:

```bash
python -u gpudrive_chocolate/baselines/ppo/evolve_hparams.py \
  --study configs/hparam_search/stage2_evolution/search_stage2_lane_safety_geom_warmstart.yaml
```

Quick TTC-fix study (warm-start from stage1 `cont0` 3M checkpoint):

```bash
python -u gpudrive_chocolate/baselines/ppo/evolve_hparams.py \
  --study configs/hparam_search/stage2_evolution/search_stage2_lane_safety_geom_ttcfix_quick_from_stage1_3m.yaml
```

Quick road-edge TTC study (same warm-start, but tunes forbidden-road TTC term):

```bash
python -u gpudrive_chocolate/baselines/ppo/evolve_hparams.py \
  --study configs/hparam_search/stage2_evolution/search_stage2_lane_safety_geom_roadedge_ttc_quick_from_stage1_3m.yaml
```

Strict-phase deep overnight study (keeps strict resume checkpoint, longer trials, larger population):

```bash
python -u gpudrive_chocolate/baselines/ppo/evolve_hparams.py \
  --study configs/hparam_search/stage2_evolution/search_stage2_lane_safety_geom_strict_deep.yaml
```

The default real preset is intentionally broader than the stage-1 search:

- `population_size: 12`
- `generations: 5`
- `timesteps_per_trial: 131072`

That gives the safety search more time to separate lineages without relying on a tiny early-training signal.
