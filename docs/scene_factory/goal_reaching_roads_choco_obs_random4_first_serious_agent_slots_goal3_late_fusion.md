# Goal-3 Late-Fusion Run

Config:
- [goal_reaching_roads_choco_obs_random4_first_serious_agent_slots_goal3_late_fusion.yaml](/home/yz8733/Github/isaac-rl/configs/scene_factory/goal_reaching_roads_choco_obs_random4_first_serious_agent_slots_goal3_late_fusion.yaml)

What changed:
- Keeps the current SceneFactory goal-reaching setup.
- Swaps the flat RSL-RL MLP policy for the old choco-style late-fusion architecture.
- Keeps the current observation layout and reward stack unchanged for this step.

Late-fusion architecture:
- `ego_layers: [64, 64]`
- `road_layers: [96, 96]`
- `vehicle_layers: [96, 96]`
- `shared_layers: [128, 64]`
- `last_layer_dim_pi: 64`
- `last_layer_dim_vf: 64`
- `activation: relu`
- `pool: max`

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.train_student_vehicle_goal_multiagent_rsl_rl \
  --config configs/scene_factory/goal_reaching_roads_choco_obs_random4_first_serious_agent_slots_goal3_late_fusion.yaml \
  --headless
```

What to compare against the flat-MLP run:
- `Metrics/success_rate`
- `Episode_Reward/goal_bonus`
- `Metrics/final_distance_to_goal`
- `Train/mean_reward`
- `Train/mean_episode_length`

Important:
- This is only the architecture port.
- The observation density, reward formulation, PPO hyperparameters, and world curriculum are still the current SceneFactory ones.
