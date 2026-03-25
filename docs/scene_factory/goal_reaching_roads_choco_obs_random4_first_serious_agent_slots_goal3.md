# Goal-3 Serious Run

Config:
- [goal_reaching_roads_choco_obs_random4_first_serious_agent_slots_goal3.yaml](/home/yz8733/Github/isaac-rl/configs/scene_factory/goal_reaching_roads_choco_obs_random4_first_serious_agent_slots_goal3.yaml)

Purpose:
- Keep the per-agent shared-policy SceneFactory setup unchanged.
- Make success easier for a car-scale task by using `goal_reached_threshold_m: 3.0`.
- Make true route completion matter more by using `reward.goal_bonus: 100.0`.

What stayed the same:
- `4` parallel SceneFactory worlds.
- `4` vehicles per world, subject to spawn clamping if a selected world cannot support all of them.
- `choco_reference` observations.
- Lane-center and forbidden-lane rewards.
- Stable explicit-road, non-Fabric training path.

Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -m src.train_student_vehicle_goal_multiagent_rsl_rl \
  --config configs/scene_factory/goal_reaching_roads_choco_obs_random4_first_serious_agent_slots_goal3.yaml \
  --headless
```

What to watch in TensorBoard:
- `Metrics/success_rate`
- `Episode_Reward/goal_bonus`
- `Metrics/final_distance_to_goal`
- `WorldEpisode/success_count`
- `WorldEpisode/lane_forbidden_count`
- `WorldEpisode/collision_count`

Expected effect:
- `success_rate` should be able to move off zero much earlier than before.
- `goal_bonus` should stop being flat at zero once vehicles begin finishing routes.
