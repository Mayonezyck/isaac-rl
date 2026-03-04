# Navigation Geometric V1

This curriculum family is the first version that assumes:

- structured ego / road / vehicle observations
- geometric lane reward and route-progress reward
- contact-based road-edge and vehicle-contact logic kept as hard guards

Stages:

1. `stage1_goal_route_geom`
   - small dry-world warm start
   - optimize route-following and goal reaching
   - no vehicle-contact termination
   - search-derived promoted winner: `ppo_stage1_goal_route_geom_best.yaml`

2. `stage2_lane_safety_geom`
   - dry safety stage
   - road-edge termination, vehicle-contact termination, TTC shaping
   - stronger lane geometry reward

3. `stage3_traffic_weather_geom`
   - denser traffic and moderate wet-road variation
   - same geometric route/lane reward plus safety constraints

These configs are separate from the older stage files on purpose so comparisons stay clean.
