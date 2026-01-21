# Dot2D RL Hello World

A minimal RL "hello world" using Gymnasium + Stable-Baselines3. A dot moves in 2D to reach a goal.

## Quickstart

```bash
cd dot2d_hello
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py
python eval.py
```

## Configuration

All key knobs live in `train.py` under the single `CONFIG` dict. `env_dot2d.py` only reads the config it is passed.

Highlights:
- Action space: `env_config.action_type` = `"continuous"` or `"discrete"`
- Dynamics: `env_config.dynamics_mode` = `"position"` or `"velocity"`
- Observation: `env_config.observation_mode` = `"absolute"`, `"relative"`, or `"relative_only"`
- Velocity in obs: `env_config.vel_enabled`
- Rewards: `env_config.reward_mode` = `"dense_neg_dist"`, `"progress"`, or `"sparse"`
- Termination: `env_config.goal_radius`, `env_config.max_steps`
- Randomization: `env_config.spawn_range`, `env_config.goal_range`, `env_config.fixed_spawn`, `env_config.fixed_goal`
- Training: `algo`, `total_timesteps`, `num_envs`, `vec_env_type`, `action_noise_std`
- Normalization: `use_vecnormalize`, `norm_obs`, `norm_reward`

`eval.py` reads the saved `config.json` from `runs/dot2d` and runs deterministic episodes.

## Files

- `env_dot2d.py`: Gymnasium env with discrete + continuous actions and multiple reward/obs/dynamics modes
- `train.py`: Training entry point with a single `CONFIG`
- `eval.py`: Evaluation entry point
- `requirements.txt`: Minimal dependencies

## Notes

- Rendering is optional and uses matplotlib only when `render_mode` is set.
- For ASCII render, set `EVAL_CONFIG["ascii_render"] = True` in `eval.py`.
