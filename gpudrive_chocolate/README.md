# gpudrive_chocolate

Minimal SB3/IPPO training scaffold that reuses the Chocolate world builder and controller.

## What this does
- Builds Isaac Sim scenes using the same chocolate Waymo builder used in `capture_choco_standalone.py`.
- Wraps `ChocolateEnv` in a vectorized SB3-style multi-agent environment.
- Trains with gpudrive's `IPPO` implementation.

## Quick start
1. Update `gpudrive_chocolate/config/ppo_choco_sb3.yaml` with the path to your chocolate YAML config.
2. Run:

```bash
python gpudrive_chocolate/baselines/ppo/ppo_sb3.py
```

## TODOs
- Add collision and off-road detection from Isaac Sim and hook them into reward shaping.
- Expand observations (see `gpudrive_chocolate/env/sb3_wrapper.py` for the TODO location).
- Add a proper logging callback (e.g., success rate per episode).

## Rollout capture
- Set `render_every_updates`, `render_rollout_steps`, and `render_dir` in `gpudrive_chocolate/config/ppo_choco_sb3.yaml`.
- Ensure your chocolate config sets `app.headless: false` and `env.render: true` so viewport frames can be captured.
