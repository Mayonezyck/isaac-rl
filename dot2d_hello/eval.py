from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_dot2d import Dot2DEnv


EVAL_CONFIG: Dict[str, Any] = {
    "run_dir": "runs/dot2d",
    "episodes": 20,
    "render": False,
    "render_mode": "human",  # "human" or "rgb_array"
    "ascii_render": False,
}


def load_config(run_dir: Path) -> Dict[str, Any]:
    with (run_dir / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def make_env(env_config: Dict[str, Any], render: bool, render_mode: str) -> Dot2DEnv:
    if render:
        env_config = dict(env_config)
        env_config["render_mode"] = render_mode
    return Dot2DEnv(env_config)


def load_model(run_dir: Path, algo: str, vec_env) -> Any:
    model_path = run_dir / "model.zip"
    if algo == "PPO":
        return PPO.load(model_path, env=vec_env)
    if algo == "A2C":
        return A2C.load(model_path, env=vec_env)
    if algo == "DQN":
        return DQN.load(model_path, env=vec_env)
    raise ValueError(f"Unsupported algo: {algo}")


def ascii_draw(pos: np.ndarray, goal: np.ndarray, size: int = 11) -> None:
    grid = np.full((size, size), ".", dtype="<U1")
    def to_idx(v: float) -> int:
        return int(np.clip((v + 2.0) / 4.0 * (size - 1), 0, size - 1))

    gx, gy = to_idx(goal[0]), to_idx(goal[1])
    px, py = to_idx(pos[0]), to_idx(pos[1])
    grid[size - 1 - gy, gx] = "G"
    grid[size - 1 - py, px] = "O"
    print("\n".join("".join(row) for row in grid))
    print("-")


def main() -> None:
    run_dir = Path(EVAL_CONFIG["run_dir"])
    config = load_config(run_dir)

    env_config = config["env_config"]
    render = EVAL_CONFIG["render"]
    render_mode = EVAL_CONFIG["render_mode"]

    env = DummyVecEnv([lambda: make_env(env_config, render, render_mode)])
    if config["use_vecnormalize"]:
        env = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), env)
        env.training = False
        env.norm_reward = False

    algo = config["algo"] or ("DQN" if env_config["action_type"] == "discrete" else "PPO")
    algo = str(algo).upper()
    model = load_model(run_dir, algo, env)

    success = 0
    returns = []
    lengths = []

    for _ in range(EVAL_CONFIG["episodes"]):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            done = bool(done[0])
            total_reward += float(reward[0])
            steps += 1
            if render:
                env.envs[0].render()
            if EVAL_CONFIG["ascii_render"]:
                base_env = env.envs[0]
                ascii_draw(base_env._pos, base_env._goal)

        returns.append(total_reward)
        lengths.append(steps)
        if bool(info[0].get("distance", 1e9) < env_config["goal_radius"]):
            success += 1

    success_rate = success / EVAL_CONFIG["episodes"]
    avg_return = float(np.mean(returns))
    avg_length = float(np.mean(lengths))

    print(f"success_rate: {success_rate:.2f}")
    print(f"avg_return: {avg_return:.2f}")
    print(f"avg_length: {avg_length:.1f}")

    env.close()


if __name__ == "__main__":
    main()
