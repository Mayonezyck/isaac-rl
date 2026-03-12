import argparse
import json
import os
import sys
import traceback
import zipfile
import yaml
from box import Box
from datetime import datetime
from typing import Callable

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gpudrive_chocolate.env.sb3_wrapper import ChocolateSB3MultiAgentEnv
from gpudrive_chocolate.networks.late_fusion_policy import LateFusionPolicy
from gpudrive_chocolate.baselines.ppo.callbacks import RolloutCaptureCallback
from gpudrive_chocolate.baselines.ppo.masked_rollout_buffer import MaskedRolloutBuffer


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return Box(yaml.safe_load(f))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="gpudrive_chocolate/config/ppo_choco_stage2.yaml",
        help="Path to the PPO experiment YAML.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore resume_from in the experiment config and start from scratch.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier used to name the TensorBoard output directory.",
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Optional root directory for TensorBoard logs. Defaults to exp_config.runs_root or runs.",
    )
    return parser.parse_args()


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule that decays to zero."""

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def build_resume_custom_objects(env: ChocolateSB3MultiAgentEnv) -> dict[str, object]:
    # SB3 checkpoints may embed NumPy-internal module paths in pickled metadata.
    # Reuse the current env spaces and force a reset instead of deserializing them.
    return {
        "_last_obs": None,
        "_last_episode_starts": None,
        "observation_space": env.observation_space,
        "action_space": env.action_space,
    }


def _load_checkpoint_data(checkpoint_path: str) -> dict:
    with zipfile.ZipFile(checkpoint_path, "r") as zf:
        raw = zf.read("data")
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to parse checkpoint metadata from {checkpoint_path}: {exc}") from exc


def _align_curriculum_obs_to_resume(exp_config: Box) -> dict | None:
    resume_from = getattr(exp_config, "resume_from", None)
    if not resume_from:
        return None
    if str(getattr(exp_config, "policy_type", "mlp")) != "late_fusion":
        return None
    if not bool(getattr(exp_config, "align_obs_with_resume", True)):
        return None

    ckpt_data = _load_checkpoint_data(str(resume_from))
    ckpt_pk = ckpt_data.get("policy_kwargs", {}) if isinstance(ckpt_data, dict) else {}
    if not isinstance(ckpt_pk, dict):
        ckpt_pk = {}

    ckpt_road_k = ckpt_pk.get("road_point_k", None)
    ckpt_vehicle_k = ckpt_pk.get("vehicle_k", None)
    ckpt_road_dim = ckpt_pk.get("road_point_dim", None)

    if ckpt_road_k is None and ckpt_vehicle_k is None and ckpt_road_dim is None:
        return ckpt_pk

    with open(str(exp_config.choco_config_path), "r", encoding="utf-8") as f:
        choco_cfg = yaml.safe_load(f)
    if not isinstance(choco_cfg, dict):
        raise RuntimeError(f"Invalid curriculum YAML at {exp_config.choco_config_path}")

    env_cfg = choco_cfg.setdefault("env", {})
    if not isinstance(env_cfg, dict):
        env_cfg = {}
        choco_cfg["env"] = env_cfg

    changed = False
    if ckpt_road_k is not None:
        cur_road_k = int(env_cfg.get("road_points_k", 0))
        if int(ckpt_road_k) != cur_road_k:
            env_cfg["road_points_k"] = int(ckpt_road_k)
            changed = True
    if ckpt_vehicle_k is not None:
        cur_vehicle_k = int(env_cfg.get("vehicle_obs_k", 0))
        if int(ckpt_vehicle_k) != cur_vehicle_k:
            env_cfg["vehicle_obs_k"] = int(ckpt_vehicle_k)
            changed = True
    if ckpt_road_dim in (3, 5):
        want_dirs = bool(int(ckpt_road_dim) == 5)
        cur_dirs = bool(env_cfg.get("road_points_include_dirs", False))
        if cur_dirs != want_dirs:
            env_cfg["road_points_include_dirs"] = want_dirs
            changed = True

    if changed:
        os.makedirs("runs/autofix_curriculum", exist_ok=True)
        src_name = os.path.basename(str(exp_config.choco_config_path))
        stem = src_name[:-5] if src_name.endswith(".yaml") else src_name
        ts = datetime.now().strftime("%m_%d_%H_%M_%S_%f")
        out_path = os.path.join("runs/autofix_curriculum", f"{stem}.resume_aligned.{ts}.yaml")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(choco_cfg, f, sort_keys=False)
        exp_config.choco_config_path = out_path
        print(
            f"[train] aligned curriculum obs layout to resume checkpoint: "
            f"road_points_k={env_cfg.get('road_points_k')} "
            f"vehicle_obs_k={env_cfg.get('vehicle_obs_k')} "
            f"road_points_include_dirs={env_cfg.get('road_points_include_dirs')} "
            f"path={out_path}",
            flush=True,
        )
    return ckpt_pk


def train(exp_config: Box, *, run_id_override: str | None = None, runs_root_override: str | None = None):
    if isinstance(exp_config.device, str) and exp_config.device.startswith("cuda") and ":" not in exp_config.device:
        exp_config.device = "cuda:0"
    if isinstance(exp_config.device, str) and exp_config.device.startswith("cuda"):
        torch.cuda.set_device(exp_config.device)

    ckpt_policy_kwargs = _align_curriculum_obs_to_resume(exp_config)

    env = ChocolateSB3MultiAgentEnv(
        choco_config_path=exp_config.choco_config_path,
        exp_config=exp_config,
        device=exp_config.device,
        reward_type=exp_config.reward_type,
        collision_weight=exp_config.collision_weight,
        goal_achieved_weight=exp_config.goal_achieved_weight,
        off_road_weight=exp_config.off_road_weight,
        log_distance_weight=exp_config.log_distance_weight,
    )

    exp_config.num_envs = env.num_envs
    exp_config.batch_size = (
        exp_config.num_envs * exp_config.n_steps
    ) // exp_config.num_minibatches

    run_id = run_id_override or getattr(exp_config, "run_id", None) or datetime.now().strftime("%m_%d_%H_%S")
    runs_root = runs_root_override or getattr(exp_config, "runs_root", None) or "runs"
    tensorboard_root = os.path.join(runs_root, run_id)
    print(f"[train] tensorboard_root={tensorboard_root}")

    policy_type = str(getattr(exp_config, "policy_type", "mlp"))
    policy_kwargs = {}
    if policy_type == "mlp":
        if hasattr(exp_config, "policy_net_arch") and exp_config.policy_net_arch:
            policy_kwargs["net_arch"] = exp_config.policy_net_arch
        policy = "MlpPolicy"
    elif policy_type == "late_fusion":
        lf_cfg = getattr(exp_config, "late_fusion", {})
        env_cfg = env.choco_cfg.get("env", {})
        road_points_enable = bool(env_cfg.get("road_points_enable", False))
        road_points_include_dirs = bool(env_cfg.get("road_points_include_dirs", False))
        vehicle_obs_enable = bool(env_cfg.get("vehicle_obs_enable", False))
        point_layers = lf_cfg.get("point_layers", [64, 64])
        policy = LateFusionPolicy
        policy_kwargs.update(
            {
                "ego_dim": int(lf_cfg.get("ego_dim", 11)),
                "point_dim": int(lf_cfg.get("point_dim", 3)),
                "point_k": lf_cfg.get("point_k", None),
                "road_point_dim": int(
                    lf_cfg.get(
                        "road_point_dim",
                        (5 if road_points_include_dirs else 3) if road_points_enable else 0,
                    )
                ),
                "road_point_k": int(
                    lf_cfg.get(
                        "road_point_k",
                        int(env_cfg.get("road_points_k", 0)) if road_points_enable else 0,
                    )
                ),
                "vehicle_dim": int(
                    lf_cfg.get("vehicle_dim", 6 if vehicle_obs_enable else 0)
                ),
                "vehicle_k": int(
                    lf_cfg.get(
                        "vehicle_k",
                        int(env_cfg.get("vehicle_obs_k", 0)) if vehicle_obs_enable else 0,
                    )
                ),
                "ego_layers": lf_cfg.get("ego_layers", [64, 64]),
                "point_layers": point_layers,
                "road_layers": lf_cfg.get("road_layers", point_layers),
                "vehicle_layers": lf_cfg.get("vehicle_layers", point_layers),
                "shared_layers": lf_cfg.get("shared_layers", [64]),
                "last_layer_dim_pi": int(lf_cfg.get("last_layer_dim_pi", 64)),
                "last_layer_dim_vf": int(lf_cfg.get("last_layer_dim_vf", 64)),
                "act": str(lf_cfg.get("act", "relu")),
                "dropout": float(lf_cfg.get("dropout", 0.0)),
                "pool": str(lf_cfg.get("pool", "max")),
            }
        )
        if ckpt_policy_kwargs is not None:
            keys = ["road_point_dim", "road_point_k", "vehicle_dim", "vehicle_k", "ego_dim"]
            mismatches = []
            for key in keys:
                ck = ckpt_policy_kwargs.get(key, None)
                cur = policy_kwargs.get(key, None)
                if ck is not None and cur is not None and int(ck) != int(cur):
                    mismatches.append((key, ck, cur))
            if mismatches:
                details = ", ".join([f"{k}: checkpoint={a} current={b}" for k, a, b in mismatches])
                raise RuntimeError(
                    "Resume checkpoint policy layout is incompatible with current late-fusion layout after "
                    f"alignment attempt. {details}"
                )
    else:
        raise ValueError(f"Unknown policy_type: {policy_type}")

    if getattr(exp_config, "resume_from", None):
        model = PPO.load(
            exp_config.resume_from,
            env=env,
            device=exp_config.device,
            custom_objects=build_resume_custom_objects(env),
            tensorboard_log=tensorboard_root,
            rollout_buffer_class=MaskedRolloutBuffer,
        )
    else:
        model = PPO(
            policy=policy,
            env=env,
            n_steps=exp_config.n_steps,
            batch_size=exp_config.batch_size,
            n_epochs=exp_config.n_epochs,
            gamma=exp_config.gamma,
            gae_lambda=exp_config.gae_lambda,
            clip_range=exp_config.clip_range,
            ent_coef=exp_config.ent_coef,
            vf_coef=exp_config.vf_coef,
            learning_rate=linear_schedule(float(exp_config.lr)),
            verbose=exp_config.verbose,
            seed=exp_config.seed,
            device=exp_config.device,
            tensorboard_log=tensorboard_root,
            policy_kwargs=policy_kwargs,
            rollout_buffer_class=MaskedRolloutBuffer,
        )

    capture_callback = RolloutCaptureCallback(
        render_every_updates=exp_config.render_every_updates,
        render_rollout_steps=exp_config.render_rollout_steps,
        render_dir=exp_config.render_dir,
        always_render=bool(getattr(exp_config, "render_during_training", False)),
        continuous_recording=bool(getattr(exp_config, "record_training_video", False)),
        video_fps=int(getattr(exp_config, "video_fps", 30)),
        video_name_prefix=str(getattr(exp_config, "video_name_prefix", "training")),
        keep_frames=bool(getattr(exp_config, "video_keep_frames", False)),
        log_step_metrics=bool(getattr(exp_config, "log_step_metrics", False)),
        log_detailed_metrics=bool(getattr(exp_config, "log_detailed_metrics", False)),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=int(exp_config.save_freq),
        save_path=str(exp_config.save_dir),
        name_prefix=str(exp_config.save_prefix),
    )

    interrupted = False
    start_num_timesteps = int(getattr(model, "num_timesteps", 0))
    print(
        "[train] learn_start "
        f"initial_num_timesteps={start_num_timesteps} "
        f"total_timesteps={int(exp_config.total_timesteps)}"
    )
    try:
        model.learn(
            total_timesteps=exp_config.total_timesteps,
            callback=[capture_callback, checkpoint_cb],
        )
        end_num_timesteps = int(getattr(model, "num_timesteps", 0))
        print(f"[train] learn_end final_num_timesteps={end_num_timesteps}")
    except KeyboardInterrupt:
        interrupted = True
        print("[train] interrupted by user, finalizing video output...")
    except Exception as exc:
        print(f"[train] fatal error during learn: {exc}")
        traceback.print_exc()
        raise
    finally:
        try:
            capture_callback.finalize()
        finally:
            env.close()

    if interrupted:
        return


if __name__ == "__main__":
    args = parse_args()
    exp_config = load_config(args.config)
    if args.fresh:
        exp_config.resume_from = None
    if args.device is not None:
        exp_config.device = args.device
    train(
        exp_config,
        run_id_override=args.run_id,
        runs_root_override=args.runs_root,
    )
