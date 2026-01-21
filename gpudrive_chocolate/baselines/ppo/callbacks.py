from __future__ import annotations

import os
from time import perf_counter
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RolloutCaptureCallback(BaseCallback):
    def __init__(
        self,
        *,
        render_every_updates: int = 1000,
        render_rollout_steps: int = 0,
        render_dir: str = "runs/capture",
        always_render: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.render_every_updates = int(render_every_updates)
        self.render_rollout_steps = int(render_rollout_steps)
        self.render_dir = render_dir
        self.always_render = bool(always_render)
        self.update_count = 0
        self.recording = False
        self.frame_idx = 0
        self.rollout_idx = 0

    def _on_rollout_start(self) -> None:
        if self.render_every_updates <= 0:
            self.recording = False
            return

        next_update = self.update_count + 1
        self.recording = (next_update % self.render_every_updates) == 0
        self.frame_idx = 0
        if self.recording:
            self.rollout_idx += 1
            self.start_time = perf_counter()
            self.rollout_dir = os.path.join(
                self.render_dir, f"rollout_{self.rollout_idx:05d}"
            )
            os.makedirs(self.rollout_dir, exist_ok=True)
            if not self.always_render:
                try:
                    self.training_env.env_method("set_render", True)
                except Exception:
                    pass

    def _on_step(self) -> bool:
        # Log per-step reward average so TensorBoard shows a series of points.
        try:
            rewards = self.locals.get("rewards", None)
            if rewards is not None:
                avg_reward = float(np.nanmean(rewards))
                self.logger.record("choco/avg_reward_step", avg_reward)
        except Exception:
            pass

        if not self.recording:
            return True

        if self.render_rollout_steps > 0 and self.frame_idx >= self.render_rollout_steps:
            return True

        frame_path = os.path.join(self.rollout_dir, f"frame_{self.frame_idx:06d}.png")
        try:
            self.training_env.capture_frame(frame_path)
        except Exception:
            pass

        self.frame_idx += 1
        return True

    def _on_rollout_end(self) -> None:
        self.update_count += 1
        self.recording = False
        if not self.always_render:
            try:
                self.training_env.env_method("set_render", False)
            except Exception:
                pass

        try:
            rewards = self.model.rollout_buffer.rewards
            avg_reward = float(np.nanmean(rewards))
            self.logger.record("choco/avg_reward", avg_reward)
        except Exception:
            pass
