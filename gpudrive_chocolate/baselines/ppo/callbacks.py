from __future__ import annotations

import os
from time import perf_counter
from stable_baselines3.common.callbacks import BaseCallback


class RolloutCaptureCallback(BaseCallback):
    def __init__(
        self,
        *,
        render_every_updates: int = 1000,
        render_rollout_steps: int = 0,
        render_dir: str = "runs/capture",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.render_every_updates = int(render_every_updates)
        self.render_rollout_steps = int(render_rollout_steps)
        self.render_dir = render_dir
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

    def _on_step(self) -> bool:
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
