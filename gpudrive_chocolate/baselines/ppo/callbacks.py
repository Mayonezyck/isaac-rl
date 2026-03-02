from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
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
        continuous_recording: bool = False,
        video_fps: int = 30,
        video_name_prefix: str = "training",
        keep_frames: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.render_every_updates = int(render_every_updates)
        self.render_rollout_steps = int(render_rollout_steps)
        self.render_dir = render_dir
        self.always_render = bool(always_render)
        self.continuous_recording = bool(continuous_recording)
        self.video_fps = max(1, int(video_fps))
        self.video_name_prefix = str(video_name_prefix)
        self.keep_frames = bool(keep_frames)
        self.update_count = 0
        self.recording = False
        self.frame_idx = 0
        self.rollout_idx = 0
        self.rollout_dir = None
        self.session_dir = None
        self.frame_dir = None
        self.video_path = None
        self._render_enabled_by_callback = False
        self._reset_rollout_stats()

    def _reset_rollout_stats(self) -> None:
        self._rollout_steps = 0
        self._rollout_reward_sum = 0.0
        self._rollout_goal_sum = 0.0
        self._rollout_off_road_sum = 0.0
        self._rollout_collision_sum = 0.0
        self._rollout_agent_steps = 0.0
        self._rollout_done_count = 0.0
        self._rollout_done_success_count = 0.0

    def _set_render_enabled(self, enabled: bool) -> None:
        try:
            self.training_env.env_method("set_render", bool(enabled))
        except Exception:
            pass

    def _start_continuous_recording(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{self.video_name_prefix}_{stamp}"
        self.session_dir = os.path.join(self.render_dir, session_name)
        self.frame_dir = os.path.join(self.session_dir, "frames")
        self.video_path = os.path.join(self.session_dir, f"{session_name}.mp4")
        os.makedirs(self.frame_dir, exist_ok=True)
        self.recording = True
        self.frame_idx = 0
        if not self.always_render:
            self._set_render_enabled(True)
            self._render_enabled_by_callback = True
        print(f"[capture] recording training video frames to {self.frame_dir}")

    def _capture_frame(self, frame_path: str) -> None:
        try:
            ok = bool(self.training_env.capture_frame(frame_path))
        except Exception:
            ok = False
        if ok:
            self.frame_idx += 1

    def _encode_video(self) -> None:
        if not self.frame_dir or self.frame_idx <= 0 or not self.video_path:
            return
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(self.video_fps),
            "-i",
            os.path.join(self.frame_dir, "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            self.video_path,
        ]
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            print(f"[capture] ffmpeg launch failed: {exc}")
            return
        if completed.returncode != 0:
            err = (completed.stderr or completed.stdout or "").strip()
            print(f"[capture] ffmpeg failed ({completed.returncode}): {err}")
            return
        print(f"[capture] wrote training video to {self.video_path}")
        if not self.keep_frames:
            shutil.rmtree(self.frame_dir, ignore_errors=True)

    def finalize(self) -> None:
        if self.continuous_recording:
            self.recording = False
            self._encode_video()
        if self._render_enabled_by_callback:
            self._set_render_enabled(False)
            self._render_enabled_by_callback = False

    def _on_training_start(self) -> None:
        if self.continuous_recording:
            self._start_continuous_recording()

    def _on_rollout_start(self) -> None:
        self._reset_rollout_stats()
        if self.continuous_recording:
            return
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
                self._rollout_reward_sum += float(np.nansum(rewards))
                self._rollout_steps += 1
        except Exception:
            pass

        try:
            env = self.locals.get("env", None)
            if env is not None and hasattr(env, "info_dict"):
                info = env.info_dict
                n_agents = float(info.get("num_controlled_agents", 0.0))
                self._rollout_agent_steps += n_agents
                self._rollout_goal_sum += float(info.get("goal_achieved", 0.0))
                self._rollout_off_road_sum += float(info.get("off_road", 0.0))
                self._rollout_collision_sum += float(info.get("collided", 0.0))
                self._rollout_done_count += float(info.get("done_count", 0.0))
                self._rollout_done_success_count += float(info.get("done_success_count", 0.0))
        except Exception:
            pass

        if not self.recording:
            return True

        if (
            not self.continuous_recording
            and self.render_rollout_steps > 0
            and self.frame_idx >= self.render_rollout_steps
        ):
            return True

        frame_root = self.frame_dir if self.continuous_recording else self.rollout_dir
        if frame_root:
            frame_path = os.path.join(frame_root, f"frame_{self.frame_idx:06d}.png")
            self._capture_frame(frame_path)
        return True

    def _on_rollout_end(self) -> None:
        self.update_count += 1
        if not self.continuous_recording:
            self.recording = False
            if not self.always_render:
                self._set_render_enabled(False)

        try:
            rewards = self.model.rollout_buffer.rewards
            avg_reward = float(np.nanmean(rewards))
            self.logger.record("choco/avg_reward", avg_reward)
        except Exception:
            pass

        # Per-rollout aggregates
        try:
            if self._rollout_agent_steps > 0:
                self.logger.record(
                    "rollout/goal_rate",
                    self._rollout_goal_sum / self._rollout_agent_steps,
                )
                self.logger.record(
                    "rollout/off_road_rate",
                    self._rollout_off_road_sum / self._rollout_agent_steps,
                )
                self.logger.record(
                    "rollout/collision_rate",
                    self._rollout_collision_sum / self._rollout_agent_steps,
                )
            if self._rollout_steps > 0:
                self.logger.record(
                    "rollout/mean_reward",
                    self._rollout_reward_sum / self._rollout_steps,
                )
            if self._rollout_done_count > 0:
                self.logger.record(
                    "rollout/mean_episode_len",
                    self._rollout_agent_steps / self._rollout_done_count,
                )
                self.logger.record(
                    "rollout/success_rate",
                    self._rollout_done_success_count / self._rollout_done_count,
                )
        except Exception:
            pass
