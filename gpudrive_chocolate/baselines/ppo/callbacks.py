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
        self._rollout_collect_start_time = None
        self._training_spawned_count_total = 0.0
        self._training_done_success_count_total = 0.0
        self._training_done_count_total = 0.0
        self._training_vehicle_collision_count_total = 0.0
        self._training_done_vehicle_collided_count_total = 0.0
        self._reset_rollout_stats()

    def _reset_rollout_stats(self) -> None:
        self._rollout_steps = 0
        self._rollout_reward_sum = 0.0
        self._rollout_base_reward_sum = 0.0
        self._rollout_spawned_count = 0.0
        self._rollout_goal_sum = 0.0
        self._rollout_success_latched_sum = 0.0
        self._rollout_road_contact_done_sum = 0.0
        self._rollout_vehicle_contact_done_sum = 0.0
        self._rollout_off_road_sum = 0.0
        self._rollout_lane_hit_sum = 0.0
        self._rollout_collision_sum = 0.0
        self._rollout_road_collision_sum = 0.0
        self._rollout_vehicle_collision_sum = 0.0
        self._rollout_vehicle_collided_sum = 0.0
        self._rollout_below_min_z_sum = 0.0
        self._rollout_agent_steps = 0.0
        self._rollout_valid_agent_steps = 0.0
        self._rollout_active_agent_steps = 0.0
        self._rollout_pending_agent_steps = 0.0
        self._rollout_done_count = 0.0
        self._rollout_done_success_count = 0.0
        self._rollout_done_vehicle_collided_sum = 0.0
        self._rollout_timeout_count = 0.0
        self._rollout_dist_sum = 0.0
        self._rollout_active_dist_sum = 0.0
        self._rollout_lane_error_sum = 0.0
        self._rollout_heading_alignment_sum = 0.0
        self._rollout_route_progress_sum = 0.0

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
        self._training_spawned_count_total = 0.0
        self._training_done_success_count_total = 0.0
        self._training_done_count_total = 0.0
        self._training_vehicle_collision_count_total = 0.0
        self._training_done_vehicle_collided_count_total = 0.0
        if self.continuous_recording:
            self._start_continuous_recording()

    def _on_rollout_start(self) -> None:
        self._reset_rollout_stats()
        self._rollout_collect_start_time = perf_counter()
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
                self._rollout_reward_sum += avg_reward
                self._rollout_steps += 1
        except Exception:
            pass

        try:
            env = self.locals.get("env", None)
            if env is not None and hasattr(env, "info_dict"):
                info = env.info_dict
                n_agents = float(info.get("num_controlled_agents", 0.0))
                spawned_count = float(info.get("spawned_count_step", 0.0))
                self._rollout_agent_steps += n_agents
                self._rollout_valid_agent_steps += float(info.get("num_valid_agents", 0.0))
                self._rollout_active_agent_steps += float(info.get("num_active_agents", 0.0))
                self._rollout_pending_agent_steps += float(info.get("pending_respawn_count", 0.0))
                self._rollout_spawned_count += spawned_count
                self._rollout_goal_sum += float(info.get("new_success_count", 0.0))
                self._rollout_success_latched_sum += float(info.get("success_latched_count", 0.0))
                self._rollout_road_contact_done_sum += float(info.get("road_contact_done_count", 0.0))
                self._rollout_vehicle_contact_done_sum += float(info.get("vehicle_contact_done_count", 0.0))
                self._rollout_off_road_sum += float(info.get("off_road", 0.0))
                self._rollout_lane_hit_sum += float(info.get("lane_hit_count", 0.0))
                self._rollout_collision_sum += float(info.get("collided", 0.0))
                self._rollout_road_collision_sum += float(info.get("road_collided_count", 0.0))
                # Done-conditioned vehicle collision numerator.
                self._rollout_vehicle_collision_sum += float(info.get("vehicle_contact_done_count", 0.0))
                # Legacy controlled-conditioned vehicle collision count (debug only).
                self._rollout_vehicle_collided_sum += float(info.get("vehicle_collided_count", 0.0))
                self._rollout_below_min_z_sum += float(info.get("below_min_z_count", 0.0))
                self._rollout_done_count += float(info.get("done_count", 0.0))
                self._rollout_done_success_count += float(info.get("done_success_count", 0.0))
                self._rollout_done_vehicle_collided_sum += float(
                    info.get("done_vehicle_collided_count", 0.0)
                )
                self._training_spawned_count_total += spawned_count
                self._training_done_success_count_total += float(info.get("done_success_count", 0.0))
                self._training_done_count_total += float(info.get("done_count", 0.0))
                self._training_vehicle_collision_count_total += float(
                    info.get("vehicle_contact_done_count", 0.0)
                )
                self._training_done_vehicle_collided_count_total += float(
                    info.get("done_vehicle_collided_count", 0.0)
                )
                self._rollout_timeout_count += float(info.get("truncated", 0.0))
                self._rollout_base_reward_sum += float(info.get("mean_base_reward_step", 0.0))
                self._rollout_dist_sum += (
                    float(info.get("mean_dist_to_goal_m", 0.0))
                    * float(info.get("num_valid_agents", 0.0))
                )
                self._rollout_active_dist_sum += (
                    float(info.get("mean_active_dist_to_goal_m", 0.0))
                    * float(info.get("num_active_agents", 0.0))
                )
                self._rollout_lane_error_sum += (
                    float(info.get("mean_lane_error_m", 0.0))
                    * float(info.get("num_active_agents", 0.0))
                )
                self._rollout_heading_alignment_sum += (
                    float(info.get("mean_heading_alignment", 0.0))
                    * float(info.get("num_active_agents", 0.0))
                )
                self._rollout_route_progress_sum += (
                    float(info.get("mean_route_progress_m", 0.0))
                    * float(info.get("num_active_agents", 0.0))
                )

                # Rich per-step scalars for TensorBoard.
                self.logger.record("choco/num_controlled_agents", n_agents)
                self.logger.record("choco/num_valid_agents", float(info.get("num_valid_agents", 0.0)))
                self.logger.record("choco/num_active_agents", float(info.get("num_active_agents", 0.0)))
                self.logger.record("choco/pending_respawn_count", float(info.get("pending_respawn_count", 0.0)))
                self.logger.record("choco/spawned_count_step", spawned_count)
                self.logger.record(
                    "choco/total_spawned_episodes",
                    float(self._training_spawned_count_total),
                )
                self.logger.record(
                    "choco/total_successful_episodes",
                    float(self._training_done_success_count_total),
                )
                self.logger.record(
                    "choco/total_vehicle_collision_episodes",
                    float(self._training_vehicle_collision_count_total),
                )
                if self._training_spawned_count_total > 0:
                    self.logger.record(
                        "choco/success_over_spawned_rate_total",
                        float(self._training_done_success_count_total)
                        / float(self._training_spawned_count_total),
                    )
                    self.logger.record(
                        "choco/veh_coll_spawned_cum",
                        float(self._training_vehicle_collision_count_total)
                        / float(self._training_spawned_count_total),
                    )
                if self._training_done_count_total > 0:
                    self.logger.record(
                        "choco/success_given_done_rate_total",
                        float(self._training_done_success_count_total)
                        / float(self._training_done_count_total),
                    )
                self.logger.record("choco/done_count_step", float(info.get("done_count", 0.0)))
                self.logger.record("choco/new_success_count_step", float(info.get("new_success_count", 0.0)))
                self.logger.record("choco/success_latched_count", float(info.get("success_latched_count", 0.0)))
                self.logger.record("choco/road_contact_done_count_step", float(info.get("road_contact_done_count", 0.0)))
                self.logger.record("choco/vehicle_contact_done_count_step", float(info.get("vehicle_contact_done_count", 0.0)))
                self.logger.record("choco/off_road_count_step", float(info.get("off_road_count", 0.0)))
                self.logger.record("choco/lane_hit_count_step", float(info.get("lane_hit_count", 0.0)))
                self.logger.record("choco/collided_count_step", float(info.get("collided_count", 0.0)))
                self.logger.record("choco/road_collided_count_step", float(info.get("road_collided_count", 0.0)))
                self.logger.record("choco/vehicle_collided_count_step", float(info.get("vehicle_collided_count", 0.0)))
                self.logger.record(
                    "choco/done_vehicle_collided_count_step",
                    float(info.get("done_vehicle_collided_count", 0.0)),
                )
                self.logger.record("choco/below_min_z_count_step", float(info.get("below_min_z_count", 0.0)))
                self.logger.record("choco/goal_rate_step", float(info.get("goal_rate_step", 0.0)))
                self.logger.record("choco/success_latched_rate_step", float(info.get("success_latched_rate_step", 0.0)))
                self.logger.record(
                    "choco/road_contact_done_rate_step",
                    float(info.get("road_contact_done_rate_step", 0.0)),
                )
                self.logger.record(
                    "choco/vehicle_contact_done_rate_step",
                    float(info.get("vehicle_contact_done_rate_step", 0.0)),
                )
                self.logger.record("choco/off_road_rate_step", float(info.get("off_road_rate_step", 0.0)))
                self.logger.record("choco/lane_hit_rate_step", float(info.get("lane_hit_rate_step", 0.0)))
                self.logger.record("choco/collision_rate_step", float(info.get("collision_rate_step", 0.0)))
                self.logger.record("choco/road_collision_rate_step", float(info.get("road_collision_rate_step", 0.0)))
                self.logger.record("choco/vehicle_collision_rate_step", float(info.get("vehicle_collision_rate_step", 0.0)))
                self.logger.record(
                    "choco/vehicle_collision_rate_step_per_controlled",
                    float(info.get("vehicle_collision_rate_step_per_controlled", 0.0)),
                )
                self.logger.record(
                    "choco/perc_veh_collisions_step",
                    float(info.get("perc_veh_collisions_step", 0.0)),
                )
                self.logger.record("choco/done_rate_step", float(info.get("done_rate_step", 0.0)))
                self.logger.record(
                    "choco/success_given_done_rate_step",
                    float(info.get("success_given_done_rate_step", 0.0)),
                )
                self.logger.record(
                    "choco/road_contact_done_given_active_rate_step",
                    float(info.get("road_contact_done_given_active_rate_step", 0.0)),
                )
                self.logger.record(
                    "choco/vehicle_contact_done_given_active_rate_step",
                    float(info.get("vehicle_contact_done_given_active_rate_step", 0.0)),
                )
                self.logger.record(
                    "choco/lane_hit_given_active_rate_step",
                    float(info.get("lane_hit_given_active_rate_step", 0.0)),
                )
                self.logger.record(
                    "choco/off_road_given_active_rate_step",
                    float(info.get("off_road_given_active_rate_step", 0.0)),
                )
                self.logger.record(
                    "choco/collision_given_active_rate_step",
                    float(info.get("collision_given_active_rate_step", 0.0)),
                )
                self.logger.record("choco/active_fraction", float(info.get("active_fraction", 0.0)))
                self.logger.record("choco/pending_fraction", float(info.get("pending_fraction", 0.0)))
                self.logger.record("choco/mean_dist_to_goal_m_step", float(info.get("mean_dist_to_goal_m", 0.0)))
                self.logger.record(
                    "choco/mean_active_dist_to_goal_m_step",
                    float(info.get("mean_active_dist_to_goal_m", 0.0)),
                )
                self.logger.record(
                    "choco/min_active_dist_to_goal_m_step",
                    float(info.get("min_active_dist_to_goal_m", 0.0)),
                )
                self.logger.record(
                    "choco/mean_lane_error_m_step",
                    float(info.get("mean_lane_error_m", 0.0)),
                )
                self.logger.record(
                    "choco/mean_heading_alignment_step",
                    float(info.get("mean_heading_alignment", 0.0)),
                )
                self.logger.record(
                    "choco/mean_route_progress_m_step",
                    float(info.get("mean_route_progress_m", 0.0)),
                )
                self.logger.record(
                    "choco/mean_base_reward_step",
                    float(info.get("mean_base_reward_step", 0.0)),
                )
                self.logger.record("choco/truncated_step", float(info.get("truncated", 0.0)))
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
            rollout_duration = None
            if self._rollout_collect_start_time is not None:
                rollout_duration = max(0.0, perf_counter() - self._rollout_collect_start_time)

            if self._rollout_agent_steps > 0:
                self.logger.record(
                    "rollout/goal_rate",
                    self._rollout_goal_sum / self._rollout_agent_steps,
                )
                self.logger.record(
                    "rollout/success_latched_rate",
                    self._rollout_success_latched_sum / self._rollout_agent_steps,
                )
                self.logger.record(
                    "rollout/road_contact_done_rate",
                    self._rollout_road_contact_done_sum / self._rollout_agent_steps,
                )
                self.logger.record(
                    "rollout/vehicle_contact_done_rate",
                    self._rollout_vehicle_contact_done_sum / self._rollout_agent_steps,
                )
                self.logger.record(
                    "rollout/off_road_rate",
                    self._rollout_off_road_sum / self._rollout_agent_steps,
                )
                self.logger.record(
                    "rollout/lane_hit_rate",
                    self._rollout_lane_hit_sum / self._rollout_agent_steps,
                )
                self.logger.record(
                    "rollout/collision_rate",
                    self._rollout_collision_sum / self._rollout_agent_steps,
                )
                self.logger.record(
                    "rollout/road_collision_rate",
                    self._rollout_road_collision_sum / self._rollout_agent_steps,
                )
                self.logger.record(
                    "rollout/vehicle_collision_rate_per_controlled_agent_step",
                    self._rollout_vehicle_collided_sum / self._rollout_agent_steps,
                )
                if self._rollout_spawned_count > 0:
                    # GPUDRIVE-style denominator: fraction of spawned agents
                    # that terminated due to vehicle collision within this rollout.
                    self.logger.record(
                        "rollout/veh_coll_spawned",
                        self._rollout_vehicle_collision_sum / self._rollout_spawned_count,
                    )
                if self._rollout_done_count > 0:
                    # 1:1 parity with GPUDRIVE `perc_veh_collisions`:
                    # fraction of finished agents that collided at least once.
                    self.logger.record(
                        "rollout/perc_veh_collisions",
                        self._rollout_done_vehicle_collided_sum / self._rollout_done_count,
                    )
                self.logger.record(
                    "rollout/below_min_z_rate",
                    self._rollout_below_min_z_sum / self._rollout_agent_steps,
                )
                self.logger.record(
                    "rollout/done_rate",
                    self._rollout_done_count / self._rollout_agent_steps,
                )
            if rollout_duration is not None:
                self.logger.record("perf/rollout_wall_time_sec", rollout_duration)
            if self._rollout_steps > 0:
                self.logger.record(
                    "rollout/mean_controlled_agents",
                    self._rollout_agent_steps / self._rollout_steps,
                )
                self.logger.record(
                    "rollout/spawned_episode_count",
                    self._rollout_spawned_count,
                )
                self.logger.record(
                    "rollout/mean_valid_agents",
                    self._rollout_valid_agent_steps / self._rollout_steps,
                )
                self.logger.record(
                    "rollout/mean_active_agents",
                    self._rollout_active_agent_steps / self._rollout_steps,
                )
                self.logger.record(
                    "rollout/mean_pending_respawns",
                    self._rollout_pending_agent_steps / self._rollout_steps,
                )
            if self._rollout_steps > 0:
                self.logger.record(
                    "rollout/mean_reward",
                    self._rollout_reward_sum / self._rollout_steps,
                )
                self.logger.record(
                    "rollout/mean_base_reward",
                    self._rollout_base_reward_sum / self._rollout_steps,
                )
                self.logger.record(
                    "rollout/timeout_rate",
                    self._rollout_timeout_count / self._rollout_steps,
                )
                if rollout_duration is not None and rollout_duration > 0.0:
                    self.logger.record(
                        "perf/env_steps_per_sec",
                        self._rollout_steps / rollout_duration,
                    )
            if self._rollout_valid_agent_steps > 0:
                self.logger.record(
                    "rollout/mean_dist_to_goal_m",
                    self._rollout_dist_sum / self._rollout_valid_agent_steps,
                )
                if rollout_duration is not None and rollout_duration > 0.0:
                    self.logger.record(
                        "perf/valid_agent_steps_per_sec",
                        self._rollout_valid_agent_steps / rollout_duration,
                    )
            if self._rollout_active_agent_steps > 0:
                self.logger.record(
                    "rollout/mean_active_dist_to_goal_m",
                    self._rollout_active_dist_sum / self._rollout_active_agent_steps,
                )
                self.logger.record(
                    "rollout/mean_lane_error_m",
                    self._rollout_lane_error_sum / self._rollout_active_agent_steps,
                )
                self.logger.record(
                    "rollout/mean_heading_alignment",
                    self._rollout_heading_alignment_sum / self._rollout_active_agent_steps,
                )
                self.logger.record(
                    "rollout/mean_route_progress_m",
                    self._rollout_route_progress_sum / self._rollout_active_agent_steps,
                )
                if rollout_duration is not None and rollout_duration > 0.0:
                    self.logger.record(
                        "perf/active_agent_steps_per_sec",
                        self._rollout_active_agent_steps / rollout_duration,
                    )
            if self._rollout_agent_steps > 0 and rollout_duration is not None and rollout_duration > 0.0:
                self.logger.record(
                    "perf/controlled_agent_steps_per_sec",
                    self._rollout_agent_steps / rollout_duration,
                )
            if self._rollout_done_count > 0:
                self.logger.record(
                    "rollout/vehicle_collision_rate",
                    self._rollout_vehicle_collision_sum / self._rollout_done_count,
                )
                self.logger.record(
                    "rollout/mean_episode_len",
                    self._rollout_agent_steps / self._rollout_done_count,
                )
                self.logger.record(
                    "rollout/success_given_done_rate",
                    self._rollout_done_success_count / self._rollout_done_count,
                )
            if self._training_done_count_total > 0:
                self.logger.record(
                    "rollout/perc_veh_collisions_total",
                    self._training_done_vehicle_collided_count_total
                    / self._training_done_count_total,
                )
            if self._training_spawned_count_total > 0:
                self.logger.record(
                    "rollout/success_rate",
                    self._training_done_success_count_total / self._training_spawned_count_total,
                )
                self.logger.record(
                    "rollout/veh_coll_spawned_cum",
                    self._training_vehicle_collision_count_total
                    / self._training_spawned_count_total,
                )
        except Exception:
            pass
