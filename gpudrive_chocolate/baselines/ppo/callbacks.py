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
        log_step_metrics: bool = False,
        log_detailed_metrics: bool = False,
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
        self.log_step_metrics = bool(log_step_metrics)
        self.log_detailed_metrics = bool(log_detailed_metrics)
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
        self._training_ep_worlds_total = 0.0
        self._training_ep_spawn_total = 0.0
        self._training_ep_done_total = 0.0
        self._training_ep_success_total = 0.0
        self._training_ep_road_done_total = 0.0
        self._training_ep_vehicle_done_total = 0.0
        self._training_ep_below_min_z_total = 0.0
        self._training_ep_other_done_total = 0.0
        self._training_ep_vehicle_any_total = 0.0
        self._reset_rollout_stats()

    def _reset_rollout_stats(self) -> None:
        self._rollout_steps = 0
        self._rollout_reward_sum = 0.0
        self._rollout_base_reward_sum = 0.0
        self._rollout_spawned_count = 0.0
        self._rollout_goal_sum = 0.0
        self._rollout_success_latched_sum = 0.0
        self._rollout_road_contact_done_sum = 0.0
        self._rollout_road_contact_hit_sum = 0.0
        self._rollout_road_edge_latched_sum = 0.0
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
        self._rollout_ep_worlds = 0.0
        self._rollout_ep_spawn = 0.0
        self._rollout_ep_done = 0.0
        self._rollout_ep_success = 0.0
        self._rollout_ep_road_done = 0.0
        self._rollout_ep_vehicle_done = 0.0
        self._rollout_ep_below_min_z = 0.0
        self._rollout_ep_other_done = 0.0
        self._rollout_ep_vehicle_any = 0.0

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
        self._training_ep_worlds_total = 0.0
        self._training_ep_spawn_total = 0.0
        self._training_ep_done_total = 0.0
        self._training_ep_success_total = 0.0
        self._training_ep_road_done_total = 0.0
        self._training_ep_vehicle_done_total = 0.0
        self._training_ep_below_min_z_total = 0.0
        self._training_ep_other_done_total = 0.0
        self._training_ep_vehicle_any_total = 0.0
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
        # Keep a rollout reward accumulator; avoid noisy per-step logging by default.
        rewards = None
        avg_reward = 0.0
        try:
            rewards = self.locals.get("rewards", None)
            if rewards is not None:
                avg_reward = float(np.nanmean(rewards))
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
                self._rollout_road_contact_hit_sum += float(info.get("road_contact_hit_count", 0.0))
                self._rollout_road_edge_latched_sum += float(info.get("road_edge_latched_count", 0.0))
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
                ep_worlds = float(info.get("episode_worlds_completed_count", 0.0))
                ep_spawn = float(info.get("episode_spawned_count", 0.0))
                ep_done = float(info.get("episode_done_count", 0.0))
                ep_success = float(info.get("episode_success_count", 0.0))
                ep_road_done = float(info.get("episode_road_done_count", 0.0))
                ep_vehicle_done = float(info.get("episode_vehicle_done_count", 0.0))
                ep_below_min_z = float(info.get("episode_below_min_z_count", 0.0))
                ep_other_done = float(info.get("episode_other_done_count", 0.0))
                ep_vehicle_any = float(info.get("episode_vehicle_collided_any_count", 0.0))

                self._rollout_ep_worlds += ep_worlds
                self._rollout_ep_spawn += ep_spawn
                self._rollout_ep_done += ep_done
                self._rollout_ep_success += ep_success
                self._rollout_ep_road_done += ep_road_done
                self._rollout_ep_vehicle_done += ep_vehicle_done
                self._rollout_ep_below_min_z += ep_below_min_z
                self._rollout_ep_other_done += ep_other_done
                self._rollout_ep_vehicle_any += ep_vehicle_any

                self._training_ep_worlds_total += ep_worlds
                self._training_ep_spawn_total += ep_spawn
                self._training_ep_done_total += ep_done
                self._training_ep_success_total += ep_success
                self._training_ep_road_done_total += ep_road_done
                self._training_ep_vehicle_done_total += ep_vehicle_done
                self._training_ep_below_min_z_total += ep_below_min_z
                self._training_ep_other_done_total += ep_other_done
                self._training_ep_vehicle_any_total += ep_vehicle_any
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

                if self.log_step_metrics:
                    self.logger.record("step/mean_reward", avg_reward if rewards is not None else 0.0)
                    self.logger.record("step/num_active_agents", float(info.get("num_active_agents", 0.0)))
                    self.logger.record(
                        "step/pending_respawn_count",
                        float(info.get("pending_respawn_count", 0.0)),
                    )
                    self.logger.record("step/done_count", float(info.get("done_count", 0.0)))
                    self.logger.record("step/new_success_count", float(info.get("new_success_count", 0.0)))
                    self.logger.record(
                        "step/road_contact_done_count",
                        float(info.get("road_contact_done_count", 0.0)),
                    )
                    self.logger.record(
                        "step/vehicle_contact_done_count",
                        float(info.get("vehicle_contact_done_count", 0.0)),
                    )
                    self.logger.record(
                        "step/mean_active_dist_to_goal_m",
                        float(info.get("mean_active_dist_to_goal_m", 0.0)),
                    )
                    self.logger.record(
                        "step/mean_lane_error_m",
                        float(info.get("mean_lane_error_m", 0.0)),
                    )

                if self.log_detailed_metrics:
                    self.logger.record("debug/num_controlled_agents", n_agents)
                    self.logger.record("debug/num_valid_agents", float(info.get("num_valid_agents", 0.0)))
                    self.logger.record("debug/spawned_count_step", spawned_count)
                    self.logger.record("debug/off_road_count_step", float(info.get("off_road_count", 0.0)))
                    self.logger.record("debug/lane_hit_count_step", float(info.get("lane_hit_count", 0.0)))
                    self.logger.record("debug/road_collided_count_step", float(info.get("road_collided_count", 0.0)))
                    self.logger.record(
                        "debug/vehicle_collided_count_step",
                        float(info.get("vehicle_collided_count", 0.0)),
                    )
                    self.logger.record("debug/below_min_z_count_step", float(info.get("below_min_z_count", 0.0)))
                    self.logger.record(
                        "debug/mean_base_reward_step",
                        float(info.get("mean_base_reward_step", 0.0)),
                    )
                    self.logger.record("debug/truncated_step", float(info.get("truncated", 0.0)))
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
                    "rollout/road_contact_hit_rate",
                    self._rollout_road_contact_hit_sum / self._rollout_agent_steps,
                )
                self.logger.record(
                    "rollout/road_edge_latched_rate",
                    self._rollout_road_edge_latched_sum / self._rollout_agent_steps,
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

            if self._rollout_ep_worlds > 0:
                ep_spawn = max(1.0, self._rollout_ep_spawn)
                self.logger.record("ep/worlds", self._rollout_ep_worlds)
                self.logger.record("ep/spawn", self._rollout_ep_spawn)
                self.logger.record("ep/done", self._rollout_ep_done)
                self.logger.record("ep/succ", self._rollout_ep_success)
                self.logger.record("ep/road", self._rollout_ep_road_done)
                self.logger.record("ep/veh_done", self._rollout_ep_vehicle_done)
                self.logger.record("ep/z_fail", self._rollout_ep_below_min_z)
                self.logger.record("ep/other", self._rollout_ep_other_done)
                self.logger.record("ep/veh_any", self._rollout_ep_vehicle_any)
                self.logger.record("ep/succ_rate", self._rollout_ep_success / ep_spawn)
                self.logger.record("ep/road_rate", self._rollout_ep_road_done / ep_spawn)
                self.logger.record("ep/veh_done_rate", self._rollout_ep_vehicle_done / ep_spawn)
                self.logger.record("ep/z_fail_rate", self._rollout_ep_below_min_z / ep_spawn)
                self.logger.record("ep/other_rate", self._rollout_ep_other_done / ep_spawn)
                self.logger.record("ep/veh_any_rate", self._rollout_ep_vehicle_any / ep_spawn)

            if self._training_ep_worlds_total > 0:
                ep_spawn_cum = max(1.0, self._training_ep_spawn_total)
                self.logger.record("ep_cum/worlds", self._training_ep_worlds_total)
                self.logger.record("ep_cum/spawn", self._training_ep_spawn_total)
                self.logger.record("ep_cum/done", self._training_ep_done_total)
                self.logger.record("ep_cum/succ", self._training_ep_success_total)
                self.logger.record("ep_cum/road", self._training_ep_road_done_total)
                self.logger.record("ep_cum/veh_done", self._training_ep_vehicle_done_total)
                self.logger.record("ep_cum/z_fail", self._training_ep_below_min_z_total)
                self.logger.record("ep_cum/other", self._training_ep_other_done_total)
                self.logger.record("ep_cum/veh_any", self._training_ep_vehicle_any_total)
                self.logger.record("ep_cum/succ_rate", self._training_ep_success_total / ep_spawn_cum)
                self.logger.record("ep_cum/road_rate", self._training_ep_road_done_total / ep_spawn_cum)
                self.logger.record(
                    "ep_cum/veh_done_rate",
                    self._training_ep_vehicle_done_total / ep_spawn_cum,
                )
                self.logger.record(
                    "ep_cum/z_fail_rate",
                    self._training_ep_below_min_z_total / ep_spawn_cum,
                )
                self.logger.record("ep_cum/other_rate", self._training_ep_other_done_total / ep_spawn_cum)
                self.logger.record("ep_cum/veh_any_rate", self._training_ep_vehicle_any_total / ep_spawn_cum)

            if self.log_detailed_metrics:
                if self._rollout_agent_steps > 0:
                    self.logger.record(
                        "debug/collision_rate",
                        self._rollout_collision_sum / self._rollout_agent_steps,
                    )
                    self.logger.record(
                        "debug/road_collision_rate",
                        self._rollout_road_collision_sum / self._rollout_agent_steps,
                    )
                    self.logger.record(
                        "debug/vehicle_collision_rate_per_controlled_agent_step",
                        self._rollout_vehicle_collided_sum / self._rollout_agent_steps,
                    )
                    self.logger.record(
                        "debug/success_latched_rate",
                        self._rollout_success_latched_sum / self._rollout_agent_steps,
                    )
                if self._rollout_done_count > 0:
                    self.logger.record(
                        "debug/mean_episode_len",
                        self._rollout_agent_steps / self._rollout_done_count,
                    )
                if self._rollout_steps > 0:
                    self.logger.record(
                        "debug/mean_controlled_agents",
                        self._rollout_agent_steps / self._rollout_steps,
                    )
                    self.logger.record(
                        "debug/mean_valid_agents",
                        self._rollout_valid_agent_steps / self._rollout_steps,
                    )
                    self.logger.record(
                        "debug/spawned_episode_count",
                        self._rollout_spawned_count,
                    )
        except Exception:
            pass
