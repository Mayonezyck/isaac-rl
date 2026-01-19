# src/chocolate_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class StepInfo:
    keys: List[object]
    mask: np.ndarray
    dist_n: np.ndarray
    success: np.ndarray
    timeout: bool
    t_env: int


class ChocolateEnv:
    """
    Minimal RL-style environment wrapper around your existing IsaacSim loop.

    Adds explicit debug prints when we attempt optional behaviors
    (e.g., clearing velocity memory), so you can see what was tried
    and what happened.
    """

    def __init__(
        self,
        *,
        sim,
        stage,
        ctrl,
        obs_builder,
        bounds_size_m: float,
        physics_dt: float,
        action_repeat: int = 4,
        max_steps: int = 600,
        goal_success_dist_norm: float = 0.01,
        reward_scale: float = 1.0,
        success_bonus: float = 1.0,
        action_l2_penalty: float = 0.0,
        render: bool = False,
        root_container: str = "/World/MiniWorlds",
        world_prefix: str = "world_",
        warmup_on_reset_steps: int = 1,
        verbose: bool = False,
    ):
        self.sim = sim
        self.stage = stage
        self.ctrl = ctrl
        self.obs_builder = obs_builder

        self.bounds_size_m = float(bounds_size_m)
        self.physics_dt = float(physics_dt)

        self.action_repeat = max(1, int(action_repeat))
        self.max_steps = int(max_steps)

        self.goal_success_dist_norm = float(goal_success_dist_norm)
        self.reward_scale = float(reward_scale)
        self.success_bonus = float(success_bonus)
        self.action_l2_penalty = float(action_l2_penalty)

        self.render = bool(render)
        self.root_container = str(root_container)
        self.world_prefix = str(world_prefix)
        self.warmup_on_reset_steps = max(0, int(warmup_on_reset_steps))
        self.verbose = bool(verbose)

        # state
        self.t = 0  # env steps
        self._keys: List[object] = []
        self._mask: np.ndarray = np.zeros((0,), dtype=bool)
        self._prev_dist_n: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._done: np.ndarray = np.zeros((0,), dtype=bool)

    # -------------------------
    # Core API
    # -------------------------

    def reset(self) -> Tuple[np.ndarray, np.ndarray, List[object]]:
        self.t = 0

        # Refresh controller registry
        if self.verbose:
            print("[env.reset] calling ctrl.refresh() ...")
        try:
            self.ctrl.refresh()
            if self.verbose:
                print(f"[env.reset] ctrl.refresh() OK. keys={len(self.ctrl.keys())}")
        except Exception as e:
            print(f"[env.reset] ctrl.refresh() FAILED: {type(e).__name__}: {e}")

        # Clear obs builder velocity memory if present
        if self.verbose:
            print("[env.reset] trying to clear velocity memory (obs_builder.state.prev_pos_xy_m) ...")
        cleared = False
        try:
            # Try the exact structure we expect
            state = getattr(self.obs_builder, "state", None)
            if state is None:
                if self.verbose:
                    print("[env.reset] velocity memory NOT found: obs_builder.state missing")
            else:
                prev_map = getattr(state, "prev_pos_xy_m", None)
                if prev_map is None:
                    if self.verbose:
                        print("[env.reset] velocity memory NOT found: state.prev_pos_xy_m missing")
                else:
                    # It's there: clear it
                    n_before = len(prev_map) if hasattr(prev_map, "__len__") else -1
                    prev_map.clear()
                    n_after = len(prev_map) if hasattr(prev_map, "__len__") else -1
                    cleared = True
                    if self.verbose:
                        print(f"[env.reset] velocity memory found -> cleared ({n_before} -> {n_after})")
        except Exception as e:
            print(f"[env.reset] tried to clear velocity memory but FAILED: {type(e).__name__}: {e}")
            if self.verbose:
                print("[env.reset] suggestion: better add velocity memory as obs_builder.state.prev_pos_xy_m (dict)")

        if self.verbose and (not cleared):
            print("[env.reset] note: velocity memory not cleared (may be OK if builder is stateless)")

        # Optional: a couple physics steps to settle
        if self.warmup_on_reset_steps > 0:
            if self.verbose:
                print(f"[env.reset] warmup_on_reset_steps={self.warmup_on_reset_steps}: stepping sim ...")
            for j in range(self.warmup_on_reset_steps):
                try:
                    self.sim.step(render=self.render)
                except Exception as e:
                    print(f"[env.reset] sim.step failed at warmup {j}: {type(e).__name__}: {e}")
                    break

        obs, mask, keys = self._build_obs()

        self._keys = keys
        self._mask = mask.copy()
        self._prev_dist_n = obs[:, 4].astype(np.float32).copy() if obs.shape[0] > 0 else np.zeros((0,), np.float32)
        self._done = np.zeros((len(keys),), dtype=bool)

        if self.verbose:
            print(f"[env.reset] done. N={len(keys)} active={int(mask.sum())}")

        return obs, mask, keys

    def step(self, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StepInfo]:
        if self.t == 0 and (self._keys is None or len(self._keys) == 0):
            if self.verbose:
                print("[env.step] called before reset(); auto-resetting ...")
            self.reset()

        U = np.asarray(U, dtype=np.float32)
        N = len(self._keys)

        if U.ndim != 2 or U.shape != (N, 3):
            raise ValueError(f"Action must be shape (N,3) matching keys. got {U.shape}, N={N}")

        # Apply controls once
        if self.verbose:
            print(f"[env.step] applying actions to N={N} vehicles ...")
        try:
            self.ctrl.apply_all(U)
        except Exception as e:
            print(f"[env.step] ctrl.apply_all FAILED: {type(e).__name__}: {e}")

        # Advance physics
        if self.verbose:
            print(f"[env.step] stepping physics action_repeat={self.action_repeat} (render={self.render}) ...")
        for j in range(self.action_repeat):
            try:
                self.sim.step(render=self.render)
            except Exception as e:
                print(f"[env.step] sim.step FAILED at substep {j}: {type(e).__name__}: {e}")
                break

        # New obs
        obs, mask, keys = self._build_obs()

        # Key consistency check
        if len(keys) != N:
            print(f"[env.step] WARNING: key count changed {N}->{len(keys)}. Reinitializing episode state.")
            self._keys = keys
            self._mask = mask.copy()
            self._prev_dist_n = obs[:, 4].astype(np.float32).copy()
            self._done = np.zeros((len(keys),), dtype=bool)
            N = len(keys)

        dist_n = obs[:, 4].astype(np.float32)

        # Reward: progress
        progress = (self._prev_dist_n - dist_n) * self.reward_scale
        reward = np.zeros((N,), dtype=np.float32)
        reward[mask] = progress[mask]

        # Optional action penalty
        if self.action_l2_penalty > 0.0:
            if self.verbose:
                print(f"[env.step] applying action L2 penalty: {self.action_l2_penalty}")
            a2 = (U[:, 0] ** 2 + U[:, 1] ** 2 + U[:, 2] ** 2).astype(np.float32)
            reward[mask] -= self.action_l2_penalty * a2[mask]

        # Done conditions
        success = (dist_n < self.goal_success_dist_norm) & mask
        timeout = (self.t + 1) >= self.max_steps

        done = self._done.copy()
        done |= ~mask
        done |= success
        if timeout:
            done |= mask

        newly_success = success & (~self._done)
        if newly_success.any():
            if self.verbose:
                print(f"[env.step] success achieved by {int(newly_success.sum())} agents, adding bonus={self.success_bonus}")
            reward[newly_success] += self.success_bonus

        # Update state
        self._prev_dist_n = dist_n.copy()
        self._mask = mask.copy()
        self._done = done.copy()
        self.t += 1

        info = StepInfo(
            keys=keys,
            mask=mask,
            dist_n=dist_n,
            success=success,
            timeout=bool(timeout),
            t_env=int(self.t),
        )

        if self.verbose:
            if mask.any():
                print(f"[env.step] t={self.t} active={int(mask.sum())} min_dist_n={float(dist_n[mask].min()):.6f} mean_dist_n={float(dist_n[mask].mean()):.6f}")
            else:
                print(f"[env.step] t={self.t} active=0 (no valid agents)")

        return obs, reward, done, info

    # -------------------------
    # Utilities
    # -------------------------

    @property
    def keys(self) -> List[object]:
        return list(self._keys)

    @property
    def mask(self) -> np.ndarray:
        return self._mask.copy()

    def _build_obs(self) -> Tuple[np.ndarray, np.ndarray, List[object]]:
        if self.verbose:
            print("[env] building obs via obs_builder.build_obs_all_controlled(...) ...")
        try:
            out = self.obs_builder.build_obs_all_controlled(
                stage=self.stage,
                bounds_size_m=self.bounds_size_m,
                ctrl=self.ctrl,
                dt=self.physics_dt,
                root_container=self.root_container,
                world_prefix=self.world_prefix,
            )
            if self.verbose:
                obs, mask, keys = out
                print(f"[env] obs built: obs_shape={tuple(obs.shape)} active={int(mask.sum())} keys={len(keys)}")
            return out
        except Exception as e:
            print(f"[env] build_obs_all_controlled FAILED: {type(e).__name__}: {e}")
            # fail closed (so caller sees it)
            raise
