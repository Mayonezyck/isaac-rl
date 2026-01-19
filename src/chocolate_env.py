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
        clear_on_done: bool = True,
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

        self.clear_on_done = bool(clear_on_done)
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
        self._success_latched: np.ndarray = np.zeros((0,), dtype=bool)

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
        self._success_latched = np.zeros((len(keys),), dtype=bool)

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

            print("[env.step] better add done memory: re-init _done due to key change")
            self._done = np.zeros((len(keys),), dtype=bool)
            self._success_latched = np.zeros((len(keys),), dtype=bool)
            N = len(keys)

        # Safety: ensure _done exists and matches N
        if self._done is None or len(self._done) != N:
            print("[env.step] better add done memory: init _done (len mismatch or None)")
            self._done = np.zeros((N,), dtype=bool)
        else:
            print("[env.step] done memory found")
        if self._success_latched is None or len(self._success_latched) != N:
            print("[env.step] init _success_latched (len mismatch or None)")
            self._success_latched = np.zeros((N,), dtype=bool)

        # dist_n is obs[:,4]
        dist_n = obs[:, 4].astype(np.float32)

        # SUCCESS LOGIC
        success_now = (dist_n < self.goal_success_dist_norm) & mask
        newly_success = success_now & (~self._success_latched)

        if mask.any():
            print("[dbg] dist_n:", dist_n)
            print("[dbg] success_now:", success_now, "thresh=", self.goal_success_dist_norm)
            print("[dbg] newly_success:", newly_success)

        # Latch SUCCESS permanently (separate from done)
        if newly_success.any():
            print(f"[env.step] success achieved by {int(newly_success.sum())} agents (latching success=True)")
            self._success_latched[newly_success] = True
        else:
            print("[env.step] no newly_success this step")

        # persistent success flag for info/logging
        success_latched = self._success_latched.copy()


        # Freeze/hide newly successful agents
        if newly_success.any():
            self._done[newly_success] = True
            print("[env.step] trying _freeze_agents(...)")
            self._freeze_agents(keys, newly_success)
            print("[env.step] _freeze_agents done")
            if self.clear_on_done:
                print("[env.step] trying _hide_agents(...) because clear_on_done=True")
                self._hide_agents(keys, newly_success)
                print("[env.step] _hide_agents done")
            else:
                print("[env.step] clear_on_done=False, skip hiding")
        else:
            print("[env.step] no newly_success -> skip freeze/hide")

        # Reward: progress
        progress = (self._prev_dist_n - dist_n) * self.reward_scale
        reward = np.zeros((N,), dtype=np.float32)
        reward[mask] = progress[mask]

        # Add success bonus ONLY for newly_success (latched)
        if newly_success.any():
            if self.verbose:
                print(f"[env.step] adding success bonus={self.success_bonus} to newly_success")
            reward[newly_success] += self.success_bonus
        else:
            if self.verbose:
                print("[env.step] no newly_success -> no bonus")

        # Optional action penalty
        if self.action_l2_penalty > 0.0:
            if self.verbose:
                print(f"[env.step] applying action L2 penalty: {self.action_l2_penalty}")
            a2 = (U[:, 0] ** 2 + U[:, 1] ** 2 + U[:, 2] ** 2).astype(np.float32)
            reward[mask] -= self.action_l2_penalty * a2[mask]

        # Done conditions:
        # - latched successes stay done
        # - invalid mask are done
        # - timeout ends all active agents
        timeout = (self.t + 1) >= self.max_steps

        done = self._done.copy()
        done |= ~mask
        if timeout:
            print("[env.step] timeout=True -> done |= mask")
            done |= mask

        # Update state
        self._prev_dist_n = dist_n.copy()
        self._mask = mask.copy()
        self._done = done.copy()
        self.t += 1

        info = StepInfo(
            keys=keys,
            mask=mask,
            dist_n=dist_n,
            success=success_latched,           # latched success (persistent)
            timeout=bool(timeout),
            t_env=int(self.t),
        )

        if self.verbose:
            if mask.any():
                print(f"[env.step] t={self.t} active={int(mask.sum())} min_dist_n={float(dist_n[mask].min()):.6f} mean_dist_n={float(dist_n[mask].mean()):.6f}")
            else:
                print(f"[env.step] t={self.t} active=0 (no valid agents)")

        return obs, reward, done, info

    
    def _freeze_agents(self, keys, freeze_mask: np.ndarray) -> None:
        """Brake+zero controls for agents in freeze_mask."""
        if not freeze_mask.any():
            print("[env] _freeze_agents: none to freeze")
            return

        K = len(keys)
        U_freeze = np.zeros((K, 3), dtype=np.float32)
        U_freeze[:, 0] = 0.0   # thr
        U_freeze[:, 1] = 0.0   # steer
        U_freeze[:, 2] = 1.0   # brake hard
        idx = np.where(freeze_mask)[0]
        print(f"[env] _freeze_agents: freezing {len(idx)} agents -> brake=1")
        # apply only to those indices
        self.ctrl.apply_batch([keys[i] for i in idx], U_freeze[idx])


    def _hide_agents(self, keys, hide_mask: np.ndarray) -> None:
        """Hide vehicle prims for agents in hide_mask (visual clear)."""
        if not hide_mask.any():
            print("[env] _hide_agents: none to hide")
            return
        from pxr import UsdGeom
        idx = np.where(hide_mask)[0]
        print(f"[env] _hide_agents: hiding {len(idx)} agents")
        for i in idx:
            h = self.ctrl.get(keys[i].world_idx, keys[i].agent_id)
            if h is None:
                continue
            try:
                # hide the moving pose prim
                UsdGeom.Imageable(h.pose_prim).MakeInvisible()
            except Exception as e:
                print("[env] _hide_agents: failed:", e)


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
