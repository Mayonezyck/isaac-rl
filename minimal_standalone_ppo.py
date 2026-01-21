#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

import numpy as np

# ----------------------------
# Isaac Sim bootstrap (same pattern you use)
# ----------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,                 # set True if you want headless
        "renderer": "RayTracedLighting",   # fine either way
    }
)

# After SimulationApp is created:
import omni.kit.app
import omni.usd

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaacsim.core.api import SimulationContext


# ----------------------------
# Enable required extension (vehicle wizard)
# ----------------------------
def _enable_ext(ext_name: str) -> None:
    em = omni.kit.app.get_app().get_extension_manager()
    if not em.is_extension_enabled(ext_name):
        em.set_extension_enabled_immediate(ext_name, True)
    if not em.is_extension_enabled(ext_name):
        raise RuntimeError(f"Failed enabling extension: {ext_name}")


_enable_ext("omni.physx.vehicle")
simulation_app.update()
simulation_app.update()

# PhysX vehicle wizard imports (same ones your builder uses)
from omni.physxvehicle.scripts.wizards import physxVehicleWizard as VehicleWizard
from omni.physxvehicle.scripts.helpers.UnitScale import UnitScale
from omni.physxvehicle.scripts.commands import PhysXVehicleWizardCreateCommand

ROOT_PATH = "/World"
SHARED_ROOT = ROOT_PATH + "/VehicleShared"


# ----------------------------
# Import your existing “ingredients”
# ----------------------------
from src.chocolate_vehicle_controller import ChocolateWorldVehicleController
from src.chocolate_obs_builder import ChocolateObsBuilder
from src.chocolate_env import ChocolateEnv


# ----------------------------
# Small USD helpers
# ----------------------------
def ensure_world_default_prim(stage: Usd.Stage) -> None:
    world = stage.GetPrimAtPath(ROOT_PATH)
    if not world.IsValid():
        world = UsdGeom.Xform.Define(stage, ROOT_PATH).GetPrim()
    if not stage.GetDefaultPrim().IsValid():
        stage.SetDefaultPrim(world)


def meters_per_unit(stage: Usd.Stage) -> float:
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    return float(mpu) if mpu and float(mpu) > 0 else 0.01  # Isaac often cm


def get_unit_scale(stage: Usd.Stage) -> Tuple[UnitScale, float]:
    """Return (UnitScale, meters_per_unit). Mirrors your builder’s logic."""
    mpu = meters_per_unit(stage)
    length_scale = 1.0 / mpu

    kpu = UsdPhysics.GetStageKilogramsPerUnit(stage)
    if not kpu or float(kpu) == 0.0:
        kpu = 1.0
    mass_scale = 1.0 / float(kpu)

    return UnitScale(length_scale, mass_scale), mpu


def ensure_physics_scene(stage: Usd.Stage) -> None:
    # If a physics scene already exists, do nothing.
    # Otherwise define a simple one.
    for prim in stage.Traverse():
        if prim.GetTypeName() == "PhysicsScene":
            return
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)


def spawn_ground_plane(stage: Usd.Stage, *, size_m: float = 200.0, z_m: float = 0.0) -> str:
    """
    Minimal static ground:
    - A big thin cube with collision.
    """
    ensure_world_default_prim(stage)
    mpu = meters_per_unit(stage)

    path = "/World/GroundPlane"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)

    prim = cube.GetPrim()
    xform = UsdGeom.XformCommonAPI(prim)

    # size in stage units
    sx = (size_m / mpu)
    sy = (size_m / mpu)
    sz = (0.2 / mpu)

    xform.SetScale(Gf.Vec3f(float(sx), float(sy), float(sz)))
    xform.SetTranslate(Gf.Vec3d(0.0, 0.0, float(z_m / mpu) - 0.5 * float(sz)))

    # Collision only (static)
    UsdPhysics.CollisionAPI.Apply(prim)
    return path


def spawn_vehicle_wizard_under(stage: Usd.Stage, parent_path: str, *, position_m: Tuple[float, float, float], yaw_deg: float) -> Optional[str]:
    """
    Exactly the same vehicle-wizard spawning pattern as your builder:
    parent_path/Vehicle is created, controller attrs end up under the wizard’s structure.
    """
    ensure_world_default_prim(stage)

    parent_xf = UsdGeom.Xform.Define(stage, parent_path)
    xapi = UsdGeom.XformCommonAPI(parent_xf)

    unit_scale, mpu = get_unit_scale(stage)
    x_m, y_m, z_m = position_m
    xapi.SetTranslate(Gf.Vec3d(x_m / mpu, y_m / mpu, z_m / mpu))
    xapi.SetRotate(Gf.Vec3f(0.0, 0.0, float(yaw_deg)), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    vehicle_data = VehicleWizard.VehicleData(
        unit_scale,
        VehicleWizard.VehicleData.AXIS_Z,
        VehicleWizard.VehicleData.AXIS_X,
    )
    vehicle_data.rootVehiclePath = parent_path + "/Vehicle"
    vehicle_data.rootSharedPath = SHARED_ROOT

    ret = PhysXVehicleWizardCreateCommand.execute(vehicle_data)
    success = bool(ret[0]) if isinstance(ret, (tuple, list)) and len(ret) > 0 else False
    if not success:
        payload = ret[1] if isinstance(ret, (tuple, list)) and len(ret) > 1 else None
        messages = payload[0] if isinstance(payload, (tuple, list)) and len(payload) > 0 else None
        print("[VEH WIZ FAIL]", parent_path, "messages:", messages)
        return None

    return vehicle_data.rootVehiclePath  # ".../Vehicle"


def spawn_goal_marker(stage: Usd.Stage, goal_path: str, *, center_m: Tuple[float, float, float], radius_m: float = 1.0) -> str:
    """
    Minimal goal prim that your obs builder can read:
    - lives under /World/MiniWorlds/world_000/Goals
    - name contains "_id{agent_id}"
    - customData includes "goal_center_m"
    """
    mpu = meters_per_unit(stage)

    sph = UsdGeom.Sphere.Define(stage, goal_path)
    sph.CreateRadiusAttr().Set(float(radius_m / mpu))  # radius in stage units

    prim = sph.GetPrim()
    xapi = UsdGeom.XformCommonAPI(prim)
    xapi.SetTranslate(Gf.Vec3d(center_m[0] / mpu, center_m[1] / mpu, center_m[2] / mpu))

    # The obs builder expects this key
    prim.SetCustomDataByKey("is_goal", True)
    prim.SetCustomDataByKey("goal_center_m", (float(center_m[0]), float(center_m[1]), float(center_m[2])))
    return goal_path


# ----------------------------
# Tiny PPO (single-env) to learn reach-goal
# ----------------------------
import torch
import torch.nn as nn
import torch.optim as optim


class PolicyValue(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, 3)   # mean for [thr, steer, brake]
        self.v  = nn.Linear(hidden, 1)

        # log-std as parameter (diagonal Gaussian)
        self.log_std = nn.Parameter(torch.tensor([-0.5, -0.5, -0.5], dtype=torch.float32))

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mean = self.pi(h)
        v = self.v(h).squeeze(-1)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        return mean, std, v


def squash_action(a_raw: torch.Tensor) -> torch.Tensor:
    """
    Map unconstrained Gaussian sample -> valid control ranges:
      thr   in [0,1]
      steer in [-1,1]
      brake in [0,1]
    """
    thr   = torch.sigmoid(a_raw[..., 0:1])
    steer = torch.tanh(a_raw[..., 1:2])
    brake = torch.sigmoid(a_raw[..., 2:3])
    return torch.cat([thr, steer, brake], dim=-1)


@dataclass
class Rollout:
    obs: torch.Tensor
    act_raw: torch.Tensor
    act_squashed: torch.Tensor
    logp: torch.Tensor
    val: torch.Tensor
    rew: torch.Tensor
    done: torch.Tensor


def gae_advantages(rews, vals, dones, gamma=0.99, lam=0.95):
    """
    rews, vals, dones: 1D tensors length T
    returns adv, ret
    """
    T = rews.shape[0]
    adv = torch.zeros_like(rews)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_nonterm = 1.0 - dones[t]
        next_val = vals[t + 1] if t + 1 < T else 0.0
        delta = rews[t] + gamma * next_val * next_nonterm - vals[t]
        last_gae = delta + gamma * lam * next_nonterm * last_gae
        adv[t] = last_gae
    ret = adv + vals
    return adv, ret


# ----------------------------
# Main
# ----------------------------
def main():
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    ensure_world_default_prim(stage)
    ensure_physics_scene(stage)

    # Minimal “miniworld” structure matching your controller’s expectations:
    # /World/MiniWorlds/world_000/Agents/Agent_0000_id0/Vehicle_Parent/Vehicle
    root_container = "/World/MiniWorlds"
    world_root = f"{root_container}/world_000"
    agents_root = f"{world_root}/Agents"
    goals_root = f"{world_root}/Goals"
    UsdGeom.Xform.Define(stage, root_container)
    UsdGeom.Xform.Define(stage, world_root)
    UsdGeom.Xform.Define(stage, agents_root)
    UsdGeom.Xform.Define(stage, goals_root)

    spawn_ground_plane(stage, size_m=200.0, z_m=0.0)

    agent_id = 0
    agent_path = f"{agents_root}/Agent_0000_id{agent_id}"
    vehicle_parent = f"{agent_path}/Vehicle_Parent"
    UsdGeom.Xform.Define(stage, agent_path)

    # Spawn car at origin, facing +X
    vehicle_root = spawn_vehicle_wizard_under(
        stage,
        vehicle_parent,
        position_m=(0.0, 0.0, 0.5),
        yaw_deg=0.0,
    )
    if vehicle_root is None:
        raise RuntimeError("Vehicle wizard spawn failed.")

    # Spawn goal (important: name contains _id{agent_id})
    goal_center_m = (25.0, 0.0, 0.2)
    spawn_goal_marker(stage, f"{goals_root}/Goal_0000_id{agent_id}", center_m=goal_center_m, radius_m=1.0)

    # Let USD settle
    simulation_app.update()
    simulation_app.update()

    # Physics context (same as your standalone)
    mpu = meters_per_unit(stage)
    physics_dt = 1.0 / 60.0
    sim = SimulationContext(stage_units_in_meters=mpu, physics_dt=physics_dt, rendering_dt=physics_dt)
    sim.initialize_physics()

    # One step helps controller prims appear (same pattern you use)
    sim.step(render=False)

    # Controller: use ctrl_suffix "/Vehicle" (known-good candidate in your capture script)
    ctrl = ChocolateWorldVehicleController(
        stage=stage,
        root_container=root_container,
        world_count=1,
        ctrl_suffix="/Vehicle",
        verbose=True,
    )
    ctrl.refresh()
    if len(ctrl.keys()) != 1:
        raise RuntimeError(
            f"Expected exactly 1 controllable vehicle, got {len(ctrl.keys())}. "
            f"Try ctrl_suffix '' or '/VehicleController' like your candidate list."
        )

    obs_builder = ChocolateObsBuilder()

    env = ChocolateEnv(
        sim=sim,
        stage=stage,
        ctrl=ctrl,
        obs_builder=obs_builder,
        bounds_size_m=200.0,
        physics_dt=physics_dt,
        action_repeat=4,
        max_steps=200,
        clear_on_done=False,
        goal_success_dist_m=2.0,
        reward_scale=1.0,
        success_bonus=10.0,
        action_l2_penalty=1e-3,
        render=True,
        verbose=False,
    )

    # Reset once to get obs dim
    obs_np, mask_np, keys = env.reset()
    if not mask_np.any():
        raise RuntimeError("Mask is invalid after reset. Goal/pose not detected by obs builder.")
    obs_dim = obs_np.shape[1]
    print("[demo] obs_dim =", obs_dim, "keys =", keys)

    device = torch.device("cpu")
    pv = PolicyValue(obs_dim).to(device)
    opt = optim.Adam(pv.parameters(), lr=3e-4)

    # PPO hyperparams (kept small & stable)
    steps_per_iter = 512
    ppo_epochs = 4
    minibatch = 128
    clip_eps = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    gamma = 0.99
    lam = 0.95

    # Training loop
    obs_np, mask_np, keys = env.reset()
    ep_ret = 0.0
    ep_len = 0

    for it in range(200):  # iterations
        # Collect rollout
        obs_buf: List[torch.Tensor] = []
        actraw_buf: List[torch.Tensor] = []
        act_buf: List[torch.Tensor] = []
        logp_buf: List[torch.Tensor] = []
        val_buf: List[torch.Tensor] = []
        rew_buf: List[torch.Tensor] = []
        done_buf: List[torch.Tensor] = []

        for t in range(steps_per_iter):
            obs_t = torch.tensor(obs_np[0], dtype=torch.float32, device=device).unsqueeze(0)  # (1,obs_dim)
            with torch.no_grad():
                mean, std, v = pv(obs_t)
                dist = torch.distributions.Normal(mean, std)
                a_raw = dist.sample()
                logp = dist.log_prob(a_raw).sum(dim=-1)
                a = squash_action(a_raw)

            # env step wants (N,3) float32 aligned with keys()
            U = np.zeros((len(keys), 3), dtype=np.float32)
            U[0, :] = a.squeeze(0).cpu().numpy()

            obs2_np, r_np, done_np, info = env.step(U)

            r = float(r_np[0])
            d = float(done_np[0])

            ep_ret += r
            ep_len += 1

            obs_buf.append(obs_t.squeeze(0))
            actraw_buf.append(a_raw.squeeze(0))
            act_buf.append(a.squeeze(0))
            logp_buf.append(logp.squeeze(0))
            val_buf.append(v.squeeze(0))
            rew_buf.append(torch.tensor(r, device=device))
            done_buf.append(torch.tensor(d, device=device))

            obs_np = obs2_np

            if done_np[0]:
                # Print: your env success is latched in info.success
                succ = bool(info.success[0])
                dist_m = float(info.dist_m[0])
                print(f"[iter {it}] episode done: len={ep_len:4d} return={ep_ret:8.2f} success={succ} dist_m={dist_m:.2f}")
                obs_np, mask_np, keys = env.reset()
                ep_ret = 0.0
                ep_len = 0

        obs_b = torch.stack(obs_buf, dim=0)         # (T,obs_dim)
        actraw_b = torch.stack(actraw_buf, dim=0)   # (T,3)
        logp_old = torch.stack(logp_buf, dim=0)     # (T,)
        val_b = torch.stack(val_buf, dim=0)         # (T,)
        rew_b = torch.stack(rew_buf, dim=0)         # (T,)
        done_b = torch.stack(done_buf, dim=0)       # (T,)

        # Compute advantages/returns
        with torch.no_grad():
            adv, ret = gae_advantages(rew_b, val_b, done_b, gamma=gamma, lam=lam)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO update
        idxs = torch.arange(steps_per_iter, device=device)
        for _ in range(ppo_epochs):
            perm = idxs[torch.randperm(steps_per_iter)]
            for start in range(0, steps_per_iter, minibatch):
                mb = perm[start : start + minibatch]

                obs_mb = obs_b[mb]
                actraw_mb = actraw_b[mb]
                logp_old_mb = logp_old[mb]
                adv_mb = adv[mb]
                ret_mb = ret[mb]

                mean, std, v = pv(obs_mb)
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(actraw_mb).sum(dim=-1)
                ratio = torch.exp(logp - logp_old_mb)

                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
                pi_loss = -torch.min(surr1, surr2).mean()

                vf_loss = ((v - ret_mb) ** 2).mean()

                ent = dist.entropy().sum(dim=-1).mean()
                loss = pi_loss + vf_coef * vf_loss - ent_coef * ent

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(pv.parameters(), 1.0)
                opt.step()

        # quick training signal
        print(f"[iter {it}] update done (mean rew={rew_b.mean().item():.3f})")

    simulation_app.close()


if __name__ == "__main__":
    main()
