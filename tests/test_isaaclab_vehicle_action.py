from __future__ import annotations

import sys
import re
from types import SimpleNamespace
import types
from dataclasses import dataclass

import torch

if "isaaclab.managers.action_manager" not in sys.modules:
    isaaclab_module = types.ModuleType("isaaclab")
    isaaclab_managers_module = types.ModuleType("isaaclab.managers")
    isaaclab_action_manager_module = types.ModuleType("isaaclab.managers.action_manager")

    @dataclass
    class _FakeActionTermCfg:
        asset_name: str
        debug_vis: bool = False
        clip: dict[str, tuple] | None = None

    class _FakeActionTerm:
        def __init__(self, cfg: _FakeActionTermCfg, env):
            self.cfg = cfg
            self._env = env
            self._asset = env.scene[cfg.asset_name]
            self._IO_descriptor = SimpleNamespace()
            self._export_IO_descriptor = True
            self.set_debug_vis(cfg.debug_vis)

        @property
        def num_envs(self) -> int:
            return self._env.num_envs

        @property
        def device(self) -> str:
            return self._env.device

        @property
        def has_debug_vis_implementation(self) -> bool:
            return False

        @property
        def IO_descriptor(self):
            return self._IO_descriptor

        def set_debug_vis(self, debug_vis: bool) -> bool:
            return False

    isaaclab_action_manager_module.ActionTerm = _FakeActionTerm
    isaaclab_action_manager_module.ActionTermCfg = _FakeActionTermCfg
    isaaclab_managers_module.action_manager = isaaclab_action_manager_module
    isaaclab_module.managers = isaaclab_managers_module

    sys.modules["isaaclab"] = isaaclab_module
    sys.modules["isaaclab.managers"] = isaaclab_managers_module
    sys.modules["isaaclab.managers.action_manager"] = isaaclab_action_manager_module

from src.isaaclab_vehicle_action import VehicleActionTerm, VehicleActionTermCfg


class _FakeArticulation:
    def __init__(self, joint_names: list[str], joint_vel: torch.Tensor):
        self._joint_names = list(joint_names)
        self.data = SimpleNamespace(joint_vel=joint_vel.clone())
        self.last_position_target = None
        self.last_position_joint_ids = None
        self.last_effort_target = None
        self.last_effort_joint_ids = None

    def find_joints(self, name_keys, preserve_order: bool = False):
        if isinstance(name_keys, str):
            patterns = [name_keys]
        else:
            patterns = list(name_keys)
        resolved_ids: list[int] = []
        resolved_names: list[str] = []
        for pattern in patterns:
            regex = re.compile(pattern)
            matches = [
                (idx, joint_name)
                for idx, joint_name in enumerate(self._joint_names)
                if regex.fullmatch(joint_name) and idx not in resolved_ids
            ]
            if preserve_order:
                for idx, joint_name in matches:
                    resolved_ids.append(idx)
                    resolved_names.append(joint_name)
            else:
                for idx, joint_name in matches:
                    resolved_ids.append(idx)
                    resolved_names.append(joint_name)
        return resolved_ids, resolved_names

    def set_joint_position_target(self, target: torch.Tensor, joint_ids=None, env_ids=None):
        self.last_position_target = target.clone()
        self.last_position_joint_ids = list(joint_ids)

    def set_joint_effort_target(self, target: torch.Tensor, joint_ids=None, env_ids=None):
        self.last_effort_target = target.clone()
        self.last_effort_joint_ids = list(joint_ids)


class _FakeEnv:
    def __init__(self, asset: _FakeArticulation, num_envs: int, device: str = "cpu"):
        self.num_envs = num_envs
        self.device = device
        self.scene = {"robot": asset}


def test_vehicle_action_term_decodes_to_steering_and_wheel_efforts():
    joint_names = [
        "front_left_steer_joint",
        "front_right_steer_joint",
        "front_left_wheel_joint",
        "front_right_wheel_joint",
        "rear_left_wheel_joint",
        "rear_right_wheel_joint",
    ]
    joint_vel = torch.tensor(
        [
            [0.0, 0.0, 2.0, -3.0, 0.5, -0.25],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    asset = _FakeArticulation(joint_names, joint_vel)
    env = _FakeEnv(asset, num_envs=2)
    cfg = VehicleActionTermCfg(
        asset_name="robot",
        steering_joint_names=["front_left_steer_joint", "front_right_steer_joint"],
        drive_joint_names=["front_left_wheel_joint", "front_right_wheel_joint"],
        brake_joint_names=[
            "front_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_left_wheel_joint",
            "rear_right_wheel_joint",
        ],
        steering_scale=0.5,
        drive_effort_scale=100.0,
        brake_effort_scale={"front_.*": 80.0, "rear_.*": 40.0},
        preserve_order=True,
    )

    term = VehicleActionTerm(cfg, env)
    term.process_actions(torch.tensor([[1.2, 2.0, 0.5], [0.6, -0.5, 0.25]], dtype=torch.float32))
    term.apply_actions()

    assert term.action_dim == 3
    assert torch.allclose(
        term.processed_actions,
        torch.tensor([[1.0, 1.0, 0.5], [0.6, -0.5, 0.25]], dtype=torch.float32),
    )
    assert asset.last_position_joint_ids == [0, 1]
    assert torch.allclose(
        asset.last_position_target,
        torch.tensor([[0.5, 0.5], [-0.25, -0.25]], dtype=torch.float32),
    )

    assert asset.last_effort_joint_ids == [2, 3, 4, 5]
    expected_efforts = torch.tensor(
        [
            [60.0, 140.0, -20.0, 20.0],
            [40.0, 40.0, -10.0, -10.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(asset.last_effort_target, expected_efforts)


def test_vehicle_action_term_keeps_last_brake_direction_when_wheels_stop():
    joint_names = [
        "front_left_steer_joint",
        "front_right_steer_joint",
        "front_left_wheel_joint",
        "front_right_wheel_joint",
        "rear_left_wheel_joint",
        "rear_right_wheel_joint",
    ]
    joint_vel = torch.tensor([[0.0, 0.0, 1.0, -2.0, 0.5, -0.25]], dtype=torch.float32)
    asset = _FakeArticulation(joint_names, joint_vel)
    env = _FakeEnv(asset, num_envs=1)
    cfg = VehicleActionTermCfg(
        asset_name="robot",
        steering_joint_names=["front_left_steer_joint", "front_right_steer_joint"],
        drive_joint_names=["front_left_wheel_joint", "front_right_wheel_joint"],
        brake_joint_names=[
            "front_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_left_wheel_joint",
            "rear_right_wheel_joint",
        ],
        steering_scale=0.5,
        drive_effort_scale=100.0,
        brake_effort_scale={"front_.*": 80.0, "rear_.*": 40.0},
        preserve_order=True,
    )

    term = VehicleActionTerm(cfg, env)
    term.process_actions(torch.tensor([[0.3, 0.0, 0.5]], dtype=torch.float32))
    term.apply_actions()

    asset.data.joint_vel.zero_()
    term.process_actions(torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32))
    term.apply_actions()

    expected_efforts = torch.tensor([[-80.0, 80.0, -40.0, 40.0]], dtype=torch.float32)
    assert torch.allclose(asset.last_effort_target, expected_efforts)
