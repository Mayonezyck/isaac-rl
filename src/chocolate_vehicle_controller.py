from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Union

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, Sdf


# Your known working attr names:
ACCEL_ATTR = "physxVehicleController:accelerator"
STEER_ATTR = "physxVehicleController:steer"
BRAKE_ATTR = "physxVehicleController:brake0"


@dataclass(frozen=True)
class AgentKey:
    world_idx: int
    agent_id: int


@dataclass
class AgentHandle:
    key: AgentKey
    vehicle_root_path: str          # .../Vehicle_Parent/Vehicle
    ctrl_path: str                  # vehicle_root_path + ctrl_suffix
    pose_path: str                  # usually vehicle_root_path (stable)
    ctrl_prim: Usd.Prim
    pose_prim: Usd.Prim
    xform: UsdGeom.Xformable
    accel: Usd.Attribute
    steer: Optional[Usd.Attribute]
    brake: Optional[Usd.Attribute]


class ChocolateWorldVehicleController:
    """
    Clean controller for ChocolateBarConstructor worlds.

    - Enumerates vehicles by walking ONLY the expected world roots:
        /World/MiniWorlds/world_000 .. world_{world_count-1:03d}
      and ONLY looking at direct agent prim children in /Agents.

    - No "poke and find out" of arbitrary schema/attrs.
      The ONLY configurable bit is ctrl_suffix that defines where the controller attrs live.

    Usage:
      ctrl = ChocolateWorldVehicleController(stage, world_count=100, ctrl_suffix="/VehicleController")
      ctrl.refresh()

      # batch (aligned to ctrl.keys()):
      A = np.zeros((len(ctrl.keys()), 3), np.float32)
      A[:,0] = 0.2  # throttle
      ctrl.apply_all(A)

      # individual:
      ctrl.apply_one(99, 3163, thr=0.3, steer=0.0, brake=0.0)
    """

    def __init__(
        self,
        stage: Optional[Usd.Stage] = None,
        *,
        root_container: str = "/World/MiniWorlds",
        world_prefix: str = "world_",
        world_count: int = 100,
        ctrl_suffix: str = "",  # "" OR "/VehicleController" (set once, no guessing)
        accel_attr: str = ACCEL_ATTR,
        steer_attr: str = STEER_ATTR,
        brake_attr: str = BRAKE_ATTR,
        verbose: bool = True,
    ):
        self.stage = stage or omni.usd.get_context().get_stage()
        self.root_container = root_container
        self.world_prefix = world_prefix
        self.world_count = int(world_count)

        self.ctrl_suffix = str(ctrl_suffix)
        self.accel_attr = accel_attr
        self.steer_attr = steer_attr
        self.brake_attr = brake_attr
        self.verbose = verbose

        self._handles: Dict[AgentKey, AgentHandle] = {}
        self._ordered_keys: List[AgentKey] = []

    # -------------------------
    # Build registry
    # -------------------------

    def refresh(self) -> None:
        self._handles.clear()

        found_agents = 0
        found_ctrl = 0

        for wi in range(self.world_count):
            world_root = f"{self.root_container}/{self.world_prefix}{wi:03d}"
            agents_root = f"{world_root}/Agents"
            agents_prim = self.stage.GetPrimAtPath(agents_root)
            if not agents_prim.IsValid():
                continue

            # Only iterate direct children: Agent_####_id####
            for agent_prim in agents_prim.GetAllChildren():
                name = agent_prim.GetName()
                agent_id = self._parse_agent_id_from_agent_name(name)
                if agent_id is None:
                    continue

                # Vehicle root is fixed by your builder:
                vehicle_root_path = f"{agent_prim.GetPath().pathString}/Vehicle_Parent/Vehicle"
                vehicle_root_prim = self.stage.GetPrimAtPath(vehicle_root_path)
                if not vehicle_root_prim.IsValid():
                    continue

                # ✅ Pose prim is the moving child:
                #pose_path = f"{agent_prim.GetPath().pathString}/Vehicle_Parent"
                pose_path = vehicle_root_path + "/Vehicle"
                pose_prim = self.stage.GetPrimAtPath(pose_path)
                if not pose_prim.IsValid():
                    # fallback to old behavior if needed
                    pose_path = vehicle_root_path
                    pose_prim = vehicle_root_prim


                found_agents += 1

                # Controller prim path is fixed by ctrl_suffix (no scanning)
                ctrl_path = vehicle_root_path + self.ctrl_suffix
                ctrl_prim = self.stage.GetPrimAtPath(ctrl_path)
                if not ctrl_prim.IsValid():
                    continue

                accel = ctrl_prim.GetAttribute(self.accel_attr)
                if not accel.IsValid():
                    continue

                steer = ctrl_prim.GetAttribute(self.steer_attr)
                brake = ctrl_prim.GetAttribute(self.brake_attr)

                key = AgentKey(world_idx=wi, agent_id=int(agent_id))
                handle = AgentHandle(
                    key=key,
                    vehicle_root_path=vehicle_root_path,
                    ctrl_path=ctrl_path,
                    pose_path=pose_path,  # stable pose prim
                    ctrl_prim=ctrl_prim,
                    pose_prim=pose_prim,
                    xform=UsdGeom.Xformable(pose_prim),
                    accel=accel,
                    steer=steer if steer.IsValid() else None,
                    brake=brake if brake.IsValid() else None,
                )

                self._handles[key] = handle
                found_ctrl += 1

        self._ordered_keys = sorted(self._handles.keys(), key=lambda k: (k.world_idx, k.agent_id))

        if self.verbose:
            print(f"[ChocoCtrl] refresh: agents_with_vehicle={found_agents}, controllable={found_ctrl}, keys={len(self._ordered_keys)}")
            if len(self._ordered_keys) == 0:
                print(f"[ChocoCtrl] NOTE: controllable=0. Check ctrl_suffix='{self.ctrl_suffix}' and attr names.")

    def keys(self) -> List[AgentKey]:
        return list(self._ordered_keys)

    def get(self, world_idx: int, agent_id: int) -> Optional[AgentHandle]:
        return self._handles.get(AgentKey(int(world_idx), int(agent_id)))

    # -------------------------
    # Apply controls
    # -------------------------

    def apply_one(self, world_idx: int, agent_id: int, *, thr: float, steer: float, brake: float) -> bool:
        h = self.get(world_idx, agent_id)
        if h is None:
            return False
        return self._apply_handle(h, thr, steer, brake)

    def apply_batch(self, keys: Iterable[AgentKey], controls: Union[np.ndarray, List[List[float]]]) -> int:
        """
        controls: (K,3) [thr, steer, brake] aligned with keys iteration order
        """
        Klist = list(keys)
        U = np.asarray(controls, dtype=np.float32)
        if U.ndim != 2 or U.shape[0] != len(Klist) or U.shape[1] != 3:
            raise ValueError(f"controls must be (K,3). got {U.shape} for K={len(Klist)}")

        ok = 0
        for i, k in enumerate(Klist):
            h = self._handles.get(k)
            if h is None:
                continue
            if self._apply_handle(h, float(U[i,0]), float(U[i,1]), float(U[i,2])):
                ok += 1
        return ok

    def apply_all(self, controls: Union[np.ndarray, List[List[float]]]) -> int:
        """Apply to all controllable vehicles in keys() order."""
        return self.apply_batch(self._ordered_keys, controls)

    def zero_all(self, brake: float = 1.0) -> int:
        Ks = self._ordered_keys
        if not Ks:
            return 0
        U = np.zeros((len(Ks), 3), dtype=np.float32)
        U[:, 2] = float(brake)
        return self.apply_all(U)

    # -------------------------
    # Pose helpers
    # -------------------------

    def get_xy_u(self, world_idx: int, agent_id: int) -> Optional[Tuple[float, float]]:
        h = self.get(world_idx, agent_id)
        if h is None:
            return None
        xf = h.xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        p = xf.ExtractTranslation()
        return (float(p[0]), float(p[1]))

    # -------------------------
    # Internals
    # -------------------------

    def _apply_handle(self, h: AgentHandle, thr: float, steer: float, brake: float) -> bool:
        try:
            h.accel.Set(float(thr))
            if h.steer is not None:
                h.steer.Set(float(steer))
            if h.brake is not None:
                h.brake.Set(float(brake))
            return True
        except Exception:
            return False

    @staticmethod
    def _parse_agent_id_from_agent_name(name: str) -> Optional[int]:
        # expects Agent_0028_id3163
        s = str(name)
        j = s.rfind("_id")
        if j < 0:
            return None
        digits = []
        for ch in s[j+3:]:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if not digits:
            return None
        try:
            return int("".join(digits))
        except Exception:
            return None
