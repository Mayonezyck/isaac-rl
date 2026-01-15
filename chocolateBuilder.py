# chocolate_waymo_builder.py
# Build a "chocolate bar" grid of mini-worlds in Isaac Sim from Waymo-scene JSONs.
#
# ✅ Key properties:
# - Each mini-world is its own prim: /World/MiniWorlds/world_000, world_001, ...
# - The grid placement is ONLY via root-prim translation (like a chocolate bar).
# - Each mini-world enforces a local bounds box (e.g., 200m x 200m). Out-of-bounds => NOT SPAWNED.
# - Road segments are grouped by category/type so the policy can "recognize category" (one instancer per type).
# - Vehicles spawn via the SAME PhysX Vehicle Wizard API you pasted (omni.physxvehicle...).
# - Each agent gets a destination cylinder marker (radius=3m).
#
# Assumed JSON schema (matches your earlier comments):
# cfg["road"]["polylines"] = list of dicts with keys:
#   - "type": int
#   - "id": int
#   - "xyz": list[[x,y,z], ...]   (meters)
#   - optional "poly_id" or similar (not required)
#
# cfg["agents"]["items"] = list of dicts with:
#   - "agent_id" (optional)
#   - "agent_type" (optional)
#   - "track_idx" (optional)
#   - "start": {"x","y","z","yaw"}   yaw in radians
#   - "end":   {"x","y","z","yaw"}   yaw in radians (yaw optional)
#
# If your JSON differs, tell me the real keys and I’ll adjust.

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

import omni.usd

# ---- PhysX vehicle wizard imports (THE WORKING ONES you pasted) ----
from omni.physxvehicle.scripts.wizards import physxVehicleWizard as VehicleWizard
from omni.physxvehicle.scripts.helpers.UnitScale import UnitScale
from omni.physxvehicle.scripts.commands import PhysXVehicleWizardCreateCommand

ROOT_PATH = "/World"
SHARED_ROOT = ROOT_PATH + "/VehicleShared"


# ----------------------------
# Utilities
# ----------------------------

def _ensure_world_default_prim(stage: Usd.Stage) -> None:
    world = stage.GetPrimAtPath(ROOT_PATH)
    if not world.IsValid():
        world = UsdGeom.Xform.Define(stage, ROOT_PATH).GetPrim()
    if not stage.GetDefaultPrim().IsValid():
        stage.SetDefaultPrim(world)

def _get_unit_scale(stage: Usd.Stage) -> Tuple[UnitScale, float]:
    """Return (UnitScale, meters_per_unit)."""
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    if meters_per_unit == 0:
        meters_per_unit = 0.01  # Isaac default is commonly cm
    length_scale = 1.0 / meters_per_unit

    kilograms_per_unit = UsdPhysics.GetStageKilogramsPerUnit(stage)
    if kilograms_per_unit == 0:
        kilograms_per_unit = 1.0
    mass_scale = 1.0 / kilograms_per_unit
    return UnitScale(length_scale, mass_scale), meters_per_unit

def _safe_int(x, default=-1) -> int:
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

def _safe_float(x, default=None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default

def _quath_from_yaw_z(yaw_rad: float) -> Gf.Quath:
    """Half-precision quat rotation about +Z."""
    half = 0.5 * float(yaw_rad)
    w = float(np.cos(half))
    z = float(np.sin(half))
    try:
        return Gf.Quath(Gf.Half(w), Gf.Half(0.0), Gf.Half(0.0), Gf.Half(z))
    except Exception:
        return Gf.Quath(w, 0.0, 0.0, z)

def _reduce_polyline_xy(points_xyz: np.ndarray, area_thresh: float) -> np.ndarray:
    """Iteratively remove points with small local triangle area (XY) under threshold."""
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.shape[0] < 3 or area_thresh <= 0:
        return pts

    keep = np.ones((pts.shape[0],), dtype=bool)
    changed = True

    def tri_area_xy(a, b, c) -> float:
        abx = float(b[0] - a[0]); aby = float(b[1] - a[1])
        acx = float(c[0] - a[0]); acy = float(c[1] - a[1])
        cross = abx * acy - aby * acx
        return 0.5 * abs(cross)

    while changed:
        changed = False
        idxs = np.nonzero(keep)[0]
        if idxs.size < 3:
            break
        for k in range(1, idxs.size - 1):
            i0 = idxs[k - 1]; i1 = idxs[k]; i2 = idxs[k + 1]
            if tri_area_xy(pts[i0], pts[i1], pts[i2]) < area_thresh:
                keep[i1] = False
                changed = True
    return pts[keep]


# ----------------------------
# Mini-world builder (ONE world)
# ----------------------------

@dataclass
class LocalBounds:
    width_m: float = 200.0
    length_m: float = 200.0
    origin_xy: Tuple[float, float] = (0.0, 0.0)   # local-space center of bounds


class WaymoJsonMiniWorldBuilder:
    """
    Builds ONE mini-world under world_root (which is already placed in the big grid by root translation).

    Origin handling (this is the part you’re unsure about):
    - origin_mode="center": compute a scene center from road points, subtract it so content is near (0,0).
    - origin_mode="zero": no recentering; use JSON coords as-is (then bounds should likely follow those coords).
    """

    def __init__(
        self,
        stage: Optional[Usd.Stage] = None,
        world_root: str = "/World/MiniWorlds/world_000",
        bounds: Optional[LocalBounds] = None,
        origin_mode: str = "center",  # "center" or "zero"
    ):
        self.stage = stage or omni.usd.get_context().get_stage()
        self.world_root = world_root
        self.bounds = bounds or LocalBounds()
        self.origin_mode = origin_mode

        # internal computed origin shift (what we subtract from JSON coords)
        self._scene_center = np.zeros((3,), dtype=np.float32)

        # ensure root prim exists
        UsdGeom.Xform.Define(self.stage, self.world_root)

        # subfolders
        self.road_root = f"{self.world_root}/Road"
        self.agents_root = f"{self.world_root}/Agents"
        self.goals_root = f"{self.world_root}/Goals"
        UsdGeom.Xform.Define(self.stage, self.road_root)
        UsdGeom.Xform.Define(self.stage, self.agents_root)
        UsdGeom.Xform.Define(self.stage, self.goals_root)

    # -------- bounds gating in LOCAL coords --------
    def _in_bounds_xy(self, x: float, y: float) -> bool:
        ox, oy = self.bounds.origin_xy
        hw = 0.5 * float(self.bounds.width_m)
        hl = 0.5 * float(self.bounds.length_m)
        return (ox - hw <= x <= ox + hw) and (oy - hl <= y <= oy + hl)

    def _filter_polyline_points(self, pts: np.ndarray) -> np.ndarray:
        if pts.shape[0] == 0:
            return pts
        m = np.array([self._in_bounds_xy(float(p[0]), float(p[1])) for p in pts], dtype=bool)
        return pts[m]

    # -------- origin shift --------
    def _compute_scene_center_from_road(self, polylines: List[Dict[str, Any]]) -> np.ndarray:
        all_pts = []
        for pl in polylines:
            xyz = pl.get("xyz", None)
            if not xyz:
                continue
            arr = np.asarray(xyz, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 1:
                if arr.shape[1] == 2:
                    arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=np.float32)], axis=1)
                all_pts.append(arr[:, :3])
        if not all_pts:
            return np.zeros((3,), dtype=np.float32)
        pts = np.concatenate(all_pts, axis=0)
        return pts.mean(axis=0)

    def _to_local_xyz(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        # subtract scene center if origin_mode="center"
        return (float(x - self._scene_center[0]),
                float(y - self._scene_center[1]),
                float(z - self._scene_center[2]))

    # -------- road segments grouped by type --------
    def build_road_segments_grouped(
        self,
        cfg: Dict[str, Any],
        polyline_reduction_area: float = 0.0,   # 0 disables reduction
        min_points_for_reduction: int = 10,
        jump_break_m: float = 3.0,
        seg_width: float = 0.10,
        seg_height: float = 0.10,
        z_lift: float = 0.02,
        max_segments_per_type: Optional[int] = None,
    ) -> None:
        road = (cfg.get("road", {}) or {})
        polylines = road.get("polylines", []) or []

        # compute scene center (for origin_mode="center")
        if self.origin_mode == "center":
            self._scene_center = self._compute_scene_center_from_road(polylines)
        else:
            self._scene_center = np.zeros((3,), dtype=np.float32)

        # group polylines by type
        by_type: Dict[int, List[np.ndarray]] = {}
        for pl in polylines:
            t = _safe_int(pl.get("type", -1), -1)
            xyz = pl.get("xyz", None)
            if not xyz:
                continue
            pts = np.asarray(xyz, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            if pts.shape[1] == 2:
                pts = np.concatenate([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)], axis=1)

            # convert to LOCAL coords
            pts_local = np.array([self._to_local_xyz(p[0], p[1], p[2]) for p in pts[:, :3]], dtype=np.float32)

            # bounds filter (local)
            pts_local = self._filter_polyline_points(pts_local)
            if pts_local.shape[0] < 2:
                continue

            # optional per-polyline reduction
            if polyline_reduction_area > 0 and pts_local.shape[0] >= int(min_points_for_reduction):
                pts_local = _reduce_polyline_xy(pts_local, float(polyline_reduction_area))
                if pts_local.shape[0] < 2:
                    continue

            by_type.setdefault(t, []).append(pts_local)

        # build one PointInstancer per type
        for t, polys in sorted(by_type.items(), key=lambda kv: kv[0]):
            type_root = f"{self.road_root}/Type_{t:02d}"
            instancer_path = f"{type_root}/Segments"

            UsdGeom.Xform.Define(self.stage, type_root)

            seg_positions_py = []
            seg_orients_py = []
            seg_scales_py = []
            proto_indices_py = []

            seg_count = 0

            for poly in polys:
                for i in range(poly.shape[0] - 1):
                    p0 = poly[i]
                    p1 = poly[i + 1]

                    # endpoint bounds gate (already point-filtered, but keep it strict)
                    if not (self._in_bounds_xy(float(p0[0]), float(p0[1])) and self._in_bounds_xy(float(p1[0]), float(p1[1]))):
                        continue

                    dx = float(p1[0] - p0[0])
                    dy = float(p1[1] - p0[1])
                    length = float(math.sqrt(dx * dx + dy * dy))
                    if length < 1e-6:
                        continue

                    # break on big jumps (prevents bad connections)
                    if length > float(jump_break_m):
                        continue

                    mid = Gf.Vec3f(
                        float((p0[0] + p1[0]) * 0.5),
                        float((p0[1] + p1[1]) * 0.5),
                        float((p0[2] + p1[2]) * 0.5 + z_lift),
                    )

                    yaw = float(math.atan2(dy, dx))
                    q = _quath_from_yaw_z(yaw)
                    scale = Gf.Vec3f(float(length), float(seg_width), float(seg_height))

                    seg_positions_py.append(mid)
                    seg_orients_py.append(q)
                    seg_scales_py.append(scale)
                    proto_indices_py.append(0)

                    seg_count += 1
                    if max_segments_per_type is not None and seg_count >= int(max_segments_per_type):
                        break
                if max_segments_per_type is not None and seg_count >= int(max_segments_per_type):
                    break

            if seg_count == 0:
                continue

            seg_positions = Vt.Vec3fArray(seg_positions_py)
            seg_orients = Vt.QuathArray(seg_orients_py)
            seg_scales = Vt.Vec3fArray(seg_scales_py)
            proto_indices = Vt.IntArray(proto_indices_py)

            instancer = UsdGeom.PointInstancer.Define(self.stage, instancer_path)
            proto_path = f"{instancer_path}/CubeProto"
            cube = UsdGeom.Cube.Define(self.stage, proto_path)
            cube.GetSizeAttr().Set(1.0)

            instancer.CreatePrototypesRel().SetTargets([cube.GetPath()])
            instancer.GetPositionsAttr().Set(seg_positions)
            instancer.GetOrientationsAttr().Set(seg_orients)
            instancer.GetScalesAttr().Set(seg_scales)
            instancer.GetProtoIndicesAttr().Set(proto_indices)

            # tag category on the instancer prim (easy to read later)
            instancer.GetPrim().SetCustomDataByKey("road_type", int(t))

    # -------- vehicles + goals --------
    def _spawn_vehicle_wizard_under(self, parent_path: str, position_m: Tuple[float, float, float], yaw_deg: float) -> Optional[str]:
        stage = self.stage
        _ensure_world_default_prim(stage)

        # create parent Xform at desired pose (LOCAL coords; root prim handles grid translation)
        parent_xform = UsdGeom.Xform.Define(stage, parent_path)
        xform_api = UsdGeom.XformCommonAPI(parent_xform)

        unit_scale, meters_per_unit = _get_unit_scale(stage)

        x_m, y_m, z_m = position_m
        pos_units = Gf.Vec3d(x_m / meters_per_unit, y_m / meters_per_unit, z_m / meters_per_unit)

        xform_api.SetTranslate(pos_units)
        xform_api.SetRotate(Gf.Vec3f(0.0, 0.0, float(yaw_deg)), UsdGeom.XformCommonAPI.RotationOrderXYZ)

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

        return vehicle_data.rootVehiclePath

    def _spawn_goal_cylinder(self, prim_path: str, x: float, y: float, z: float, radius_m: float = 3.0) -> None:
        cyl = UsdGeom.Cylinder.Define(self.stage, prim_path)
        cyl.CreateRadiusAttr(float(radius_m))
        cyl.CreateHeightAttr(0.2)

        xf = UsdGeom.Xform.Define(self.stage, prim_path + "_Xform")
        api = UsdGeom.XformCommonAPI(xf)
        api.SetTranslate(Gf.Vec3d(float(x), float(y), float(z + 0.1)))
        # re-parent cylinder under xform for stable transforms
        self.stage.GetPrimAtPath(prim_path).GetParent().GetChildren()
        # easiest: just set translate directly on cylinder if you prefer:
        # UsdGeom.XformCommonAPI(cyl).SetTranslate(Gf.Vec3d(x,y,z+0.1))

        # Instead of re-parenting complexity, simplest:
        api2 = UsdGeom.XformCommonAPI(cyl)
        api2.SetTranslate(Gf.Vec3d(float(x), float(y), float(z + 0.1)))

    def build_agents_with_goals(
        self,
        cfg: Dict[str, Any],
        max_agents: Optional[int] = None,
        spawn_z_m: float = 1.0,   # PhysX vehicles often need a small lift
        goal_radius_m: float = 3.0,
        require_goal_in_bounds: bool = True,
    ) -> None:
        agents = (cfg.get("agents", {}) or {}).get("items", []) or []
        kept = 0
        skipped = 0

        for idx, a in enumerate(agents):
            if max_agents is not None and kept >= int(max_agents):
                break

            s = a.get("start", {}) or {}
            e = a.get("end", {}) or {}

            sx = _safe_float(s.get("x", None))
            sy = _safe_float(s.get("y", None))
            sz = _safe_float(s.get("z", 0.0), 0.0)
            syaw = _safe_float(s.get("yaw", 0.0), 0.0)  # radians

            ex = _safe_float(e.get("x", None))
            ey = _safe_float(e.get("y", None))
            ez = _safe_float(e.get("z", 0.0), 0.0)

            if sx is None or sy is None or ex is None or ey is None:
                skipped += 1
                continue

            # convert to LOCAL (apply origin_mode center shift)
            sx, sy, sz = self._to_local_xyz(sx, sy, sz)
            ex, ey, ez = self._to_local_xyz(ex, ey, ez)

            # bounds gate (local)
            if not self._in_bounds_xy(float(sx), float(sy)):
                skipped += 1
                continue
            if require_goal_in_bounds and (not self._in_bounds_xy(float(ex), float(ey))):
                skipped += 1
                continue

            agent_id = _safe_int(a.get("agent_id", idx), idx)
            agent_path = f"{self.agents_root}/Agent_{kept:04d}_id{agent_id}"
            UsdGeom.Xform.Define(self.stage, agent_path)

            # vehicle under agent folder
            veh_parent = f"{agent_path}/Vehicle_Parent"
            yaw_deg = float(np.degrees(float(syaw)))
            veh_outer = self._spawn_vehicle_wizard_under(
                veh_parent,
                position_m=(float(sx), float(sy), float(spawn_z_m)),
                yaw_deg=yaw_deg,
            )
            if veh_outer is None:
                skipped += 1
                continue

            # tag agent metadata
            prim_outer = self.stage.GetPrimAtPath(veh_outer)
            prim_outer.SetCustomDataByKey("agent_id", int(agent_id))
            prim_outer.SetCustomDataByKey("agent_type", int(_safe_int(a.get("agent_type", -1), -1)))
            prim_outer.SetCustomDataByKey("track_idx", int(_safe_int(a.get("track_idx", kept), kept)))

            # destination cylinder (radius 3m)
            goal_path = f"{self.goals_root}/Goal_{kept:04d}_id{agent_id}"
            cyl = UsdGeom.Cylinder.Define(self.stage, goal_path)
            cyl.CreateRadiusAttr(float(goal_radius_m))
            cyl.CreateHeightAttr(0.2)
            UsdGeom.XformCommonAPI(cyl).SetTranslate(Gf.Vec3d(float(ex), float(ey), float(ez + 0.1)))

            kept += 1

        print(f"[MiniWorldBuilder] Agents kept={kept} skipped={skipped}")

    # -------- entry point --------
    def build_from_json(
        self,
        json_path: Union[str, Path],
        *,
        max_agents: int = 50,
        # road params:
        polyline_reduction_area: float = 0.0,
        min_points_for_reduction: int = 10,
        jump_break_m: float = 3.0,
        seg_width: float = 0.10,
        seg_height: float = 0.10,
        z_lift: float = 0.02,
        # agent params:
        spawn_z_m: float = 1.0,
        goal_radius_m: float = 3.0,
    ) -> None:
        p = Path(json_path).expanduser().resolve()
        with p.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        # safe scalar metadata only
        root_prim = self.stage.GetPrimAtPath(self.world_root)
        root_prim.SetCustomDataByKey("scene_json", str(p))
        root_prim.SetCustomDataByKey("origin_mode", str(self.origin_mode))
        root_prim.SetCustomDataByKey("bounds_w_m", float(self.bounds.width_m))
        root_prim.SetCustomDataByKey("bounds_l_m", float(self.bounds.length_m))

        self.build_road_segments_grouped(
            cfg,
            polyline_reduction_area=polyline_reduction_area,
            min_points_for_reduction=min_points_for_reduction,
            jump_break_m=jump_break_m,
            seg_width=seg_width,
            seg_height=seg_height,
            z_lift=z_lift,
        )
        self.build_agents_with_goals(
            cfg,
            max_agents=max_agents,
            spawn_z_m=spawn_z_m,
            goal_radius_m=goal_radius_m,
            require_goal_in_bounds=True,
        )


# ----------------------------
# Chocolate-bar multi-world constructor (GRID of mini-worlds)
# ----------------------------

@dataclass
class GridLayout:
    world_size_m: Tuple[float, float] = (200.0, 200.0)
    padding_m: float = 0.0          # 0 => chocolate tiles touching
    grid_cols: int = 5
    base_z_m: float = 0.0


class ChocolateBarConstructor:
    """
    Places N mini-world roots in a grid and spawns each from a JSON.
    IMPORTANT: The per-world filtering is LOCAL, grid translation is on the root prim only.
    """

    def __init__(
        self,
        stage: Optional[Usd.Stage] = None,
        root_container: str = "/World/MiniWorlds",
        layout: GridLayout = GridLayout(),
        origin_mode: str = "center",   # recommended
    ):
        self.stage = stage or omni.usd.get_context().get_stage()
        self.root_container = root_container
        self.layout = layout
        self.origin_mode = origin_mode

        UsdGeom.Xform.Define(self.stage, self.root_container)

    def _world_root_path(self, i: int) -> str:
        return f"{self.root_container}/world_{i:03d}"

    def _world_translation(self, i: int) -> Tuple[float, float, float]:
        cols = int(self.layout.grid_cols)
        sx, sy = self.layout.world_size_m
        pitch_x = float(sx + self.layout.padding_m)
        pitch_y = float(sy + self.layout.padding_m)
        r = i // cols
        c = i % cols
        return (float(c * pitch_x), float(r * pitch_y), float(self.layout.base_z_m))

    def _set_root_translation(self, prim_path: str, txyz: Tuple[float, float, float]) -> None:
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            UsdGeom.Xform.Define(self.stage, prim_path)
            prim = self.stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.XformCommonAPI(prim)
        xform.SetTranslate(Gf.Vec3d(*txyz))

    def clear_all(self) -> None:
        for p in self.stage.Traverse():
            # fast-ish delete approach: delete the container children by name
            pass
        # easiest: just RemovePrim the container and recreate it
        self.stage.RemovePrim(Sdf.Path(self.root_container))
        UsdGeom.Xform.Define(self.stage, self.root_container)

    def build(
        self,
        json_paths: Sequence[Union[str, Path]],
        world_count: int,
        *,
        bounds_size_m: float = 200.0,
        max_agents_per_world: int = 50,
        # road params:
        polyline_reduction_area: float = 0.0,
        min_points_for_reduction: int = 10,
        jump_break_m: float = 3.0,
        seg_width: float = 0.10,
        seg_height: float = 0.10,
        z_lift: float = 0.02,
        # agent params:
        spawn_z_m: float = 1.0,
        goal_radius_m: float = 3.0,
    ) -> None:
        json_list = [str(Path(p).expanduser().resolve()) for p in json_paths]
        if len(json_list) == 0:
            raise ValueError("json_paths is empty")

        for i in range(int(world_count)):
            root = self._world_root_path(i)
            UsdGeom.Xform.Define(self.stage, root)

            # place root in grid (chocolate layout)
            txyz = self._world_translation(i)
            self._set_root_translation(root, txyz)

            # mini-world local bounds centered at (0,0)
            bounds = LocalBounds(width_m=float(bounds_size_m), length_m=float(bounds_size_m), origin_xy=(0.0, 0.0))

            builder = WaymoJsonMiniWorldBuilder(
                stage=self.stage,
                world_root=root,
                bounds=bounds,
                origin_mode=self.origin_mode,
            )

            json_path = json_list[i % len(json_list)]
            builder.build_from_json(
                json_path,
                max_agents=max_agents_per_world,
                polyline_reduction_area=polyline_reduction_area,
                min_points_for_reduction=min_points_for_reduction,
                jump_break_m=jump_break_m,
                seg_width=seg_width,
                seg_height=seg_height,
                z_lift=z_lift,
                spawn_z_m=spawn_z_m,
                goal_radius_m=goal_radius_m,
            )

        print(f"[ChocolateBarConstructor] Built {world_count} worlds under {self.root_container}")
