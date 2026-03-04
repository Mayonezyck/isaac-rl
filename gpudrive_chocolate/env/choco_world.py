from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML not installed. Install it or switch to JSON config.") from exc

    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping/dict")
    return cfg


def meters_per_unit(stage) -> float:
    from pxr import UsdGeom

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    return float(mpu) if mpu and float(mpu) > 0 else 0.01


def ensure_world_default_prim(stage) -> None:
    from pxr import UsdGeom

    root = stage.GetPrimAtPath("/World")
    if not root.IsValid():
        root = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    if not stage.GetDefaultPrim().IsValid():
        stage.SetDefaultPrim(root)


def _safe_get_physx_attr(api, getter_names: Sequence[str]):
    for getter_name in getter_names:
        getter = getattr(api, getter_name, None)
        if callable(getter):
            try:
                attr = getter()
                if attr and attr.IsValid():
                    return attr.Get()
            except Exception:
                continue
    return None


def _safe_set_physx_attr(api, creator_names: Sequence[str], value) -> bool:
    for creator_name in creator_names:
        creator = getattr(api, creator_name, None)
        if callable(creator):
            try:
                attr = creator()
                if attr and attr.IsValid():
                    attr.Set(value)
                    return True
            except Exception:
                continue
    return False


def get_physx_scene_status(stage, scene_path: str = "/World/PhysicsScene") -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "scene_path": str(scene_path),
        "physx_scene_api_available": False,
        "enable_gpu_dynamics": None,
        "broadphase_type": None,
        "enable_ccd": None,
        "enable_contact_report": None,
    }

    scene_prim = stage.GetPrimAtPath(scene_path)
    if not scene_prim.IsValid():
        return status

    try:
        from pxr import PhysxSchema
    except Exception:
        return status

    try:
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    except Exception:
        return status

    status["physx_scene_api_available"] = True
    status["enable_gpu_dynamics"] = _safe_get_physx_attr(
        physx_scene,
        ("GetEnableGPUDynamicsAttr",),
    )
    status["broadphase_type"] = _safe_get_physx_attr(
        physx_scene,
        ("GetBroadphaseTypeAttr",),
    )
    status["enable_ccd"] = _safe_get_physx_attr(
        physx_scene,
        ("GetEnableCCDAttr",),
    )
    status["enable_contact_report"] = _safe_get_physx_attr(
        physx_scene,
        ("GetEnableContactReportAttr",),
    )
    return status


def ensure_physics_scene(
    stage,
    scene_path: str = "/World/PhysicsScene",
    *,
    physics_cfg: Dict[str, Any] | None = None,
    app_cfg: Dict[str, Any] | None = None,
) -> str:
    from pxr import Gf, UsdPhysics
    try:
        from pxr import PhysxSchema
    except Exception:
        PhysxSchema = None

    scene_prim = stage.GetPrimAtPath(scene_path)
    if not scene_prim.IsValid():
        scene = UsdPhysics.Scene.Define(stage, scene_path)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        scene_prim = scene.GetPrim()

    if PhysxSchema is not None:
        try:
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
            _safe_set_physx_attr(
                physx_scene,
                ("CreateEnableContactReportAttr",),
                True,
            )

            physics_cfg = physics_cfg or {}
            if physics_cfg.get("enable_gpu_dynamics", None) is not None:
                _safe_set_physx_attr(
                    physx_scene,
                    ("CreateEnableGPUDynamicsAttr",),
                    bool(physics_cfg["enable_gpu_dynamics"]),
                )
            if physics_cfg.get("broadphase_type", None) is not None:
                _safe_set_physx_attr(
                    physx_scene,
                    ("CreateBroadphaseTypeAttr",),
                    physics_cfg["broadphase_type"],
                )
            if physics_cfg.get("enable_ccd", None) is not None:
                _safe_set_physx_attr(
                    physx_scene,
                    ("CreateEnableCCDAttr",),
                    bool(physics_cfg["enable_ccd"]),
                )
        except Exception:
            pass

    physics_cfg = physics_cfg or {}
    if bool(physics_cfg.get("report_gpu_dynamics_once", False)):
        status = get_physx_scene_status(stage, scene_path)
        requested_active_gpu = None if app_cfg is None else app_cfg.get("active_gpu", None)
        requested_physics_gpu = None if app_cfg is None else app_cfg.get("physics_gpu", None)
        print(
            "[physx] "
            f"scene={status['scene_path']} "
            f"enable_gpu_dynamics={status['enable_gpu_dynamics']} "
            f"broadphase_type={status['broadphase_type']} "
            f"enable_ccd={status['enable_ccd']} "
            f"enable_contact_report={status['enable_contact_report']} "
            f"app_active_gpu={requested_active_gpu} "
            f"app_physics_gpu={requested_physics_gpu}"
        )
    return scene_path


def create_invisible_ground_plane(
    stage,
    *,
    prim_path: str = "/World/__GroundPlane",
    size_m: Tuple[float, float] = (2000.0, 2000.0),
    thickness_m: float = 0.2,
    z_m: float = 0.0,
    invisible: bool = True,
) -> str:
    from pxr import Gf, UsdGeom, UsdPhysics

    mpu = meters_per_unit(stage)

    w_m, l_m = float(size_m[0]), float(size_m[1])
    h_m = float(thickness_m)

    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.GetSizeAttr().Set(1.0)

    api = UsdGeom.XformCommonAPI(cube)
    api.SetTranslate(Gf.Vec3d(0.0, 0.0, (z_m - 0.5 * h_m) / mpu))
    api.SetScale(Gf.Vec3f(w_m / mpu, l_m / mpu, h_m / mpu))

    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if invisible:
        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()

    return prim_path


def _srgb_to_linear(c: float) -> float:
    c = float(c)
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _rgb_srgb_to_linear(rgb: Tuple[float, float, float]):
    from pxr import Gf

    return Gf.Vec3f(
        _srgb_to_linear(rgb[0]),
        _srgb_to_linear(rgb[1]),
        _srgb_to_linear(rgb[2]),
    )


def _get_or_create_preview_material(
    stage,
    mat_path: str,
    *,
    rgb_srgb: Tuple[float, float, float],
    emissive_strength: float = 0.0,
):
    from pxr import Sdf, UsdShade

    mat = UsdShade.Material.Get(stage, mat_path)
    if not mat:
        mat = UsdShade.Material.Define(stage, mat_path)

    shader_path = f"{mat_path}/PreviewSurface"
    shader = UsdShade.Shader.Get(stage, shader_path)
    if not shader:
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")

    rgb_lin = _rgb_srgb_to_linear(rgb_srgb)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(rgb_lin)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    if emissive_strength > 0.0:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            type(rgb_lin)(
                rgb_lin[0] * float(emissive_strength),
                rgb_lin[1] * float(emissive_strength),
                rgb_lin[2] * float(emissive_strength),
            )
        )

    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _bind_material(prim, material) -> None:
    from pxr import UsdShade

    try:
        UsdShade.MaterialBindingAPI(prim).Bind(material)
    except Exception:
        pass


def create_physics_preview_material(
    stage,
    *,
    mat_path: str,
    static_friction: float,
    dynamic_friction: float,
    restitution: float = 0.0,
    rgb_srgb: Tuple[float, float, float] = (0.6, 0.6, 0.6),
    emissive_strength: float = 0.0,
):
    from pxr import PhysxSchema, UsdPhysics

    mat = _get_or_create_preview_material(
        stage,
        mat_path,
        rgb_srgb=rgb_srgb,
        emissive_strength=emissive_strength,
    )
    prim = mat.GetPrim()

    physics_mat = UsdPhysics.MaterialAPI.Apply(prim)
    physics_mat.CreateStaticFrictionAttr().Set(float(static_friction))
    physics_mat.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    physics_mat.CreateRestitutionAttr().Set(float(restitution))
    PhysxSchema.PhysxMaterialAPI.Apply(prim)
    return mat


def create_ground_surface_cube(
    stage,
    *,
    prim_path: str,
    material,
    size_m: Tuple[float, float],
    thickness_m: float,
    z_m: float,
    invisible: bool,
) -> str:
    from pxr import Gf, UsdGeom, UsdPhysics

    mpu = meters_per_unit(stage)
    w_m, l_m = float(size_m[0]), float(size_m[1])
    h_m = float(thickness_m)

    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.GetSizeAttr().Set(1.0)

    api = UsdGeom.XformCommonAPI(cube)
    api.SetTranslate(Gf.Vec3d(0.0, 0.0, (z_m - 0.5 * h_m) / mpu))
    api.SetScale(Gf.Vec3f(w_m / mpu, l_m / mpu, h_m / mpu))

    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    _bind_material(prim, material)
    if invisible:
        UsdGeom.Imageable(prim).MakeInvisible()
    return prim_path


def _upsert_friction_table_entry(stage, *, table_path: str, material_path: str, friction_value: float) -> None:
    from pxr import PhysxSchema, Sdf

    table = PhysxSchema.PhysxVehicleTireFrictionTable.Get(stage, table_path)
    if not table:
        table = PhysxSchema.PhysxVehicleTireFrictionTable.Define(stage, table_path)

    rel = table.CreateGroundMaterialsRel()
    targets = list(rel.GetTargets())
    material_target = Sdf.Path(material_path)
    values_attr = table.CreateFrictionValuesAttr()
    values = list(values_attr.Get() or [])

    if material_target in targets:
        idx = targets.index(material_target)
        while len(values) <= idx:
            values.append(float(friction_value))
        values[idx] = float(friction_value)
    else:
        rel.AddTarget(material_target)
        targets = list(rel.GetTargets())
        while len(values) < len(targets) - 1:
            values.append(float(friction_value))
        values.append(float(friction_value))

    values_attr.Set(values)


def patch_vehicle_shared_friction_tables(
    stage,
    *,
    shared_root_path: str,
    material_path: str,
    friction_value: float,
) -> None:
    candidate_names = (
        "SummerTireFrictionTable",
        "AllSeasonFrictionTable",
        "SlickTireFrictionTable",
    )
    existing_paths = [
        f"{shared_root_path}/{name}"
        for name in candidate_names
        if stage.GetPrimAtPath(f"{shared_root_path}/{name}").IsValid()
    ]
    table_paths = existing_paths or [f"{shared_root_path}/AllSeasonFrictionTable"]

    for table_path in table_paths:
        _upsert_friction_table_entry(
            stage,
            table_path=table_path,
            material_path=material_path,
            friction_value=float(friction_value),
        )


_SEVEN_SEGMENT_MAP = {
    "0": "abcedf",
    "1": "bc",
    "2": "abdeg",
    "3": "abcdg",
    "4": "bcfg",
    "5": "acdfg",
    "6": "acdefg",
    "7": "abc",
    "8": "abcdefg",
    "9": "abcdfg",
    ".": ".",
    "-": "g",
}


def create_friction_label(
    stage,
    *,
    root_path: str,
    value_text: str,
    world_size_m: Tuple[float, float],
    ground_z_m: float,
    corner: str = "top_right",
    margin_m: float = 8.0,
    char_height_m: float = 8.0,
    char_spacing_m: float = 1.5,
    thickness_m: float = 0.08,
    z_offset_m: float = 0.04,
) -> str:
    from pxr import Gf, UsdGeom

    mpu = meters_per_unit(stage)
    world_w, world_l = float(world_size_m[0]), float(world_size_m[1])
    char_h = float(char_height_m)
    char_w = 0.62 * char_h
    seg_t = 0.16 * char_h
    h_len = max(char_w - 1.5 * seg_t, 0.5 * seg_t)
    v_len = max(0.5 * char_h - 1.5 * seg_t, 0.5 * seg_t)
    spacing = float(char_spacing_m)
    text = str(value_text)
    total_w = max(0.0, len(text) * char_w + max(0, len(text) - 1) * spacing)
    board_pad = 0.8
    board_w = total_w + 2.0 * board_pad
    board_h = char_h + 2.0 * board_pad
    label_z = float(ground_z_m) + float(z_offset_m)

    if str(corner).lower() == "top_left":
        cx = -0.5 * world_w + float(margin_m) + 0.5 * board_w
        cy = +0.5 * world_l - float(margin_m) - 0.5 * board_h
    elif str(corner).lower() == "bottom_left":
        cx = -0.5 * world_w + float(margin_m) + 0.5 * board_w
        cy = -0.5 * world_l + float(margin_m) + 0.5 * board_h
    elif str(corner).lower() == "bottom_right":
        cx = +0.5 * world_w - float(margin_m) - 0.5 * board_w
        cy = -0.5 * world_l + float(margin_m) + 0.5 * board_h
    else:
        cx = +0.5 * world_w - float(margin_m) - 0.5 * board_w
        cy = +0.5 * world_l - float(margin_m) - 0.5 * board_h

    label_root = UsdGeom.Xform.Define(stage, root_path)
    root_api = UsdGeom.XformCommonAPI(label_root)
    root_api.SetTranslate(Gf.Vec3d(cx / mpu, cy / mpu, label_z / mpu))

    bg_mat = _get_or_create_preview_material(
        stage,
        f"{root_path}/Materials/Board",
        rgb_srgb=(0.08, 0.08, 0.08),
        emissive_strength=0.15,
    )
    fg_mat = _get_or_create_preview_material(
        stage,
        f"{root_path}/Materials/Digits",
        rgb_srgb=(0.98, 0.98, 0.98),
        emissive_strength=0.40,
    )

    board = UsdGeom.Cube.Define(stage, f"{root_path}/Board")
    board.GetSizeAttr().Set(1.0)
    board_api = UsdGeom.XformCommonAPI(board)
    board_api.SetTranslate(Gf.Vec3d(0.0, 0.0, -0.5 * thickness_m / mpu))
    board_api.SetScale(Gf.Vec3f(board_w / mpu, board_h / mpu, (0.5 * thickness_m) / mpu))
    _bind_material(board.GetPrim(), bg_mat)

    x_cursor = -0.5 * total_w + 0.5 * char_w
    y_top = +0.5 * char_h - 0.5 * seg_t
    y_mid = 0.0
    y_bottom = -0.5 * char_h + 0.5 * seg_t
    y_upper = 0.25 * char_h
    y_lower = -0.25 * char_h
    x_left = -0.5 * char_w + 0.5 * seg_t
    x_right = +0.5 * char_w - 0.5 * seg_t

    for char_idx, char in enumerate(text):
        segments = _SEVEN_SEGMENT_MAP.get(char, "")
        char_root = UsdGeom.Xform.Define(stage, f"{root_path}/Chars/C{char_idx:02d}_{ord(char):03d}")
        char_api = UsdGeom.XformCommonAPI(char_root)
        char_api.SetTranslate(Gf.Vec3d(x_cursor / mpu, 0.0, 0.0))

        for seg_name in segments:
            if seg_name == ".":
                seg_x = x_right
                seg_y = y_bottom
                sx = seg_t
                sy = seg_t
                seg_token = "dot"
            elif seg_name == "a":
                seg_x, seg_y, sx, sy = 0.0, y_top, h_len, seg_t
                seg_token = seg_name
            elif seg_name == "b":
                seg_x, seg_y, sx, sy = x_right, y_upper, seg_t, v_len
                seg_token = seg_name
            elif seg_name == "c":
                seg_x, seg_y, sx, sy = x_right, y_lower, seg_t, v_len
                seg_token = seg_name
            elif seg_name == "d":
                seg_x, seg_y, sx, sy = 0.0, y_bottom, h_len, seg_t
                seg_token = seg_name
            elif seg_name == "e":
                seg_x, seg_y, sx, sy = x_left, y_lower, seg_t, v_len
                seg_token = seg_name
            elif seg_name == "f":
                seg_x, seg_y, sx, sy = x_left, y_upper, seg_t, v_len
                seg_token = seg_name
            else:
                seg_x, seg_y, sx, sy = 0.0, y_mid, h_len, seg_t
                seg_token = seg_name if seg_name != "." else "dot"

            seg = UsdGeom.Cube.Define(
                stage,
                f"{char_root.GetPath().pathString}/Seg_{seg_token}",
            )
            seg.GetSizeAttr().Set(1.0)
            seg_api = UsdGeom.XformCommonAPI(seg)
            seg_api.SetTranslate(Gf.Vec3d(seg_x / mpu, seg_y / mpu, 0.0))
            seg_api.SetScale(Gf.Vec3f(sx / mpu, sy / mpu, thickness_m / mpu))
            _bind_material(seg.GetPrim(), fg_mat)

        x_cursor += char_w + spacing

    return root_path


def _sanitize_ground_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            clean[str(key)] = value
    return clean


def _build_ground_friction_materials(world_specs: Sequence[Any]) -> Optional[List[Optional[Dict[str, Any]]]]:
    materials: List[Optional[Dict[str, Any]]] = []
    has_estimates = False

    for spec in world_specs:
        estimate = getattr(spec, "friction_estimate", None)
        request = getattr(spec, "friction_request", None)
        if estimate is None:
            materials.append(None)
            continue

        has_estimates = True
        scene_json_name = str(getattr(spec, "scene_json_name", ""))
        world_index = int(getattr(spec, "world_index", len(materials)))
        model_mu_static = float(estimate.mu_static)
        model_mu_dynamic = float(estimate.mu_dynamic)
        applied_static_friction = max(model_mu_static, model_mu_dynamic)
        applied_dynamic_friction = min(model_mu_static, model_mu_dynamic)
        metadata = _sanitize_ground_metadata(
            {
                "scene_json": scene_json_name,
                "road_type": estimate.road_type,
                "precip_type": getattr(request, "precip_type", None),
                "precip_intensity_mmph": getattr(request, "precip_intensity_mmph", None),
                "water_film_mm": estimate.water_film_mm,
                "theta_texture": estimate.theta_texture,
                "texture_amplitude_mm": estimate.texture_amplitude_mm,
                "model_mu_static": model_mu_static,
                "model_mu_dynamic": model_mu_dynamic,
                "applied_static_friction": applied_static_friction,
                "applied_dynamic_friction": applied_dynamic_friction,
                "mu_eff": estimate.mu_eff,
                "tire_model_id": estimate.tire_model_id,
                "params_source": estimate.params_source,
            }
        )
        water_desc = f"water:{estimate.water_film_mm:.3f}mm"
        precip_desc = (
            f" precip={metadata['precip_type']}:{metadata['precip_intensity_mmph']:.2f}mmph"
            if "precip_type" in metadata and "precip_intensity_mmph" in metadata
            else ""
        )
        print(
            f"[ground] world_{world_index:03d} scene={scene_json_name} "
            f"road={estimate.road_type or 'unknown'} {water_desc}{precip_desc} "
            f"applied_static={applied_static_friction:.4f} "
            f"applied_dynamic={applied_dynamic_friction:.4f} "
            f"mu_eff={estimate.mu_eff:.4f}"
        )
        materials.append(
            {
                "static_friction": applied_static_friction,
                "dynamic_friction": applied_dynamic_friction,
                "effective_friction": float(estimate.mu_eff),
                "label_value": float(estimate.mu_eff),
                "metadata": metadata,
            }
        )

    return materials if has_estimates else None


def create_per_world_ground_surfaces(
    stage,
    *,
    root_container: str,
    world_count: int,
    world_size_m: Tuple[float, float],
    thickness_m: float,
    z_m: float,
    invisible: bool,
    friction_values: Sequence[float],
    friction_materials: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    surface_color_srgb: Tuple[float, float, float] = (0.18, 0.18, 0.20),
    shared_root_path: str = "/World/VehicleShared",
    label_enable: bool = True,
    label_format: str = "{value:.2f}",
    label_corner: str = "top_right",
    label_margin_m: float = 8.0,
    label_char_height_m: float = 8.0,
    label_char_spacing_m: float = 1.5,
) -> None:
    values = [float(v) for v in friction_values if v is not None]
    if not values:
        values = [0.5]
    material_specs = list(friction_materials or [])
    if material_specs and len(material_specs) < int(world_count):
        raise ValueError(
            "friction_materials must include at least one entry per world "
            f"({len(material_specs)} < {world_count})"
        )

    for world_idx in range(int(world_count)):
        world_root = f"{root_container}/world_{world_idx:03d}"
        world_prim = stage.GetPrimAtPath(world_root)
        if not world_prim.IsValid():
            continue

        material_spec = material_specs[world_idx] if material_specs else None
        if material_spec is None:
            friction_value = values[world_idx % len(values)]
            static_friction = friction_value
            dynamic_friction = friction_value
            effective_friction = friction_value
            label_value = friction_value
            metadata: Dict[str, Any] = {}
        else:
            effective_friction = float(material_spec.get("effective_friction", 0.5))
            static_friction = float(
                material_spec.get("static_friction", effective_friction)
            )
            dynamic_friction = float(
                material_spec.get("dynamic_friction", effective_friction)
            )
            label_value = float(material_spec.get("label_value", effective_friction))
            metadata = _sanitize_ground_metadata(
                dict(material_spec.get("metadata", {}) or {})
            )
        material = create_physics_preview_material(
            stage,
            mat_path=f"{world_root}/Materials/GroundSurface",
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=0.0,
            rgb_srgb=surface_color_srgb,
            emissive_strength=0.04,
        )

        create_ground_surface_cube(
            stage,
            prim_path=f"{world_root}/GroundSurface",
            material=material,
            size_m=world_size_m,
            thickness_m=thickness_m,
            z_m=z_m,
            invisible=bool(invisible),
        )
        patch_vehicle_shared_friction_tables(
            stage,
            shared_root_path=shared_root_path,
            material_path=str(material.GetPath()),
            friction_value=effective_friction,
        )
        world_prim.SetCustomDataByKey("ground_friction", float(effective_friction))
        world_prim.SetCustomDataByKey("ground_static_friction", float(static_friction))
        world_prim.SetCustomDataByKey("ground_dynamic_friction", float(dynamic_friction))
        if metadata:
            world_prim.SetCustomDataByKey("ground_friction_metadata", metadata)
            if "road_type" in metadata:
                world_prim.SetCustomDataByKey("ground_road_type", str(metadata["road_type"]))
            if "water_film_mm" in metadata:
                world_prim.SetCustomDataByKey(
                    "ground_water_film_mm",
                    float(metadata["water_film_mm"]),
                )
            if "precip_intensity_mmph" in metadata:
                world_prim.SetCustomDataByKey(
                    "ground_precip_intensity_mmph",
                    float(metadata["precip_intensity_mmph"]),
                )

        if bool(label_enable):
            label_text = str(label_format).format(value=float(label_value))
            create_friction_label(
                stage,
                root_path=f"{world_root}/FrictionLabel",
                value_text=label_text,
                world_size_m=world_size_m,
                ground_z_m=z_m,
                corner=label_corner,
                margin_m=label_margin_m,
                char_height_m=label_char_height_m,
                char_spacing_m=label_char_spacing_m,
            )


def _enable_ext(ext_name: str) -> bool:
    import omni.kit.app

    try:
        em = omni.kit.app.get_app().get_extension_manager()
        if em.is_extension_enabled(ext_name):
            return True
        em.set_extension_enabled_immediate(ext_name, True)
        return em.is_extension_enabled(ext_name)
    except Exception as exc:
        print(f"[ext] failed enabling {ext_name}: {exc}")
        return False


def ensure_viewport_window(simulation_app) -> bool:
    try:
        import omni.kit.viewport.utility as vutil
        vw = vutil.get_active_viewport_window()
        if vw is not None:
            return True

        try:
            import omni.kit.window.viewport as vpwin
            vpwin.ViewportWindow()
        except Exception:
            pass

        simulation_app.update()
        simulation_app.update()

        vw = vutil.get_active_viewport_window()
        return vw is not None
    except Exception as exc:
        print("[viewport] ensure_viewport_window failed:", exc)
        return False


def configure_active_viewport(
    simulation_app,
    *,
    cam_path: str | None = None,
    width_px: int | None = None,
    height_px: int | None = None,
) -> bool:
    try:
        import omni.kit.viewport.utility as vutil

        if not ensure_viewport_window(simulation_app):
            return False

        vw = vutil.get_active_viewport_window()
        if vw is None:
            return False
        vp = vw.viewport_api

        if cam_path:
            try:
                vp.set_active_camera(cam_path)
            except Exception:
                try:
                    vp.set_active_camera_path(cam_path)
                except Exception:
                    pass

        if width_px is not None and height_px is not None:
            w = max(1, int(width_px))
            h = max(1, int(height_px))

            for setter in (
                lambda: vp.set_texture_resolution((w, h)),
                lambda: vp.set_texture_resolution(w, h),
                lambda: setattr(vp, "texture_resolution", (w, h)),
                lambda: setattr(vp, "resolution", (w, h)),
                lambda: vw.set_texture_resolution((w, h)),
                lambda: vw.resize(w, h),
                lambda: setattr(vw, "width", w),
                lambda: setattr(vw, "height", h),
            ):
                try:
                    setter()
                except Exception:
                    continue

        simulation_app.update()
        simulation_app.update()
        return True
    except Exception as exc:
        print("[viewport] configure_active_viewport failed:", exc)
        return False


def capture_active_viewport_png(filepath: str) -> bool:
    try:
        import omni.kit.viewport.utility as vutil
        vw = vutil.get_active_viewport_window()
        if vw is None:
            return False
        vp = vw.viewport_api
        if hasattr(vutil, "capture_viewport_to_file"):
            vutil.capture_viewport_to_file(vp, filepath)
            return True
        if hasattr(vutil, "capture_viewport_to_file_async"):
            vutil.capture_viewport_to_file_async(vp, filepath)
            return True
    except Exception:
        pass

    try:
        import omni.kit.capture.viewport as cap
        cap.capture_viewport_to_file(filepath)
        return True
    except Exception:
        return False


def create_dome_light(
    stage,
    path="/World/__DomeLight",
    intensity=3000.0,
    exposure=0.0,
    texture_file=None,
    rotation_deg_y=0.0,
):
    from pxr import UsdLux, UsdGeom

    dome = UsdLux.DomeLight.Define(stage, path)
    prim = dome.GetPrim()

    dome.CreateIntensityAttr(float(intensity))
    dome.CreateExposureAttr(float(exposure))

    if texture_file:
        dome.CreateTextureFileAttr(texture_file)

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    rot_op = xform.AddRotateYOp()
    rot_op.Set(float(rotation_deg_y))

    print(
        f"[light] created dome light {path} intensity={intensity} exposure={exposure} tex={texture_file}"
    )
    return prim


def _set_camera_world_matrix(cam_prim, M_world_units):
    from pxr import UsdGeom

    xf = UsdGeom.Xformable(cam_prim)
    ops = xf.GetOrderedXformOps()
    op = ops[0] if ops else xf.AddTransformOp()
    op.Set(M_world_units)


def apply_camera_pose_from_cfg(stage, cam_cfg: Dict[str, Any]) -> str:
    from pxr import Gf, UsdGeom

    cam_path = str(cam_cfg.get("prim_path", "/World/CaptureCam"))
    pose = cam_cfg.get("pose", {}) or {}

    pos = pose.get("pos_units", [0.0, 0.0, 100.0])
    axis = pose.get("rot_axis", [0.0, 0.0, 1.0])
    deg = float(pose.get("rot_deg", 0.0))

    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.CreateFocalLengthAttr(float(cam_cfg.get("focal_length", 24.0)))
    cam.CreateHorizontalApertureAttr(float(cam_cfg.get("horiz_aperture", 20.955)))
    cam.CreateVerticalApertureAttr(float(cam_cfg.get("vert_aperture", 15.2908)))

    R = Gf.Rotation(Gf.Vec3d(*map(float, axis)), deg)
    M = Gf.Matrix4d(1.0)
    M.SetRotate(R)
    M.SetTranslateOnly(Gf.Vec3d(*map(float, pos)))

    _set_camera_world_matrix(cam.GetPrim(), M)
    return cam_path


def compute_center_extent_from_world_roots(
    stage,
    *,
    root_container: str,
    world_count: int,
    bounds_size_m: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    from pxr import UsdGeom, Usd

    mpu = meters_per_unit(stage)
    half = 0.5 * float(bounds_size_m)

    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")

    found = 0
    for i in range(int(world_count)):
        wp = f"{root_container}/world_{i:03d}"
        prim = stage.GetPrimAtPath(wp)
        if not prim.IsValid():
            continue

        xf = UsdGeom.Xformable(prim)
        M = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t_u = M.ExtractTranslation()

        tx_m = float(t_u[0] * mpu)
        ty_m = float(t_u[1] * mpu)
        tz_m = float(t_u[2] * mpu)

        min_x = min(min_x, tx_m - half)
        max_x = max(max_x, tx_m + half)
        min_y = min(min_y, ty_m - half)
        max_y = max(max_y, ty_m + half)
        min_z = min(min_z, tz_m - half)
        max_z = max(max_z, tz_m + half)

        found += 1

    if found == 0:
        raise RuntimeError("No valid world roots found; cannot place camera.")

    center_m = (0.5 * (min_x + max_x), 0.5 * (min_y + max_y), 0.5 * (min_z + max_z))
    extent_m = (max_x - min_x, max_y - min_y, max_z - min_z)
    return center_m, extent_m


def ensure_camera_lookat(
    stage,
    *,
    cam_path: str,
    eye_m: Tuple[float, float, float],
    target_m: Tuple[float, float, float],
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    focal_length: float = 24.0,
    horiz_ap: float = 20.955,
    vert_ap: float = 15.2908,
) -> str:
    from pxr import Gf, UsdGeom

    mpu = meters_per_unit(stage)

    cam = UsdGeom.Camera.Define(stage, cam_path)
    cam.CreateFocalLengthAttr(float(focal_length))
    cam.CreateHorizontalApertureAttr(float(horiz_ap))
    cam.CreateVerticalApertureAttr(float(vert_ap))

    eye_u = Gf.Vec3d(eye_m[0] / mpu, eye_m[1] / mpu, eye_m[2] / mpu)
    tgt_u = Gf.Vec3d(target_m[0] / mpu, target_m[1] / mpu, target_m[2] / mpu)

    M = Gf.Matrix4d(1.0)
    M.SetLookAt(eye_u, tgt_u, Gf.Vec3d(*up))

    prim = cam.GetPrim()
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    op = ops[0] if ops else xf.AddTransformOp()
    op.Set(M)

    return cam_path


def place_camera_for_bbox(
    stage,
    cam_path: str,
    center_m: Tuple[float, float, float],
    extent_m: Tuple[float, float, float],
    *,
    tilt_deg: float = 35.0,
    yaw_deg: float = 225.0,
    margin: float = 1.25,
    look_at_z_offset_m: float = 0.0,
) -> None:
    import math
    from pxr import Gf, UsdGeom

    cx, cy, cz = center_m
    dx, dy, dz = extent_m
    span = max(dx, dy, dz, 1.0)

    dist = margin * span

    yaw = math.radians(yaw_deg)
    tilt = math.radians(tilt_deg)

    ex = cx + dist * math.cos(yaw) * math.cos(tilt)
    ey = cy + dist * math.sin(yaw) * math.cos(tilt)
    ez = cz + dist * math.sin(tilt) + 0.10 * span

    target = (cx, cy, cz + look_at_z_offset_m)

    ensure_camera_lookat(
        stage,
        cam_path=cam_path,
        eye_m=(ex, ey, ez),
        target_m=target,
        up=(0.0, 0.0, 1.0),
        focal_length=24.0,
        horiz_ap=20.955,
        vert_ap=15.2908,
    )

    mpu = meters_per_unit(stage)
    cam_prim = stage.GetPrimAtPath(cam_path)
    cam = UsdGeom.Camera(cam_prim)
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.1, float((dist + 2.0 * span) / max(mpu, 1e-6))))


def force_set_active_camera(cam_path: str) -> None:
    try:
        import omni.kit.viewport.utility as vutil
        vw = vutil.get_active_viewport_window()
        if vw is None:
            return
        vp = vw.viewport_api
        try:
            vp.set_active_camera(cam_path)
        except Exception:
            try:
                vp.set_active_camera_path(cam_path)
            except Exception:
                return
    except Exception:
        return


class ChocoWorldBuilder:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.simulation_app = None
        self.sim = None
        self.stage = None
        self.capture_camera_path = None
        self.capture_width_px = None
        self.capture_height_px = None

    def start(self):
        from isaacsim import SimulationApp

        app_cfg = self.cfg.get("app", {})
        sim_app_cfg = {
            "headless": bool(app_cfg.get("headless", True)),
            "renderer": str(app_cfg.get("renderer", "RayTracedLighting")),
        }
        for key in ("active_gpu", "physics_gpu", "multi_gpu", "max_gpu_count"):
            if key in app_cfg:
                sim_app_cfg[key] = app_cfg[key]
        self.simulation_app = SimulationApp(sim_app_cfg)

        import omni.usd
        from isaacsim.core.api import SimulationContext

        if not _enable_ext("omni.physx.vehicle"):
            raise RuntimeError("Could not enable omni.physx.vehicle extension")

        self.simulation_app.update()
        self.simulation_app.update()

        from src.chocolate_waymo_builder import ChocolateBarConstructor, GridLayout
        from src.chocolate_vehicle_controller import ChocolateWorldVehicleController
        from src.chocolate_obs_builder import ChocolateObsBuilder
        from src.trfc import prepare_stage_world_specs

        usd_ctx = omni.usd.get_context()
        usd_ctx.new_stage()
        self.stage = usd_ctx.get_stage()
        ensure_world_default_prim(self.stage)
        ensure_physics_scene(
            self.stage,
            "/World/PhysicsScene",
            physics_cfg=self.cfg.get("physics", {}),
            app_cfg=app_cfg,
        )

        world_specs = prepare_stage_world_specs(self.cfg)
        json_paths = [spec.scene_json_path for spec in world_specs]

        wcfg = self.cfg["world"]
        env_cfg = self.cfg.get("env", {})
        layout = GridLayout(
            world_size_m=tuple(map(float, wcfg["world_size_m"])),
            padding_m=float(wcfg["padding_m"]),
            grid_cols=int(wcfg["grid_cols"]),
            base_z_m=float(wcfg["base_z_m"]),
        )

        ctor = ChocolateBarConstructor(
            stage=self.stage,
            root_container=str(wcfg["root_container"]),
            layout=layout,
            origin_mode=str(wcfg.get("origin_mode", "center")),
        )

        road = self.cfg.get("road", {})
        agents = self.cfg.get("agents", {})

        ctor.build(
            json_paths=json_paths,
            world_count=int(wcfg["world_count"]),
            bounds_size_m=float(wcfg["bounds_size_m"]),
            max_agents_per_world=int(wcfg["max_agents_per_world"]),
            jump_break_m=float(road.get("jump_break_m", 3.0)),
            seg_width=float(road.get("seg_width", 0.10)),
            seg_height=float(road.get("seg_height", 0.10)),
            z_lift=float(road.get("z_lift", 0.02)),
            flatten_road_z=bool(road.get("flatten_road", True)),
            road_z_m=float(road.get("road_z_m", 0.0)),
            polyline_reduction_area=float(road.get("polyline_reduction_area", 0.0)),
            min_points_for_reduction=int(road.get("min_points_for_reduction", 10)),
            enable_segment_collision=bool(road.get("enable_segment_collision", False)),
            trigger_enable=bool(road.get("trigger_enable", False)),
            trigger_height_m=float(road.get("trigger_height_m", 1.0)),
            trigger_width_scale=float(road.get("trigger_width_scale", 1.0)),
            trigger_offset_z_m=float(road.get("trigger_offset_z_m", 0.5)),
            trigger_match_segment=bool(road.get("trigger_match_segment", True)),
            trigger_script_enable=bool(road.get("trigger_script_enable", True)),
            spawn_z_m=float(agents.get("spawn_z_m", 1.0)),
            goal_radius_m=float(agents.get("goal_radius_m", 3.0)),
            parked_if_start_in_goal=bool(agents.get("parked_if_start_in_goal", True)),
            skip_if_start_in_goal=bool(agents.get("skip_if_start_in_goal", False)),
            start_goal_thresh_m=float(agents.get("start_goal_thresh_m", 10.0)),
            parked_ground_z_m=float(agents.get("parked_ground_z_m", 0.0)),
            parked_chassis_size_m=tuple(
                map(float, agents.get("parked_chassis_size_m", [4.0, 2.0, 1.0]))
            ),
            parked_wheel_radius_m=float(agents.get("parked_wheel_radius_m", 0.35)),
            parked_wheel_thickness_m=float(
                agents.get("parked_wheel_thickness_m", 0.15)
            ),
            parked_wheel_inset_x_m=float(agents.get("parked_wheel_inset_x_m", 0.6)),
            parked_wheel_inset_y_m=float(agents.get("parked_wheel_inset_y_m", 0.05)),
            parked_ground_clearance_m=float(
                agents.get("parked_ground_clearance_m", 0.25)
            ),
            goal_ring_z_m=float(agents.get("goal_ring_z_m", 0.0)),
            goal_ring_tube_radius_m=float(
                agents.get("goal_ring_tube_radius_m", 0.12)
            ),
            goal_trigger_height_m=float(
                agents.get("goal_trigger_height_m", 0.6)
            ),
            vehicle_trigger_enable=bool(agents.get("vehicle_trigger_enable", False)),
            vehicle_trigger_offset_m=tuple(
                map(float, agents.get("vehicle_trigger_offset_m", [0.0, 0.0, 0.0]))
            ),
            vehicle_trigger_size_m=tuple(
                map(float, agents.get("vehicle_trigger_size_m", [1.0, 1.0, 1.0]))
            ),
            vehicle_trigger_script_enable=bool(
                agents.get("vehicle_trigger_script_enable", True)
            ),
        )

        gcfg = self.cfg.get("ground", {})
        if bool(gcfg.get("enable", True)):
            ground_mode = str(gcfg.get("mode", "global")).lower()
            ground_size_m = tuple(
                map(
                    float,
                    gcfg.get(
                        "size_m",
                        [float(wcfg["bounds_size_m"]), float(wcfg["bounds_size_m"])],
                    ),
                )
            )
            if ground_mode == "per_world":
                label_cfg = gcfg.get("label", {}) or {}
                friction_materials = _build_ground_friction_materials(world_specs)
                create_per_world_ground_surfaces(
                    self.stage,
                    root_container=str(wcfg["root_container"]),
                    world_count=int(wcfg["world_count"]),
                    world_size_m=ground_size_m,
                    thickness_m=float(gcfg.get("thickness_m", 0.2)),
                    z_m=float(gcfg.get("z_m", 0.0)),
                    invisible=bool(gcfg.get("invisible", False)),
                    friction_values=gcfg.get("friction_values", [0.5]),
                    friction_materials=friction_materials,
                    surface_color_srgb=tuple(
                        map(float, gcfg.get("color_srgb", [0.18, 0.18, 0.20]))
                    ),
                    shared_root_path="/World/VehicleShared",
                    label_enable=bool(label_cfg.get("enable", True)),
                    label_format=str(label_cfg.get("format", "{value:.2f}")),
                    label_corner=str(label_cfg.get("corner", "top_right")),
                    label_margin_m=float(label_cfg.get("margin_m", 8.0)),
                    label_char_height_m=float(label_cfg.get("char_height_m", 8.0)),
                    label_char_spacing_m=float(label_cfg.get("char_spacing_m", 1.5)),
                )
            else:
                create_invisible_ground_plane(
                    self.stage,
                    prim_path="/World/__GroundPlane",
                    size_m=ground_size_m,
                    thickness_m=float(gcfg.get("thickness_m", 0.2)),
                    z_m=float(gcfg.get("z_m", 0.0)),
                    invisible=bool(gcfg.get("invisible", True)),
                )

        light_cfg = self.cfg.get("light", {}) or {}
        if bool(light_cfg.get("enable", True)):
            create_dome_light(
                self.stage,
                path=str(light_cfg.get("path", "/World/__DomeLight")),
                intensity=float(light_cfg.get("intensity", 3000.0)),
                exposure=float(light_cfg.get("exposure", 0.0)),
                texture_file=light_cfg.get("texture_file", None),
                rotation_deg_y=float(light_cfg.get("rotation_deg_y", 0.0)),
            )

        cam_cfg = self.cfg.get("camera", {}) or {}
        cam_mode = str(cam_cfg.get("mode", "bbox"))
        cam_path = str(cam_cfg.get("prim_path", "/World/CaptureCam"))

        if cam_mode == "pose":
            cam_path = apply_camera_pose_from_cfg(self.stage, cam_cfg)
        else:
            center_m, extent_m = compute_center_extent_from_world_roots(
                self.stage,
                root_container=str(wcfg["root_container"]),
                world_count=int(wcfg["world_count"]),
                bounds_size_m=float(wcfg["bounds_size_m"]),
            )
            place_camera_for_bbox(
                self.stage,
                cam_path=cam_path,
                center_m=center_m,
                extent_m=extent_m,
                tilt_deg=float(cam_cfg.get("tilt_deg", 35.0)),
                yaw_deg=float(cam_cfg.get("yaw_deg", 225.0)),
                margin=float(cam_cfg.get("margin", 1.10)),
                look_at_z_offset_m=float(cam_cfg.get("look_at_z_offset_m", 0.0)),
            )

        self.capture_camera_path = cam_path
        self.capture_width_px = (
            int(cam_cfg["viewport_width"])
            if cam_cfg.get("viewport_width", None) is not None
            else None
        )
        self.capture_height_px = (
            int(cam_cfg["viewport_height"])
            if cam_cfg.get("viewport_height", None) is not None
            else None
        )

        configure_active_viewport(
            self.simulation_app,
            cam_path=cam_path,
            width_px=self.capture_width_px,
            height_px=self.capture_height_px,
        )

        self.simulation_app.update()
        self.simulation_app.update()

        phys_cfg = self.cfg.get("physics", {})
        physics_dt = float(phys_cfg.get("physics_dt", 1.0 / 60.0))
        render_dt = float(phys_cfg.get("rendering_dt", physics_dt))

        mpu = meters_per_unit(self.stage)
        self.sim = SimulationContext(
            stage_units_in_meters=mpu,
            physics_dt=physics_dt,
            rendering_dt=render_dt,
        )
        self.sim.initialize_physics()
        self.sim.step(render=False)

        ctrl_cfg = self.cfg.get("control", {})
        ctrl_suffix_candidates = ["", "/Vehicle", "/VehicleController"]
        ctrl = None
        for suf in ctrl_suffix_candidates:
            _ctrl = ChocolateWorldVehicleController(
                stage=self.stage,
                root_container=str(wcfg["root_container"]),
                world_count=int(wcfg["world_count"]),
                ctrl_suffix=suf,
                verbose=bool(env_cfg.get("verbose", False)),
            )
            _ctrl.refresh()
            if len(_ctrl.keys()) > 0:
                ctrl = _ctrl
                break

        if ctrl is None:
            raise RuntimeError("No controllable vehicles found; controller attrs not located.")

        obs_builder = ChocolateObsBuilder()

        return {
            "sim": self.sim,
            "stage": self.stage,
            "ctrl": ctrl,
            "obs_builder": obs_builder,
            "physics_dt": physics_dt,
            "action_repeat": int(ctrl_cfg.get("action_repeat", 4)),
        }

    def close(self) -> None:
        if self.simulation_app is not None:
            self.simulation_app.close()
            self.simulation_app = None

    def capture_frame(self, filepath: str) -> bool:
        if self.simulation_app is None:
            return False
        ensure_viewport_window(self.simulation_app)
        if self.capture_camera_path:
            force_set_active_camera(self.capture_camera_path)
        return capture_active_viewport_png(filepath)
