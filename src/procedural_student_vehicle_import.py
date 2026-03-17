from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import traceback

from src.physx_teacher_patch_track import _enable_extensions, _ensure_physics_scene, _ensure_world_default_prim, _set_stage_units
from src.procedural_student_vehicle import (
    StudentVehicleSpec,
    build_default_student_vehicle_spec,
    load_student_vehicle_spec,
    nominal_root_height_m,
    write_student_vehicle_spec,
    write_student_vehicle_urdf,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and import a procedural student vehicle into Isaac Sim.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", type=str, default="artifacts/student_vehicle_assets/default")
    parser.add_argument("--spec-json", type=str, default="", help="Optional JSON file overriding StudentVehicleSpec fields.")
    parser.add_argument("--urdf-path", type=str, default="", help="Optional existing URDF to import instead of generating one.")
    parser.add_argument("--save-stage-usd", type=str, default="", help="Optional explicit USD export path.")
    parser.add_argument("--sim-steps", type=int, default=180)
    parser.add_argument("--spawn-x-m", type=float, default=0.0)
    parser.add_argument("--spawn-y-m", type=float, default=0.0)
    parser.add_argument("--spawn-yaw-deg", type=float, default=0.0)
    return parser.parse_args()


def _build_spec(args: argparse.Namespace) -> StudentVehicleSpec:
    if str(args.spec_json):
        return load_student_vehicle_spec(args.spec_json)
    return build_default_student_vehicle_spec()


def _write_artifacts(output_dir: str | Path, spec: StudentVehicleSpec, urdf_path_arg: str) -> tuple[Path, Path]:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    spec_path = output_root / "student_vehicle_spec.json"
    write_student_vehicle_spec(spec_path, spec)
    if str(urdf_path_arg):
        return Path(urdf_path_arg).expanduser().resolve(), spec_path
    urdf_path = output_root / "student_fwd_vehicle.urdf"
    write_student_vehicle_urdf(urdf_path, spec)
    return urdf_path, spec_path


def main() -> int:
    args = _parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    spec = _build_spec(args)
    urdf_path, spec_path = _write_artifacts(output_root, spec, args.urdf_path)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})
    try:
        _enable_extensions(["isaacsim.asset.importer.urdf"])

        import omni.kit.app
        import omni.kit.commands
        import omni.usd
        from pxr import Gf, PhysicsSchemaTools, Sdf, UsdGeom, UsdLux

        app = omni.kit.app.get_app()
        for _ in range(10):
            app.update()

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        _ensure_world_default_prim(stage)
        _set_stage_units(stage)
        _ensure_physics_scene(stage)

        PhysicsSchemaTools.addGroundPlane(stage, "/World/GroundPlane", "Z", 50.0, Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.5))
        light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
        light.CreateIntensityAttr(700.0)

        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status:
            raise RuntimeError("URDFCreateImportConfig failed")
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = False
        import_config.collision_from_visuals = False
        import_config.parse_mimic = True
        try:
            from isaacsim.asset.importer.urdf._urdf import UrdfJointTargetType

            import_config.default_drive_type = UrdfJointTargetType.JOINT_DRIVE_NONE
        except Exception:
            pass

        import_result = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(urdf_path),
            import_config=import_config,
        )
        if isinstance(import_result, tuple) and len(import_result) == 2 and isinstance(import_result[0], bool):
            import_status, robot_root_path = import_result
            if not import_status:
                raise RuntimeError(f"URDFParseAndImportFile failed: {import_result!r}")
        else:
            robot_root_path = import_result
        if not isinstance(robot_root_path, str) or not robot_root_path:
            raise RuntimeError(f"URDFParseAndImportFile returned an invalid robot path: {import_result!r}")

        for _ in range(5):
            app.update()

        root_prim = stage.GetPrimAtPath(robot_root_path)
        if not root_prim.IsValid():
            raise RuntimeError(f"Imported robot root does not exist: {robot_root_path}")

        xform_api = UsdGeom.XformCommonAPI(root_prim)
        xform_api.SetTranslate(
            Gf.Vec3d(
                float(args.spawn_x_m),
                float(args.spawn_y_m),
                float(nominal_root_height_m(spec)),
            )
        )
        xform_api.SetRotate(
            Gf.Vec3f(0.0, 0.0, float(args.spawn_yaw_deg)),
            UsdGeom.XformCommonAPI.RotationOrderXYZ,
        )

        for _ in range(max(1, int(args.sim_steps))):
            simulation_app.update()

        stage_usd_path = Path(args.save_stage_usd).expanduser().resolve() if str(args.save_stage_usd) else output_root / "student_vehicle_stage.usd"
        stage.Export(str(stage_usd_path))

        meta = {
            "robot_root_path": str(robot_root_path),
            "urdf_path": str(urdf_path),
            "spec_path": str(spec_path),
            "stage_usd_path": str(stage_usd_path),
            "sim_steps": int(args.sim_steps),
            "spawn": {
                "x_m": float(args.spawn_x_m),
                "y_m": float(args.spawn_y_m),
                "yaw_deg": float(args.spawn_yaw_deg),
                "z_m": float(nominal_root_height_m(spec)),
            },
            "spec": asdict(spec),
        }
        meta_path = output_root / "student_vehicle_import_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"[procedural_student_vehicle_import] imported student vehicle to {robot_root_path}", flush=True)
        print(f"[procedural_student_vehicle_import] wrote stage to {stage_usd_path}", flush=True)
        print(f"[procedural_student_vehicle_import] wrote metadata to {meta_path}", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
