# main_chocolate_isaac.py
#
# Run Isaac Sim headless or with UI, build the chocolate-bar worlds, then tick the sim.
#
# Usage examples:
#   # UI mode (recommended first):
#   ./python.sh chocolate_main.py --json_dir /path/to/jsons --worlds 16 --cols 4
#
#   # Headless:
#   ./python.sh main_chocolate_isaac.py --headless --json_dir /path/to/jsons --worlds 64 --cols 8
#
# Notes:
# - This assumes chocolateBuilder.py is importable and contains ChocolateBarConstructor, GridLayout.
# - If it's in the same folder, this script will import it fine.
# - Vehicle Wizard requires PhysX Vehicle extension enabled (you already have it working in Isaac UI).

import argparse
import time
from pathlib import Path

# Isaac Sim app
from isaacsim import SimulationApp

def enable_exts(ext_names):
    import omni.kit.app
    app = omni.kit.app.get_app()
    em = app.get_extension_manager()

    for name in ext_names:
        try:
            em.set_extension_enabled_immediate(name, True)
            print(f"[ext] enabled: {name}")
        except Exception as e:
            print(f"[ext] failed to enable {name}: {e}")

    # IMPORTANT: let Kit load them
    for _ in range(5):
        app.update()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="Run without UI.")
    ap.add_argument("--json_dir", type=str, required=True, help="Folder containing scene_*.json (Waymo scene jsons).")
    ap.add_argument("--glob", type=str, default="*.json", help="Glob pattern inside json_dir.")
    ap.add_argument("--worlds", type=int, default=16, help="How many mini-worlds to build.")
    ap.add_argument("--cols", type=int, default=4, help="Grid columns.")
    ap.add_argument("--world_size", type=float, default=200.0, help="Each world is world_size x world_size meters.")
    ap.add_argument("--padding", type=float, default=0.0, help="Gap between worlds (meters). 0 => chocolate tiles touch.")
    ap.add_argument("--max_agents", type=int, default=30, help="Max agents per world.")
    ap.add_argument("--steps", type=int, default=600, help="How many simulation ticks to run.")
    ap.add_argument("--dt", type=float, default=1.0 / 60.0, help="Sim dt.")
    ap.add_argument("--substeps", type=int, default=1, help="Physics substeps.")
    ap.add_argument("--play", action="store_true", help="Start timeline playing (usually not required, but can help).")
    ap.add_argument("--warmup_steps", type=int, default=5, help="A few updates after building (lets UI/prim settle).")
    return ap.parse_args()


def main():
    args = parse_args()

    # 1) Start Isaac Sim
    sim_app = SimulationApp(
        {
            "headless": bool(args.headless),
            # These can help determinism / performance; keep minimal.
            "renderer": "RayTracedLighting" if not args.headless else "None",
        }
    )

    import omni.kit.app

    app = omni.kit.app.get_app()
    em = app.get_extension_manager()

    # List all extension IDs that contain "vehicle" or "physx"
    all_exts = em.get_extensions()  # list of dicts
    hits = []
    for e in all_exts:
        ext_id = e.get("id") or ""
        name = e.get("name") or ""
        if ("vehicle" in ext_id.lower()) or ("vehicle" in name.lower()) or ("physx" in ext_id.lower()):
            hits.append((ext_id, name, e.get("version")))

    hits = sorted(hits, key=lambda x: x[0])
    print("\n".join([f"{a} | {b} | {c}" for a,b,c in hits[:200]]))
    print(f"\n[dbg] total hits: {len(hits)}")

    # 2) Now safe to import omni/pxr things
    import omni.usd
    import omni.kit.app
    import omni.timeline
    from pxr import UsdGeom, UsdPhysics, PhysxSchema

    # ✅ enable extensions BEFORE importing your builder module
    enable_exts([
        "omni.physxvehicle",          # the one your working code uses
        "omni.physx.vehicle",         # sometimes present in other installs
    ])

    # Your builder module
    from chocolateBuilder import ChocolateBarConstructor, GridLayout

    stage = omni.usd.get_context().get_stage()

    # 3) (Optional) Stage unit sanity (meters recommended)
    # If your stage is cm (0.01), your code still works, but setting meters avoids confusion.
    try:
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    except Exception as e:
        print("[WARN] Could not set metersPerUnit to 1.0:", e)

    # 4) Ensure physics scene exists
    # Isaac often already has one, but we create if missing.
    physics_scene_path = "/World/physicsScene"
    if not stage.GetPrimAtPath(physics_scene_path).IsValid():
        scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
        scene.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

    # 5) Configure PhysX scene params (optional but helpful)
    try:
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(physics_scene_path))
        physx_scene.CreateEnableCCDAttr(True)
        physx_scene.CreateEnableStabilizationAttr(True)
    except Exception as e:
        print("[WARN] PhysX scene tuning skipped:", e)

    # 6) Collect json paths
    json_dir = Path(args.json_dir).expanduser().resolve()
    json_paths = sorted(json_dir.glob(args.glob))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in {json_dir} with pattern {args.glob}")

    # 7) Build chocolate-bar worlds
    layout = GridLayout(
        world_size_m=(float(args.world_size), float(args.world_size)),
        padding_m=float(args.padding),
        grid_cols=int(args.cols),
        base_z_m=0.0,
    )

    ctor = ChocolateBarConstructor(
        stage=stage,
        root_container="/World/MiniWorlds",
        layout=layout,
        origin_mode="center",
    )

    ctor.build(
        json_paths=json_paths,
        world_count=int(args.worlds),
        bounds_size_m=float(args.world_size),
        max_agents_per_world=int(args.max_agents),
        # road viz tuning (same defaults you used)
        jump_break_m=3.0,
        seg_width=0.10,
        seg_height=0.10,
        z_lift=0.02,
        polyline_reduction_area=0.0,
    )

    # 8) Warm-up a few UI/app updates so everything materializes
    app = omni.kit.app.get_app()
    for _ in range(int(args.warmup_steps)):
        app.update()

    # 9) Set simulation settings
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_time_codes_per_second(1.0 / float(args.dt))

    # Physics dt is usually controlled by /physicsScene; but timeline dt works fine for ticking.
    # If you use Isaac Lab later, you’ll switch to World/SimulationContext.

    if args.play:
        timeline.play()

    # 10) Tick simulator
    print(f"[main] ticking {args.steps} steps (dt={args.dt}, headless={args.headless})")
    for i in range(int(args.steps)):
        app.update()  # advances sim + renders if UI
        if (i % 60) == 0:
            print(f"[main] step {i}/{args.steps}")

    # 11) Stop and close
    if args.play:
        timeline.stop()

    sim_app.close()
    print("[main] done.")


if __name__ == "__main__":
    main()
