"""Pre-process the low-poly car USDZ: remove ground plane, fix orientation, save as .usd."""
from pxr import Usd, UsdGeom, Gf, Sdf

SRC = "/home/yz8733/Github/isaac-rl/Simple_Car_Low_Poly_-_Rigged.usdz"
DST = "/home/yz8733/Github/isaac-rl/artifacts/low_poly_car_proxy.usd"

stage = Usd.Stage.Open(SRC)

# --- Remove ground plane ---
ground = stage.GetPrimAtPath("/scene/Meshes/Sketchfab_model/root/GLTF_SceneRootNode/Ground_2")
if ground:
    stage.RemovePrim(ground.GetPath())
    print("Removed Ground_2 prim")

# --- Check current bbox after ground removal ---
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
root = stage.GetPseudoRoot()
bbox = bbox_cache.ComputeWorldBound(root)
rng = bbox.ComputeAlignedRange()
mn, mx = rng.GetMin(), rng.GetMax()
mpu = UsdGeom.GetStageMetersPerUnit(stage)
size = mx - mn
print(f"Car bbox (meters): X={size[0]*mpu:.3f}, Y={size[1]*mpu:.3f}, Z={size[2]*mpu:.3f}")
print(f"Car bbox min (stage): ({mn[0]:.1f}, {mn[1]:.1f}, {mn[2]:.1f})")
print(f"Car bbox max (stage): ({mx[0]:.1f}, {mx[1]:.1f}, {mx[2]:.1f})")
print(f"Up axis: {UsdGeom.GetStageUpAxis(stage)}, metersPerUnit: {mpu}")

# --- Add a corrective Xform wrapper ---
# USDZ: Y-up, car long axis = Z, car width = X
# Isaac Sim: Z-up, car forward = +X
# Need: USDZ-Y -> Isaac-Z (up), USDZ-Z -> Isaac-X (forward), USDZ-X -> Isaac-Y (left)
# This is a rotation: first -90 deg around X, then -90 deg around new-Z
# Or equivalently: permute axes (X,Y,Z) -> (Y,Z,X) with sign adjustments.
#
# Rotation matrix we want:
#   Isaac-X = USDZ-Z  (forward)
#   Isaac-Y = USDZ-X  (left) 
#   Isaac-Z = USDZ-Y  (up)
#
# As rotation: Rz(-90) * Rx(-90)
# Rx(-90) quat = (cos(-45), sin(-45), 0, 0) = (0.7071, -0.7071, 0, 0)
# Rz(-90) quat = (cos(-45), 0, 0, sin(-45)) = (0.7071, 0, 0, -0.7071)
# Combined = Rz * Rx

from math import sqrt
s = sqrt(0.5)
# Rx(-90): w=s, x=-s, y=0, z=0
qx = Gf.Quatd(s, Gf.Vec3d(-s, 0, 0))
# Rz(-90): w=s, x=0, y=0, z=-s  
qz = Gf.Quatd(s, Gf.Vec3d(0, 0, s))
q_fix = qz * qx
print(f"Correction quat (w,x,y,z): ({q_fix.GetReal():.4f}, {q_fix.GetImaginary()[0]:.4f}, {q_fix.GetImaginary()[1]:.4f}, {q_fix.GetImaginary()[2]:.4f})")

# Apply rotation to the scene root
scene_prim = stage.GetPrimAtPath("/scene")
xformable = UsdGeom.Xformable(scene_prim)
xformable.ClearXformOpOrder()
rot_op = xformable.AddRotateXYZOp(opSuffix="fix")
# USDZ axes: X=width(2.1), Y=height(1.6), Z=length(4.1), Up=Y
# Isaac axes: X=forward(length), Y=left(width), Z=up(height)
# Step 1: Rx(-90) rotates Y-up to Z-up:  X->X, Y->Z, Z->-Y
#   After: X=width, Y=-length, Z=height  
# Step 2: Rz(+90) rotates to get length on X: X->Y, Y->-X, Z->Z
#   After: X=length, Y=width, Z=height  ✓
# But car faces -Z originally, so after Rx(-90) it faces +Y, after Rz(+90) it faces -X
# Try Rz(-90) instead: X->-Y, Y->X
#   After Rx(-90): X=width, Y=-length, Z=height
#   After Rz(-90): X=length, Y=-width, Z=height  ... close but let's just try
rot_op.Set(Gf.Vec3f(90.0, 0.0, 90.0))

# Save first, then reload to get accurate bbox
stage.GetRootLayer().Export(DST)

# Reload and verify
stage2 = Usd.Stage.Open(DST)
bbox_cache2 = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
bbox2 = bbox_cache2.ComputeWorldBound(stage2.GetPseudoRoot())
rng2 = bbox2.ComputeAlignedRange()
mn2, mx2 = rng2.GetMin(), rng2.GetMax()
size2 = mx2 - mn2
mpu2 = UsdGeom.GetStageMetersPerUnit(stage2)
print(f"After Rx(-90) bbox (meters): X={size2[0]*mpu2:.3f}, Y={size2[1]*mpu2:.3f}, Z={size2[2]*mpu2:.3f}")
print(f"  (expect: X=width~2.1, Y=length~4.1, Z=height~1.6)")
print(f"Saved to {DST}")
