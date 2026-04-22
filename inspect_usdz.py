from pxr import Usd, UsdGeom, Gf

stage = Usd.Stage.Open("/home/yz8733/Github/isaac-rl/Simple_Car_Low_Poly_-_Rigged.usdz")

print("Up axis:", UsdGeom.GetStageUpAxis(stage))
print("Meters per unit:", UsdGeom.GetStageMetersPerUnit(stage))

bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
root = stage.GetPseudoRoot()
bbox = bbox_cache.ComputeWorldBound(root)
rng = bbox.ComputeAlignedRange()
mn, mx = rng.GetMin(), rng.GetMax()
size = mx - mn
mpu = UsdGeom.GetStageMetersPerUnit(stage)
print(f"BBox min: ({mn[0]:.4f}, {mn[1]:.4f}, {mn[2]:.4f})")
print(f"BBox max: ({mx[0]:.4f}, {mx[1]:.4f}, {mx[2]:.4f})")
print(f"Size (stage units): X={size[0]:.4f}, Y={size[1]:.4f}, Z={size[2]:.4f}")
print(f"Size (meters): X={size[0]*mpu:.4f}, Y={size[1]*mpu:.4f}, Z={size[2]*mpu:.4f}")

for p in stage.GetPseudoRoot().GetChildren():
    print(f"Root prim: {p.GetPath()} type={p.GetTypeName()}")

# Walk the tree and print bounding boxes of key prims
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Xform) or prim.IsA(UsdGeom.Mesh):
        bb = bbox_cache.ComputeWorldBound(prim)
        r = bb.ComputeAlignedRange()
        s = r.GetMax() - r.GetMin()
        sm = (s[0]*mpu, s[1]*mpu, s[2]*mpu)
        if max(sm) > 0.01:
            print(f"  {prim.GetPath()} [{prim.GetTypeName()}] size_m=({sm[0]:.3f}, {sm[1]:.3f}, {sm[2]:.3f})")
