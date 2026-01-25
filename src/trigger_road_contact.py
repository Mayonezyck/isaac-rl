import sys
from pxr import Usd, UsdGeom, UsdUtils, Vt


def _find_road_type(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    while prim and prim.IsValid():
        try:
            cd = prim.GetCustomData()
        except Exception:
            cd = {}
        if isinstance(cd, dict) and "road_type" in cd:
            try:
                return int(cd["road_type"])
            except Exception:
                return None
        prim = prim.GetParent()
    return None


def _find_vehicle_prim(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    while prim and prim.IsValid():
        try:
            cd = prim.GetCustomData()
        except Exception:
            cd = {}
        if isinstance(cd, dict) and "agent_id" in cd:
            return prim
        prim = prim.GetParent()
    return None


def _update_contact_list(veh_prim, road_type: int, enter: bool):
    try:
        cd = veh_prim.GetCustomData()
    except Exception:
        cd = {}
    if not isinstance(cd, dict):
        cd = {}

    cur = cd.get("road_contact_types", None)
    cur_set = set()
    if cur is not None:
        try:
            for v in cur:
                cur_set.add(int(v))
        except Exception:
            cur_set = set()

    if enter:
        cur_set.add(int(road_type))
    else:
        cur_set.discard(int(road_type))

    updated = Vt.IntArray(sorted(cur_set))
    veh_prim.SetCustomDataByKey("road_contact_types", updated)


def main():
    if not hasattr(sys, "argv"):
        sys.argv = [""]
    if len(sys.argv) != 6:
        return

    stage_id = int(sys.argv[1])
    trigger_path = sys.argv[2]
    other_path = sys.argv[3]
    event_name = sys.argv[4]

    cache = UsdUtils.StageCache.Get()
    stage = cache.Find(Usd.StageCache.Id.FromLongInt(stage_id))
    if not stage:
        return

    road_type = _find_road_type(stage, trigger_path)
    if road_type is None:
        return

    veh_prim = _find_vehicle_prim(stage, other_path)
    if veh_prim is None:
        return

    enter = event_name != "LeaveEvent"
    _update_contact_list(veh_prim, road_type, enter)


main()
