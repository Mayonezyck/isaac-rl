import os
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

def _update_vehicle_contact_list(veh_prim, other_agent_id: int, enter: bool):
    if not enter:
        return
    try:
        cd = veh_prim.GetCustomData()
    except Exception:
        cd = {}
    if not isinstance(cd, dict):
        cd = {}

    cur = cd.get("vehicle_contact_ids", None)
    cur_set = set()
    if cur is not None:
        try:
            for v in cur:
                cur_set.add(int(v))
        except Exception:
            cur_set = set()

    cur_set.add(int(other_agent_id))

    updated = Vt.IntArray(sorted(cur_set))
    veh_prim.SetCustomDataByKey("vehicle_contact_ids", updated)
    veh_prim.SetCustomDataByKey("vehicle_collided", True)


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
    if road_type is not None:
        veh_prim = _find_vehicle_prim(stage, other_path)
        if veh_prim is None:
            return
        enter = event_name != "LeaveEvent"
        _update_contact_list(veh_prim, road_type, enter)
        return

    a_veh_prim = _find_vehicle_prim(stage, other_path)
    b_veh_prim = _find_vehicle_prim(stage, trigger_path)
    if a_veh_prim is None or b_veh_prim is None:
        return

    try:
        b_cd = b_veh_prim.GetCustomData()
    except Exception:
        b_cd = {}
    if not isinstance(b_cd, dict):
        return
    b_agent_id = b_cd.get("agent_id", None)
    if b_agent_id is None:
        return

    try:
        a_cd = a_veh_prim.GetCustomData()
    except Exception:
        a_cd = {}
    if isinstance(a_cd, dict) and a_cd.get("agent_id", None) == b_agent_id:
        return

    enter = event_name != "LeaveEvent"
    if enter:
        a_agent_id = None
        try:
            a_agent_id = int(a_cd.get("agent_id", None))
        except Exception:
            a_agent_id = None
        if os.environ.get("CHOCO_TRIGGER_DEBUG", "").strip() == "1":
            print(f"--------Triggered -------- A={a_agent_id} hit B={int(b_agent_id)}")
    _update_vehicle_contact_list(a_veh_prim, int(b_agent_id), enter)


main()
