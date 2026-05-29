"""
Per-camera preferences for what to show on the dashboard Camera Detail page only.
Does not affect the main violations table or database.
"""
import json
import os
from datetime import datetime
from threading import Lock

import violations_log as vlog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PREFS_JSON = os.path.join(DATA_DIR, "camera_detail_prefs.json")
_lock = Lock()


def _now_cutoff():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(PREFS_JSON):
        with open(PREFS_JSON, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)


def _load():
    with _lock:
        _ensure_file()
        try:
            with open(PREFS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}


def _save(data):
    with _lock:
        _ensure_file()
        with open(PREFS_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def get_prefs(camera_id):
    cid = (camera_id or "").strip()
    if not cid:
        return {"detail_violations_since": "", "hidden_violation_images": []}
    entry = _load().get(cid) or {}
    hidden = entry.get("hidden_violation_images") or entry.get("hidden_images") or []
    return {
        "detail_violations_since": (entry.get("detail_violations_since") or "").strip(),
        "hidden_violation_images": [str(h).strip() for h in hidden if str(h).strip()],
    }


def merge_into_camera(cam):
    """Attach detail-view prefs to a camera dict for API/dashboard."""
    if not cam:
        return cam
    out = dict(cam)
    prefs = get_prefs(out.get("id") or "")
    out["detail_violations_since"] = prefs["detail_violations_since"]
    out["hidden_violation_images"] = prefs["hidden_violation_images"]
    return out


def clear_camera_detail(camera_id, location_name=""):
    """Hide all current snapshots for this camera's detail page."""
    cid = (camera_id or "").strip()
    if not cid:
        return None
    loc = (location_name or "").strip()
    hidden = set(get_prefs(cid)["hidden_violation_images"])
    if loc:
        hidden.update(vlog.list_violation_images_for_location(loc))
    data = _load()
    data[cid] = {
        "detail_violations_since": _now_cutoff(),
        "hidden_violation_images": sorted(hidden),
    }
    _save(data)
    return {
        "detail_violations_since": data[cid]["detail_violations_since"],
        "location": loc,
        "hidden_count": len(hidden),
    }


def clear_all_camera_details(cameras):
    """Clear camera detail for every camera in the list."""
    results = []
    for cam in cameras or []:
        cid = (cam.get("id") or "").strip()
        if not cid:
            continue
        loc = (cam.get("location_name") or "").strip()
        result = clear_camera_detail(cid, loc)
        if result:
            results.append({"camera_id": cid, **result})
    return results


def _violation_on_or_after(datetime_str, since_str):
    dt = (datetime_str or "").strip().replace("T", " ")
    since = (since_str or "").strip().replace("T", " ")
    if not since:
        return True
    if not dt:
        return False
    return dt >= since


def filter_violations_for_detail(camera_id, violations):
    """Return violations visible on camera detail (prefs applied)."""
    prefs = get_prefs(camera_id)
    hidden = set(prefs["hidden_violation_images"])
    since = prefs["detail_violations_since"]
    visible = []
    for item in violations or []:
        img = (item.get("image") or "").strip()
        if img and img in hidden:
            continue
        if since and not _violation_on_or_after(item.get("datetime"), since):
            continue
        visible.append(item)
    return visible


def hide_snapshot(camera_id, image_basename):
    cid = (camera_id or "").strip()
    img = (image_basename or "").strip()
    if not cid or not img:
        return False
    data = _load()
    entry = dict(data.get(cid) or {})
    hidden = {str(h).strip() for h in (entry.get("hidden_violation_images") or []) if str(h).strip()}
    hidden.add(img)
    entry["hidden_violation_images"] = sorted(hidden)
    if not (entry.get("detail_violations_since") or "").strip():
        entry["detail_violations_since"] = ""
    data[cid] = entry
    _save(data)
    return True
