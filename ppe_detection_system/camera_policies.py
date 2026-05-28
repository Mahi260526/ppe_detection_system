"""
Camera setup: cameras and area-wise violation policies.
Each camera can have multiple areas; each area has a rule type that defines what counts as a violation.
"""
import os
import json
import uuid
from threading import Lock

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
POLICIES_JSON = os.path.join(DATA_DIR, "camera_policies.json")
_lock = Lock()

# Rule types: what violation is generated when condition is met in that area
RULE_TYPES = [
    {"id": "ppe", "label": "PPE violation", "description": "No helmet, no vest, no glasses, no mask"},
    {"id": "person_detected", "label": "Person detected", "description": "Any person in this area is a violation"},
    {"id": "person_out_of_lane", "label": "Person out of lane", "description": "Person outside defined walking lane"},
    {"id": "vehicle_detected", "label": "Vehicle detected", "description": "Any vehicle in this area is a violation"},
]

def _ensure_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(POLICIES_JSON):
        with open(POLICIES_JSON, "w") as f:
            json.dump({"cameras": [], "areas": []}, f, indent=2)

def _load():
    with _lock: #Lock to prevent multiple threads from accessing the file at the same time.
        _ensure_file()
        try:
            with open(POLICIES_JSON, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"cameras": [], "areas": []}

def _save(data):
    with _lock:
        _ensure_file()
        with open(POLICIES_JSON, "w") as f:
            json.dump(data, f, indent=2)

def get_rule_types():
    return list(RULE_TYPES)

def get_cameras():
    """Return list of cameras (id, name, source, location_name)."""
    data = _load()
    return list(data.get("cameras", []))

def get_camera(camera_id):
    data = _load()
    for c in data.get("cameras", []):
        if (c.get("id") or "") == camera_id:
            return c
    return None

def save_camera(camera_id=None, name="", source=0, location_name=""):
    """Create or update camera. source can be int (index) or string (URL). Returns camera id."""
    data = _load()
    cameras = list(data.get("cameras", []))
    if camera_id:
        for i, c in enumerate(cameras):
            if (c.get("id") or "") == camera_id:
                cameras[i] = {
                    "id": camera_id,
                    "name": (name or "").strip() or c.get("name", ""),
                    "source": source if isinstance(source, (int, float)) else str(source),
                    "location_name": (location_name or "").strip() or c.get("location_name", ""),
                }
                data["cameras"] = cameras
                _save(data)
                return camera_id
    new_id = "cam_" + uuid.uuid4().hex[:8]
    cameras.append({
        "id": new_id,
        "name": (name or "").strip() or "Camera",
        "source": source if isinstance(source, (int, float)) else str(source),
        "location_name": (location_name or "").strip() or "",
    })
    data["cameras"] = cameras
    _save(data)
    return new_id

def delete_camera(camera_id):
    """Remove camera and all its areas."""
    data = _load()
    data["cameras"] = [c for c in data.get("cameras", []) if (c.get("id") or "") != camera_id]
    data["areas"] = [a for a in data.get("areas", []) if (a.get("camera_id") or "") != camera_id]
    _save(data)

def get_areas(camera_id):
    """Return list of areas for a camera. Each area: id, camera_id, name, bounds, rule."""
    data = _load()
    return [a for a in data.get("areas", []) if (a.get("camera_id") or "") == camera_id]

def save_area(area_id=None, camera_id="", name="", bounds=None, rule=""):
    """bounds: [x1, y1, x2, y2] normalized 0-1. rule: one of RULE_TYPES id."""
    data = _load()
    areas = list(data.get("areas", []))
    bounds = bounds or [0, 0, 1, 1]
    if len(bounds) != 4:
        bounds = [0, 0, 1, 1]
    rule = (rule or "").strip() or "ppe"
    if area_id:
        for i, a in enumerate(areas):
            if (a.get("id") or "") == area_id:
                areas[i] = {
                    "id": area_id,
                    "camera_id": (camera_id or a.get("camera_id", "")),
                    "name": (name or "").strip() or a.get("name", "Area"),
                    "bounds": bounds,
                    "rule": rule,
                }
                data["areas"] = areas
                _save(data)
                return area_id
    new_id = "area_" + uuid.uuid4().hex[:8]
    areas.append({
        "id": new_id,
        "camera_id": (camera_id or "").strip(),
        "name": (name or "").strip() or "Area",
        "bounds": bounds,
        "rule": rule,
    })
    data["areas"] = areas
    _save(data)
    return new_id

def delete_area(area_id):
    data = _load()
    data["areas"] = [a for a in data.get("areas", []) if (a.get("id") or "") != area_id]
    _save(data)

def get_policies_for_camera(camera_id):
    """Return camera dict with areas list (for detection app)."""
    cam = get_camera(camera_id)
    if not cam:
        return None
    cam["areas"] = get_areas(camera_id)
    return cam
