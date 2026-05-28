"""
Tracks which violations each user has reviewed by submitting remarks.
Once remarks are provided, the violation is removed from the review section.
"""
import os
import json
from datetime import datetime
from threading import Lock

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
READ_JSON = os.path.join(DATA_DIR, "notifications_read.json")
_lock = Lock()

def _ensure_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(READ_JSON):
        with open(READ_JSON, "w") as f:
            json.dump({}, f)

def _normalize_user_data(raw):
    """Convert legacy list format to dict format. Returns dict of basename -> { remarks, timestamp }."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {str(b): {"remarks": "", "timestamp": ""} for b in raw if b}
    return {}

def get_read_ids(username):
    """Return set of violation image basenames that this user has reviewed (submitted remarks)."""
    with _lock:
        _ensure_file()
        try:
            with open(READ_JSON, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return set()
    key = (username or "").strip().lower()
    raw = data.get(key, {})
    return set(_normalize_user_data(raw).keys())

def submit_remarks(username, image_basename, remarks):
    """Save remarks for a violation; once submitted, it is considered reviewed and removed from review list."""
    with _lock:
        _ensure_file()
        try:
            with open(READ_JSON, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {}
        key = (username or "").strip().lower()
        user_data = _normalize_user_data(data.get(key, {}))
        user_data[(image_basename or "").strip()] = {
            "remarks": (remarks or "").strip(),
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        # Keep last 500 entries
        keys = list(user_data.keys())
        if len(keys) > 500:
            for k in keys[:-500]:
                del user_data[k]
        data[key] = user_data
        with open(READ_JSON, "w") as f:
            json.dump(data, f, indent=2)

def get_remarks(username, image_basename):
    """Return remarks dict for a violation, or None if not reviewed."""
    with _lock:
        _ensure_file()
        try:
            with open(READ_JSON, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    key = (username or "").strip().lower()
    user_data = _normalize_user_data(data.get(key, {}))
    return user_data.get((image_basename or "").strip())

def mark_read(username, image_basename):
    """Mark a violation as reviewed (no remarks). Kept for backward compatibility."""
    submit_remarks(username, image_basename, "")

def mark_all_read(username, image_basenames):
    """Mark multiple violations as reviewed. Kept for backward compatibility."""
    for b in image_basenames or []:
        submit_remarks(username, b, "")
