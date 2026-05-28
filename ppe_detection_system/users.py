"""
User store for login and role-based access. Users have username, password, role, and assigned locations.
"""
import os
import json
import hashlib
from threading import Lock

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_JSON = os.path.join(BASE_DIR, "data", "users.json")
_lock = Lock()

def _hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def _ensure_file():
    os.makedirs(os.path.dirname(USERS_JSON), exist_ok=True)
    if not os.path.exists(USERS_JSON):
        default_users = {
            "users": [
                {
                    "username": "admin",
                    "password_hash": _hash_password("admin"),
                    "role": "admin",
                    "locations": [],  # admin sees all
                },
                {
                    "username": "supervisor",
                    "password_hash": _hash_password("supervisor"),
                    "role": "supervisor",
                    "locations": ["Main Gate", "Site A"],
                },
                {
                    "username": "viewer",
                    "password_hash": _hash_password("viewer"),
                    "role": "viewer",
                    "locations": ["Main Gate"],
                },
            ]
        }
        with open(USERS_JSON, "w") as f:
            json.dump(default_users, f, indent=2)

def get_user(username, password):
    """Return user dict (without password_hash) if credentials are valid, else None."""
    with _lock:
        _ensure_file()
        with open(USERS_JSON, "r") as f:
            data = json.load(f)
    for u in data.get("users", []):
        if (u.get("username") or "").strip().lower() == (username or "").strip().lower():
            if u.get("password_hash") == _hash_password(password):
                return {
                    "username": u.get("username"),
                    "role": u.get("role") or "viewer",
                    "locations": list(u.get("locations") or []),
                }
            return None
    return None

def get_user_locations(username):
    """Return list of locations assigned to user, or None if admin (all locations)."""
    with _lock:
        _ensure_file()
        with open(USERS_JSON, "r") as f:
            data = json.load(f)
    for u in data.get("users", []):
        if (u.get("username") or "").strip().lower() == (username or "").strip().lower():
            if (u.get("role") or "").lower() == "admin":
                return None  # all locations
            return list(u.get("locations") or [])
    return []
