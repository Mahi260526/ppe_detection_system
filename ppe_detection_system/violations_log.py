"""
SQLite-backed violations store.
The dashboard reads violation metadata from the database and serves images from
image blobs stored in the same database.
"""
import json
import mimetypes
import os
import re
import sqlite3
from threading import Lock

BASE_DIR = os.path.dirname(__file__)
VIOLATIONS_DIR = os.path.join(BASE_DIR, "violations")
VIOLATIONS_JSON = os.path.join(VIOLATIONS_DIR, "violations_data.json")  # legacy migration source
VIOLATIONS_DB = os.path.join(VIOLATIONS_DIR, "violations.db")
_lock = Lock()

PROCESS_CATEGORY_UNSAFE_ACT = "unsafe_act"
PROCESS_CATEGORY_NEAR_MISS = "near_miss"
PROCESS_CATEGORY_MINOR_INJURY = "minor_injury"
PROCESS_CATEGORY_LTI = "lost_time_injury"
PROCESS_CATEGORY_FATALITY = "fatality"

PROCESS_CATEGORY_CHOICES = [
    PROCESS_CATEGORY_UNSAFE_ACT,
    PROCESS_CATEGORY_NEAR_MISS,
    PROCESS_CATEGORY_MINOR_INJURY,
    PROCESS_CATEGORY_LTI,
    PROCESS_CATEGORY_FATALITY,
]

PROCESS_CATEGORY_LABELS = {
    PROCESS_CATEGORY_UNSAFE_ACT: "Unsafe Act",
    PROCESS_CATEGORY_NEAR_MISS: "Near Miss",
    PROCESS_CATEGORY_MINOR_INJURY: "Minor Injury",
    PROCESS_CATEGORY_LTI: "Lost Time Injury",
    PROCESS_CATEGORY_FATALITY: "Fatality",
}


def _get_connection():
    conn = sqlite3.connect(VIOLATIONS_DB)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_process_category(value):
    category = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if category in PROCESS_CATEGORY_CHOICES:
        return category
    return PROCESS_CATEGORY_UNSAFE_ACT


def _init_db(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS violations (
            id TEXT PRIMARY KEY,
            datetime TEXT NOT NULL,
            image TEXT NOT NULL UNIQUE,
            no_helmet INTEGER NOT NULL DEFAULT 0,
            no_vest INTEGER NOT NULL DEFAULT 0,
            no_glasses INTEGER NOT NULL DEFAULT 0,
            no_mask INTEGER NOT NULL DEFAULT 0,
            location TEXT NOT NULL DEFAULT '',
            clip TEXT NOT NULL DEFAULT '',
            process_category TEXT NOT NULL DEFAULT 'unsafe_act',
            image_blob BLOB,
            image_mime TEXT NOT NULL DEFAULT 'image/jpeg',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    _ensure_columns(conn)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_violations_datetime ON violations(datetime DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_violations_location ON violations(location)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_violations_process_category ON violations(process_category)")
    conn.commit()


def _ensure_columns(conn):
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(violations)").fetchall()
    }
    if "process_category" not in columns:
        conn.execute(
            "ALTER TABLE violations ADD COLUMN process_category TEXT NOT NULL DEFAULT 'unsafe_act'"
        )
        conn.commit()


def _ensure_db():
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    with _get_connection() as conn:
        _init_db(conn)
        _migrate_from_legacy_json(conn)


def _read_legacy_json():
    if not os.path.exists(VIOLATIONS_JSON) or os.path.getsize(VIOLATIONS_JSON) == 0:
        return []
    try:
        with open(VIOLATIONS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    items = data.get("violations", [])
    return items if isinstance(items, list) else []


def _migrate_from_legacy_json(conn):
    row = conn.execute("SELECT COUNT(*) AS count FROM violations").fetchone()
    if row and row["count"] > 0:
        return
    legacy_rows = _read_legacy_json()
    if not legacy_rows:
        return
    for item in legacy_rows:
        image_name = (item.get("image") or "").strip()
        if not image_name:
            continue
        image_path = os.path.join(VIOLATIONS_DIR, image_name)
        image_blob, image_mime = _read_image_payload(image_path)
        conn.execute(
            """
            INSERT OR IGNORE INTO violations (
                id, datetime, image, no_helmet, no_vest, no_glasses, no_mask,
                location, clip, process_category, image_blob, image_mime
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (item.get("id") or "").strip() or None,
                (item.get("datetime") or "").strip(),
                image_name,
                int(bool(item.get("no_helmet"))),
                int(bool(item.get("no_vest"))),
                int(bool(item.get("no_glasses"))),
                int(bool(item.get("no_mask"))),
                (item.get("location") or "").strip(),
                (item.get("clip") or "").strip(),
                normalize_process_category(item.get("process_category")),
                image_blob,
                image_mime,
            ),
        )
    conn.commit()


def _read_image_payload(image_path):
    if not image_path or not os.path.isfile(image_path):
        return None, "image/jpeg"
    try:
        with open(image_path, "rb") as f:
            blob = f.read()
    except OSError:
        return None, "image/jpeg"
    mime, _ = mimetypes.guess_type(image_path)
    return blob, mime or "image/jpeg"


def _next_id(conn):
    row = conn.execute(
        """
        SELECT id
        FROM violations
        WHERE id GLOB 'VID[0-9]*'
        ORDER BY CAST(SUBSTR(id, 4) AS INTEGER) DESC
        LIMIT 1
        """
    ).fetchone()
    if not row or not row["id"]:
        return "VID001"
    match = re.match(r"VID(\d+)", row["id"].strip().upper())
    next_num = (int(match.group(1)) + 1) if match else 1
    return f"VID{next_num:03d}"


def _row_to_violation(row):
    process_category = normalize_process_category(row["process_category"] if "process_category" in row.keys() else None)
    return {
        "id": row["id"],
        "datetime": row["datetime"],
        "image": row["image"],
        "no_helmet": bool(row["no_helmet"]),
        "no_vest": bool(row["no_vest"]),
        "no_glasses": bool(row["no_glasses"]),
        "no_mask": bool(row["no_mask"]),
        "location": row["location"] or "",
        "clip": row["clip"] or "",
        "process_category": process_category,
        "process_category_label": PROCESS_CATEGORY_LABELS.get(process_category, "Unsafe Act"),
    }


def add_violation(
    datetime_str,
    image_basename,
    no_helmet=False,
    no_vest=False,
    no_glasses=False,
    no_mask=False,
    location=None,
    clip=None,
    image_path=None,
    process_category=PROCESS_CATEGORY_UNSAFE_ACT,
):
    """Insert one violation row and store the image bytes in the database."""
    with _lock:
        _ensure_db()
        image_blob, image_mime = _read_image_payload(image_path or os.path.join(VIOLATIONS_DIR, image_basename))
        with _get_connection() as conn:
            _init_db(conn)
            vid = _next_id(conn)
            conn.execute(
                """
                INSERT INTO violations (
                    id, datetime, image, no_helmet, no_vest, no_glasses, no_mask,
                    location, clip, process_category, image_blob, image_mime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vid,
                    datetime_str,
                    image_basename,
                    int(bool(no_helmet)),
                    int(bool(no_vest)),
                    int(bool(no_glasses)),
                    int(bool(no_mask)),
                    (location or "").strip(),
                    (clip or "").strip(),
                    normalize_process_category(process_category),
                    image_blob,
                    image_mime,
                ),
            )
            conn.commit()


def update_violation_clip(image_basename, clip_basename):
    """Set the clip filename for the violation with the given image basename."""
    with _lock:
        _ensure_db()
        with _get_connection() as conn:
            conn.execute(
                "UPDATE violations SET clip = ? WHERE image = ?",
                ((clip_basename or "").strip(), (image_basename or "").strip()),
            )
            conn.commit()


def update_violation_process_category(image_basename, process_category):
    """Update the process safety category for a violation."""
    with _lock:
        _ensure_db()
        with _get_connection() as conn:
            conn.execute(
                "UPDATE violations SET process_category = ? WHERE image = ?",
                (
                    normalize_process_category(process_category),
                    (image_basename or "").strip(),
                ),
            )
            conn.commit()


def _get_all_raw():
    """Return list of violation records (newest first), no filter."""
    with _lock:
        _ensure_db()
        with _get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, datetime, image, no_helmet, no_vest, no_glasses, no_mask, location, clip, process_category
                FROM violations
                ORDER BY datetime DESC, id DESC
                """
            ).fetchall()
    return [_row_to_violation(row) for row in rows]


def get_all_violations(locations=None):
    """Return list of violation records (newest first)."""
    items = _get_all_raw()
    if not locations:
        return items
    allowed = {str(location).strip() for location in locations if location}
    if not allowed:
        return items
    return [item for item in items if (item.get("location") or "").strip() in allowed]


def get_kpis(locations=None):
    """Return counts: total, no_helmet, no_vest, no_glasses, no_mask."""
    items = get_all_violations(locations)
    process_triangle = {
        category: 0 for category in PROCESS_CATEGORY_CHOICES
    }
    for item in items:
        process_triangle[normalize_process_category(item.get("process_category"))] += 1
    return {
        "total": len(items),
        "no_helmet": sum(1 for item in items if item.get("no_helmet")),
        "no_vest": sum(1 for item in items if item.get("no_vest")),
        "no_glasses": sum(1 for item in items if item.get("no_glasses")),
        "no_mask": sum(1 for item in items if item.get("no_mask")),
        "process_triangle": process_triangle,
    }


def get_violation_by_file(filename):
    """Return violation metadata for an image filename or clip filename."""
    with _lock:
        _ensure_db()
        with _get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, datetime, image, no_helmet, no_vest, no_glasses, no_mask, location, clip, process_category
                FROM violations
                WHERE image = ? OR clip = ?
                LIMIT 1
                """,
                ((filename or "").strip(), (filename or "").strip()),
            ).fetchone()
    return _row_to_violation(row) if row else None


def get_violation_image_blob(image_basename):
    """Return (blob, mime_type) for a violation image stored in the DB."""
    with _lock:
        _ensure_db()
        with _get_connection() as conn:
            row = conn.execute(
                "SELECT image_blob, image_mime FROM violations WHERE image = ? LIMIT 1",
                ((image_basename or "").strip(),),
            ).fetchone()
    if not row:
        return None, None
    return row["image_blob"], row["image_mime"] or "image/jpeg"


def get_process_category_choices():
    """Return process category ids and labels for the UI."""
    return [
        {"id": category, "label": PROCESS_CATEGORY_LABELS[category]}
        for category in PROCESS_CATEGORY_CHOICES
    ]
