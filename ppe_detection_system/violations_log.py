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
_metadata_backfill_complete = False

IMAGE_FILENAME_RE = re.compile(
    r"^violation_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})(?:_(\d{3}))?\.jpg$",
    re.IGNORECASE,
)

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
    conn = sqlite3.connect(VIOLATIONS_DB, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
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
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(violations)").fetchall()
    }
    if "person_detected" not in columns:
        conn.execute(
            "ALTER TABLE violations ADD COLUMN person_detected INTEGER NOT NULL DEFAULT 0"
        )
        conn.commit()
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(violations)").fetchall()
    }
    if "violation_summary" not in columns:
        conn.execute(
            "ALTER TABLE violations ADD COLUMN violation_summary TEXT NOT NULL DEFAULT ''"
        )
        conn.commit()


PERSON_DETECTED_SUMMARY = "Violation - person is detected in the restricted area"


def build_violation_summary(
    no_helmet=False,
    no_vest=False,
    no_glasses=False,
    no_mask=False,
    person_detected=False,
):
    if person_detected:
        return PERSON_DETECTED_SUMMARY
    tags = []
    if no_helmet:
        tags.append("No Helmet")
    if no_vest:
        tags.append("No Vest")
    if no_glasses:
        tags.append("No Glasses")
    if no_mask:
        tags.append("No Mask")
    return " | ".join(tags)


def _snapshot_has_ppe_violation_title(frame):
    """True when the snapshot header is the wide red 'PPE VIOLATION' banner."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False

    h, w = frame.shape[:2]
    title = frame[int(h * 0.04) : int(h * 0.10), 0:int(w)]
    red = cv2.inRange(title, np.array([0, 0, 150]), np.array([100, 100, 255]))
    cols = np.where(red.any(axis=0))[0]
    if cols.size == 0:
        return False
    title_span = (cols[-1] - cols[0] + 1) / float(max(w, 1))
    return title_span > 0.25


def _location_uses_ppe_area_rule(location_name):
    """True when this location has no person-in-zone area rules (PPE checks apply)."""
    try:
        import camera_policies as cam_policies
    except ImportError:
        return True

    areas = cam_policies.get_areas_for_location(location_name)
    if not areas:
        return True
    rules = [(a.get("rule") or "ppe").strip().lower() for a in areas]
    return "person_detected" not in rules


def _snapshot_is_person_restricted_area_overlay(frame):
    """
    True only for snapshots saved with the person-in-restricted-area overlay
    (narrow red 'VIOLATION' title + long orange subtitle). PPE snapshots use a
    wide 'PPE VIOLATION' title and must not match this check.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False

    if _snapshot_has_ppe_violation_title(frame):
        return False

    h, w = frame.shape[:2]
    title = frame[int(h * 0.045) : int(h * 0.078), 0:int(w)]
    red = cv2.inRange(title, np.array([0, 0, 150]), np.array([100, 100, 255]))
    if int(red.sum()) < 1500:
        return False
    cols = np.where(red.any(axis=0))[0]
    if cols.size == 0:
        return False
    title_span = (cols[-1] - cols[0] + 1) / float(max(w, 1))
    if title_span > 0.15:
        return False
    # Person subtitle sits above PPE tag line (y≈88 vs y≈95 on 1080p frames).
    person_line = frame[int(h * 0.078) : int(h * 0.092), int(w * 0.01) : int(w * 0.75)]
    orange = cv2.inRange(person_line, np.array([0, 130, 190]), np.array([60, 210, 255]))
    return int(orange.sum()) > 150000


def infer_violation_flags_from_snapshot(image_path, location=None):
    """Read violation type from snapshot overlay text layout or re-run detection."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return {}

    frame = cv2.imread(image_path)
    if frame is None:
        return {}

    h, w = frame.shape[:2]

    loc = (location or "").strip()
    person_overlay = _snapshot_is_person_restricted_area_overlay(frame)
    if person_overlay and not (loc and _location_uses_ppe_area_rule(loc)):
        return {
            "no_helmet": False,
            "no_vest": False,
            "no_glasses": False,
            "no_mask": False,
            "person_detected": True,
            "violation_summary": PERSON_DETECTED_SUMMARY,
        }

    try:
        from ultralytics import YOLO
        from app import get_violation_types_from_results

        model = YOLO("models/best.pt")
        results = model(frame, conf=0.25, verbose=False)
        nh, nv, ng, nm, _ = get_violation_types_from_results(results, frame=frame)
        if any([nh, nv, ng, nm]):
            return {
                "no_helmet": nh,
                "no_vest": nv,
                "no_glasses": ng,
                "no_mask": nm,
                "person_detected": False,
                "violation_summary": build_violation_summary(nh, nv, ng, nm, False),
            }
    except Exception:
        pass

    title = frame[int(h * 0.04) : int(h * 0.09), int(w * 0.01) : int(w * 0.45)]
    red_mask = cv2.inRange(title, np.array([0, 0, 150]), np.array([100, 100, 255]))
    if int(red_mask.sum()) > 2000:
        return {
            "no_helmet": False,
            "no_vest": False,
            "no_glasses": False,
            "no_mask": False,
            "person_detected": False,
            "violation_summary": "PPE violation (imported)",
        }
    return {}


def backfill_violation_metadata():
    """Fill violation type labels for imported or legacy rows."""
    generic_summaries = {"", "Violation", "Violation (imported)", "PPE violation (imported)"}
    with _lock:
        _ensure_db()
        updated = 0
        with _get_connection() as conn:
            rows = conn.execute(
                """
                SELECT image, no_helmet, no_vest, no_glasses, no_mask, person_detected, violation_summary, location
                FROM violations
                """
            ).fetchall()
            for row in rows:
                summary = (row["violation_summary"] or "").strip()
                nh = bool(row["no_helmet"])
                nv = bool(row["no_vest"])
                ng = bool(row["no_glasses"])
                nm = bool(row["no_mask"])
                pd = bool(row["person_detected"])
                image_path = os.path.join(VIOLATIONS_DIR, row["image"])
                row_location = (row["location"] if "location" in row.keys() else "") or ""

                # Re-check person_detected rows (old backfill often mislabeled PPE snapshots).
                if pd or "restricted area" in summary.lower():
                    inferred = infer_violation_flags_from_snapshot(image_path, location=row_location)
                    if inferred:
                        nh = bool(inferred.get("no_helmet"))
                        nv = bool(inferred.get("no_vest"))
                        ng = bool(inferred.get("no_glasses"))
                        nm = bool(inferred.get("no_mask"))
                        pd = bool(inferred.get("person_detected"))
                        summary = inferred.get("violation_summary") or build_violation_summary(
                            nh, nv, ng, nm, pd
                        )
                    conn.execute(
                        """
                        UPDATE violations
                        SET no_helmet = ?, no_vest = ?, no_glasses = ?, no_mask = ?,
                            person_detected = ?, violation_summary = ?
                        WHERE image = ?
                        """,
                        (int(nh), int(nv), int(ng), int(nm), int(pd), summary, row["image"]),
                    )
                    updated += 1
                    continue

                if summary not in generic_summaries and (nh or nv or ng or nm or pd):
                    continue
                if summary not in generic_summaries:
                    continue

                if (nh or nv or ng or nm or pd) and summary in generic_summaries:
                    summary = build_violation_summary(nh, nv, ng, nm, pd)
                    conn.execute(
                        "UPDATE violations SET violation_summary = ? WHERE image = ?",
                        (summary, row["image"]),
                    )
                    updated += 1
                    continue

                inferred = infer_violation_flags_from_snapshot(image_path, location=row_location)
                if inferred:
                    nh = bool(inferred.get("no_helmet"))
                    nv = bool(inferred.get("no_vest"))
                    ng = bool(inferred.get("no_glasses"))
                    nm = bool(inferred.get("no_mask"))
                    pd = bool(inferred.get("person_detected"))
                    summary = inferred.get("violation_summary") or build_violation_summary(nh, nv, ng, nm, pd)
                else:
                    summary = "Violation (imported)"

                conn.execute(
                    """
                    UPDATE violations
                    SET no_helmet = ?, no_vest = ?, no_glasses = ?, no_mask = ?,
                        person_detected = ?, violation_summary = ?
                    WHERE image = ?
                    """,
                    (int(nh), int(nv), int(ng), int(nm), int(pd), summary, row["image"]),
                )
                updated += 1
            conn.commit()
        return updated


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


def _datetime_from_image_match(match):
    date = match.group(1)
    h, mi, s = match.group(2), match.group(3), match.group(4)
    return f"{date} {h}:{mi}:{s}"


def _clip_for_image(image_basename):
    base, _ = os.path.splitext(image_basename)
    clip_name = f"{base}_clip.mp4"
    clip_path = os.path.join(VIOLATIONS_DIR, clip_name)
    return clip_name if os.path.isfile(clip_path) else ""


def sync_violations_from_disk():
    """Import violation images/clips saved on disk into the SQLite database."""
    with _lock:
        _ensure_db()
        imported = 0
        updated = 0
        with _get_connection() as conn:
            known = {
                row["image"]: (row["clip"] or "")
                for row in conn.execute("SELECT image, clip FROM violations").fetchall()
            }
            for name in sorted(os.listdir(VIOLATIONS_DIR)):
                match = IMAGE_FILENAME_RE.match(name)
                if not match:
                    continue
                clip = _clip_for_image(name)
                if name in known:
                    if clip and not known[name]:
                        conn.execute(
                            "UPDATE violations SET clip = ? WHERE image = ?",
                            (clip, name),
                        )
                        updated += 1
                    continue
                dt = _datetime_from_image_match(match)
                image_path = os.path.join(VIOLATIONS_DIR, name)
                image_blob, image_mime = _read_image_payload(image_path)
                vid = _next_id(conn)
                inferred = infer_violation_flags_from_snapshot(image_path)
                nh = bool(inferred.get("no_helmet")) if inferred else False
                nv = bool(inferred.get("no_vest")) if inferred else False
                ng = bool(inferred.get("no_glasses")) if inferred else False
                nm = bool(inferred.get("no_mask")) if inferred else False
                pd = bool(inferred.get("person_detected")) if inferred else False
                summary = (
                    (inferred.get("violation_summary") if inferred else "")
                    or build_violation_summary(nh, nv, ng, nm, pd)
                    or "Violation (imported)"
                )
                conn.execute(
                    """
                    INSERT INTO violations (
                        id, datetime, image, no_helmet, no_vest, no_glasses, no_mask,
                        person_detected, location, clip, process_category, image_blob, image_mime,
                        violation_summary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        vid,
                        dt,
                        name,
                        int(nh),
                        int(nv),
                        int(ng),
                        int(nm),
                        int(pd),
                        "",
                        clip,
                        PROCESS_CATEGORY_UNSAFE_ACT,
                        image_blob,
                        image_mime,
                        summary,
                    ),
                )
                imported += 1
            conn.commit()
        if imported:
            backfill_violation_metadata()
        return {"imported": imported, "updated": updated}


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
        "person_detected": bool(row["person_detected"]) if "person_detected" in row.keys() else False,
        "violation_summary": (row["violation_summary"] or "").strip() if "violation_summary" in row.keys() else "",
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
    person_detected=False,
    location=None,
    clip=None,
    image_path=None,
    process_category=PROCESS_CATEGORY_UNSAFE_ACT,
    violation_summary=None,
):
    """Insert one violation row and store the image bytes in the database."""
    summary = violation_summary
    if summary is None:
        summary = build_violation_summary(
            no_helmet, no_vest, no_glasses, no_mask, person_detected
        )
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
                    person_detected, location, clip, process_category, image_blob, image_mime,
                    violation_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vid,
                    datetime_str,
                    image_basename,
                    int(bool(no_helmet)),
                    int(bool(no_vest)),
                    int(bool(no_glasses)),
                    int(bool(no_mask)),
                    int(bool(person_detected)),
                    (location or "").strip(),
                    (clip or "").strip(),
                    normalize_process_category(process_category),
                    image_blob,
                    image_mime,
                    (summary or "").strip(),
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
                SELECT id, datetime, image, no_helmet, no_vest, no_glasses, no_mask,
                       person_detected, location, clip, process_category, violation_summary
                FROM violations
                ORDER BY datetime DESC, id DESC
                """
            ).fetchall()
    return [_row_to_violation(row) for row in rows]


def get_all_violations(locations=None):
    """Return list of violation records (newest first)."""
    global _metadata_backfill_complete
    sync_result = sync_violations_from_disk()
    if not _metadata_backfill_complete or sync_result.get("imported"):
        backfill_violation_metadata()
        _metadata_backfill_complete = True
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
                SELECT id, datetime, image, no_helmet, no_vest, no_glasses, no_mask,
                       person_detected, location, clip, process_category, violation_summary
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
