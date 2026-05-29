"""
Flask server for PPE Violations Dashboard with login and role-based access.
Run: python server.py
Then open http://127.0.0.1:5000 — you will be redirected to login, then to dashboard.
"""
import io
import os
from flask import Flask, send_from_directory, send_file, jsonify, request, redirect, url_for, session

import violations_log as vlog
import users
import notifications_store as notif_store
import camera_policies as cam_policies

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=None)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "ppe-dashboard-secret-change-in-production")

def get_user_locations():
    """Return None (all) for admin, else list of locations for current user."""
    u = session.get("user")
    if not u:
        return None
    if (u.get("role") or "").lower() == "admin":
        return None
    return u.get("locations") or []

def require_admin():
    """Return (None, None) if admin, else (error_response, status_code)."""
    if not session.get("user"):
        return jsonify({"error": "Not logged in"}), 401
    if (session["user"].get("role") or "").lower() != "admin":
        return jsonify({"error": "Admin only"}), 403
    return None, None

@app.route("/")
def index():
    if session.get("user"):
        return redirect(url_for("dashboard_v2"))
    return redirect(url_for("login_page"))

@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "GET":
        if session.get("user"):
            return redirect(url_for("dashboard_v2"))
        return send_from_directory(BASE_DIR, "login.html")
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    if not username:
        return send_from_directory(BASE_DIR, "login.html"), 400
    user = users.get_user(username, password)
    if not user:
        return redirect(url_for("login_page") + "?error=invalid")
    session["user"] = user
    return redirect(url_for("dashboard_v2"))

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login_page"))

@app.route("/dashboard")
def dashboard():
    if not session.get("user"):
        return redirect(url_for("login_page"))
    return send_from_directory(BASE_DIR, "dashboard.html")



    
#dashboard-v2






@app.route("/dashboard-v2")
def dashboard_v2():
    if not session.get("user"):
        return redirect(url_for("login_page"))
    return send_from_directory(BASE_DIR, "ppe.html")








@app.route("/camera-setup")
def camera_setup_page():
    if not session.get("user"):
        return redirect(url_for("login_page"))
    err, code = require_admin()
    if err is not None:
        return redirect(url_for("dashboard_v2"))
    return send_from_directory(BASE_DIR, "camera_setup.html")

@app.route("/api/me")
def api_me():
    if not session.get("user"):
        return jsonify({"error": "Not logged in"}), 401
    return jsonify(session["user"])

@app.route("/api/kpis")
def api_kpis():
    if not session.get("user"):
        return jsonify({"error": "Not logged in"}), 401
    locations = get_user_locations()
    return jsonify(vlog.get_kpis(locations))

@app.route("/api/process-categories")
def api_process_categories():
    if not session.get("user"):
        return jsonify({"error": "Not logged in"}), 401
    return jsonify(vlog.get_process_category_choices())

@app.route("/api/violations")
def api_violations():
    if not session.get("user"):
        return jsonify({"error": "Not logged in"}), 401
    locations = get_user_locations()
    violations = vlog.get_all_violations(locations)
    username = session["user"].get("username")
    for v in violations:
        r = notif_store.get_remarks(username, v.get("image"))
        v["user_remarks"] = (r.get("remarks") or "").strip() if r else ""
    return jsonify(violations)

@app.route("/api/notifications")
def api_notifications():
    """Recent violations for user's locations with read/unread flag. Limit 50."""
    if not session.get("user"):
        return jsonify({"error": "Not logged in"}), 401
    locations = get_user_locations()
    violations = vlog.get_all_violations(locations)
    read_ids = notif_store.get_read_ids(session["user"].get("username"))
    limit = min(int(request.args.get("limit", 50)), 100)
    result = []
    for v in violations[:limit]:
        img = v.get("image") or ""
        result.append({
            **v,
            "read": img in read_ids,
        })
    unread_count = sum(1 for r in result if not r.get("read"))
    return jsonify({"violations": result, "unread_count": unread_count})

@app.route("/api/notifications/mark_read", methods=["POST"])
def api_mark_read():
    if not session.get("user"):
        return jsonify({"error": "Not logged in"}), 401
    username = session["user"].get("username")
    data = request.get_json(silent=True) or {}
    if data.get("images"):
        notif_store.mark_all_read(username, data["images"])
    elif data.get("image"):
        notif_store.mark_read(username, data["image"])
    else:
        return jsonify({"error": "Provide 'image' or 'images'"}), 400
    return jsonify({"ok": True})

@app.route("/api/notifications/submit_remarks", methods=["POST"])
def api_submit_remarks():
    """Submit remarks for a violation; once submitted, it is removed from the review list."""
    if not session.get("user"):
        return jsonify({"error": "Not logged in"}), 401
    username = session["user"].get("username")
    data = request.get_json(silent=True) or {}
    image_basename = (data.get("image") or "").strip()
    remarks = (data.get("remarks") or "").strip()
    process_category = (data.get("process_category") or "").strip()
    if not image_basename:
        return jsonify({"error": "Provide 'image' (violation image basename)"}), 400
    if not remarks:
        return jsonify({"error": "Remarks are required"}), 400
    if not process_category:
        return jsonify({"error": "Process safety category is required"}), 400
    vlog.update_violation_process_category(image_basename, process_category)
    notif_store.submit_remarks(username, image_basename, remarks)
    return jsonify({"ok": True})

@app.route("/api/camera-policies/rule-types")
def api_rule_types():
    err, code = require_admin()
    if err is not None:
        return err, code
    return jsonify(cam_policies.get_rule_types())

@app.route("/api/camera-policies/cameras", methods=["GET", "POST"])
def api_cameras():
    if request.method == "GET":
        if not session.get("user"):
            return jsonify({"error": "Not logged in"}), 401
        return jsonify(cam_policies.get_cameras())
    err, code = require_admin()
    if err is not None:
        return err, code
    data = request.get_json(silent=True) or {}
    cam_id = data.get("id") or None
    name = (data.get("name") or "").strip()
    source = data.get("source", 0)
    location_name = (data.get("location_name") or "").strip()
    probe = {"source": source}
    if isinstance(source, str) and source.strip() and not source.strip().lower().startswith(("http://", "https://", "rtsp://")):
        if source.strip().lower().endswith(cam_policies.VIDEO_EXTENSIONS) and not cam_policies.camera_has_video_source(probe):
            return jsonify({
                "error": f"Video file not found: {source}. Check the path and filename (e.g. sample_video3.mp4).",
            }), 400
    new_id = cam_policies.save_camera(cam_id, name=name, source=source, location_name=location_name)
    return jsonify({"ok": True, "id": new_id})

@app.route("/api/camera-policies/cameras/<camera_id>", methods=["DELETE"])
def api_delete_camera(camera_id):
    err, code = require_admin()
    if err is not None:
        return err, code
    cam_policies.delete_camera(camera_id or "")
    return jsonify({"ok": True})

@app.route("/api/camera-policies/cameras/<camera_id>/areas", methods=["GET"])
def api_camera_areas(camera_id):
    err, code = require_admin()
    if err is not None:
        return err, code
    return jsonify(cam_policies.get_areas(camera_id or ""))

@app.route("/api/camera-policies/areas", methods=["POST"])
def api_save_area():
    err, code = require_admin()
    if err is not None:
        return err, code
    data = request.get_json(silent=True) or {}
    area_id = data.get("id") or None
    camera_id = (data.get("camera_id") or "").strip()
    name = (data.get("name") or "").strip()
    bounds = data.get("bounds")
    if isinstance(bounds, list) and len(bounds) == 4:
        bounds = [float(b) for b in bounds]
    else:
        bounds = [0, 0, 1, 1]
    rule = (data.get("rule") or "ppe").strip()
    new_id = cam_policies.save_area(area_id, camera_id=camera_id, name=name, bounds=bounds, rule=rule)
    return jsonify({"ok": True, "id": new_id})

@app.route("/api/camera-policies/areas/<area_id>", methods=["DELETE"])
def api_delete_area(area_id):
    err, code = require_admin()
    if err is not None:
        return err, code
    cam_policies.delete_area(area_id or "")
    return jsonify({"ok": True})

@app.route("/camera-media/<camera_id>")
def serve_camera_media(camera_id):
    """Serve a camera's configured video file for All Cameras preview."""
    if not session.get("user"):
        return jsonify({"error": "Not logged in"}), 401
    cam = cam_policies.get_camera(camera_id or "")
    if not cam:
        return jsonify({"error": "Not found"}), 404
    source = str(cam.get("source") or "")
    if not source.lower().endswith((".mp4", ".avi", ".webm", ".mov", ".mkv")):
        return jsonify({"error": "No video source configured"}), 404
    filepath = os.path.abspath(source)
    if not os.path.isfile(filepath):
        return jsonify({"error": "Video file not found"}), 404
    if filepath.lower().endswith(".mp4"):
        mime = "video/mp4"
    elif filepath.lower().endswith(".webm"):
        mime = "video/webm"
    elif filepath.lower().endswith(".mov"):
        mime = "video/quicktime"
    elif filepath.lower().endswith(".mkv"):
        mime = "video/x-matroska"
    else:
        mime = "video/x-msvideo"
    return send_file(
        filepath,
        mimetype=mime,
        as_attachment=False,
        conditional=True,
        etag=False,
    )

@app.route("/violations/<path:filename>")
def serve_violation_file(filename):
    """Serve a violation image from DB or a clip from disk, with location access checks."""
    if not session.get("user"):
        return jsonify({"error": "Not logged in"}), 401
    locations = get_user_locations()
    violation = vlog.get_violation_by_file(filename)
    if not violation:
        return jsonify({"error": "Not found"}), 404
    if locations is not None and (violation.get("location") or "").strip() not in locations:
        return jsonify({"error": "Forbidden"}), 403

    if (violation.get("image") or "") == filename:
        blob, mime_type = vlog.get_violation_image_blob(filename)
        if blob:
            return send_file(
                io.BytesIO(blob),
                mimetype=mime_type or "image/jpeg",
                download_name=filename,
                as_attachment=False,
                etag=False,
            )
        filepath = os.path.join(vlog.VIOLATIONS_DIR, filename)
        if os.path.isfile(filepath):
            return send_file(
                filepath,
                mimetype=mime_type or "image/jpeg",
                download_name=filename,
                as_attachment=False,
                etag=False,
            )
        return jsonify({"error": "Image not found"}), 404

    filepath = os.path.join(vlog.VIOLATIONS_DIR, filename)
    if not os.path.isfile(filepath):
        return jsonify({"error": "Not found"}), 404

    # Serve video with Range support so browsers can stream/seek
    if filename.lower().endswith(".mp4"):
        return send_file(
            filepath,
            mimetype="video/mp4",
            as_attachment=False,
            conditional=True,
            etag=False,
        )
    if filename.lower().endswith(".avi"):
        return send_file(
            filepath,
            mimetype="video/x-msvideo",
            as_attachment=False,
            conditional=True,
            etag=False,
        )
    return jsonify({"error": "Unsupported file type"}), 400

if __name__ == "__main__":
    cam_policies.sync_demo_cameras_from_config()
    sync_result = vlog.sync_violations_from_disk()
    if sync_result.get("imported") or sync_result.get("updated"):
        print(
            f"Synced violations from disk: imported={sync_result.get('imported', 0)}, "
            f"clips_updated={sync_result.get('updated', 0)}"
        )
    filled = vlog.backfill_violation_metadata()
    if filled:
        print(f"Updated violation labels for {filled} record(s).")
    print("PPE Violations Dashboard: http://127.0.0.1:5000 (login required)")
    app.run(host="0.0.0.0", port=5000, debug=False)
