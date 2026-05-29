"""
Register demo cameras and process all configured videos in parallel.

Run (with server.py already running in another terminal):
    python run_demo_cameras.py

Each video runs in its own process. Violations are saved under that camera's
location_name so they appear under the correct camera in Dashboard -> All Cameras.
"""
import multiprocessing as mp
import os

import camera_policies as cam_policies
from config import DEMO_CAMERAS


def sync_demo_cameras():
    """Ensure camera_policies.json has one entry per DEMO_CAMERA (matched by location)."""
    cam_policies.sync_demo_cameras_from_config()


def _process_camera_worker(cam):
    """Process one camera video in a separate process."""
    from ultralytics import YOLO
    from app import main

    src = cam.get("source")
    loc = (cam.get("location_name") or "").strip()
    name = (cam.get("name") or "Camera").strip()
    if not src:
        print(f"[{name}] SKIP: no source path configured.")
        return False
    if not os.path.isfile(str(src)):
        print(f"[{name}] SKIP: video not found at {src}")
        return False

    print(f"[{name}] Starting | Location: {loc}")
    model = YOLO("models/best.pt")
    ok = main(source=src, location_name=loc, show_gui=False, model=model)
    print(f"[{name}] Finished.")
    return ok is not False


if __name__ == "__main__":
    mp.freeze_support()

    if not DEMO_CAMERAS:
        print("No DEMO_CAMERAS configured in config.py")
        raise SystemExit(1)

    sync_demo_cameras()
    print(f"Registered {len(DEMO_CAMERAS)} camera(s) for the dashboard.\n")

    valid_cams = []
    for cam in DEMO_CAMERAS:
        src = cam.get("source")
        name = (cam.get("name") or "Camera").strip()
        if not src:
            print(f"SKIP {name}: no source path configured.")
            continue
        if not os.path.isfile(str(src)):
            print(f"SKIP {name}: video not found at {src}")
            continue
        valid_cams.append(cam)

    if not valid_cams:
        print("No valid video files found. Check paths in config.py -> DEMO_CAMERAS.")
        raise SystemExit(1)

    print(f"Starting {len(valid_cams)} video(s) in parallel...\n")
    with mp.Pool(processes=len(valid_cams)) as pool:
        results = pool.map(_process_camera_worker, valid_cams)

    processed = sum(1 for ok in results if ok)
    print(f"\nDone. Processed {processed}/{len(valid_cams)} camera video(s) in parallel.")
    print("Open http://127.0.0.1:5000/dashboard-v2 -> All Cameras -> Refresh.")
