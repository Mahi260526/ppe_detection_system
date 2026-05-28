import cv2
import sys
import torch
from types import ModuleType # used to dynamically create fake Python modules

# Allow loading checkpoints that reference old 'ultralytics.yolo' module path
# (e.g. Construction-PPE-Detection ppe.pt). Map to current ultralytics.nn.
def _register_ultralytics_yolo_compat():
    if "ultralytics.yolo" in sys.modules:
        return
    try:
        from ultralytics import nn
        import ultralytics.utils as ul_utils
        yolo = ModuleType("ultralytics.yolo")
        yolo.__path__ = []  # so Python treats it as a package (e.g. for ultralytics.yolo.utils)
        yolo.v8 = ModuleType("ultralytics.yolo.v8")
        yolo.v8.detect = ModuleType("ultralytics.yolo.v8.detect")
        yolo.v8.detect.DetectionModel = nn.tasks.DetectionModel
        yolo.v8.modules = ModuleType("ultralytics.yolo.v8.modules")
        yolo.v8.modules.block = nn.modules.block
        yolo.v8.modules.conv = nn.modules.conv
        yolo.v8.modules.head = nn.modules.head
        for name in dir(nn.modules):
            if not name.startswith("_"):
                setattr(yolo.v8.modules, name, getattr(nn.modules, name))
        yolo.utils = ul_utils
        sys.modules["ultralytics.yolo"] = yolo
        sys.modules["ultralytics.yolo.v8"] = yolo.v8
        sys.modules["ultralytics.yolo.v8.detect"] = yolo.v8.detect
        sys.modules["ultralytics.yolo.v8.modules"] = yolo.v8.modules
        sys.modules["ultralytics.yolo.v8.modules.block"] = nn.modules.block
        sys.modules["ultralytics.yolo.v8.modules.conv"] = nn.modules.conv
        sys.modules["ultralytics.yolo.v8.modules.head"] = nn.modules.head
        sys.modules["ultralytics.yolo.utils"] = ul_utils
    except Exception:
        pass

_register_ultralytics_yolo_compat()

# Fix for PyTorch 2.6+ weights_only security feature
# Patch ultralytics torch_safe_load to use weights_only=False for trusted model files
def patch_ultralytics_loader():
    """Patch ultralytics to work with PyTorch 2.6+ security changes"""
    try:
        from ultralytics.nn import tasks
        
        # Store original function
        original_torch_safe_load = tasks.torch_safe_load
        
        # Create patched version
        def patched_torch_safe_load(file):
            """Patched version that uses weights_only=False for trusted YOLOv8 models"""
            # Use weights_only=False for trusted model files from GitHub
            # This is safe since we're loading from a known trusted source
            return torch.load(file, map_location='cpu', weights_only=False), file
        
        # Apply patch
        tasks.torch_safe_load = patched_torch_safe_load
    except (ImportError, AttributeError):
        # If patching fails, continue - newer ultralytics versions may handle this
        pass

# Apply patch before importing YOLO
patch_ultralytics_loader()

from ultralytics import YOLO
from config import (
    VIDEO_SOURCE,
    CONFIDENCE_THRESHOLD,
    ENABLE_HELMET_COLOR_HEURISTIC,
    HARDHAT_MIN_CONFIDENCE,
    HARDHAT_HEAD_REGION_RATIO,
    HARDHAT_HEAD_OVERLAP_THRESHOLD,
    HARDHAT_SKIN_RATIO_THRESHOLD,
    HARDHAT_MAX_MEAN_SATURATION,
    ENABLE_EMAIL_ALERTS,
    EMAIL_COOLDOWN_SEC,
    SAME_PERSON_THRESHOLD,
    PERSON_LEFT_FRAME_SEC,
    REPORTED_PERSON_IOU_THRESHOLD,
    REPORTED_PERSON_DUPLICATE_IOU_THRESHOLD,
    LOCATION_NAME,
    CLIP_FPS,
    CLIP_PRE_SEC,
    CLIP_POST_SEC,
)
import os
import time
import queue
import threading
import numpy as np
from datetime import datetime
import requests
from urllib.parse import urlparse #Used for parsing camera URLs.

def check_opencv_gui():
    """Check if OpenCV GUI (cv2.imshow) is available"""
    try:
        # Try to create a test window
        test_img = cv2.imread("test.jpg") if os.path.exists("test.jpg") else None
        if test_img is None:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.imshow("test", test_img)
        cv2.waitKey(1)
        cv2.destroyWindow("test")
        return True
    except cv2.error:
        return False

def save_frame(frame, frame_count, output_dir="output_frames"):
    """Save frame to file as fallback when GUI is not available"""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
    cv2.imwrite(filename, frame)
    return filename

VIOLATIONS_DIR = "violations"

# Map model class names to dashboard categories (case-insensitive substring match)
# Supports: Hansung-Cho/yolov8-ppe-detection (No-Hardhat, No-Safety Vest, No-Mask)
# and other PPE models with no helmet/vest/glasses or mask.
VIOLATION_CLASS_MAP = {
    "no_helmet": [
        "no helmet", "no_helmet", "without helmet", "missing helmet", "helmet off",
        "no-hardhat", "no hardhat", "no_hardhat", "without hardhat",
    ],
    "no_vest": [
        "no vest", "no_vest", "without vest", "missing vest", "vest off", "no safety vest",
        "no-safety vest", "no safety vest", "no_safety_vest", "without safety vest",
    ],
    "no_glasses": [
        "no glasses", "no_glasses", "noglasses", "no glass", "no_glass",
        "without glasses", "without_glasses", "withoutglass", "withoutglasss",
        "missing glasses", "missing_glasses",
        "no goggles", "no_goggles", "without goggles", "goggles off",
        "no eye protection", "no_eye_protection", "without eye protection",
        "no eye", "no_eyes", "no eyewear", "without eyewear",
    ],
    "no_mask": [
        "no mask", "no_mask", "no-mask", "without mask", "missing mask",
        "mask off", "no face mask", "without face mask",
    ],
}

# Class name keywords that indicate a human/person detection (required in frame before raising violations)
PERSON_CLASS_KEYWORDS = ["person", "human", "worker", "people", "man", "woman"]
POSITIVE_HARDHAT_KEYWORDS = ["hardhat", "hard hat", "helmet"]
NEGATIVE_PPE_KEYWORDS = ["no ", "no-", "no_", "without", "missing"]


def _normalize_label(label):
    return (label or "").strip().lower()


def _is_negative_ppe_label(label):
    label = _normalize_label(label)
    return any(token in label for token in NEGATIVE_PPE_KEYWORDS)


def _is_positive_hardhat_label(label):
    label = _normalize_label(label)
    return any(token in label for token in POSITIVE_HARDHAT_KEYWORDS) and not _is_negative_ppe_label(label)


def extract_detections(results):
    """Return normalized detections with labels, confidences, and boxes."""
    if len(results[0].boxes) == 0:
        return []
    names = results[0].names or {}
    try:
        xyxy = results[0].boxes.xyxy.cpu().numpy() #boundary boxes and .cpu as yolo gives in gpu
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int) #get the class ids (Hardhat,NO-Mask,Person)
        confs = results[0].boxes.conf.cpu().numpy() #confidence scores
    except Exception:
        return []
    detections = []
    for box, cid, conf in zip(xyxy, cls_ids, confs):
        detections.append({
            "label": _normalize_label(names.get(int(cid))),
            "confidence": float(conf),
            "box": tuple(float(v) for v in box),
        })
    return detections


def _intersection_area(box_a, box_b): #detect overlap between two boxes
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float(ix2 - ix1) * float(iy2 - iy1)


def _box_area(box):  #calculate the box area
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def _head_region_for_person(person_box):
    """Approximate the head region as the upper middle portion of a person box."""
    x1, y1, x2, y2 = person_box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    inset_x = width * 0.15
    head_height = height * HARDHAT_HEAD_REGION_RATIO
    return (
        x1 + inset_x,
        y1,
        x2 - inset_x,
        min(y2, y1 + head_height),
    )


def _hardhat_looks_like_skin(frame, hardhat_box):
    """Return True when a hardhat ROI looks more like bare skin than PPE."""
    if not ENABLE_HELMET_COLOR_HEURISTIC:
        return False
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = hardhat_box
    x1 = max(0, min(int(x1), fw - 1))
    y1 = max(0, min(int(y1), fh - 1))
    x2 = max(0, min(int(x2), fw))
    y2 = max(0, min(int(y2), fh))
    if x2 <= x1 or y2 <= y1:
        return False
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    skin_mask = (
        (y >= 80) & (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)
    )
    skin_ratio = float(np.mean(skin_mask))
    mean_saturation = float(np.mean(hsv[:, :, 1])) / 255.0
    return (
        skin_ratio >= HARDHAT_SKIN_RATIO_THRESHOLD
        and mean_saturation <= HARDHAT_MAX_MEAN_SATURATION
    )


def _person_has_suspicious_hardhat(frame, person_detection, hardhat_detections):
    """Return True when all matched hardhat detections for a person look like bare head."""
    person_box = person_detection["box"]
    head_box = _head_region_for_person(person_box)
    matched_hardhats = []
    suspicious_matches = 0
    for hardhat in hardhat_detections:
        if hardhat["confidence"] < HARDHAT_MIN_CONFIDENCE:
            continue
        overlap = _intersection_area(hardhat["box"], head_box)
        hardhat_area = _box_area(hardhat["box"])
        if hardhat_area <= 0:
            continue
        if (overlap / hardhat_area) < HARDHAT_HEAD_OVERLAP_THRESHOLD:
            continue
        matched_hardhats.append(hardhat)
        if _hardhat_looks_like_skin(frame, hardhat["box"]):
            suspicious_matches += 1
    return bool(matched_hardhats) and suspicious_matches == len(matched_hardhats)


def draw_runtime_notes(frame, notes):
    """Overlay short runtime notes below the main PPE violation heading."""
    if not notes:
        return frame
    y = 125
    for note in notes:
        cv2.putText(frame, note, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        y += 24
    return frame

def frame_has_person(results):
    """Return True only if at least one detection is a person/human. Ignore frames with only machinery, cones, etc."""
    for detection in extract_detections(results):
        if any(kw in detection["label"] for kw in PERSON_CLASS_KEYWORDS):
            return True
    return False

def get_violation_types_from_results(results, frame=None):
    """Extract no_helmet, no_vest, no_glasses, no_mask from detections plus runtime heuristics."""
    no_helmet, no_vest, no_glasses, no_mask = False, False, False, False
    runtime_notes = []
    detections = extract_detections(results)
    if not detections:
        return no_helmet, no_vest, no_glasses, no_mask, runtime_notes
    for detection in detections:
        label = detection["label"]
        for key, keywords in VIOLATION_CLASS_MAP.items():
            if any(kw in label for kw in keywords):
                if key == "no_helmet":
                    no_helmet = True
                elif key == "no_vest":
                    no_vest = True
                elif key == "no_glasses":
                    no_glasses = True
                elif key == "no_mask":
                    no_mask = True
    # Fallback: infer from first class name if no match yet
    if not (no_helmet or no_vest or no_glasses or no_mask):
        first_name = detections[0]["label"]
        if "helmet" in first_name and _is_negative_ppe_label(first_name):
            no_helmet = True
        if "vest" in first_name and _is_negative_ppe_label(first_name):
            no_vest = True
        if (("glass" in first_name or "goggle" in first_name or "eye" in first_name)
                and _is_negative_ppe_label(first_name)):
            no_glasses = True
        if "mask" in first_name and _is_negative_ppe_label(first_name):
            no_mask = True

    # Runtime override for a common edge case: bald heads or exposed skin misread as "Hardhat".
    # We only override when a positive hardhat detection overlaps a person's head and all matched
    # hardhat boxes look like skin-toned bare head rather than colored PPE.
    if frame is not None and not no_helmet:
        person_detections = [
            d for d in detections
            if any(kw in d["label"] for kw in PERSON_CLASS_KEYWORDS)
        ]
        hardhat_detections = [d for d in detections if _is_positive_hardhat_label(d["label"])]
        suspicious_persons = sum(
            1 for person in person_detections
            if _person_has_suspicious_hardhat(frame, person, hardhat_detections)
        )
        if suspicious_persons:
            no_helmet = True
            runtime_notes.append("Helmet override: suspicious hardhat detection")

    if no_helmet and no_vest and not no_glasses:
        no_glasses = True
    return no_helmet, no_vest, no_glasses, no_mask, runtime_notes

def _box_iou(box_a, box_b): #How much two boxes overlap (0 = none, 1 = same box).
    inter = _intersection_area(box_a, box_b)
    if inter <= 0:
        return 0.0
    union = _box_area(box_a) + _box_area(box_b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _center_distance(box_a, box_b): #Straight-line distance between two box centers.
    ax, ay = _box_center(box_a)
    bx, by = _box_center(box_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def get_person_detections(results):
    """Return deduplicated person detections sorted by area/confidence."""
    person_detections = [
        d for d in extract_detections(results)
        if any(kw in d["label"] for kw in PERSON_CLASS_KEYWORDS)
    ]
    person_detections.sort(
        key=lambda d: (_box_area(d["box"]), d["confidence"]),
        reverse=True,
    )
    unique = []
    for detection in person_detections:
        if any(_box_iou(detection["box"], kept["box"]) >= REPORTED_PERSON_DUPLICATE_IOU_THRESHOLD for kept in unique):
            continue
        unique.append(detection)
    return unique


def get_primary_person_detection(results):
    """Return the most stable person box for duplicate suppression."""
    persons = get_person_detections(results)
    return persons[0] if persons else None


def refresh_reported_and_check_same(person_detection, reported_list, frame_width, frame_height, threshold_frac, left_frame_sec):
    """Refresh tracked reported persons and return whether this person matches one already reported."""
    if person_detection is None:
        return reported_list, False
    now = time.time()
    dist_limit = threshold_frac * min(frame_width, frame_height) if (frame_width and frame_height) else 100
    updated = []
    same_person = False
    current_box = person_detection["box"]
    best_match_idx = None
    best_match_score = -1.0

    for idx, tracked in enumerate(reported_list):
        if now - tracked["last_seen"] > left_frame_sec:
            continue  # Person left frame (not seen near this position for a while)
        tracked_box = tracked["box"]
        iou = _box_iou(current_box, tracked_box)
        dist = _center_distance(current_box, tracked_box)
        if iou >= REPORTED_PERSON_IOU_THRESHOLD or dist <= dist_limit:
            score = max(iou, 1.0 - min(1.0, dist / max(dist_limit, 1.0)))
            if score > best_match_score:
                best_match_score = score
                best_match_idx = idx
        updated.append(tracked)

    if best_match_idx is not None:
        updated[best_match_idx]["last_seen"] = now
        updated[best_match_idx]["box"] = current_box
        updated[best_match_idx]["confidence"] = person_detection["confidence"]
        same_person = True

    return updated, same_person

def save_violation_image(annotated_frame, no_helmet=False, no_vest=False, no_glasses=False, no_mask=False, extra_label=""):
    """Save violation snapshot with date/time; return (full_path, datetime_str, image_basename)."""
    import violations_log as vlog
    out_dir = vlog.VIOLATIONS_DIR
    os.makedirs(out_dir, exist_ok=True)
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    base = now.strftime("%Y-%m-%d_%H-%M-%S")
    if extra_label:
        base = f"{base}_{extra_label}"
    filename = os.path.join(out_dir, f"violation_{base}.jpg")
    if os.path.exists(filename):
        n = 1
        while os.path.exists(filename):
            filename = os.path.join(out_dir, f"violation_{base}_{n:03d}.jpg")
            n += 1
    img = annotated_frame.copy()
    h, w = img.shape[:2]
    cv2.putText(img, date_time_str, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(img, "PPE VIOLATION", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    tags = []
    if no_helmet:
        tags.append("No Helmet")
    if no_vest:
        tags.append("No Vest")
    if no_glasses:
        tags.append("No Glasses")
    if no_mask:
        tags.append("No Mask")
    if tags:
        cv2.putText(img, " | ".join(tags), (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    cv2.imwrite(filename, img)
    basename = os.path.basename(filename)
    vlog.add_violation(
        date_time_str,
        basename,
        no_helmet=no_helmet,
        no_vest=no_vest,
        no_glasses=no_glasses,
        no_mask=no_mask,
        location=LOCATION_NAME,
        image_path=filename,
    )
    return filename, date_time_str, basename

def _write_clip_thread(pre_frames, clip_path, post_queue, width, height, fps, post_sec, image_basename, clip_basename):
    """Background thread: write pre_frames then post_queue frames to clip_path, then update violation."""
    import violations_log as vlog
    # Some codecs require even dimensions
    w, h = width & ~1, height & ~1
    if w <= 0 or h <= 0:
        w, h = width, height
    try:
        # Prefer H.264 in MP4 for browser playback; fallback to mp4v then AVI
        out = None
        for ext, fourcc in [(".mp4", "avc1"), (".mp4", "mp4v"), (".avi", "XVID")]:
            path = clip_path if clip_path.endswith(ext) else (os.path.splitext(clip_path)[0] + ext)
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            if out.isOpened():
                clip_path = path
                clip_basename = os.path.basename(path)
                break
            out.release()
        if out is None or not out.isOpened():
            print("Clip: could not open VideoWriter for", clip_path)
            return
        for f in pre_frames:
            if f.shape[1] != w or f.shape[0] != h:
                f = cv2.resize(f, (w, h))
            out.write(f)
        target_post = int(post_sec * fps)
        for _ in range(target_post):
            try:
                frame = post_queue.get(timeout=post_sec + 10)
                if frame is None:
                    break
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h))
                out.write(frame)
            except queue.Empty:
                break
        out.release()
        # Re-encode to H.264 MP4 for browser playback if ffmpeg is available
        final_basename = _reencode_clip_for_browser(clip_path, clip_basename) or clip_basename
        vlog.update_violation_clip(image_basename, final_basename)
        print(f"Clip saved: {final_basename}")
    except Exception as e:
        import traceback
        print("Clip recording error:", e)
        traceback.print_exc()

def _reencode_clip_for_browser(clip_path, clip_basename):
    """If ffmpeg is available, re-encode to H.264 MP4 for browser playback. Returns new basename or None."""
    import subprocess
    base = os.path.splitext(clip_path)[0]
    out_path = base + "_h264.mp4"
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", clip_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            out_path,
        ], capture_output=True, timeout=300)
        if result.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if os.path.isfile(out_path):
        try:
            os.remove(clip_path)
        except OSError:
            pass
        return os.path.basename(out_path)
    return None

def test_stream_accessible(url):
    """Test if the stream URL is accessible via HTTP"""
    try:
        response = requests.get(url, stream=True, timeout=5)
        if response.status_code == 200:
            print(f"✓ Stream is accessible via HTTP (Status: {response.status_code})")
            return True
        else:
            print(f"✗ Stream returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot access stream via HTTP: {e}")
        return False

class MJPEGStreamReader:
    """
    Read MJPEG stream over HTTP when OpenCV VideoCapture fails (e.g. on Windows).
    Parses multipart/x-mixed-replace or raw JPEG stream and yields frames.
    """
    def __init__(self, url, timeout=10):
        self.url = url
        self.timeout = timeout
        self._response = None
        self._stream = None
        self._buffer = b""
        self._frame_size = None  # (width, height) from first frame
        self._open()

    def _open(self):
        self._response = requests.get(self.url, stream=True, timeout=self.timeout)
        self._response.raise_for_status()
        self._stream = self._response.iter_content(chunk_size=8192)
        self._buffer = b""
        # Parse boundary from Content-Type if present
        ct = self._response.headers.get("Content-Type", "")
        self._boundary = None
        if "boundary=" in ct:
            self._boundary = ct.split("boundary=")[-1].strip().strip('"').encode()

    def _find_jpeg_in_buffer(self):
        # JPEG starts with FFD8, ends with FFD9
        start = self._buffer.find(b"\xff\xd8")
        if start == -1:
            return None, None
        end = self._buffer.find(b"\xff\xd9", start)
        if end == -1:
            return None, start  # keep from start for next time
        end += 2
        jpeg_data = self._buffer[start:end]
        self._buffer = self._buffer[end:]
        return jpeg_data, None

    def read(self):
        """Return (ret, frame) like cv2.VideoCapture.read()"""
        while True:
            jpeg_data, keep_from = self._find_jpeg_in_buffer()
            if jpeg_data:
                arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    self._frame_size = (frame.shape[1], frame.shape[0])
                    return True, frame
            if keep_from is not None:
                self._buffer = self._buffer[keep_from:]
            try:
                chunk = next(self._stream)
                if not chunk:
                    return False, None
                self._buffer += chunk
            except StopIteration:
                return False, None
            except Exception:
                return False, None

    def get(self, prop):
        """Emulate cv2.CAP_PROP_* for compatibility"""
        if self._frame_size is None:
            return 0
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._frame_size[0])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._frame_size[1])
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        return 0

    def release(self):
        if self._response is not None:
            try:
                self._response.close()
            except Exception:
                pass
            self._response = None
        self._stream = None

def open_ip_camera(url):
    """Try OpenCV first, then fall back to HTTP MJPEG stream reader"""
    print(f"Attempting to connect to IP camera: {url}")
    print("Testing stream accessibility...")
    test_stream_accessible(url)

    # Try OpenCV backends first
    backends = [
        (cv2.CAP_FFMPEG, "FFMPEG"),
        (cv2.CAP_ANY, "ANY"),
        (cv2.CAP_DSHOW, "DSHOW"),
    ]
    print(f"\nTrying URL: {url}")
    for backend_id, backend_name in backends:
        try:
            print(f"  Trying backend: {backend_name}...", end=" ")
            cap = cv2.VideoCapture(url, backend_id)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            if cap.isOpened() and ret and frame is not None:
                print("Success! Using OpenCV")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return cap
            cap.release()
            print("Failed (cannot read frame)")
        except Exception as e:
            print(f"Error: {e}")
            try:
                cap.release()
            except Exception:
                pass

    # OpenCV failed: use HTTP MJPEG stream reader
    print("\nUsing HTTP MJPEG stream reader (OpenCV cannot open this stream on this system)...")
    try:
        reader = MJPEGStreamReader(url)
        ret, frame = reader.read()
        if ret and frame is not None:
            print("Success! Connected via HTTP MJPEG stream.")
            return reader
        reader.release()
    except Exception as e:
        print(f"HTTP MJPEG reader failed: {e}")
    return None

def main():
    # Load YOLOv8 custom PPE model (ensure best.pt is in models folder)
    print("Loading PPE Detection model...")
    model = YOLO("models/best.pt")
    print("Model loaded successfully!")
    try:
        import email_alert
        if ENABLE_EMAIL_ALERTS and email_alert.is_email_configured():
            print("Email alerts enabled (CCTV/video source will be included).")
        elif not ENABLE_EMAIL_ALERTS:
            print("Email alerts disabled (set ENABLE_EMAIL_ALERTS = True in config.py to enable).")
        else:
            print("Email alerts disabled. Set SENDER_EMAIL, RECEIVER_EMAIL, EMAIL_PASSWORD in .env to enable.")
    except Exception:
        pass

    # Check if OpenCV GUI is available
    gui_available = check_opencv_gui()
    if not gui_available:
        print("Warning: OpenCV GUI (cv2.imshow) is not available.")
        print("Frames will be saved to 'output_frames' directory instead.")
        print("To enable GUI, install: pip install opencv-contrib-python")
        print("Press Ctrl+C to stop processing.\n")
    else:
        print("GUI available. Press Q or Esc in the video window to quit.\n")

    # Initialize video capture
    print(f"\n{'='*60}")
    print(f"Connecting to video source: {VIDEO_SOURCE}")
    print(f"{'='*60}\n")
    
    # For IP camera streams, try multiple methods (OpenCV then HTTP MJPEG reader)
    if isinstance(VIDEO_SOURCE, str) and ("http://" in VIDEO_SOURCE or "https://" in VIDEO_SOURCE or "rtsp://" in VIDEO_SOURCE):
        video_source = open_ip_camera(VIDEO_SOURCE)
        if video_source is None:
            print(f"\n{'='*60}")
            print("ERROR: Could not open IP camera stream")
            print(f"{'='*60}")
            print("\nTroubleshooting: Verify the URL works in a browser and check firewall/network.")
            return
    else:
        # Local camera (integer index) or recorded video file (string path)
        if isinstance(VIDEO_SOURCE, str):
            print(f"Opening video file: {VIDEO_SOURCE}")
        else:
            print(f"Opening local camera: {VIDEO_SOURCE}")
        video_source = cv2.VideoCapture(VIDEO_SOURCE)
        if not video_source.isOpened():
            print(f"Error: Could not open video source: {VIDEO_SOURCE}")
            if isinstance(VIDEO_SOURCE, str) and not VIDEO_SOURCE.isdigit():
                print("For a video file, ensure the path is correct and the file exists (e.g. recorded_footage.mp4).")
            return

    print("Video source opened successfully!")
    width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_source.get(cv2.CAP_PROP_FPS)
    if width and height:
        print(f"Video properties: {width}x{height} @ {fps} FPS\n")
    else:
        print("Video properties: will show from first frame\n")

    frame_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    stop_requested = False
    wait_ms = 30
    last_violation_save_time = 0
    last_email_time = 0
    violation_save_cooldown_sec = 2  # Min seconds between saving violation images
    reported_persons = []  # [{"last_seen": ts, "box": (x1, y1, x2, y2), "confidence": conf}, ...]

    # Clip recording: rolling buffer (1 min before) + post queue (1 min after)
    clip_buffer = []
    clip_buffer_max = CLIP_PRE_SEC * CLIP_FPS
    last_clip_frame_time = 0
    clip_post_state = {"queue": None, "end_time": 0}  # shared with main loop and violation block

    try:
        while not stop_requested:
            ret, frame = video_source.read()
            if not ret:
                print("End of video stream or failed to read frame.")
                break

            # Run inference
            results = model(frame, conf=CONFIDENCE_THRESHOLD)
            if len(results[0].boxes) == 0:
                continue  # Skip frames with no detections
            if not frame_has_person(results):
                continue  # Skip frames where no human is detected (ignore machinery, cones, false PPE on objects)

            no_helmet, no_vest, no_glasses, no_mask, runtime_notes = get_violation_types_from_results(results, frame=frame)


            has_violation = any([no_helmet, no_vest, no_glasses, no_mask])

            annotated_frame = results[0].plot()
            if runtime_notes:
                annotated_frame = draw_runtime_notes(annotated_frame, runtime_notes)
            now = time.time()
            # Feed clip buffer at CLIP_FPS (pre-violation)
            if now - last_clip_frame_time >= 1.0 / CLIP_FPS:
                clip_buffer.append(annotated_frame.copy())
                if len(clip_buffer) > clip_buffer_max:
                    clip_buffer.pop(0)
                last_clip_frame_time = now
            # Feed post-violation queue at CLIP_FPS if we're recording a clip
            if clip_post_state["queue"] is not None:
                if now < clip_post_state["end_time"]:
                    if now - clip_post_state.get("last_put", 0) >= 1.0 / CLIP_FPS:
                        try:
                            clip_post_state["queue"].put(annotated_frame.copy(), block=False)
                            clip_post_state["last_put"] = now
                        except queue.Full:
                            pass
                else:
                    try:
                        clip_post_state["queue"].put(None, block=False)
                    except queue.Full:
                        pass
                    clip_post_state["queue"] = None

            has_detection = len(results[0].boxes) > 0
            same_person_still_in_frame = False
            if has_detection:
                h, w = frame.shape[:2]
                primary_person = get_primary_person_detection(results)
                if primary_person:
                    reported_persons, same_person_still_in_frame = refresh_reported_and_check_same(
                        primary_person,
                        reported_persons,
                        w, h,
                        SAME_PERSON_THRESHOLD,
                        PERSON_LEFT_FRAME_SEC,
                    )

            # Save violation only when: detection, cooldown passed, and not already reported while still in frame
            if has_detection and has_violation and (time.time() - last_violation_save_time) >= violation_save_cooldown_sec:
                if same_person_still_in_frame:
                    pass  # Same person still in frame – don't report again until they leave and come back
                else:
                    path, dt_str, basename = save_violation_image(
                        annotated_frame, no_helmet=no_helmet, no_vest=no_vest, no_glasses=no_glasses, no_mask=no_mask
                    )
                    print(f"Violation saved: {basename} ({dt_str})")
                    last_violation_save_time = time.time()
                    primary_person = get_primary_person_detection(results)
                    if primary_person:
                        reported_persons.append({
                            "last_seen": time.time(),
                            "box": primary_person["box"],
                            "confidence": primary_person["confidence"],
                        })
                    # Start clip recording (1 min before + 1 min after) if not already recording
                    if clip_post_state["queue"] is None and len(clip_buffer) > 0:
                        import violations_log as vlog
                        pre_frames = list(clip_buffer)
                        base = os.path.splitext(basename)[0]
                        clip_basename = base + "_clip.mp4"
                        clip_path = os.path.join(vlog.VIOLATIONS_DIR, clip_basename)
                        os.makedirs(vlog.VIOLATIONS_DIR, exist_ok=True)
                        post_q = queue.Queue(maxsize=0)  # unbounded so we don't drop frames
                        clip_post_state["queue"] = post_q
                        clip_post_state["end_time"] = time.time() + CLIP_POST_SEC
                        clip_post_state["last_put"] = time.time()
                        h, w = annotated_frame.shape[:2]
                        t = threading.Thread(
                            target=_write_clip_thread,
                            args=(pre_frames, clip_path, post_q, w, h, CLIP_FPS, CLIP_POST_SEC, basename, clip_basename),
                            daemon=True,
                        )
                        t.start()
                    # Email alert (non-blocking) when configured; use cooldown to avoid spam
                    try:
                        import email_alert
                        if ENABLE_EMAIL_ALERTS and email_alert.is_email_configured() and (time.time() - last_email_time) >= EMAIL_COOLDOWN_SEC:
                            tags = []
                            if no_helmet:
                                tags.append("No Helmet")
                            if no_vest:
                                tags.append("No Vest")
                            if no_glasses:
                                tags.append("No Glasses")
                            if no_mask:
                                tags.append("No Mask")
                            email_alert.send_email_in_background(
                                path,
                                video_source=str(VIDEO_SOURCE),
                                violation_tags=tags or ["PPE violation"],
                                location=LOCATION_NAME,
                                datetime_str=dt_str,
                            )
                            last_email_time = time.time()
                    except Exception:
                        pass

            # Calculate and display FPS
            fps_frame_count += 1
            if fps_frame_count % 30 == 0:
                elapsed = time.time() - fps_start_time
                fps = 30 / elapsed if elapsed > 0 else 0
                print(f"Processing at {fps:.1f} FPS (Frame {frame_count})")
                fps_start_time = time.time()

            if gui_available:
                # Draw quit hint on frame (top-left)
                cv2.putText(annotated_frame, "Press Q or Esc to quit", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("PPE Detection System", annotated_frame)
                key = cv2.waitKey(wait_ms)
                if key != -1:
                    key = key & 0xFF
                    if key == ord("q") or key == ord("Q") or key == 27:  # 27 = Esc
                        print("Quit key pressed. Stopping...")
                        stop_requested = True
                        break
            else:
                filename = save_frame(annotated_frame, frame_count)
                if frame_count % 30 == 0:
                    print(f"Saved frame {frame_count} to {filename}. Press Ctrl+C to stop.")
                time.sleep(0.033)

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")

    try:
        video_source.release()
    except Exception:
        pass
    if gui_available:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print(f"\nProcessing complete. Processed {frame_count} frames.")
    if not gui_available:
        print(f"All frames saved to 'output_frames' directory.")
    sys.exit(0)

if __name__ == "__main__":
    main()
