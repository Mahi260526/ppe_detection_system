# Video input: use one of the following:
# - Integer: camera index (0 = default webcam, 1 = first USB, etc.)
# - URL string: IP/CCTV stream e.g. "http://192.168.1.214:4747/mjpegfeed"
# - File path: recorded footage e.g. "recorded_footage.mp4" or "videos/camera_recording.mp4"
VIDEO_SOURCE = r"C:\Users\Mahi Agrawal\Downloads\sample_video4.mp4"  # recorded video file (place your .mp4/.avi in project folder or use full path)
# Location name for this camera/source (used for filtering by user-assigned locations)
LOCATION_NAME = "Main Gate"

# Demo cameras for parallel processing (run: python run_demo_cameras.py)
# Each entry = one process. Use a unique location_name per video.
DEMO_CAMERAS = [
    {
        "name": "Camera 1",
        "location_name": "Site A",
        "source": r"C:\Users\Mahi Agrawal\Downloads\sample_video7.mp4",
    },
    {
        "name": "Camera 2",
        "location_name": "Site B",
        "source": r"C:\Users\Mahi Agrawal\Downloads\sample_video6.mp4",
    },
    {
        "name": "Camera 3",
        "location_name": "Site C",
        "source": r"C:\Users\Mahi Agrawal\Downloads\sample_video4.mp4",
    },
]

# Lowering this helps catch more misses (esp. helmet) but can increase noise.
# Keep it modest on CPU, and rely on the confirmation logic in app.py.
CONFIDENCE_THRESHOLD = 0.35

# Accuracy tuning (CPU-friendly)
# - Per-class minimum confidence: ignore weak/noisy "no_*" detections.
# - Multi-frame confirmation: violation must persist across a few frames (reduces flicker).

# Minimum confidence to treat a detection as a "person" for gating/tracking logic.
PERSON_MIN_CONFIDENCE = 0.35

# Minimum confidence to accept each violation label from the model.
NO_HELMET_MIN_CONFIDENCE = 0.45
NO_VEST_MIN_CONFIDENCE = 0.45
NO_MASK_MIN_CONFIDENCE = 0.45
NO_GLASSES_MIN_CONFIDENCE = 0.45

# Multi-frame confirmation (N-of-M frames)
HELMET_CONFIRM_WINDOW_FRAMES = 5
HELMET_CONFIRM_REQUIRED_FRAMES = 3
VEST_CONFIRM_WINDOW_FRAMES = 5
VEST_CONFIRM_REQUIRED_FRAMES = 3
MASK_CONFIRM_WINDOW_FRAMES = 5
MASK_CONFIRM_REQUIRED_FRAMES = 3
GLASSES_CONFIRM_WINDOW_FRAMES = 5
GLASSES_CONFIRM_REQUIRED_FRAMES = 3


# Hardhat validation:
# Some PPE models can mistake bald heads / skin-colored regions for a hardhat.
# These heuristics are conservative and only override a hardhat detection when
# it overlaps the top of a detected person's box and the ROI looks strongly like skin.
ENABLE_HELMET_COLOR_HEURISTIC = True
HARDHAT_MIN_CONFIDENCE = 0.6
HARDHAT_HEAD_REGION_RATIO = 0.35
HARDHAT_HEAD_OVERLAP_THRESHOLD = 0.5
HARDHAT_SKIN_RATIO_THRESHOLD = 0.45
HARDHAT_MAX_MEAN_SATURATION = 0.35

# Set to True to send email alerts on violations (requires .env configured)
ENABLE_EMAIL_ALERTS = True
# Minimum seconds between email alerts (avoid spam)
EMAIL_COOLDOWN_SEC = 60

# Same-person dedup: once reported, same person is not reported again while they stay in frame.
# Distance as fraction of min(frame width, height) to treat as same person.
# 0.2 = 20% of frame; increase to allow more repeats, decrease to be stricter.
SAME_PERSON_THRESHOLD = 0.2
# If we don't see a detection near a reported position for this many seconds, treat as "left frame";
# then if they come back they can be reported again.
PERSON_LEFT_FRAME_SEC = 30
REPORTED_PERSON_IOU_THRESHOLD = 0.25
REPORTED_PERSON_DUPLICATE_IOU_THRESHOLD = 0.7

# Clip recording: 1 min before + 1 min after violation (saved with violation, viewable on dashboard)
CLIP_FPS = 5
CLIP_PRE_SEC = 60   # seconds of video to keep before violation
CLIP_POST_SEC = 60  # seconds to record after violation
