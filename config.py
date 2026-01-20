import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()
DATABASE_PATH = BASE_DIR / "data" / "faces.db"
CHECKIN_LOG_PATH = BASE_DIR / "data" / "checkins.db"

# =============================================================================
# Device Settings (GPU/CPU)
# =============================================================================
# Set to 'cuda' for GPU, 'cpu' for CPU
# Auto-detect: use GPU if available, otherwise CPU
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# =============================================================================
# Face Detection Settings
# =============================================================================
FACE_DETECTION_CONFIDENCE = 0.9
FACE_SIZE = (112, 112)
MAX_IMAGE_SIZE = (1920, 1080)

# =============================================================================
# Face Recognition Settings
# =============================================================================
FACE_RECOGNITION_THRESHOLD = 0.5

# =============================================================================
# Anti-Spoofing Settings
# =============================================================================
BLINK_THRESHOLD = 0.25
TEXTURE_THRESHOLD = 0.5

# =============================================================================
# Real-time Pipeline Settings
# =============================================================================
STREAM_FPS = 15
MAX_QUEUE_SIZE = 10  # Drop old frames when queue exceeds this

# =============================================================================
# Tracking Settings
# =============================================================================
TRACK_MAX_AGE = 30  # Frames without detection before track becomes EXIT
TRACK_MIN_HITS = 3  # Frames to transition from NEW to STABLE
TRACK_IOU_THRESHOLD = 0.3  # IOU threshold for matching detections to tracks

# =============================================================================
# Quality Filtering Thresholds
# =============================================================================
MIN_FACE_SIZE = 80  # Minimum face size in pixels
BLUR_THRESHOLD = 100  # Laplacian variance threshold (lower = more blurry)
POSE_THRESHOLD = 0.7  # Frontal pose ratio threshold
BRIGHTNESS_MIN = 40  # Minimum average brightness
BRIGHTNESS_MAX = 220  # Maximum average brightness

# =============================================================================
# FAS Accumulation Settings
# =============================================================================
FAS_MIN_FRAMES = 5  # Minimum frames before making FAS decision
FAS_REJECT_THRESHOLD = 0.3  # Below this = spoof
FAS_ACCEPT_THRESHOLD = 0.7  # Above this = real
FAS_EARLY_REJECT_THRESHOLD = 0.2  # Early reject if score consistently below this
FAS_EARLY_REJECT_FRAMES = 3  # Frames for early reject decision

# =============================================================================
# FR Accumulation Settings
# =============================================================================
FR_EMA_ALPHA = 0.7  # Exponential moving average weight for new embeddings
FR_MIN_FRAMES = 3  # Minimum frames before FR decision
FR_SIMILARITY_THRESHOLD = 0.5  # Match threshold for recognition

# =============================================================================
# Session & Check-in Settings
# =============================================================================
CHECKIN_COOLDOWN_MINUTES = 5  # Prevent re-check-in within this time
DECISION_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for final decision
SESSION_TIMEOUT_SECONDS = 30  # Max time for a single check-in session

# =============================================================================
# Geolocation Settings
# =============================================================================
# Default: Hanoi Center (Example) - Replace with actual company coordinates
COMPANY_LOCATION = (21.0285, 105.8542)  # (Latitude, Longitude)
MAX_CHECKIN_DISTANCE = 1000  # Maximum allowed distance in meters (1km for testing)

# =============================================================================
# API Settings
# =============================================================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# =============================================================================
# Create directories
# =============================================================================
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
EVIDENCE_DIR = DATA_DIR / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)
