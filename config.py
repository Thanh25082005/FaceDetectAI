import os
from pathlib import Path
import torch
from utils.config_utils import load_dynamic_config

# === Path Settings ===
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
EVIDENCE_DIR = DATA_DIR / "evidence"
DATABASE_PATH = DATA_DIR / "faces.db"
CHECKIN_LOG_PATH = DATA_DIR / "checkins.db"

# Create essential directories
DATA_DIR.mkdir(exist_ok=True)
EVIDENCE_DIR.mkdir(exist_ok=True)

# === Device Settings (GPU/CPU) ===
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# === Face Detection & Recognition ===
FACE_DETECTION_CONFIDENCE = 0.9
FACE_SIZE = (112, 112)
MAX_IMAGE_SIZE = (1920, 1080)
FR_THRESHOLD = 0.5  # Face Recognition similarity threshold

# === Quality Filtering Thresholds ===
MIN_FACE_SIZE = 80      # Minimum face size in pixels
BLUR_THRESHOLD = 100    # Laplacian variance threshold (lower = more blurry)
POSE_THRESHOLD = 0.7    # Frontal pose ratio threshold (1.0 = perfect frontal)
BRIGHTNESS_MIN = 40     # Minimum average brightness
BRIGHTNESS_MAX = 220    # Maximum average brightness

# === Anti-Spoofing (FAS) Settings ===
FAS_ENABLE = True
FAS_MIN_FRAMES = 5              # Minimum frames before making FAS decision
FAS_REJECT_THRESHOLD = 0.3      # Below this = spoof
FAS_ACCEPT_THRESHOLD = 0.7      # Above this = real
FAS_EARLY_REJECT_THRESHOLD = 0.2 # Early reject if score consistently below this
FAS_EARLY_REJECT_FRAMES = 3      # Frames for early reject decision

# === Real-time Pipeline & Tracking ===
STREAM_FPS = 15
MAX_QUEUE_SIZE = 10
TRACK_MAX_AGE = 30              # Frames without detection before track ends
TRACK_MIN_HITS = 3               # Frames to transition to STABLE track
TRACK_IOU_THRESHOLD = 0.3        # IOU threshold for matching tracks

# === Decision & Accumulation Logic ===
FR_EMA_ALPHA = 0.7              # Exponential moving average for embeddings
FR_MIN_FRAMES = 3               # Minimum frames before recognition decision
SESSION_TIMEOUT_SECONDS = 30     # Max time for a single check-in session
CHECKIN_COOLDOWN_MINUTES = 5     # Reset check-in for user after this time
DECISION_CONFIDENCE_THRESHOLD = 0.7 # Minimum fusion confidence

# === Score Fusion Weights (FR + FAS + Quality) ===
FUSION_WEIGHT_FR = 0.5
FUSION_WEIGHT_FAS = 0.4
FUSION_WEIGHT_QUALITY = 0.1

# === API & Network Settings ===
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# === SQL Server Settings ===
MSSQL_HOST = os.getenv("MSSQL_HOST", "localhost")
MSSQL_PORT = int(os.getenv("MSSQL_PORT", "1433"))
MSSQL_USER = os.getenv("MSSQL_USER", "sa")
MSSQL_PASSWORD = os.getenv("MSSQL_PASSWORD", "YourStrong@Passw0rd")
MSSQL_DATABASE = os.getenv("MSSQL_DATABASE", "FaceCheckDB")

# === Geolocation Settings ===
_dynamic_config = load_dynamic_config()
COMPANY_LOCATION = tuple(_dynamic_config.get("COMPANY_LOCATION", [21.0285, 105.8542]))  # (Latitude, Longitude)
MAX_CHECKIN_DISTANCE = _dynamic_config.get("MAX_CHECKIN_DISTANCE", 1000)            # Maximum allowed distance in meters

# === Backward Compatibility Aliases (Do not remove) ===
FACE_RECOGNITION_THRESHOLD = FR_THRESHOLD
FR_SIMILARITY_THRESHOLD = FR_THRESHOLD
CHECKIN_RECOGNITION_THRESHOLD = FR_THRESHOLD
BLINK_THRESHOLD = 0.2
TEXTURE_THRESHOLD = 0.5
