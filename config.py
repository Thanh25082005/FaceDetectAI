import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()
DATABASE_PATH = BASE_DIR / "data" / "faces.db"

FACE_DETECTION_CONFIDENCE = 0.9
FACE_RECOGNITION_THRESHOLD = 0.5

BLINK_THRESHOLD = 0.25
TEXTURE_THRESHOLD = 0.5

MAX_IMAGE_SIZE = (1920, 1080)
FACE_SIZE = (112, 112)

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Create data directory if not exists
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
