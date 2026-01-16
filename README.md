# Face Recognition API

A face recognition system with detection, recognition, and anti-spoofing capabilities.

## Features

- **Face Detection** - MTCNN for detecting faces and landmarks
- **Face Recognition** - InsightFace/ArcFace for 512-d embeddings
- **Anti-Spoofing** - Texture analysis, FFT moire detection, color & blur checks
- **REST API** - FastAPI with OpenAPI documentation

## Requirements

- Python 3.10+
- GPU: **Optional** (runs on CPU by default, GPU makes it faster)

## Installation

```bash
# Clone repository
git clone ...
cd detection-face

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Start Server

```bash
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/detect_face` | POST | Detect faces in image |
| `/api/v1/recognize_face` | POST | Recognize face against database |
| `/api/v1/anti_spoofing` | POST | Check if face is real or spoofed |
| `/api/v1/add_face` | POST | Add face to database |
| `/api/v1/get_face/{user_id}` | GET | Get face data by user_id |
| `/api/v1/delete_face/{user_id}` | DELETE | Delete face from database |
| `/api/v1/update_face` | POST | Update face data |

### Examples

**Add a face:**
```bash
curl -X POST "http://localhost:8000/api/v1/add_face" \
  -F "file=@photo.jpg" \
  -F "user_id=user001" \
  -F "name=John Doe"
```

**Recognize a face:**
```bash
curl -X POST "http://localhost:8000/api/v1/recognize_face" \
  -F "file=@test.jpg"
```

**Check anti-spoofing:**
```bash
curl -X POST "http://localhost:8000/api/v1/anti_spoofing" \
  -F "file=@face.jpg"
```

## Project Structure

```
detection-face/
├── main.py                 # FastAPI entry point
├── config.py               # Configuration
├── requirements.txt        # Dependencies
├── models/
│   ├── face_detector.py    # MTCNN detection
│   ├── face_recognizer.py  # InsightFace recognition
│   ├── anti_spoofing.py    # Liveness detection
│   └── database.py         # SQLite storage
├── api/
│   ├── routes.py           # API endpoints
│   └── schemas.py          # Pydantic models
├── utils/
│   └── image_utils.py      # Image processing
└── tests/
    └── test_api.py         # API tests
```

## Testing

```bash
python -m pytest tests/test_api.py -v
```
