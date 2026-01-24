# 🎯 Face Recognition Check-in System

Hệ thống nhận diện khuôn mặt và chấm công sử dụng **InsightFace (ArcFace)** với tích hợp **Face Anti-Spoofing (FAS)**.

---

## 📁 Cấu trúc thư mục

```
FaceDetectAI/
├── main.py                 # Entry point - Khởi động FastAPI server
├── config.py               # Cấu hình hệ thống (thresholds, paths, device)
├── requirements.txt        # Dependencies
│
├── api/                    # API Layer
│   ├── routes.py           # Định nghĩa các endpoints
│   ├── auth.py             # Xác thực người dùng
│   └── schemas.py          # Pydantic models cho request/response
│
├── models/                 # AI Models & Business Logic
│   ├── face_detector.py    # Phát hiện khuôn mặt (MTCNN)
│   ├── face_recognizer.py  # Nhận diện khuôn mặt (ArcFace)
│   ├── anti_spoofing.py    # Chống giả mạo (Silent-FAS)
│   ├── quality_filter.py   # Đánh giá chất lượng ảnh
│   ├── database.py         # Quản lý database khuôn mặt
│   └── checkin_logger.py   # Ghi log check-in
│
├── utils/                  # Tiện ích
│   ├── image_utils.py      # Xử lý ảnh
│   └── geo_utils.py        # Tính toán vị trí GPS
│
├── data/                   # Lưu trữ dữ liệu
│   ├── faces.db            # Database embeddings
│   ├── checkins.db         # Database lịch sử check-in
│   └── evidence/           # Ảnh bằng chứng check-in
│
└── libs/                   # Thư viện AI models
    └── Silent-Face-Anti-Spoofing/  # FAS model weights
```

---

## 🧠 Cấu trúc Model

### 1. Face Detector (`models/face_detector.py`)
- **Model**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Chức năng**: Phát hiện và căn chỉnh khuôn mặt trong ảnh
- **Output**: Bounding box, 5 landmarks (mắt, mũi, miệng), confidence score

### 2. Face Recognizer (`models/face_recognizer.py`)
- **Model**: InsightFace ArcFace (`buffalo_l`)
- **Chức năng**: Trích xuất embedding vector 512 chiều từ khuôn mặt
- **So khớp**: Cosine similarity với ngưỡng `FR_THRESHOLD = 0.5`

### 3. Anti-Spoofing (`models/anti_spoofing.py`)
- **Model**: Silent-Face-Anti-Spoofing (MiniFASNet)
- **Chức năng**: Phát hiện ảnh giả, video, mask
- **Ngưỡng**: 
  - `< 0.3` → Spoof (giả mạo)
  - `> 0.7` → Real (thật)

---

## 🔄 Luồng xử lý

### Đăng ký khuôn mặt (`/api/v1/add_face`)

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────┐
│  Upload     │───▶│ Face Detect  │───▶│ Anti-Spoof  │───▶│ Extract      │───▶│ Save to  │
│  Image      │    │ (MTCNN)      │    │ (FAS)       │    │ Embedding    │    │ Database │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘    └──────────┘
                         │                   │                  │
                         ▼                   ▼                  ▼
                   Detect face,        Reject if         512-d vector
                   align to 112x112    score < 0.3       (ArcFace)
```

### Check-in (`/api/v1/mobile_checkin`)

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────┐
│  Upload     │───▶│ Verify       │───▶│ Face Detect │───▶│ Anti-Spoof   │───▶│ Face     │
│  Image+GPS  │    │ Location     │    │ + Align     │    │ Check        │    │ Match    │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘    └──────────┘
                         │                                                          │
                         ▼                                                          ▼
                   Distance ≤ 1000m                                          Compare với DB
                   từ công ty                                                Log kết quả
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/add_face` | Đăng ký khuôn mặt mới |
| `GET` | `/api/v1/get_face/{user_id}` | Lấy thông tin khuôn mặt |
| `DELETE` | `/api/v1/delete_face/{user_id}` | Xóa khuôn mặt |
| `POST` | `/api/v1/mobile_checkin` | Chấm công GPS + Khuôn mặt |
| `GET` | `/api/v1/config` | Lấy cấu hình GPS hiện tại |
| `POST` | `/api/v1/config` | Cập nhật cấu hình GPS |
| `GET` | `/api/v1/health` | Kiểm tra trạng thái hệ thống |

---

### `GET /api/v1/health`
Kiểm tra trạng thái hệ thống.

```json
// Response
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cuda"
}
```

---

### `POST /api/v1/add_face`
Đăng ký khuôn mặt mới với kiểm tra anti-spoofing.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✅ | Ảnh khuôn mặt |
| `user_id` | String | ✅ | ID người dùng |
| `name` | String | ❌ | Tên hiển thị |

```json
// Response
{
  "success": true,
  "message": "Face added for user user123",
  "fas_score": 0.85
}
```

---

### `GET /api/v1/get_face/{user_id}`
Lấy thông tin khuôn mặt đã đăng ký.

```json
// Response
{
  "user_id": "user123",
  "name": "Nguyen Van A",
  "created_at": "2026-01-24T10:30:00"
}
```

---

### `DELETE /api/v1/delete_face/{user_id}`
Xóa khuôn mặt khỏi database.

```json
// Response
{
  "success": true,
  "message": "Face deleted for user user123"
}
```

---

### `POST /api/v1/mobile_checkin`
Chấm công với xác thực khuôn mặt + GPS.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✅ | Ảnh khuôn mặt |
| `latitude` | Float | ✅ | Vĩ độ GPS |
| `longitude` | Float | ✅ | Kinh độ GPS |
| `expected_user_id` | String | ❌ | ID người dùng mong đợi |

```json
// Response
{
  "success": true,
  "user_id": "user123",
  "similarity": 0.89,
  "fas_score": 0.92,
  "location_verified": true,
  "distance_meters": 50.5
}
```

---

### `GET /api/v1/config`
Lấy cấu hình vị trí công ty và khoảng cách check-in hiện tại.

```json
// Response
{
  "success": true,
  "message": "Configuration retrieved successfully",
  "company_location": [21.0285, 105.8542],
  "max_checkin_distance": 1000.0
}
```

---

### `POST /api/v1/config`
Cập nhật cấu hình vị trí công ty và khoảng cách check-in.

**Request:** `application/json`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `company_location` | List[Float] | ❌ | Vị trí công ty `[latitude, longitude]` |
| `max_checkin_distance` | Float | ❌ | Khoảng cách tối đa (mét) |

**Example Body:**
```json
{
  "company_location": [21.0285, 105.8542],
  "max_checkin_distance": 1000
}
```

```json
// Response
{
  "success": true,
  "message": "Configuration updated successfully",
  "company_location": [21.0285, 105.8542],
  "max_checkin_distance": 1000.0
}
```

---

## 🚀 Cách sử dụng

### 1. Cài đặt

```bash
# Clone và tạo môi trường
git clone https://github.com/Thanh25082005/FaceDetectAI.git
cd FaceDetectAI
python3 -m venv venv
source venv/bin/activate

# Cài đặt chuyên cho Mobile
pip install -r requirements_mobile.txt
```

### 2. Chạy server (Mobile)

```bash
uvicorn main_mobile:app --host 0.0.0.0 --port 8000
```

### 3. Truy cập API docs

```
http://localhost:8000/docs
```

---

## ⚙️ Cấu hình (`config.py`)

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `FR_THRESHOLD` | `0.5` | Ngưỡng nhận diện khuôn mặt |
| `FAS_ACCEPT_THRESHOLD` | `0.7` | Ngưỡng chấp nhận anti-spoofing |
| `FAS_REJECT_THRESHOLD` | `0.3` | Ngưỡng từ chối anti-spoofing |
| `MAX_CHECKIN_DISTANCE` | `1000` | Khoảng cách tối đa (mét) |
| `CHECKIN_COOLDOWN_MINUTES` | `5` | Cooldown giữa các lần check-in |
| `DEVICE` | auto | `cuda` nếu có GPU, `cpu` nếu không |