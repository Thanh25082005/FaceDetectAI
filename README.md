# 📸 Face Auth & Check-in System (Lite Version)

Hệ thống điểm danh, chấm công trực tuyến tối ưu hóa cho tốc độ và hiệu năng, sử dụng công nghệ Nhận diện khuôn mặt (Face Recognition).
Dự án được xây dựng theo mô hình Full-Stack hiện đại với **ReactJS** (Frontend) và **FastAPI** (Backend).

> **Lưu ý:** Phiên bản này tập trung vào tốc độ nhận diện nhanh, đã loại bỏ các module kiểm tra giả mạo (Anti-Spoofing) phức tạp để tối ưu độ trễ.

---

## 🚀 Tính năng nổi bật

*   **Xác thực khuôn mặt (Face Authentication):** Nhận diện chính xác nhân viên qua khuôn mặt sử dụng InsightFace/ArcFace.
*   **Chấm công Live Stream:** Chế độ quét thời gian thực qua WebSockets (10-15 FPS), mang lại trải nghiệm mượt mà không độ trễ.
*   **Quản lý Người dùng:**
    *   Đăng ký kèm lấy mẫu khuôn mặt (Face Enrollment).
    *   Cập nhật thông tin và dữ liệu khuôn mặt.
    *   Đăng nhập hệ thống bảo mật.
*   **Giao diện Hiện đại:** Dashboard trực quan, hỗ trợ Mobile/Desktop, vẽ khung nhận diện (Bounding Box) thời gian thực.
*   **Lịch sử Chấm công:** Lưu trữ log điểm danh chi tiết, bao gồm hình ảnh bằng chứng (Evidence).

---

## 🛠️ Công nghệ sử dụng

### Backend (Python)
*   **FastAPI:** Framework API hiệu năng cao, hỗ trợ tốt Async/Await.
*   **WebSockets:** Truyền tải video stream thời gian thực.
*   **OpenCV & InsightFace:** Core xử lý ảnh và trích xuất đặc trưng khuôn mặt.
*   **SQLite:** Cơ sở dữ liệu nhẹ, không cần cài đặt server DB phức tạp.

### Frontend (JavaScript)
*   **ReactJS (Vite):** Tốc độ khởi động nhanh, trải nghiệm SPA (Single Page App).
*   **TailwindCSS:** Mọi style đều được viết bằng utility classes tiện lợi.
*   **Axios:** Giao tiếp HTTP API.

---

## ⚙️ Cài đặt & Chạy dự án

### 1. Yêu cầu hệ thống
*   **Python:** 3.8 trở lên.
*   **Node.js:** 16 trở lên (Recommended: v18+).
*   **GPU (Optional):** NVIDIA GPU + CUDA để đạt tốc độ nhận diện <50ms (Nếu không có sẽ chạy CPU vẫn ổn định).

### 2. Cài đặt Backend
```bash
# Di chuyển vào thư mục gốc dự án
cd /path/to/detection-face

# Tạo môi trường ảo (khuyên dùng)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Cài đặt thư viện
pip install -r requirements.txt
```

### 3. Cài đặt Frontend
```bash
# Di chuyển vào thư mục frontend
cd frontend

# Cài đặt gói npm
npm install
```

---

## ▶️ Hướng dẫn Chạy (Run)

Bạn cần mở **2 Terminal** riêng biệt để chạy song song Backend và Frontend.

**Terminal 1: Chạy Backend (API Server)**
```bash
cd /path/to/detection-face
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
*Backend sẽ chạy tại: `http://localhost:8000`*

**Terminal 2: Chạy Frontend (Giao diện)**
```bash
cd /path/to/detection-face/frontend
npm run dev
```
*Frontend sẽ chạy tại: `http://localhost:3000`*

---

## 📖 Hướng dẫn Sử dụng nhanh

1.  **Truy cập:** Mở trình duyệt vào `http://localhost:3000`.
2.  **Đăng ký Mới:** Chọn "Đăng ký", điền thông tin và thực hiện quét khuôn mặt lần đầu (giữ khuôn mặt trong khung xanh).
3.  **Đăng nhập:** Dùng User/Pass vừa tạo.
4.  **Chấm công:**
    *   Tại màn hình chính, nhấn nút **"⚡ Chế độ Live Stream"**.
    *   Hệ thống sẽ bật Camera và tự động nhận diện.
    *   Khi hiện thông báo **"Thành công"** (Khung xanh lá), bạn đã chấm công xong!

---

## 📂 Cấu trúc thư mục

```
/detection-face
├── api/                # Các API Endpoints (Auth, Checkin, Face CRUD)
├── models/             # Core Logic (Detector, Recognizer, Session Manager)
├── streaming/          # Xử lý luồng Video WebSocket
├── database/           # SQLite (faces.db, checkins.db)
├── frontend/           # Source code ReactJS
│   ├── src/
│   │   ├── components/ # Các thành phần UI (LiveCamera, Dashboard...)
│   │   └── ...
├── main.py             # File khởi chạy chính
└── config.py           # Cấu hình hệ thống (Device, Threshold, Paths...)
```