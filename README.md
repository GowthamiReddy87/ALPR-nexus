# 🔍 ALPR NEXUS — Automatic License Plate Recognition System

> **Hackathon-grade full-stack ALPR platform** with real-time image processing, multi-condition detection, live camera support, and an analytics dashboard.

---

## 🏗 Architecture

```
alpr-system/
├── backend/
│   ├── app.py              # Flask REST API + OpenCV ALPR engine
│   └── requirements.txt
└── frontend/
    └── index.html          # Single-page app (no build step needed)
```

## 🚀 Quick Start

### 1. Install Python dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the backend server
```bash
python app.py
# Server starts at http://localhost:5050
```

### 3. Open the frontend
```bash
# Option A: Open directly
open ../frontend/index.html

# Option B: Serve with Python
cd ../frontend
python3 -m http.server 3000
# Visit http://localhost:3000
```

---

## 🔧 API Reference

| Method | Endpoint              | Description                            |
|--------|-----------------------|----------------------------------------|
| GET    | `/api/health`         | Server health check                    |
| POST   | `/api/detect`         | Detect plates in base64 image          |
| POST   | `/api/video`          | Process uploaded video file            |
| POST   | `/api/video/stream`   | Real-time video frame processing       |
| GET    | `/api/logs`           | Fetch detection log (paginated)        |
| POST   | `/api/logs/clear`     | Clear all logs                         |
| GET    | `/api/stats`          | Aggregated analytics                   |
| POST   | `/api/simulate`       | Simulate a detection (demo)            |

### POST /api/detect
```json
{
  "image": "data:image/jpeg;base64,...",
  "region": "Toll Plaza A",
  "lane": "Lane-2"
}
```
**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "plate_number": "MH12AB1234",
      "confidence": 0.92,
      "bbox": [120, 80, 310, 85],
      "timestamp": "2024-01-01T12:00:00",
      "lighting_condition": "daytime",
      "image_quality": "Good",
      "vehicle_speed": "78 km/h",
      "lane": "Lane-2",
      "blur_score": 312.5,
      "brightness": 145.8
    }
  ],
  "annotated_image": "data:image/jpeg;base64,...",
  "processing_time_ms": 112
}
```
### POST /api/video
Upload and process a video file for license plate detection.
```json
{
  "video": "multipart/form-data file"
}
```
**Response:**
```json
{
  "success": true,
  "detections": [...],
  "total_frames": 150,
  "processed_frames": 30,
  "video_stats": {
    "night_frames": 45,
    "blurry_frames": 12,
    "night_percentage": 30.0,
    "blur_percentage": 8.0
  }
}
```

### POST /api/video/stream
Process a batch of video frames in real-time.
```json
{
  "frames": ["data:image/jpeg;base64,...", ...]
}
```
**Response:**
```json
{
  "success": true,
  "detections": [...],
  "stats": {
    "total_frames": 10,
    "processed_frames": 10,
    "night_frames": 3,
    "blurry_frames": 1,
    "avg_brightness": 85.2,
    "avg_blur_score": 245.8,
    "night_percentage": 30.0,
    "blur_percentage": 10.0
  }
}
```

---

## 🧠 Detection Pipeline

```
Input Image
    │
    ▼
┌─────────────────────┐
│  PREPROCESSING      │  CLAHE + FastNlMeans denoise
└──────────┬──────────┘
           │
    ┌──────▼──────┐
    │  STAGE 1    │  Canny edges → Morphological close → Contour filter
    │  (Edges)    │  Aspect ratio: 1.5–6.5, area > 800px²
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  STAGE 2    │  Adaptive threshold → Vertical edge filter
    │  (Chars)    │  Character clustering into plate regions
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  NMS        │  IoU > 0.3 → deduplicate candidates
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  DESKEW     │  Hough lines → rotation correction
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  OCR        │  3 enhancement versions → char segmentation
    │  ENGINE     │  Vertical projection → char count → plate format
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  SCORING    │  Confidence = region_score + char_count_score
    └─────────────┘
```

---

## 🎯 Challenges Addressed

| Challenge               | Solution                                               |
|-------------------------|--------------------------------------------------------|
| Motion blur             | Laplacian variance detection + sharpening filter       |
| Low light / nighttime   | Brightness analysis + CLAHE enhancement               |
| Glare / overexposure    | Adaptive thresholding + quality assessment            |
| Video processing        | Frame-by-frame analysis + batch streaming             |
| Real-time streaming     | Buffer-based frame processing + stats aggregation     |
| Skewed angles           | Hough-based deskew correction                          |
| Multiple vehicles       | NMS deduplication, top-5 candidates                    |
| Different plate formats | Flexible char-count OCR (4–10 chars)                   |
| Real-time constraints   | Optimized OpenCV pipeline (~50–180ms per frame)        |

---

## 🖥 Frontend Features

- **Image/Video upload** (drag & drop, multi-format support)
- **Live camera** with auto-capture and real-time streaming
- **Video processing** — batch analysis with quality metrics
- **Annotated output** with bounding boxes, confidence scores
- **Detection log** with CSV export and enhanced metadata
- **Analytics dashboard** — lighting breakdown, blur stats, unique plates
- **Simulation mode** — test daytime/nighttime/blur/rain scenarios
- **Live terminal** — real-time system logs with processing stats
- **Responsive** — works on mobile browsers too

---

## 📦 Tech Stack

| Layer     | Technology                         |
|-----------|------------------------------------|
| Backend   | Python 3.10+, Flask, OpenCV 4.x    |
| CV Engine | OpenCV (Canny, CLAHE, Hough, NMS)  |
| Frontend  | Vanilla HTML/CSS/JS (no framework) |
| Fonts     | Orbitron, Rajdhani, Share Tech Mono|
| Protocol  | REST JSON API over HTTP            |

---

## 🏆 Hackathon Highlights

1. **Zero external API dependencies** — fully offline once installed
2. **Multi-stage detection** — combines edge + character-region approaches
3. **Production-grade UI** — sci-fi terminal aesthetic, real-time updates
4. **Extensible** — drop in EasyOCR or PaddleOCR for production-grade OCR
5. **Camera integration** — works with webcam or phone camera via browser

---

## 🔮 Upgrade Path (Production)

To deploy at a real toll plaza:
1. Replace `synthesize_plate_text_from_image()` with **EasyOCR** or **PaddleOCR**
2. Add **YOLOv8** for vehicle + plate localization
3. Connect to a **PostgreSQL** database for persistent logging
4. Add **WebSocket** (`socket.io`) for true real-time streaming
5. Deploy backend on **Gunicorn + Nginx**

---

*Built for the hackathon — ALPR NEXUS v2.0*
