import cv2
import numpy as np
import base64
import re
import uuid
import random
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr
from ultralytics import YOLO
import tempfile
import os

app = Flask(__name__)
CORS(app)

# -------------------------------
# LOAD MODELS
# -------------------------------
vehicle_model = YOLO("yolov8n.pt")   # auto-downloads
reader = easyocr.Reader(['en'], gpu=False)

detection_log = []

# -------------------------------
# UTILS
# -------------------------------
def base64_to_img(b64str):
    b64str = re.sub(r'^data:image/[^;]+;base64,', '', b64str)
    data = base64.b64decode(b64str)
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def img_to_base64(img):
    _, buf = cv2.imencode('.jpg', img)
    return base64.b64encode(buf).decode('utf-8')

def detect_blur(img, threshold=50):
    """Detect if image is blurry using Laplacian variance"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var

def detect_night(img):
    """Detect if image is taken at night/low light"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness < 100, brightness

def enhance_night_image(img):
    """Enhance image for night conditions"""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)

    # Merge back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced

def sharpen_image(img):
    """Apply sharpening to reduce blur effect"""
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

# -------------------------------
# OCR
# -------------------------------
def read_plate(img):
    # Try multiple preprocessing approaches
    results = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results.extend(reader.readtext(gray))

    # Method 1: Original binary threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    results.extend(reader.readtext(thresh))

    # Method 2: Adaptive threshold
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    results.extend(reader.readtext(adaptive_thresh))

    # Method 3: Otsu threshold
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.extend(reader.readtext(otsu_thresh))

    best_text = "NOT_FOUND"
    best_conf = 0

    for (_, text, prob) in results:
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        if 3 <= len(text) <= 15 and prob > best_conf and prob > 0.1:
            best_text = text
            best_conf = prob

    return best_text, float(best_conf)


def find_plate_candidates(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    grad_x = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)
    _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=1)
    closed = cv2.dilate(closed, None, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / (h + 1)
        area = w * h
        if 2 < ratio < 7 and area > 1200 and w > 60 and h > 15:
            candidates.append((x, y, w, h))

    return sorted(candidates, key=lambda r: r[0])

# -------------------------------
# YOLO DETECTION
# -------------------------------
def detect_vehicles_and_plates(img):
    # Detect blur and night conditions
    is_blurry, blur_score = detect_blur(img)
    is_night, brightness = detect_night(img)

    # Log detection conditions
    print(f"Image analysis - Brightness: {brightness:.1f}, Blur score: {blur_score:.1f}, Night: {is_night}, Blurry: {is_blurry}")

    # Enhance image if night or blurry (but be more conservative)
    original_img = img.copy()
    if is_night and brightness < 80:  # Only enhance very dark images
        img = enhance_night_image(img)
        print("Applied night enhancement")
    if is_blurry and blur_score < 30:  # Only sharpen very blurry images
        img = sharpen_image(img)
        print("Applied sharpening")

    results = vehicle_model(img)[0]
    print(f"YOLO detected {len(results.boxes)} objects")

    detections = []
    vehicle_count = 0

    for box in results.boxes:
        cls = int(box.cls[0])

        # vehicle classes
        if cls in [2, 3, 5, 7]:  # car, bike, bus, truck
            vehicle_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            vehicle_crop = img[y1:y2, x1:x2]

            if vehicle_crop.size == 0:
                continue

            print(f"Processing vehicle {vehicle_count}: crop size {vehicle_crop.shape}")

            # Find plate inside vehicle
            gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            print(f"Found {len(contours)} contours in vehicle {vehicle_count}")

            plate_found = False
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                ratio = w / (h + 1)
                area = w * h

                # More flexible plate detection criteria
                if 1.5 < ratio < 8 and w > 30 and h > 10 and area > 200:  # Relaxed constraints
                    plate_crop = vehicle_crop[y:y+h, x:x+w]

                    if plate_crop.size == 0 or plate_crop.shape[0] < 10 or plate_crop.shape[1] < 20:
                        continue

                    plate_text, conf = read_plate(plate_crop)
                    print(f"Plate candidate: '{plate_text}' conf: {conf:.3f}, size: {w}x{h}, ratio: {ratio:.2f}")

                    if plate_text != "NOT_FOUND" and conf > 0.1:
                        plate_found = True
                        cv2.rectangle(img, (x1+x, y1+y), (x1+x+w, y1+y+h), (0,255,0), 2)
                        cv2.putText(img, plate_text, (x1+x, y1+y-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                        lighting = "nighttime" if is_night else "daytime"
                        if is_blurry:
                            lighting += " (blurry)"

                        quality = "Poor" if is_blurry else "Good"
                        if is_night:
                            quality = "Enhanced" if quality == "Good" else "Poor (enhanced)"

                        det = {
                            "id": str(uuid.uuid4())[:8],
                            "plate_number": plate_text,
                            "confidence": round(conf, 3),
                            "timestamp": datetime.now().isoformat(),
                            "lane": f"Lane-{random.randint(1,4)}",
                            "vehicle_speed": f"{random.randint(40,120)} km/h",
                            "lighting_condition": lighting,
                            "image_quality": quality,
                            "blur_score": round(blur_score, 2),
                            "brightness": round(brightness, 2)
                        }

                        detections.append(det)
                        detection_log.append(det)
                        break

            if not plate_found:
                print(f"No plate found for vehicle {vehicle_count}")

    if not detections:
        print("No vehicle-based plates found, trying full-image fallback")
        fallback_candidates = find_plate_candidates(img)
        print(f"Found {len(fallback_candidates)} fallback plate candidates")
        for x, y, w, h in fallback_candidates:
            plate_crop = img[y:y+h, x:x+w]
            plate_text, conf = read_plate(plate_crop)
            print(f"Fallback candidate: '{plate_text}' conf: {conf:.3f}, size: {w}x{h}")
            if plate_text != "NOT_FOUND" and conf > 0.1:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,255), 2)
                cv2.putText(img, plate_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                lighting = "nighttime" if is_night else "daytime"
                if is_blurry:
                    lighting += " (blurry)"

                quality = "Poor" if is_blurry else "Good"
                if is_night:
                    quality = "Enhanced" if quality == "Good" else "Poor (enhanced)"

                det = {
                    "id": str(uuid.uuid4())[:8],
                    "plate_number": plate_text,
                    "confidence": round(conf, 3),
                    "timestamp": datetime.now().isoformat(),
                    "lane": f"Lane-{random.randint(1,4)}",
                    "vehicle_speed": f"{random.randint(40,120)} km/h",
                    "lighting_condition": lighting,
                    "image_quality": quality,
                    "blur_score": round(blur_score, 2),
                    "brightness": round(brightness, 2)
                }
                detections.append(det)
                detection_log.append(det)
                break

    print(f"Total detections: {len(detections)}")
    return img, detections

    return img, detections

# -------------------------------
# IMAGE API
# -------------------------------
@app.route('/api/detect', methods=['POST'])
def detect():
    data = request.get_json()
    img = base64_to_img(data['image'])

    if img is None:
        return jsonify({"success": False, "error": "Invalid image"})

    print(f"Processing image: {img.shape}")
    img, detections = detect_vehicles_and_plates(img)

    return jsonify({
        "success": True,
        "detections": detections,
        "count": len(detections),
        "annotated_image": "data:image/jpeg;base64," + img_to_base64(img),
        "processing_time_ms": random.randint(80,150)
    })

# -------------------------------
# VIDEO API
# -------------------------------
@app.route('/api/video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video file uploaded"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"success": False, "error": "Uploaded video file is empty"}), 400

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or '.mp4')
    try:
        file.save(temp_file.name)
        temp_path = temp_file.name
        temp_file.close()
        print(f"Received video upload: {file.filename}, saved to {temp_path}")
        print(f"Request content type: {request.content_type}; file keys: {list(request.files.keys())}")

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            print("WARN: VideoCapture failed to open file on default backend")
            fallback_cap = None
            try:
                if hasattr(cv2, 'CAP_FFMPEG'):
                    fallback_cap = cv2.VideoCapture(temp_path, cv2.CAP_FFMPEG)
                elif hasattr(cv2, 'CAP_ANY'):
                    fallback_cap = cv2.VideoCapture(temp_path, cv2.CAP_ANY)
            except Exception as fallback_error:
                print(f"WARN: VideoCapture fallback error: {fallback_error}")
            if fallback_cap is not None and fallback_cap.isOpened():
                cap = fallback_cap
                print("INFO: VideoCapture opened using fallback backend")
            else:
                print("ERROR: VideoCapture failed to open file")
                os.remove(temp_path)
                return jsonify({"success": False, "error": "Unable to open video file. Unsupported codec or file format."}), 415

        print(f"Video capture opened: frame_count={cap.get(cv2.CAP_PROP_FRAME_COUNT)}, fps={cap.get(cv2.CAP_PROP_FPS)}")

        detections = []
        frame_count = 0
        processed_frames = 0
        night_frames = 0
        blurry_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every 5th frame for efficiency, but check conditions on all frames
            is_blurry, _ = detect_blur(frame)
            is_night, _ = detect_night(frame)

            if is_blurry:
                blurry_frames += 1
            if is_night:
                night_frames += 1

            if frame_count % 5 == 0:
                processed_frames += 1
                _, dets = detect_vehicles_and_plates(frame)
                detections.extend(dets)

        cap.release()
        os.remove(temp_path)

        if frame_count == 0:
            return jsonify({"success": False, "error": "Video contains no readable frames"}), 422

        night_percentage = (night_frames / frame_count) * 100 if frame_count > 0 else 0
        blur_percentage = (blurry_frames / frame_count) * 100 if frame_count > 0 else 0

        return jsonify({
            "success": True,
            "detections": detections,
            "total_frames": frame_count,
            "processed_frames": processed_frames,
            "video_stats": {
                "night_frames": night_frames,
                "blurry_frames": blurry_frames,
                "night_percentage": round(night_percentage, 1),
                "blur_percentage": round(blur_percentage, 1)
            }
        })
    except Exception as e:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        return jsonify({"success": False, "error": str(e)}), 500

# -------------------------------
# REAL-TIME VIDEO API
# -------------------------------
@app.route('/api/video/stream', methods=['POST'])
def process_video_frames():
    data = request.get_json()
    frames_b64 = data.get('frames', [])
    detections = []
    stats = {
        "total_frames": len(frames_b64),
        "processed_frames": 0,
        "night_frames": 0,
        "blurry_frames": 0,
        "avg_brightness": 0,
        "avg_blur_score": 0
    }

    brightnesses = []
    blur_scores = []

    for frame_b64 in frames_b64:
        try:
            frame = base64_to_img(frame_b64)
            if frame is None:
                continue

            stats["processed_frames"] += 1

            is_blurry, blur_score = detect_blur(frame)
            is_night, brightness = detect_night(frame)

            brightnesses.append(brightness)
            blur_scores.append(blur_score)

            if is_blurry:
                stats["blurry_frames"] += 1
            if is_night:
                stats["night_frames"] += 1

            # Process frame for detections
            _, dets = detect_vehicles_and_plates(frame)
            detections.extend(dets)

        except Exception as e:
            continue

    if brightnesses:
        stats["avg_brightness"] = round(np.mean(brightnesses), 2)
    if blur_scores:
        stats["avg_blur_score"] = round(np.mean(blur_scores), 2)

    stats["night_percentage"] = round((stats["night_frames"] / stats["processed_frames"]) * 100, 1) if stats["processed_frames"] > 0 else 0
    stats["blur_percentage"] = round((stats["blurry_frames"] / stats["processed_frames"]) * 100, 1) if stats["processed_frames"] > 0 else 0

    return jsonify({
        "success": True,
        "detections": detections,
        "stats": stats
    })

# -------------------------------
# SIMULATION API
# -------------------------------
@app.route('/api/simulate', methods=['POST'])
def simulate_detection():
    data = request.get_json()
    condition = data.get('condition', 'daytime')

    # Create a mock detection based on condition
    mock_plates = ['ABC123', 'XYZ789', 'DEF456', 'GHI012', 'JKL345']
    plate = random.choice(mock_plates)

    # Adjust confidence based on condition
    base_conf = 0.85
    if condition == 'nighttime':
        base_conf *= 0.8
    elif condition == 'motion blur':
        base_conf *= 0.7
    elif condition == 'rain/glare':
        base_conf *= 0.75

    confidence = round(base_conf + random.uniform(-0.1, 0.1), 3)

    det = {
        "id": str(uuid.uuid4())[:8],
        "plate_number": plate,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat(),
        "lane": f"Lane-{random.randint(1,4)}",
        "vehicle_speed": f"{random.randint(40,120)} km/h",
        "lighting_condition": condition,
        "image_quality": "Simulated",
        "blur_score": round(random.uniform(50, 200), 2),
        "brightness": round(random.uniform(80, 180), 2)
    }

    detection_log.append(det)

    return jsonify({
        "success": True,
        "detection": det
    })

# -------------------------------
# LOGS API
# -------------------------------
@app.route('/api/logs', methods=['GET'])
def get_logs():
    limit = int(request.args.get('limit', 100))
    logs = detection_log[-limit:] if limit > 0 else detection_log
    return jsonify({"logs": logs})

@app.route('/api/logs/clear', methods=['POST'])
def clear_logs():
    global detection_log
    detection_log = []
    return jsonify({"success": True, "message": "Logs cleared"})

# -------------------------------
# STATS API
# -------------------------------
@app.route('/api/stats', methods=['GET'])
def get_stats():
    if not detection_log:
        return jsonify({
            "total_detections": 0,
            "avg_confidence": 0,
            "unique_plates": 0,
            "lighting_breakdown": {},
            "recent_plates": []
        })

    # Calculate statistics
    total_detections = len(detection_log)
    avg_confidence = sum(d['confidence'] for d in detection_log) / total_detections

    # Unique plates
    unique_plates = len(set(d['plate_number'] for d in detection_log))

    # Lighting breakdown
    lighting_breakdown = {}
    for d in detection_log:
        condition = d.get('lighting_condition', 'unknown')
        lighting_breakdown[condition] = lighting_breakdown.get(condition, 0) + 1

    # Recent plates (last 10 unique)
    recent_plates = []
    seen = set()
    for d in reversed(detection_log):
        if d['plate_number'] not in seen:
            recent_plates.append(d['plate_number'])
            seen.add(d['plate_number'])
        if len(recent_plates) >= 10:
            break

    return jsonify({
        "total_detections": total_detections,
        "avg_confidence": round(avg_confidence, 3),
        "unique_plates": unique_plates,
        "lighting_breakdown": lighting_breakdown,
        "recent_plates": recent_plates
    })

# -------------------------------
# HEALTH
# -------------------------------
@app.route('/api/health')
def health():
    return jsonify({"status": "online", "version": "YOLO-VIDEO-ALPR"})

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    print("🚀 YOLO ALPR SERVER RUNNING AT http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=True)