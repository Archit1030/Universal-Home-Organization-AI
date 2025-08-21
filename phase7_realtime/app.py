from flask import Flask, render_template, Response
import cv2
import json
from ultralytics import YOLO
from datetime import datetime
from flask_cors import CORS
import webbrowser
import threading
from flask import jsonify
import json, os



app = Flask(__name__)
CORS(app)

@app.route('/detections')
def detections():
    # safe guard: if file exists, return it; else empty
    path = "detected_objects.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"objects": [], "timestamp": None})
# ------------------------------
# Category Mapping
# ------------------------------
CATEGORY_MAP = {
    "bed": "furniture",
    "chair": "furniture",
    "sofa": "furniture",
    "potted plant": "decor",
    "vase": "decor",
    "tv": "electronics",
    "laptop": "electronics",
    "refrigerator": "appliances",
    "microwave": "appliances",
    "oven": "appliances",
    "person": "people",
}

# ------------------------------
# Zone Assignment Function
# ------------------------------
def assign_zone(y_center, frame_height):
    """Assign a zone based on the object's vertical position."""
    if y_center < frame_height / 3:
        return "shelf"
    elif y_center < 2 * frame_height / 3:
        return "table"
    else:
        return "floor"

# ------------------------------
# Load YOLO Model
# ------------------------------
model = YOLO("yolov8n.pt")

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ------------------------------
# Frame generator for video streaming
# ------------------------------
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]

        # Run YOLO inference
        results = model(frame)
        detected_objects = []
        annotated_frame = frame.copy()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]

                category = CATEGORY_MAP.get(cls_name, "unknown")
                y_center = (y1 + y2) / 2
                zone = assign_zone(y_center, frame_height)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{cls_name} ({category})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                detected_objects.append({
                    "name": cls_name,
                    "bbox": [x1, y1, x2, y2],
                    "category": category,
                    "zone": zone
                })

        # Save detections to JSON
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "objects": detected_objects
        }
        with open("detected_objects.json", "w") as f:
            json.dump(output_data, f, indent=2)

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ------------------------------
# Flask Routes
# ------------------------------
@app.route('/')
def index():
    return "<h1>YOLO Live Stream</h1><img src='/video_feed'>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------------------
# Auto-open browser
# ------------------------------
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    app.run(host="0.0.0.0", port=5000, debug=False)

