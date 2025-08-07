import json

from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Run inference
results = model("room.jpeg", save=True)

# Process results
for result in results:
    boxes = result.boxes  # bounding boxes
    names = result.names  # class ID to name map

    print("Detected Objects:")
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])         # class ID
        conf = float(boxes.conf[i])        # confidence score
        xyxy = boxes.xyxy[i].tolist()      # bounding box coordinates
        name = names[cls_id]               # class name

        print(f"- {name}: {xyxy}, Confidence: {conf:.2f}")
object_data = []

for i in range(len(boxes)):
    cls_id = int(boxes.cls[i])
    conf = float(boxes.conf[i])
    xyxy = boxes.xyxy[i].tolist()
    name = names[cls_id]

    object_data.append({
        "name": name,
        "bbox": xyxy,
        "confidence": conf
    })

# Save to JSON
with open("detected_objects.json", "w") as f:
    json.dump(object_data, f, indent=4)

print("Saved detection data to detected_objects.json")

