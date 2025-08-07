# detect_objects.py

from ultralytics import YOLO
import cv2

# Load the YOLOv8 pretrained model
model = YOLO("D:/MyProjects/universal-home-ai/phase1_object-detection-setup/yolov8n.pt")  # Use 'yolov8n' for fastest inference (n = nano)

# Load your image
image_path = "room.jpeg"
image = cv2.imread(image_path)

# Run object detection
results = model(image)

# Visualize the detection
annotated_frame = results[0].plot()

# Save the result to file
cv2.imwrite("detected_output.jpg", annotated_frame)

# Show the result
cv2.imshow("Detected Objects", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Print detected object names
for box in results[0].boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    print("Detected:", class_name)
