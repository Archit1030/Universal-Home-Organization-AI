import cv2
import json

# Load image and data
image = cv2.imread("../phase2_detection-integration/room.jpeg")  # Update path as needed
with open("../phase3_result-parsing-display/detected_objects.json", "r") as f:
    objects = json.load(f)

# Draw bounding boxes
for obj in objects:
    x1, y1, x2, y2 = map(int, obj["bbox"])
    label = f"{obj['name']} ({obj['confidence']:.2f})"
    
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the image
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
