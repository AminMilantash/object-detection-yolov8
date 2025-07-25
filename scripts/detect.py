from ultralytics import YOLO
import sys
import matplotlib.pyplot as plt

MODEL_PATH = 'runs/detect/yolov8-vehicles/weights/best.pt'

if len(sys.argv) > 1:
    IMAGE_PATH = sys.argv[1]
else:
    IMAGE_PATH = 'test.jpg'

# Load model
model = YOLO(MODEL_PATH)

# Run prediction
results = model(IMAGE_PATH, save=True, conf=0.25)

# Print number of detections
print(f'Detections: {len(results[0].boxes)}')

# Show image with bounding boxes
img_with_boxes = results[0].plot()
plt.imshow(img_with_boxes)
plt.axis('off')
plt.show()
