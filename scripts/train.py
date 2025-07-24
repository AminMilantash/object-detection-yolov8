from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.yaml')  # or 'yolov8n.pt' for pretrained

# Train model
model.train(data='dataset/data.yaml', epochs=50, imgsz=640)
