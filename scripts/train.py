from ultralytics import YOLO

model = YOLO('yolov8n.yaml')

model.train(
    data='dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    name='car-detection-fixed',
    augment=True
)
