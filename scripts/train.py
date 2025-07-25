from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
model.train(data='dataset/data.yaml', epochs=30, imgsz=640, name='yolo-test')
