from ultralytics import YOLO

model = YOLO('models/best.pt')  # use your trained model

results = model('test_image.jpg', save=True)
