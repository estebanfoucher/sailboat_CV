from ultralytics import YOLO

model = YOLO('runs/detect/train-03/weights/best.pt')
print(model.model.args)        # YOLOv8
print(model.model.yaml)        # Contains imgsz if available