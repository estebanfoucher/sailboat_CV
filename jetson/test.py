from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train-03-nano/weights/best.engine")

# Validate the model
metrics = model.val(data='data/splitted_datasets/pennon-label-yolo-03/data.yaml', split='val', imgsz=1080, conf=0.25)
print(metrics.box.map)  # map50-95