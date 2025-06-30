from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train-03-nano/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="engine", half=True, data='data/splitted_datasets/pennon-label-yolo-03/data.yaml')