from ultralytics import YOLO

# Load the exported TensorRT model
trt_model = YOLO("runs/detect/train-03/weights/best.engine")

# Run inference
results = trt_model("data/datasets/pennon-label-yolo-03/images/frame_2Ce-CKKCtV4_4.00s.jpg")