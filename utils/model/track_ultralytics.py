from ultralytics import YOLO

model = YOLO('runs/detect/train-03/weights/best.engine')

results = model.track(source='data/videos/example/2Ce-CKKCtV4_35.0_40.0.mkv', show=True, tracker='trackers/bytetrack.yaml', save=True, conf=0.45)