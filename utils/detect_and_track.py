from ultralytics import YOLO

model = YOLO('docker/yolo/models/custom-03.pt')

results = model.track(source='data/videos/2Ce-CKKCtV4_38.0_39.0.mkv', show=True, tracker='trackers/bytetrack.yaml', save=True, conf=0.45)