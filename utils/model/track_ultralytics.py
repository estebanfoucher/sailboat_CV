from ultralytics import YOLO

model = YOLO('runs/detect/train-03/weights/best.engine')

source='data/videos/example/2Ce-CKKCtV4_35.0_40.0_fps25.mkv'


results = model.track(source=source, show=True, tracker='trackers/bytetrack.yaml', save=True, conf=0.45)