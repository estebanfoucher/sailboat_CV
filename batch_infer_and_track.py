import os
import sys
import shutil
import argparse
import tempfile
import yaml
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np

# Import custom modules from utils/track
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils', 'track'))
from render import draw_tracks
from track import DetectionProcessor, BaseClassMapper

# Supported formats
VIDEO_EXTS = {'.mp4', '.mkv', '.mov'}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.heic'}

CONFIG_PATH = os.path.join('utils', 'track', 'config.yml')
PIPELINE_PATH = os.path.join('utils', 'track', 'pipeline.py')

def is_video(filename):
    return os.path.splitext(filename)[1].lower() in VIDEO_EXTS

def is_image(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTS

def copy_structure_and_get_output_path(input_path, input_root, output_root):
    rel_path = os.path.relpath(input_path, input_root)
    out_path = os.path.join(output_root, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, path):
    with open(path, 'w') as f:
        yaml.safe_dump(config, f)

def run_tracking_on_video(video_path, output_path, config_template, yolo_model_path):
    # Prepare temp config
    with tempfile.NamedTemporaryFile('w', suffix='.yml', delete=False) as tmp:
        config = config_template.copy()
        config['input_video'] = os.path.abspath(video_path)
        config['output_video'] = os.path.abspath(output_path)
        config['yolo_model_path'] = yolo_model_path
        save_config(config, tmp.name)
        tmp_config_path = tmp.name
    print(f"[TRACK] {video_path} -> {output_path}")
    print(f"  Config: {tmp_config_path}")
    # Run pipeline
    os.system(f"python {PIPELINE_PATH} --config {tmp_config_path}")
    # Optionally, remove temp config
    os.remove(tmp_config_path)

def run_inference_on_image(image_path, output_path, yolo_model, detection_processor, class_info, confidence_threshold):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Could not read image: {image_path}")
        return
    results = yolo_model(img)
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        bboxes = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        # Filter by confidence
        keep = confidences >= confidence_threshold
        bboxes = bboxes[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]
        detections = detection_processor.process_detections(bboxes, confidences, class_ids)
    else:
        detections = []
    rendered = draw_tracks(img.copy(), detections, class_info, show_confidence=True, show_class_name=True)
    cv2.imwrite(output_path, rendered)
    print(f"[INFER] {image_path} -> {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch tracking and image inference script.")
    parser.add_argument('--input', required=True, help='Input folder with videos and images')
    parser.add_argument('--output', required=True, help='Output folder for rendered results')
    parser.add_argument('--confidence', type=float, default=None, help='Confidence threshold for image inference')
    parser.add_argument('--yolo_model', type=str, default=None, help='Path to YOLO model (overrides config)')
    args = parser.parse_args()

    input_folder = os.path.abspath(args.input)
    output_folder = os.path.abspath(args.output)
    os.makedirs(output_folder, exist_ok=True)

    # Load config template
    config_template = load_config(CONFIG_PATH)
    yolo_model_path = args.yolo_model or config_template['yolo_model_path']

    # Ask for confidence threshold if not provided
    confidence_threshold = args.confidence
    if confidence_threshold is None:
        try:
            confidence_threshold = float(input('Enter confidence threshold for image inference (e.g. 0.25): '))
        except Exception:
            print('Invalid input. Using default 0.25.')
            confidence_threshold = 0.25

    # Prepare YOLO model and detection processor for images
    yolo_model = YOLO(yolo_model_path)
    base_mapper = BaseClassMapper()
    detection_processor = DetectionProcessor(base_mapper)
    class_info = {k: {'name': v['name'], 'color': tuple(v['color'])} for k, v in config_template['class_info'].items()}

    # Walk input folder
    for root, dirs, files in os.walk(input_folder):
        for fname in files:
            in_path = os.path.join(root, fname)
            if is_video(fname):
                out_path = copy_structure_and_get_output_path(in_path, input_folder, output_folder)
                run_tracking_on_video(in_path, out_path, config_template, yolo_model_path)
            elif is_image(fname):
                out_path = copy_structure_and_get_output_path(in_path, input_folder, output_folder)
                run_inference_on_image(in_path, out_path, yolo_model, detection_processor, class_info, confidence_threshold)
            else:
                # Optionally, copy other files
                pass

if __name__ == '__main__':
    main() 