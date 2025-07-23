import argparse
from typing import Dict, Any
import yaml
import os
import json
import numpy as np
from tqdm import tqdm
import cv2
from ultralytics import YOLO, RTDETR
from video_io import get_video_properties, read_video_frames, open_video_writer
from render import draw_tracks
from tracking import DetectionProcessor, BaseClassMapper, ByteTrackWrapper
from byte_tracker import BYTETracker
from loguru import logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load pipeline configuration from a YAML file.
    Args:
        config_path (str): Path to the YAML config file.
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def resolve_all_paths(config, project_root):
    def resolve_path(path):
        return path if os.path.isabs(path) else os.path.abspath(os.path.join(project_root, path))
    config['input_video'] = resolve_path(config['input_video'])
    config['output_video'] = resolve_path(config['output_video'])
    config['model_path'] = resolve_path(config['model_path'])
    return config


def load_model(model_path, archi='yolo'):
    if archi=='yolo':
        return YOLO(model_path)
    if archi=='rt-detr':
        return RTDETR(model_path)
    return None


def init_detection_processor(base_class_mapping):
    base_mapper = BaseClassMapper()
    base_mapper.class_mapping = base_class_mapping
    return DetectionProcessor(base_mapper), base_mapper


def init_tracker(fps, bytetrack_params, base_mapper):
    tracker = BYTETracker(
        frame_rate=fps,
        track_thresh=bytetrack_params.get('track_thresh', 0.2),
        track_buffer=bytetrack_params.get('track_buffer', 30),
        match_thresh=bytetrack_params.get('match_thresh', 0.8)
    )
    return ByteTrackWrapper(tracker, base_mapper)


def init_writer(output_video, fps, frame_size):
    return open_video_writer(output_video, fps, frame_size)


def init_class_info(config):
    # Enforce coloring convention
    return {
        0: {"name": config['class_info'][0]['name'], "color": (0, 255, 0)},   # green
        1: {"name": config['class_info'][1]['name'], "color": (0, 0, 255)},   # red
        2: {"name": config['class_info'][2]['name'], "color": (255, 0, 0)},   # blue
    }


def make_json_serializable(obj):
    """Convert numpy arrays to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    return obj


def save_tracks_timeline(tracks_timeline, output_video_path):
    """Save tracks timeline as JSON with same name as video."""
    base_path = os.path.splitext(output_video_path)[0]
    tracks_path = f"{base_path}_tracks_timeline.json"
    
    serializable_tracks = make_json_serializable(tracks_timeline)
    with open(tracks_path, 'w') as f:
        json.dump(serializable_tracks, f, indent=2)
    
    logger.info(f"Tracks timeline saved to {tracks_path}")


def run_yolo_detection(model, frame):
    """
    Run YOLO detection on a frame and return bboxes, confidences, class_ids, and class_probs.
    Args:
        model: YOLO model.
        frame (np.ndarray): Input frame (BGR).
    Returns:
        bboxes (np.ndarray): (N, 4) array of [x1, y1, x2, y2].
        confidences (np.ndarray): (N,) array of confidence scores.
        class_ids (np.ndarray): (N,) array of class IDs.
        class_probs (np.ndarray): (N, num_classes) array of class probabilities.
    """
    results = model(frame)
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        bboxes = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        # Extract class probabilities from boxes.data if available
        raw = boxes.data.cpu().numpy()
        print(f"boxes.data shape: {raw.shape}")
        if raw.shape[1] > 6:
            class_probs = raw[:, 6:]
        else:
            class_probs = None
        return bboxes, confidences, class_ids, class_probs
    else:
        return None, None, None, None


def process_frame(frame, model, detection_processor, tracker_wrapper, class_info,  frame_idx):
    bboxes, confidences, class_ids, class_probs = run_yolo_detection(model, frame)
    detections = []
    if bboxes is not None:
        detections = detection_processor.process_detections(bboxes, confidences, class_ids)
    tracks = tracker_wrapper.update(detections)
    rendered = draw_tracks(frame, tracks, class_info)
    return {'rendered':rendered, 'tracks':tracks}


def run_tracking_pipeline(config: Dict[str, Any]) -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    config = resolve_all_paths(config, project_root)
    fps, frame_size, frame_count = get_video_properties(config['input_video'])
    writer = init_writer(config['output_video'], fps, frame_size)
    logger.debug(f"[Pipeline] Video FPS detected: {fps}")
    model = load_model(config['model_path'], config['archi'])
    detection_processor, base_mapper = init_detection_processor(config['base_class_mapping'])
    tracker_wrapper = init_tracker(fps, config.get('bytetrack_params', {}), base_mapper)
    class_info = init_class_info(config)
    tracks_timeline = []
    for i, frame in enumerate(tqdm(read_video_frames(config['input_video']), total=frame_count, desc="Tracking video")):
        result_frame = process_frame(frame, model, detection_processor, tracker_wrapper, class_info, i)
        writer.write(result_frame['rendered'])
        tracks_timeline.append(result_frame['tracks'])
    writer.release()
    
    # Save tracks timeline data
    save_tracks_timeline(tracks_timeline, config['output_video'])
    
    logger.info(f"Tracked video written to {config['output_video']}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + ByteTrack video tracking pipeline (YAML config)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    run_tracking_pipeline(config) 