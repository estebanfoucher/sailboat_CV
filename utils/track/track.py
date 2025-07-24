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
from byte_tracker import ByteTracker
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

def init_tracker(fps, bytetrack_params):
    tracker = ByteTracker(
        frame_rate=fps,
        track_thresh=bytetrack_params.get('track_thresh'),
        track_buffer=bytetrack_params.get('track_buffer'),
        match_thresh=bytetrack_params.get('match_thresh')
    )
    return tracker

def load_model(model_path, archi='yolo'):
    if archi=='yolo':
        return YOLO(model_path)
    if archi=='rt-detr':
        return RTDETR(model_path)
    return None

def init_writer(output_video, fps, frame_size):
    return open_video_writer(output_video, fps, frame_size)


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


def format_inference_results(results):
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        bboxes = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        
        return bboxes, confidences, class_ids
    else:
        return None, None, None

def process_detections(bboxes: np.ndarray, 
                          confidences: np.ndarray, 
                          class_ids: np.ndarray):
        """
        Convert raw detection arrays to structured format.
        
        Args:
            bboxes: (N, 4) array of bounding boxes [x1, y1, x2, y2]
            confidences: (N,) array of confidence scores
            class_ids: (N,) array of class IDs
        """
        detections = []
        
        for i in range(len(bboxes)):
            detection = {
                'bbox': bboxes[i],
                'confidence': confidences[i],
                'class_id': int(class_ids[i]),
                'original_class_id': int(class_ids[i])
            }
            detections.append(detection)
        
        return detections

def process_frame(frame, model, tracker, class_info):
    results = model(frame)
    bboxes, confidences, class_ids = format_inference_results(results)
    tracks = tracker.update(process_detections(bboxes, confidences, class_ids))
    rendered = draw_tracks(frame, tracks, class_info)
    return {'rendered':rendered, 'tracks':tracks}


def run_tracking_pipeline(config: Dict[str, Any]) -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    config = resolve_all_paths(config, project_root)
    fps, frame_size, frame_count = get_video_properties(config['input_video'])
    writer = init_writer(config['output_video'], fps, frame_size)
    logger.debug(f"[Pipeline] Video FPS detected: {fps}")
    model = load_model(config['model_path'], config['archi'])
    tracker = init_tracker(fps, config['bytetrack_params'])
    class_info = config['class_info']
    tracks_timeline = []
    
    for i, frame in enumerate(tqdm(read_video_frames(config['input_video']), total=frame_count, desc="Tracking video")):
        result_frame = process_frame(frame, model, tracker, class_info)
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