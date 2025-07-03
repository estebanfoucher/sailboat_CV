import argparse
from typing import Dict, Any
import yaml
import os
from tqdm import tqdm
import cv2
from ultralytics import YOLO
from video_io import get_video_properties, read_video_frames, open_video_writer
from render import draw_tracks
from track import DetectionProcessor, BaseClassMapper
from byte_tracker import BYTETracker
from track import ByteTrackWrapper


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


def load_yolo_model(model_path: str):
    """
    Load a YOLOv8 model from the given path.
    Args:
        model_path (str): Path to the YOLOv8 .pt model.
    Returns:
        YOLO: Loaded YOLO model.
    """
    return YOLO(model_path)


def run_yolo_detection(model, frame):
    """
    Run YOLO detection on a frame and return bboxes, confidences, class_ids.
    Args:
        model: YOLO model.
        frame (np.ndarray): Input frame (BGR).
    Returns:
        bboxes (np.ndarray): (N, 4) array of [x1, y1, x2, y2].
        confidences (np.ndarray): (N,) array of confidence scores.
        class_ids (np.ndarray): (N,) array of class IDs.
    """
    results = model(frame)
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        bboxes = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        return bboxes, confidences, class_ids
    else:
        return None, None, None


def resolve_path(path, project_root):
    """
    Resolve a path relative to the project root if not already absolute.
    """
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(project_root, path))


def run_tracking_pipeline(
    input_video: str,
    output_video: str,
    yolo_model_path: str,
    class_info: Dict[int, Dict[str, Any]],
    base_class_mapping: Dict[int, int],
    bytetrack_params: Dict[str, Any] = None
) -> None:
    """
    Run the full tracking and rendering pipeline.
    Args:
        input_video (str): Path to the input video file.
        output_video (str): Path to the output video file.
        yolo_model_path (str): Path to the YOLOv8 .pt model.
        class_info (dict): Mapping of class_id to {"name", "color"}.
        base_class_mapping (dict): Mapping of detector class_id to base class for tracking.
        bytetrack_params (dict, optional): Parameters for ByteTrack initialization.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    input_video = resolve_path(input_video, project_root)
    output_video = resolve_path(output_video, project_root)
    yolo_model_path = resolve_path(yolo_model_path, project_root)

    fps, frame_size, frame_count = get_video_properties(input_video)
    writer = open_video_writer(output_video, fps, frame_size)

    # Load YOLO model
    model = load_yolo_model(yolo_model_path)
    # Prepare detection processor and class mapper
    base_mapper = BaseClassMapper()
    base_mapper.class_mapping = base_class_mapping
    detection_processor = DetectionProcessor(base_mapper)

    # Initialize ByteTrack
    tracker = BYTETracker(
        frame_rate=bytetrack_params.get('frame_rate', 25),
        track_thresh=bytetrack_params.get('track_thresh', 0.2),
        track_buffer=bytetrack_params.get('track_buffer', 30),
        match_thresh=bytetrack_params.get('match_thresh', 0.8)
    )
    tracker_wrapper = ByteTrackWrapper(tracker, base_mapper)

    for frame in tqdm(read_video_frames(input_video), total=frame_count, desc="Tracking video"):
        bboxes, confidences, class_ids = run_yolo_detection(model, frame)
        if bboxes is not None:
            detections = detection_processor.process_detections(bboxes, confidences, class_ids)
            tracks = tracker_wrapper.update(detections)
            # Each track should have 'bbox', 'track_id', 'original_class_id', 'confidence'
            rendered = draw_tracks(frame, tracks, class_info, show_confidence=True)
        else:
            rendered = frame
        writer.write(rendered)
    writer.release()
    print(f"Tracked video written to {output_video}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + ByteTrack video tracking pipeline (YAML config)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    # Enforce coloring convention
    class_info = {
        0: {"name": config['class_info'][0]['name'], "color": (0, 255, 0)},   # green
        1: {"name": config['class_info'][1]['name'], "color": (0, 0, 255)},   # red
        2: {"name": config['class_info'][2]['name'], "color": (255, 0, 0)},   # blue
    }
    run_tracking_pipeline(
        input_video=config['input_video'],
        output_video=config['output_video'],
        yolo_model_path=config['yolo_model_path'],
        class_info=class_info,
        base_class_mapping=config['base_class_mapping'],
        bytetrack_params=config.get('bytetrack_params', {})
    ) 