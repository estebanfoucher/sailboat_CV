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
    config['yolo_model_path'] = resolve_path(config['yolo_model_path'])
    return config


def init_yolo_model(yolo_model_path):
    """
    Load a YOLOv8 model from the given path.
    Args:
        model_path (str): Path to the YOLOv8 .pt model.
    Returns:
        YOLO: Loaded YOLO model.
    """
    return YOLO(yolo_model_path)


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


def process_frame(frame, model, detection_processor, tracker_wrapper, class_info, blend_pairs, frame_idx, show_blend_value=False):
    bboxes, confidences, class_ids, class_probs = run_yolo_detection(model, frame)
    if frame_idx < 5:
        logger.debug(f"Frame {frame_idx}: YOLO detections:")
        logger.debug(f"  bboxes: {bboxes}")
        logger.debug(f"  class_ids: {class_ids}")
        if class_probs is not None:
            for i, probs in enumerate(class_probs):
                print(f"  Detection {i} class_probs: {probs}")
    detections = []
    if bboxes is not None:
        detections = detection_processor.process_detections(bboxes, confidences, class_ids)
        # Add blend metrics for each detection
        if class_probs is not None:
            for i, det in enumerate(detections):
                det['blend_metrics'] = {}
                for pair in blend_pairs:
                    a, b = pair
                    pa = class_probs[i, a] if a < class_probs.shape[1] else 0.0
                    pb = class_probs[i, b] if b < class_probs.shape[1] else 0.0
                    if pa == 0 and pb == 0:
                        blend = 0.5
                    else:
                        blend = pb / (pa + pb)
                    det['blend_metrics'][tuple(pair)] = blend
    if frame_idx < 5:
        logger.debug(f"Frame {frame_idx}: Processed detections:")
        for det in detections:
            logger.debug(f"  {det}")
    tracks = tracker_wrapper.update(detections)
    if frame_idx < 5:
        logger.debug(f"Frame {frame_idx}: Tracker output:")
        for tr in tracks:
            logger.debug(f"  {tr}")
    rendered = draw_tracks(frame, tracks, class_info, blend_pairs=blend_pairs, show_blend_value=show_blend_value)
    return rendered


def run_tracking_pipeline(config: Dict[str, Any]) -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    config = resolve_all_paths(config, project_root)
    fps, frame_size, frame_count = get_video_properties(config['input_video'])
    writer = init_writer(config['output_video'], fps, frame_size)
    logger.debug(f"[Pipeline] Video FPS detected: {fps}")
    model = init_yolo_model(config['yolo_model_path'])
    detection_processor, base_mapper = init_detection_processor(config['base_class_mapping'])
    tracker_wrapper = init_tracker(fps, config.get('bytetrack_params', {}), base_mapper)
    class_info = init_class_info(config)

    for i, frame in enumerate(tqdm(read_video_frames(config['input_video']), total=frame_count, desc="Tracking video")):
        rendered = process_frame(frame, model, detection_processor, tracker_wrapper, class_info, config['blend_pairs'], i)
        writer.write(rendered)
    writer.release()
    print(f"Tracked video written to {config['output_video']}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + ByteTrack video tracking pipeline (YAML config)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    run_tracking_pipeline(config) 