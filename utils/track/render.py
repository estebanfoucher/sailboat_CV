import numpy as np
import cv2
from typing import List, Dict, Tuple, Any


def get_color_for_class(class_id: int, class_info: Dict[int, Dict[str, Any]]) -> Tuple[int, int, int]:
    """
    Get the color for a given class ID from class_info.
    Args:
        class_id (int): The class ID.
        class_info (dict): Mapping from class_id to {"name", "color"}.
    Returns:
        tuple: (B, G, R) color.
    """
    return tuple(class_info[class_id]["color"])


def blend_colors(color_a, color_b, blend):
    """
    Linearly blend two BGR colors by blend (0=all a, 1=all b).
    """
    return tuple([int((1-blend)*a + blend*b) for a, b in zip(color_a, color_b)])


def draw_single_track(
    frame: np.ndarray,
    track: Dict[str, Any],
    class_info: Dict[int, Dict[str, Any]],
    blend_pairs=None,
    show_confidence: bool = False,
    show_class_name: bool = False,
    show_blend_value: bool = False
) -> None:
    """
    Draw a single track (bounding box, track ID, class name if enabled) on the frame.
    Args:
        frame (np.ndarray): The image to draw on (BGR format).
        track (dict): Track dict with keys: 'bbox', 'track_id', 'original_class_id', (optional) 'confidence'.
        class_info (dict): Mapping from class_id to {"name", "color"}.
        blend_pairs (list): List of pairs of class IDs to blend.
        show_confidence (bool): Whether to display confidence score.
        show_class_name (bool): Whether to display class name.
        show_blend_value (bool): Whether to display blend value.
    """
    bbox = track['bbox']
    class_id = track['original_class_id']
    color = get_color_for_class(class_id, class_info)
    text = ""
    blend_metrics = track.get('blend_metrics', {})
    # If blend_pairs and blend_metrics are present, blend colors
    if blend_pairs and blend_metrics:
        for pair in blend_pairs:
            pair_t = tuple(pair)
            if pair_t in blend_metrics:
                a, b = pair
                color_a = get_color_for_class(a, class_info)
                color_b = get_color_for_class(b, class_info)
                blend = blend_metrics[pair_t]
                color = blend_colors(color_a, color_b, blend)
                if show_blend_value:
                    text += f"blend({a},{b})={blend:.2f} "
                break  # Only use the first matching pair
    track_id = track.get('track_id', None)
    conf = track.get('confidence', None)
    label = class_info[class_id]['name'] if show_class_name else ""

    if show_class_name:
        text += f"{label}"
    if track_id is not None:
        if text:
            text += f" ID:{track_id}"
        else:
            text = f"ID:{track_id}"
    if show_confidence and conf is not None:
        text += f" {conf:.2f}"

    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if text:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Dict[str, Any]],
    class_info: Dict[int, Dict[str, Any]],
    blend_pairs=None,
    show_confidence: bool = False,
    show_class_name: bool = False,
    show_blend_value: bool = False
) -> np.ndarray:
    """
    Draw bounding boxes, track IDs, and (optionally) class names on a frame using the original class color.
    Args:
        frame (np.ndarray): The image to draw on (BGR format).
        tracks (List[Dict]): List of track dicts.
        class_info (dict): Mapping from class_id to {"name", "color"}.
        blend_pairs (list): List of pairs of class IDs to blend.
        show_confidence (bool): Whether to display confidence score on the box.
        show_class_name (bool): Whether to display class name (default: False).
        show_blend_value (bool): Whether to display blend value.
    Returns:
        np.ndarray: The frame with drawings.
    """
    for track in tracks:
        draw_single_track(frame, track, class_info, blend_pairs=blend_pairs, show_confidence=show_confidence, show_class_name=show_class_name, show_blend_value=show_blend_value)
    return frame


