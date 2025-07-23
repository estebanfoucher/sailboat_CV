import os
import cv2
import numpy as np
from render import draw_tracks

# Use the same class_info as in your config
CLASS_INFO = {
    0: {"name": "pennon_attached", "color": (0, 255, 0)},   # green
    1: {"name": "pennon_detached", "color": (0, 0, 255)}, # red (OpenCV uses BGR)
    2: {"name": "pennon_leech", "color": (255, 0, 0)},    # blue
}

IMG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/datasets/labels-04/images/frame_sebperso24062025gx010163cba39a_37.20s.jpg'))
LABEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/datasets/labels-04/labels/frame_sebperso24062025gx010163cba39a_37.20s.txt'))
OUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/datasets/labels-04/images/test_render.jpg'))

def load_yolo_labels(label_path, img_shape):
    """
    Load YOLO-format labels and convert to pixel coordinates.
    Args:
        label_path (str): Path to YOLO label file.
        img_shape (tuple): (height, width, channels) of the image.
    Returns:
        List[dict]: List of track dicts for rendering.
    """
    h, w = img_shape[:2]
    tracks = []
    with open(label_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, xc, yc, bw, bh = map(float, parts)
            class_id = int(class_id)
            # Convert normalized to pixel coordinates
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            tracks.append({
                "bbox": [x1, y1, x2, y2],
                "track_id": i + 1,  # Just use line number as dummy track ID
                "original_class_id": class_id,
                "confidence": None
            })
    return tracks

def test_render():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {IMG_PATH}")
    tracks = load_yolo_labels(LABEL_PATH, img.shape)
    rendered = draw_tracks(img, tracks, CLASS_INFO, show_confidence=False)
    cv2.imwrite(OUT_PATH, rendered)
    print(f"Rendered image saved to {OUT_PATH}")

if __name__ == "__main__":
    test_render() 