import os
import cv2
from tqdm import tqdm
from video_io import get_video_properties, read_video_frames, open_video_writer
from render import draw_tracks

# Use the same class_info as in your config
CLASS_INFO = {
    0: {"name": "pennon_attached", "color": (0, 255, 0)},   # green
    1: {"name": "pennon_detached", "color": (0, 0, 255)}, # red
    2: {"name": "pennon_leech", "color": (255, 0, 0)},    # blue
}

EXAMPLE_VIDEO = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/videos/example/2Ce-CKKCtV4_35.0_40.0_fps25.mkv'))
OUTPUT_VIDEO = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/videos/example/test_video_pipeline.mkv'))


def test_video_pipeline():
    fps, frame_size, frame_count = get_video_properties(EXAMPLE_VIDEO)
    cap = cv2.VideoCapture(EXAMPLE_VIDEO)
    writer = open_video_writer(OUTPUT_VIDEO, fps, frame_size)

    for i, frame in enumerate(tqdm(read_video_frames(EXAMPLE_VIDEO), total=frame_count, desc="Rendering video")):
        # Dummy track: moving box
        tracks = [{
            "bbox": [50 + i*2, 50, 200 + i*2, 200],
            "track_id": 1,
            "original_class_id": 0,
            "confidence": 0.99
        }]
        rendered = draw_tracks(frame, tracks, CLASS_INFO, show_confidence=True)
        writer.write(rendered)
        if i >= 49:  # Only process first 50 frames for test
            break
    writer.release()
    print(f"Test video written to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    test_video_pipeline() 