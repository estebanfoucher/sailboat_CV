import os
from video_io import get_video_properties, read_video_frames, write_video_frames

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
EXAMPLE_VIDEO = os.path.join(PROJECT_ROOT, "data/videos/example/2Ce-CKKCtV4_35.0_40.0_fps25.mkv")
OUTPUT_VIDEO = os.path.join(PROJECT_ROOT, "data/videos/example/test_output.mkv")


def test_video_io():
    fps, frame_size, frame_count = get_video_properties(EXAMPLE_VIDEO)
    print(f"FPS: {fps}, Size: {frame_size}, Total frames: {frame_count}")

    frames = []
    for i, frame in enumerate(read_video_frames(EXAMPLE_VIDEO)):
        if i >= 10:
            break
        frames.append(frame)
    print(f"Read {len(frames)} frames.")

    write_video_frames(frames, OUTPUT_VIDEO, fps, frame_size)
    print(f"Wrote {len(frames)} frames to {OUTPUT_VIDEO}.")

if __name__ == "__main__":
    test_video_io() 