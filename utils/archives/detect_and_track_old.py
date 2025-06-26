import cv2
import sys
import os
import torch
from ultralytics import YOLO

def process_video_with_stride(model_path, video_path, output_dir, vid_stride=1, tracker="bytetrack.yaml"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Automatically use GPU if available
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(output_dir, 'output.mp4')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_path}")
    print(f"Saving output to: {out_path}")
    print(f"Total frames: {total_frames}, Processing every {vid_stride} frame(s)")

    frame_count = 0
    last_output_frame = None  # For basic interpolation

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % vid_stride == 0:
            # Run detection & tracking
            results = model.track(frame, tracker=tracker, device=device)
            output_frame = results[0].plot()
            last_output_frame = output_frame.copy()
        else:
            # Naive reuse of last frame's boxes (no interpolation logic)
            output_frame = last_output_frame if last_output_frame is not None else frame

        out.write(output_frame)
        frame_count += 1

        if frame_count % 50 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    # Release resources
    cap.release()
    out.release()
    print("✅ Video processing complete.")

if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 6:
        print("Usage: python script.py <model_path> <video_path> <output_dir> [stride] [tracker]")
        sys.exit(1)

    model_path = sys.argv[1]
    video_path = sys.argv[2]
    output_dir = sys.argv[3]
    vid_stride = int(sys.argv[4]) if len(sys.argv) >= 5 else 1
    tracker = sys.argv[5] if len(sys.argv) == 6 else "bytetrack.yaml"

    process_video_with_stride(model_path, video_path, output_dir, vid_stride, tracker)
