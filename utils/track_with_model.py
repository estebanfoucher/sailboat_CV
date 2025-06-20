import cv2
import sys
import os
from ultralytics import YOLO

def process_video_with_stride(model_path, video_path, output_dir, vid_stride=1, tracker="bytetrack.yaml"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'), fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % vid_stride == 0:
            # Run detection on this frame
            results = model.track(frame, tracker=tracker)
            frame = results[0].plot()
        else:
            # Use tracker to interpolate object positions
            # This part is conceptual; you need to implement interpolation logic
            # For example, you might use a motion model to predict object positions
            pass

        # Write the frame to the output video
        out.write(frame)
        frame_count += 1

    # Release everything when done
    cap.release()
    out.release()
    print(f"Processed video saved to {output_dir}")

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