import cv2
from typing import Generator, Tuple, List
import numpy as np
import subprocess
import os


def open_video_capture(video_path: str) -> cv2.VideoCapture:
    """
    Open a video file for reading.
    Args:
        video_path (str): Path to the input video file.
    Returns:
        cv2.VideoCapture: OpenCV video capture object.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")
    return cap


def get_video_properties(video_path: str) -> Tuple[int, Tuple[int, int], int]:
    """
    Get the frames per second (FPS), frame size, and total frame count of a video file.
    Args:
        video_path (str): Path to the input video file.
    Returns:
        fps (int): Frames per second.
        frame_size (Tuple[int, int]): (width, height) of the video frames.
        frame_count (int): Total number of frames in the video.
    """
    cap = open_video_capture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, (width, height), frame_count


def read_video_frames(video_path: str) -> Generator[np.ndarray, None, None]:
    """
    Generator that yields frames from a video file.
    Args:
        video_path (str): Path to the input video file.
    Yields:
        frame (np.ndarray): The next video frame (BGR format).
    """
    cap = open_video_capture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def open_video_writer(output_path: str, fps: int, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    """
    Open a video file for writing.
    Args:
        output_path (str): Path to the output video file.
        fps (int): Frames per second for the output video.
        frame_size (Tuple[int, int]): (width, height) of the video frames.
    Returns:
        cv2.VideoWriter: OpenCV video writer object.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if output_path.endswith('.mp4') else cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise IOError(f"Cannot open video writer for: {output_path}")
    return writer


def write_video_frames(frames: List[np.ndarray], output_path: str, fps: int, frame_size: Tuple[int, int]) -> None:
    """
    Write a list of frames to a video file.
    Args:
        frames (List[np.ndarray]): List of frames to write (BGR format).
        output_path (str): Path to the output video file.
        fps (int): Frames per second for the output video.
        frame_size (Tuple[int, int]): (width, height) of the video frames.
    """
    writer = open_video_writer(output_path, fps, frame_size)
    for frame in frames:
        writer.write(frame)
    writer.release()


def fix_video_with_ffmpeg(input_path, output_path=None):
    """
    Remux or re-encode a video file using ffmpeg to fix metadata and compatibility issues (e.g., for WhatsApp).
    Converts to .mp4 (H.264 + AAC) by default for maximum compatibility.
    Args:
        input_path (str): Path to the input video file.
        output_path (str, optional): Path for the fixed output file. If None, uses .mp4 extension and '_fixed' before extension.
    Returns:
        str: Path to the fixed video file.
    Raises:
        RuntimeError: If ffmpeg fails.
    """
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_fixed.mp4"
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-c:v', 'libx264', '-c:a', 'aac', '-movflags', '+faststart',
        output_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}")
    return output_path 