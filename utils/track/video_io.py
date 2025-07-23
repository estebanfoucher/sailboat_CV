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


class FFmpegVideoWriter:
    """Video writer using ffmpeg for better compatibility"""
    def __init__(self, output_path: str, fps: int, frame_size: Tuple[int, int]):
        self.output_path = output_path
        self.fps = fps
        self.width, self.height = frame_size
        self.process = None
        self._start_ffmpeg()
    
    def _start_ffmpeg(self):
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}', '-pix_fmt', 'bgr24',
            '-r', str(self.fps), '-i', '-', '-c:v', 'libx264',
            '-movflags', '+faststart', '-pix_fmt', 'yuv420p',
            self.output_path
        ]
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    
    def write(self, frame):
        if self.process:
            self.process.stdin.write(frame.tobytes())
    
    def release(self):
        if self.process:
            self.process.stdin.close()
            self.process.wait()
            self.process = None
    
    def isOpened(self):
        return self.process is not None

def open_video_writer(output_path: str, fps: int, frame_size: Tuple[int, int]):
    """
    Open a video file for writing with ffmpeg integration for better compatibility.
    Args:
        output_path (str): Path to the output video file.
        fps (int): Frames per second for the output video.
        frame_size (Tuple[int, int]): (width, height) of the video frames.
    Returns:
        FFmpegVideoWriter: Custom video writer object.
    """
    return FFmpegVideoWriter(output_path, fps, frame_size)


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

