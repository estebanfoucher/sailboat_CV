import os
import subprocess
from typing import Tuple

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.mpeg', '.mpg'}

def is_video_file(filepath: str) -> bool:
    """Check if the file is a video based on its extension."""
    _, ext = os.path.splitext(filepath)
    return ext.lower() in VIDEO_EXTENSIONS


def get_video_resolution(filepath: str) -> Tuple[int, int]:
    """Return (width, height) of the video using ffprobe."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=s=x:p=0',
        filepath
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    width, height = map(int, result.stdout.strip().split('x'))
    return width, height


def downsample_fps(input_path, output_path=None, fps=25):
    """Downsample a video's FPS using ffmpeg."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_fps{fps}{ext}"
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-r", str(fps),
        "-y",  # Overwrite output file without asking
        output_path
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Saved downsampled video to {output_path}") 