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


def get_video_fps(filepath: str) -> float:
    """Return the FPS of the video using ffprobe."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'csv=s=x:p=0',
        filepath
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    
    # Parse frame rate (can be in format like "30/1" or "29.97")
    fps_str = result.stdout.strip()
    if '/' in fps_str:
        numerator, denominator = map(float, fps_str.split('/'))
        fps = numerator / denominator
    else:
        fps = float(fps_str)
    
    return fps


def downsample_fps(input_path, output_path=None, fps=25):
    """Downsample a video's FPS using ffmpeg."""
    import tempfile
    import shutil
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # If no output path specified, replace the original file
    replace_original = output_path is None
    if replace_original:
        # Create a temporary file for processing
        base, ext = os.path.splitext(input_path)
        temp_fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="temp_fps_")
        os.close(temp_fd)
        output_path = temp_path
    
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-r", str(fps),
        "-y",  # Overwrite output file without asking
        output_path
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    if replace_original:
        # Replace original file with processed version
        shutil.move(output_path, input_path)
        print(f"Replaced original file {input_path} with FPS downsampled to {fps}")
        return input_path
    else:
        print(f"Saved downsampled video to {output_path}")
        return output_path


def downsample_resolution(input_path, output_path=None, resolution=None):
    """Downsample a video's resolution using ffmpeg.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video (if None, replaces original)
        resolution: Tuple of (height, width) or None to keep original
    """
    import tempfile
    import shutil
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # If no output path specified, replace the original file
    replace_original = output_path is None
    if replace_original:
        # Create a temporary file for processing
        base, ext = os.path.splitext(input_path)
        temp_fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="temp_res_")
        os.close(temp_fd)
        output_path = temp_path
    
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-y",  # Overwrite output file without asking
    ]
    
    if resolution:
        height, width = resolution
        cmd.extend(["-vf", f"scale={width}:{height}"])
    
    cmd.append(output_path)
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    if replace_original:
        # Replace original file with processed version
        shutil.move(output_path, input_path)
        if resolution:
            print(f"Replaced original file {input_path} with resolution downsampled to {width}x{height}")
        else:
            print(f"Processed file {input_path} (no resolution change)")
        return input_path
    else:
        if resolution:
            print(f"Saved resolution downsampled video to {output_path}")
        else:
            print(f"Saved processed video to {output_path}")
        return output_path


def downsample_video(input_path, output_path=None, fps=None, resolution=None):
    """Downsample a video's FPS and/or resolution using ffmpeg.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video (if None, replaces original)
        fps: Target FPS (if None, keeps original FPS)
        resolution: Tuple of (height, width) or None to keep original resolution
    """
    import tempfile
    import shutil
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # If no processing requested, return original path
    if fps is None and resolution is None:
        print(f"No processing requested for {input_path}")
        return input_path
    
    # If no output path specified, replace the original file
    replace_original = output_path is None
    if replace_original:
        # Create a temporary file for processing
        base, ext = os.path.splitext(input_path)
        temp_fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="temp_vid_")
        os.close(temp_fd)
        output_path = temp_path
    
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-y",  # Overwrite output file without asking
    ]
    
    # Add FPS parameter if specified
    if fps:
        cmd.extend(["-r", str(fps)])
    
    # Add resolution scaling if specified
    if resolution:
        height, width = resolution
        cmd.extend(["-vf", f"scale={width}:{height}"])
    
    cmd.append(output_path)
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    if replace_original:
        # Replace original file with processed version
        shutil.move(output_path, input_path)
        changes = []
        if fps:
            changes.append(f"FPS to {fps}")
        if resolution:
            height, width = resolution
            changes.append(f"resolution to {width}x{height}")
        print(f"Replaced original file {input_path} with {' and '.join(changes)}")
        return input_path
    else:
        changes = []
        if fps:
            changes.append(f"FPS to {fps}")
        if resolution:
            height, width = resolution
            changes.append(f"resolution to {width}x{height}")
        print(f"Saved processed video ({' and '.join(changes)}) to {output_path}")
        return output_path