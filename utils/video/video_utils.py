import os
import subprocess
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional
from loguru import logger

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
    # Only consider the first two non-empty parts as width and height, ignore the rest
    parts = [p for p in result.stdout.strip().split('x') if p.strip()]
    if len(parts) < 2:
        raise ValueError(f"Could not parse resolution from ffprobe output: '{result.stdout.strip()}'")
    width, height = map(int, parts[:2])
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
    fps_str = result.stdout.strip()
    # If output is like '30000/1001x', split and use the first non-empty part
    fps_part = [p for p in fps_str.split('x') if p.strip()]
    if not fps_part:
        raise ValueError(f"Could not parse FPS from ffprobe output: '{fps_str}'")
    fps_str = fps_part[0]
    if '/' in fps_str:
        numerator, denominator = map(float, fps_str.split('/'))
        fps = numerator / denominator
    else:
        fps = float(fps_str)
    return fps


def get_video_duration(filepath: str) -> float:
    """Return the duration of the video in seconds using ffprobe."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'csv=p=0',
        filepath
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    
    duration = float(result.stdout.strip())
    return duration


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


def find_video_files(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    Find all video files in directory tree
    
    Args:
        directory: Directory to search
        recursive: Search recursively in subdirectories
        
    Returns:
        List of video file paths
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    video_files = []
    pattern = "**/*" if recursive else "*"
    
    for file_path in directory.glob(pattern):
        if file_path.is_file() and is_video_file(str(file_path)):
            video_files.append(file_path)
    
    # Sort for consistent processing order
    video_files.sort()
    return video_files


def get_video_frame_count(filepath: str) -> int:
    """
    Get total frame count of video using OpenCV (faster than ffprobe for this)
    
    Args:
        filepath: Path to video file
        
    Returns:
        Total number of frames
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {filepath}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def get_comprehensive_video_info(filepath: str) -> Dict:
    """
    Get comprehensive video information using both OpenCV and ffprobe
    
    Args:
        filepath: Path to video file
        
    Returns:
        Dictionary with comprehensive video metadata
    """
    filepath = str(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    file_path = Path(filepath)
    info = {
        "path": filepath,
        "filename": file_path.name,
        "size_bytes": file_path.stat().st_size,
        "size_mb": file_path.stat().st_size / (1024 * 1024),
    }
    
    # Get OpenCV-based information
    try:
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            info.update({
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            })
            cap.release()
        else:
            # Fallback to ffprobe if OpenCV fails
            info.update({
                "fps": get_video_fps(filepath),
                "width": get_video_resolution(filepath)[0],
                "height": get_video_resolution(filepath)[1],
                "resolution": f"{get_video_resolution(filepath)[0]}x{get_video_resolution(filepath)[1]}"
            })
            info["frame_count"] = int(info["fps"] * get_video_duration(filepath))
    except Exception as e:
        print(f"Warning: Could not get complete video info for {filepath}: {e}")
        # Minimal fallback
        info.update({
            "fps": 0,
            "frame_count": 0,
            "width": 0,
            "height": 0,
            "resolution": "unknown"
        })
    
    # Calculate duration
    if info.get("fps", 0) > 0 and info.get("frame_count", 0) > 0:
        info["duration_seconds"] = info["frame_count"] / info["fps"]
    else:
        try:
            info["duration_seconds"] = get_video_duration(filepath)
        except:
            info["duration_seconds"] = 0
    
    return info


def calculate_total_frames_for_batch(video_paths: List[Union[str, Path]], step: int = 1) -> Dict:
    """
    Calculate total frames that would be extracted for a batch of videos
    
    Args:
        video_paths: List of video file paths
        step: Frame extraction step (every Nth frame)
        
    Returns:
        Dictionary with batch statistics
    """
    total_frames = 0
    total_duration = 0
    total_size = 0
    video_count = 0
    failed_count = 0
    
    for video_path in video_paths:
        try:
            info = get_comprehensive_video_info(str(video_path))
            
            # Calculate frames that would be extracted with this step
            total_video_frames = info.get("frame_count", 0)
            frames_to_extract = len(list(range(0, total_video_frames, step)))
            
            total_frames += frames_to_extract
            total_duration += info.get("duration_seconds", 0)
            total_size += info.get("size_mb", 0)
            video_count += 1
            
        except Exception as e:
            print(f"Warning: Could not analyze {video_path}: {e}")
            failed_count += 1
    
    return {
        "total_videos": len(video_paths),
        "analyzable_videos": video_count,
        "failed_analysis": failed_count,
        "total_frames_to_extract": total_frames,
        "total_duration_hours": total_duration / 3600,
        "total_size_gb": total_size / 1024,
        "average_frames_per_video": total_frames / max(video_count, 1),
        "estimated_extraction_time_minutes": total_frames / (60 * 5)  # Assume ~5 fps processing
    }


def validate_video_files(video_paths: List[Union[str, Path]]) -> Dict:
    """
    Validate a list of video files for processing
    
    Args:
        video_paths: List of video file paths
        
    Returns:
        Dictionary with validation results
    """
    valid_videos = []
    invalid_videos = []
    missing_videos = []
    
    for video_path in video_paths:
        video_path = Path(video_path)
        
        if not video_path.exists():
            missing_videos.append({
                "path": str(video_path),
                "error": "File not found"
            })
            continue
        
        if not is_video_file(str(video_path)):
            invalid_videos.append({
                "path": str(video_path),
                "error": "Not a recognized video format"
            })
            continue
        
        try:
            # Quick validation - try to open with OpenCV
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                if frame_count > 0 and fps > 0:
                    valid_videos.append(str(video_path))
                else:
                    invalid_videos.append({
                        "path": str(video_path),
                        "error": "Invalid video metadata (0 frames or fps)"
                    })
            else:
                invalid_videos.append({
                    "path": str(video_path),
                    "error": "Cannot open video file"
                })
                
        except Exception as e:
            invalid_videos.append({
                "path": str(video_path),
                "error": f"Validation error: {str(e)}"
            })
    
    return {
        "valid_videos": valid_videos,
        "invalid_videos": invalid_videos,
        "missing_videos": missing_videos,
        "total_valid": len(valid_videos),
        "total_invalid": len(invalid_videos) + len(missing_videos),
        "validation_rate": len(valid_videos) / max(len(video_paths), 1) * 100
    }


def estimate_frame_extraction_size(video_paths: List[Union[str, Path]], step: int = 1,
                                 quality: int = 95) -> Dict:
    """
    Estimate disk space needed for frame extraction
    
    Args:
        video_paths: List of video file paths
        step: Frame extraction step
        quality: JPEG quality (affects file size)
        
    Returns:
        Dictionary with size estimates
    """
    total_frames = 0
    total_resolution_pixels = 0
    
    for video_path in video_paths:
        try:
            info = get_comprehensive_video_info(str(video_path))
            frame_count = info.get("frame_count", 0)
            width = info.get("width", 0)
            height = info.get("height", 0)
            
            frames_to_extract = len(list(range(0, frame_count, step)))
            total_frames += frames_to_extract
            total_resolution_pixels += frames_to_extract * width * height
            
        except Exception:
            continue
    
    # Estimate JPEG file size based on resolution and quality
    # Rough estimate: 0.1-0.3 bytes per pixel depending on quality
    if quality >= 95:
        bytes_per_pixel = 0.25
    elif quality >= 85:
        bytes_per_pixel = 0.15
    else:
        bytes_per_pixel = 0.1
    
    estimated_size_bytes = total_resolution_pixels * bytes_per_pixel
    estimated_size_mb = estimated_size_bytes / (1024 * 1024)
    estimated_size_gb = estimated_size_mb / 1024
    
    return {
        "total_frames_estimated": total_frames,
        "estimated_size_mb": estimated_size_mb,
        "estimated_size_gb": estimated_size_gb,
        "total_pixels": total_resolution_pixels,
        "quality_factor": quality,
        "bytes_per_pixel_estimate": bytes_per_pixel
    }