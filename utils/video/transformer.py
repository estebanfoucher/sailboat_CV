"""
Concise FFmpeg video converter to harmonize formats to .mp4
"""

import subprocess
import json
from pathlib import Path
from typing import Union, Optional, Tuple


def convert_to_mp4(input_path: Union[str, Path], 
                  output_path: Optional[Union[str, Path]] = None,
                  quality: str = 'medium',
                  overwrite: bool = False) -> Path:
    """
    Convert any video format to .mp4 using FFmpeg.
    
    Args:
        input_path: Input video file path
        output_path: Output .mp4 path (auto-generated if None)
        quality: 'low', 'medium', 'high' 
        overwrite: Overwrite existing files
        
    Returns:
        Path to converted .mp4 file
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")
    
    # Generate output path
    if output_path is None:
        output_path = input_path.with_suffix('.mp4')
    else:
        output_path = Path(output_path).with_suffix('.mp4')
    
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}")
    
    # Quality settings
    crf_map = {'low': '28', 'medium': '23', 'high': '18', 'best': '0'}
    crf = crf_map.get(quality, '23')
    
    # FFmpeg command
    cmd = [
        'ffmpeg', '-y' if overwrite else '-n',
        '-i', str(input_path),
        '-c:v', 'libx264', '-crf', crf,
        '-c:a', 'aac', '-b:a', '128k',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Conversion failed: {e}")


def get_video_format(input_path: Union[str, Path]) -> Tuple[str, str]:
    """
    Get video container and codec info.
    
    Returns:
        (container_format, video_codec)
    """
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', str(input_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    
    container = info.get('format', {}).get('format_name', 'unknown')
    video_codec = 'unknown'
    
    for stream in info.get('streams', []):
        if stream.get('codec_type') == 'video':
            video_codec = stream.get('codec_name', 'unknown')
            break
    
    return container, video_codec


def batch_convert_directory(input_dir: Union[str, Path], 
                          output_dir: Optional[Union[str, Path]] = None,
                          **kwargs) -> int:
    """
    Convert all videos in directory to .mp4
    
    Returns:
        Number of files converted
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_exts = {'.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
    video_files = [f for f in input_dir.iterdir() 
                   if f.suffix.lower() in video_exts]
    
    converted = 0
    for video_file in video_files:
        try:
            output_path = output_dir / f"{video_file.stem}.mp4"
            convert_to_mp4(video_file, output_path, **kwargs)
            converted += 1
        except Exception as e:
            print(f"Failed to convert {video_file}: {e}")
    
    return converted


def main():
    """CLI interface for video conversion"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert video to .mp4 format')
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('-o', '--output', help='Output .mp4 file path')
    parser.add_argument('-q', '--quality', choices=['low', 'medium', 'high'],
                       default='high', help='Conversion quality')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output file')
    
    args = parser.parse_args()
    
    try:
        output_path = convert_to_mp4(
            args.input,
            args.output,
            args.quality,
            args.overwrite
        )
        print(f"Successfully converted to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())