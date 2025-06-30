#!/usr/bin/env python3
"""
Video Folder Report

Analyzes videos in a folder and provides statistics on:
- Resolution distribution (height x width)
- FPS distribution
- Duration statistics
- Detailed information for each video

Usage:
    python video_folder_report.py /path/to/videos/folder
"""

import argparse
import os
from collections import Counter
from pathlib import Path
from loguru import logger

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.video.video_utils import get_video_resolution, get_video_fps, get_video_duration, is_video_file

def analyze_videos(folder_path):
    """Analyze videos by resolution, FPS, and duration in a folder."""
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        logger.error(f"Error: Folder not found: {folder_path}")
        return
    
    if not folder_path.is_dir():
        logger.error(f"Error: Path is not a directory: {folder_path}")
        return
    
    resolution_counter = Counter()
    fps_counter = Counter()
    total_files = 0
    processed_files = 0
    error_files = 0
    total_duration = 0.0
    video_details = []
    
    # Get all video files recursively
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if is_video_file(file):
                total_files += 1
                video_path = os.path.join(root, file)
                try:
                    width, height = get_video_resolution(video_path)
                    fps = get_video_fps(video_path)
                    duration = get_video_duration(video_path)
                    
                    # Format as heightxwidth (consistent with image report)
                    resolution_str = f"{height}x{width}"
                    resolution_counter[resolution_str] += 1
                    
                    # Round FPS to nearest integer for counting
                    fps_rounded = round(fps)
                    fps_counter[fps_rounded] += 1
                    
                    total_duration += duration
                    
                    # Store detailed info for each video
                    video_details.append({
                        'file': os.path.relpath(video_path, folder_path),
                        'resolution': resolution_str,
                        'fps': fps,
                        'duration': duration
                    })
                    
                    processed_files += 1
                except Exception as e:
                    logger.warning(f"Could not analyze {video_path}: {e}")
                    error_files += 1
    
    if processed_files == 0:
        logger.warning("No valid videos found!")
        return
    
    logger.info(f"Found {total_files} video files")
    logger.info(f"Successfully processed {processed_files} videos")
    if error_files > 0:
        logger.warning(f"Failed to process {error_files} videos")
    
    # Summary statistics
    print(f"\n=== VIDEO ANALYSIS SUMMARY ===")
    print(f"Total videos processed: {processed_files}")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    if processed_files > 0:
        avg_duration = total_duration / processed_files
        print(f"Average duration: {avg_duration:.2f} seconds")
    
    # Resolution distribution
    print("\n=== RESOLUTION DISTRIBUTION ===")
    print("Resolution (height x width) | Count")
    print("-" * 40)
    
    for resolution in sorted(resolution_counter.keys()):
        count = resolution_counter[resolution]
        print(f"{resolution:<25} | {count}")
    
    total_count = sum(resolution_counter.values())
    print("-" * 40)
    print(f"{'TOTAL':<25} | {total_count}")
    
    # FPS distribution
    print("\n=== FPS DISTRIBUTION ===")
    print("FPS        | Count")
    print("-" * 20)
    
    for fps in sorted(fps_counter.keys()):
        count = fps_counter[fps]
        print(f"{fps:<10} | {count}")
    
    print("-" * 20)
    print(f"{'TOTAL':<10} | {sum(fps_counter.values())}")
    
    # Detailed video information
    print("\n=== DETAILED VIDEO INFORMATION ===")
    print(f"{'Filename':<40} | {'Resolution':<12} | {'FPS':<6} | {'Duration (s)':<12}")
    print("-" * 80)
    
    for video in sorted(video_details, key=lambda x: x['file']):
        print(f"{video['file']:<40} | {video['resolution']:<12} | {video['fps']:<6.1f} | {video['duration']:<12.2f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze videos by resolution, FPS, and duration in a folder"
    )
    
    parser.add_argument('folder_path', 
                       help='Path to the folder containing videos')
    
    args = parser.parse_args()
    
    analyze_videos(args.folder_path)


if __name__ == '__main__':
    main()