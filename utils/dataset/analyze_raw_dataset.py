#!/usr/bin/env python3
"""
Raw Dataset Analysis Script

Analyzes a raw dataset with the following structure:
dataset/
├── images/
└── videos/

This script will:
1. Run image resolution report on the images/ folder
2. Run video resolution report on the videos/ folder
3. Provide a combined summary

Usage:
    python analyze_raw_dataset.py /path/to/dataset/
"""

import argparse
import os
import sys
from pathlib import Path
from loguru import logger

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.dataset.image_folder_report import count_resolutions as analyze_images
from utils.dataset.video_folder_report import analyze_videos


def analyze_raw_dataset(dataset_path):
    """Analyze a raw dataset containing images/ and videos/ directories."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        logger.error(f"Error: Dataset path not found: {dataset_path}")
        return False
    
    if not dataset_path.is_dir():
        logger.error(f"Error: Path is not a directory: {dataset_path}")
        return False
    
    images_dir = dataset_path / "images"
    videos_dir = dataset_path / "videos"
    
    print(f"\n{'='*60}")
    print(f"ANALYZING RAW DATASET: {dataset_path}")
    print(f"{'='*60}")
    
    # Check if images/ directory exists
    if images_dir.exists() and images_dir.is_dir():
        print(f"\n{'='*40}")
        print("IMAGES ANALYSIS")
        print(f"{'='*40}")
        print(f"Analyzing images in: {images_dir}")
        analyze_images(str(images_dir))
    else:
        logger.warning(f"Images directory not found or not a directory: {images_dir}")
    
    # Check if videos/ directory exists
    if videos_dir.exists() and videos_dir.is_dir():
        print(f"\n{'='*40}")
        print("VIDEOS ANALYSIS")
        print(f"{'='*40}")
        print(f"Analyzing videos in: {videos_dir}")
        analyze_videos(str(videos_dir))
    else:
        logger.warning(f"Videos directory not found or not a directory: {videos_dir}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("DATASET ANALYSIS COMPLETE")
    print(f"{'='*60}")
    
    # Count total files
    total_images = 0
    total_videos = 0
    
    if images_dir.exists():
        from utils.images.image_utils import is_image_file
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if is_image_file(file):
                    total_images += 1
    
    if videos_dir.exists():
        from utils.video.video_utils import is_video_file
        for root, dirs, files in os.walk(videos_dir):
            for file in files:
                if is_video_file(file):
                    total_videos += 1
    
    print(f"Dataset summary:")
    print(f"  - Total images: {total_images}")
    print(f"  - Total videos: {total_videos}")
    print(f"  - Total files: {total_images + total_videos}")
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze a raw dataset containing images/ and videos/ directories"
    )
    
    parser.add_argument('dataset_path', 
                       help='Path to the dataset directory (should contain images/ and/or videos/ folders)')
    
    args = parser.parse_args()
    
    success = analyze_raw_dataset(args.dataset_path)
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()