#!/usr/bin/env python3
"""
Image Folder Report

Counts images by resolution (height x width) in a folder.

Usage:
    python image_folder_report.py /path/to/images/folder
"""

import argparse
import os
from collections import Counter
from pathlib import Path
from loguru import logger

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.images.image_utils import get_image_resolution, is_image_file

def count_resolutions(folder_path):
    """Count images by resolution in a folder."""
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        logger.error(f"Error: Folder not found: {folder_path}")
        return
    
    if not folder_path.is_dir():
        logger.error(f"Error: Path is not a directory: {folder_path}")
        return
    
    resolution_counter = Counter()
    total_files = 0
    processed_files = 0
    
    # Get all image files recursively
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if is_image_file(file):
                total_files += 1
                image_path = os.path.join(root, file)
                resolution = get_image_resolution(image_path)
                if resolution:
                    width, height = resolution
                    # Format as heightxwidth
                    resolution_str = f"{height}x{width}"
                    resolution_counter[resolution_str] += 1
                    processed_files += 1
    
    if processed_files == 0:
        logger.warning("No valid images found!")
        return
    
    logger.info(f"Found {total_files} image files")
    logger.info(f"Successfully processed {processed_files} images")
    
    print("\nResolution distribution:")
    print("Resolution (height x width) | Count")
    print("-" * 40)
    
    # Sort by resolution string for consistent output
    for resolution in sorted(resolution_counter.keys()):
        count = resolution_counter[resolution]
        print(f"{resolution:<25} | {count}")
    
    # Add total line
    total_count = sum(resolution_counter.values())
    print("-" * 40)
    print(f"{'TOTAL':<25} | {total_count}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Count images by resolution in a folder"
    )
    
    parser.add_argument('folder_path', 
                       help='Path to the folder containing images')
    
    args = parser.parse_args()
    
    count_resolutions(args.folder_path)


if __name__ == '__main__':
    main()