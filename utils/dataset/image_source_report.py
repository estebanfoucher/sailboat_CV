#!/usr/bin/env python3
"""
Image Source Counter

Counts images by their source video ID in a folder.

Usage:
    python image_source_report.py /path/to/images/folder
"""

import argparse
import os
from collections import Counter
from pathlib import Path
from loguru import logger

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.images.image_utils import is_image_file, get_id_string_and_time_sec

def count_sources(folder_path):
    """Count images by source video ID in a folder."""
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        logger.error(f"Error: Folder not found: {folder_path}")
        return
    
    if not folder_path.is_dir():
        logger.error(f"Error: Path is not a directory: {folder_path}")
        return
    
    source_counter = Counter()
    total_files = 0
    processed_files = 0
    
    # Get all image files recursively
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if is_image_file(file):
                total_files += 1
                try:
                    image_info = get_id_string_and_time_sec(file)
                    source_id = image_info['id_string']
                    source_counter[source_id] += 1
                    processed_files += 1
                except (IndexError, KeyError):
                    logger.warning(f"Skipping malformed filename: {file}")
                    continue
    
    if processed_files == 0:
        logger.warning("No valid images found!")
        return
    
    logger.info(f"Found {total_files} image files")
    logger.info(f"Successfully processed {processed_files} images")
    
    print("\nSource video distribution:")
    print("Video ID                    | Frame Count")
    print("-" * 40)
    
    # Sort by video ID for consistent output
    for source_id in sorted(source_counter.keys()):
        count = source_counter[source_id]
        print(f"{source_id:<25} | {count}")
    
    # Add total line
    total_count = sum(source_counter.values())
    print("-" * 40)
    print(f"{'TOTAL':<25} | {total_count}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Count images by source video ID in a folder"
    )
    
    parser.add_argument('folder_path', 
                       help='Path to the folder containing images')
    
    args = parser.parse_args()
    
    count_sources(args.folder_path)


if __name__ == '__main__':
    main() 