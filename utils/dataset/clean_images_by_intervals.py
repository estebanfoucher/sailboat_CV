#!/usr/bin/env python3
"""
Clean Images by Intervals

Deletes images outside specified intervals for each YouTube video in a dataset folder.

Usage:
    python clean_images_by_intervals.py urls.txt /path/to/folder --backup True

- urls.txt: file with lines of the form url;start;end, url, url;start, or url;;end
- folder: dataset folder with structure folder/video_id/image_time_second
- --backup: if True, makes a backup of the folder before deleting
"""
import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime
from loguru import logger

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.images.image_utils import get_id_string_and_time_sec
from utils.youtube.youtube_utils import get_video_id_from_url

def parse_time(timestr):
    if not timestr or timestr.strip() == '':
        return None
    h, m, s = [int(x) for x in timestr.strip().split(':')]
    return h * 3600 + m * 60 + s

def parse_urls_txt(txt_path):
    """Parse the txt file and return a dict: video_id -> list of (start, end) intervals (in seconds)"""
    intervals = {}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(';')
            url = parts[0].strip()
            start = parse_time(parts[1]) if len(parts) > 1 else None
            end = parse_time(parts[2]) if len(parts) > 2 else None
            video_id = get_video_id_from_url(url)
            if not video_id:
                logger.warning(f"Could not extract video ID from URL: {url}. Skipping.")
                continue
            if video_id not in intervals:
                intervals[video_id] = []
            intervals[video_id].append((start, end))
    return intervals

def should_keep(time_sec, intervals):
    """Return True if time_sec is within any of the intervals (start, end)."""
    for start, end in intervals:
        if (start is None or time_sec >= start) and (end is None or time_sec <= end):
            return True
    return False

def backup_folder(folder):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{folder}_backup_{timestamp}"
    shutil.copytree(folder, backup_path)
    logger.info(f"Backup created at {backup_path}")

def clean_images(folder, intervals_by_video):
    folder = Path(folder)
    for video_id, intervals in intervals_by_video.items():
        video_dir = folder / video_id
        if not video_dir.exists() or not video_dir.is_dir():
            logger.warning(f"Video directory not found: {video_dir}")
            continue
        for image_file in video_dir.iterdir():
            if not image_file.is_file():
                continue
            try:
                info = get_id_string_and_time_sec(image_file.name)
                time_sec = int(info['time_sec'])
            except Exception as e:
                logger.warning(f"Could not parse image name {image_file.name}: {e}")
                continue
            if not should_keep(time_sec, intervals):
                logger.info(f"Deleting {image_file}")
                image_file.unlink()

def main():
    parser = argparse.ArgumentParser(description="Delete images outside specified intervals for each YouTube video.")
    parser.add_argument('txt_file', help='Path to the txt file with URLs and intervals')
    parser.add_argument('folder', help='Path to the dataset folder (video_id/image_time_second)')
    parser.add_argument('--backup', type=bool, default=False, help='If True, backup the folder before deleting')
    args = parser.parse_args()

    if args.backup:
        backup_folder(args.folder)

    intervals_by_video = parse_urls_txt(args.txt_file)
    clean_images(args.folder, intervals_by_video)
    logger.info("Done.")

if __name__ == '__main__':
    main() 