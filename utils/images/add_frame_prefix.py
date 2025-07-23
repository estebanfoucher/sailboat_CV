#!/usr/bin/env python3
"""
Add 'frame_' prefix to all image names in a folder.

Usage:
    python add_frame_prefix.py /path/to/folder
    python add_frame_prefix.py /path/to/folder --dry-run
"""
import argparse
from pathlib import Path
from image_utils import is_image_file


def add_frame_prefix_to_images(folder_path, dry_run=False):
    """Add 'frame_' prefix to all image files in the specified folder."""
    folder = Path(folder_path)
    
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder: {folder}")
    
    image_files = [f for f in folder.iterdir() if f.is_file() and is_image_file(f.name)]
    renamed = skipped = errors = 0
    
    for img_file in image_files:
        if img_file.name.startswith('frame_'):
            print(f"SKIP: {img_file.name}")
            skipped += 1
            continue
        
        new_name = f"frame_{img_file.name}"
        new_path = img_file.parent / new_name
        
        if new_path.exists():
            print(f"ERROR: {new_name} already exists")
            errors += 1
            continue
        
        try:
            if not dry_run:
                img_file.rename(new_path)
            print(f"RENAME: {img_file.name} -> {new_name}")
            renamed += 1
        except Exception as e:
            print(f"ERROR: {img_file.name}: {e}")
            errors += 1
    
    print(f"\nRenamed: {renamed}, Skipped: {skipped}, Errors: {errors}")
    return renamed, skipped, errors


def main():
    parser = argparse.ArgumentParser(description="Add 'frame_' prefix to image names")
    parser.add_argument('folder', help='Folder containing images')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes only')
    
    args = parser.parse_args()
    add_frame_prefix_to_images(args.folder, args.dry_run)


if __name__ == '__main__':
    main()