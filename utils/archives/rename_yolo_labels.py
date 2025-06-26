#!/usr/bin/env python3

import os
from pathlib import Path
import shutil
import re
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo_cleanup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def handle_duplicates(labels_folder_path, images_folder_path):
    """
    Find and handle duplicate frame numbers by deleting all instances
    from both labels and images directories.
    """
    labels_path = Path(labels_folder_path)
    images_path = Path(images_folder_path)

    # Create backup directories
    backup_root = labels_path.parent / 'duplicates_backup'
    if not backup_root.exists():
        backup_root.mkdir()
        (backup_root / 'labels').mkdir()
        (backup_root / 'images').mkdir()

    # Get all images and their frame numbers
    image_frame_map = {}
    frame_to_files = {}
    
    # Map frames to all their corresponding files (both images and labels)
    for image_file in images_path.glob('*'):
        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            if '__' not in image_file.stem:
                logger.warning(f"Image {image_file.name} doesn't follow video_id__frame convention")
                continue
                
            frame_part = image_file.stem.split('__')[1]
            if frame_part not in frame_to_files:
                frame_to_files[frame_part] = {'images': [], 'labels': []}
            frame_to_files[frame_part]['images'].append(image_file)

    for label_file in labels_path.glob('*.txt'):
        if label_file.name == 'classes.txt':
            continue
            
        if '__' not in label_file.stem:
            logger.warning(f"Label {label_file.name} doesn't follow task_id__frame convention")
            continue
            
        frame_part = label_file.stem.split('__')[1]
        if frame_part not in frame_to_files:
            frame_to_files[frame_part] = {'images': [], 'labels': []}
        frame_to_files[frame_part]['labels'].append(label_file)

    # Find duplicates
    duplicates = {
        frame: files for frame, files in frame_to_files.items() 
        if len(files['images']) > 1 or len(files['labels']) > 1
    }

    if not duplicates:
        logger.info("No duplicates found!")
        return False

    # Handle duplicates
    logger.info(f"\nFound {len(duplicates)} frames with duplicates:")
    for frame, files in duplicates.items():
        logger.info(f"\nFrame {frame}:")
        logger.info(f"  Images ({len(files['images'])}): {[f.name for f in files['images']]}")
        logger.info(f"  Labels ({len(files['labels'])}): {[f.name for f in files['labels']]}")
        
        # Backup and delete images
        for img in files['images']:
            backup_path = backup_root / 'images' / img.name
            shutil.copy2(img, backup_path)
            img.unlink()
            logger.info(f"  Deleted image: {img.name}")
            
        # Backup and delete labels
        for lbl in files['labels']:
            backup_path = backup_root / 'labels' / lbl.name
            shutil.copy2(lbl, backup_path)
            lbl.unlink()
            logger.info(f"  Deleted label: {lbl.name}")

    logger.info(f"\nAll duplicates have been moved to backup directory: {backup_root}")
    return True

def rename_label_files(labels_folder_path, images_folder_path):
    """
    Rename label files to match image naming convention by:
    1. Extract frame number from label filename
    2. Find matching image with same frame number
    3. Rename label to match image name
    """
    # First handle any duplicates
    if handle_duplicates(labels_folder_path, images_folder_path):
        logger.info("\nDuplicates were found and removed. Please rerun the script to rename the remaining files.")
        return

    labels_path = Path(labels_folder_path)
    images_path = Path(images_folder_path)

    # Create backup directory
    backup_path = labels_path.parent / 'labels_backup'
    if not backup_path.exists():
        backup_path.mkdir()

    # First, create a backup of all label files
    logger.info("Creating backup of label files...")
    for label_file in labels_path.glob('*.txt'):
        if label_file.name == 'classes.txt':  # Skip classes.txt if it exists
            continue
        shutil.copy2(label_file, backup_path / label_file.name)

    # Get all images and their frame numbers
    image_frame_map = {}
    for image_file in images_path.glob('*'):
        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Extract frame part from image name
            if '__' not in image_file.stem:
                logger.warning(f"Image {image_file.name} doesn't follow video_id__frame convention")
                continue
                
            frame_part = image_file.stem.split('__')[1]
            image_frame_map[frame_part] = image_file.stem

    # Rename the label files
    renamed_count = 0
    errors = []
    logger.info("\nRenaming label files...")
    
    for label_file in labels_path.glob('*.txt'):
        if label_file.name == 'classes.txt':  # Skip classes.txt if it exists
            continue
        
        # Extract frame part from label name
        if '__' not in label_file.stem:
            logger.warning(f"Label {label_file.name} doesn't follow task_id__frame convention")
            errors.append(f"Label file {label_file.name} doesn't follow task_id__frame convention")
            continue
        frame_part = label_file.stem.split('__')[1]
        # Find corresponding image name
        if frame_part not in image_frame_map:
            errors.append(f"No matching image found for label {label_file.name} (frame {frame_part})")
            continue
        new_stem = image_frame_map[frame_part]
        new_name = f"{new_stem}.txt"
        new_path = label_file.parent / new_name
        try:
            # If the destination file already exists, remove it
            if new_path.exists():
                new_path.unlink()
            label_file.rename(new_path)
            renamed_count += 1
            logger.info(f"Renamed: {label_file.name} -> {new_name}")
        except Exception as e:
            errors.append(f"Error renaming {label_file.name} to {new_name}: {str(e)}")

    # Print summary
    logger.info(f"\nRenamed {renamed_count} files")
    if errors:
        logger.info("\nErrors encountered:")
        for error in errors:
            logger.info(f"- {error}")
    
    logger.info(f"\nBackup of original files saved in: {backup_path}")

if __name__ == "__main__":
    labels_folder_path = 'docker/label_studio/data/export/yolo/pennon-label-yolo-01/labels'
    images_folder_path = './docker/label_studio/data/export/yolo/pennon-label-yolo-01/images'
    
    logger.info(f"Starting label file processing...")
    rename_label_files(labels_folder_path, images_folder_path) 