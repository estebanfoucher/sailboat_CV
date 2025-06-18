#!/usr/bin/env python3

import os
import logging
from pathlib import Path
from typing import Set, Dict, List
import sys

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('yolo_dataset_validation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_file_extensions(images_dir: Path) -> Set[str]:
    """Get all image file extensions in the directory without the dot."""
    extensions = set()
    for file in images_dir.iterdir():
        if file.is_file():
            ext = file.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                extensions.add(ext)
    return extensions

def validate_yolo_dataset(dataset_path: str) -> bool:
    """
    Validate that there is a one-to-one correspondence between images and labels in the YOLO dataset.
    Expects structure:
        dataset_path/
            images/
                image1.jpg
                image2.jpg
                ...
            labels/
                image1.txt
                image2.txt
                ...
    
    Args:
        dataset_path: Path to the YOLO dataset directory
        
    Returns:
        bool: True if the dataset is valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    dataset_dir = Path(dataset_path)
    
    # Check if the dataset directory exists
    if not dataset_dir.exists():
        logger.error(f"Dataset directory {dataset_path} does not exist")
        return False
    
    # Check for images and labels subdirectories
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    if not images_dir.exists():
        logger.error(f"Images directory not found at {images_dir}")
        return False
    
    if not labels_dir.exists():
        logger.error(f"Labels directory not found at {labels_dir}")
        return False
    
    # Get all image extensions in the directory
    image_extensions = get_file_extensions(images_dir)
    logger.info(f"Found image extensions: {image_extensions}")
    
    # Get all image and label files
    image_files = set()
    for ext in image_extensions:
        image_files.update(f.stem for f in images_dir.glob(f"*{ext}"))
    
    label_files = set(f.stem for f in labels_dir.glob("*.txt"))
    
    # Log the counts
    logger.info(f"Found {len(image_files)} images in {images_dir}")
    logger.info(f"Found {len(label_files)} labels in {labels_dir}")
    
    # Check for images without labels
    images_without_labels = image_files - label_files
    if images_without_labels:
        logger.error(f"Found {len(images_without_labels)} images without labels:")
        for img in sorted(images_without_labels):
            logger.error(f"  Missing label for image: {img}")
        # Prompt user for suppression
        suppress = input("Would you like to suppress (move) these images? (y/n): ").strip().lower()
        if suppress == 'y':
            suppressed_dir = images_dir / 'suppressed_images'
            suppressed_dir.mkdir(exist_ok=True)
            for img in sorted(images_without_labels):
                # Find the actual file with extension
                for ext in image_extensions:
                    img_file = images_dir / f"{img}{ext}"
                    if img_file.exists():
                        img_file.rename(suppressed_dir / img_file.name)
                        logger.info(f"Suppressed image: {img_file.name}")
                        break

    # Check for labels without images
    labels_without_images = label_files - image_files
    if labels_without_images:
        logger.error(f"Found {len(labels_without_images)} labels without images:")
        for lbl in sorted(labels_without_images):
            logger.error(f"  Missing image for label: {lbl}")
    
    # Final validation result
    is_valid = len(images_without_labels) == 0 and len(labels_without_images) == 0
    if is_valid:
        logger.info("✅ Dataset validation passed: Perfect bijection between images and labels")
    else:
        logger.error("❌ Dataset validation failed: No bijection between images and labels")
    
    return is_valid

if __name__ == "__main__":
    logger = setup_logging()
    
    # Default to the pennon-label-yolo-00 directory
    default_dataset_path = "./docker/label_studio/data/export/yolo/pennon-label-yolo-01"
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else default_dataset_path
    
    logger.info(f"Validating YOLO dataset at: {dataset_path}")
    validate_yolo_dataset(dataset_path) 