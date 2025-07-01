#!/usr/bin/env python3
"""
Harmonize Source Distribution Script

This script takes a flat folder of images and randomly selects X images for each source,
copying the selected images to a new output directory while leaving the input directory untouched.

Images are expected to be named in the format: frame_<id_string>_<time_sec>.suffix
where id_string identifies the source video.

Usage:
    python harmonize_source_distribution.py /path/to/input --output-dir /path/to/output --max-per-source 100
    python harmonize_source_distribution.py /path/to/input --output-dir /path/to/output --max-per-source 50
"""

import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from loguru import logger

# Add the project root to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.images.image_utils import is_image_file, get_id_string_and_time_sec


def analyze_source_distribution(folder_path: Path) -> Dict[str, List[Path]]:
    """
    Analyze the distribution of images by source ID.
    
    Args:
        folder_path: Path to the folder containing images
        
    Returns:
        Dictionary mapping source_id to list of image file paths
    """
    source_images = defaultdict(list)
    total_files = 0
    processed_files = 0
    malformed_files = []
    
    logger.info(f"Analyzing images in: {folder_path}")
    
    # Get all image files in the folder
    for file_path in folder_path.iterdir():
        if file_path.is_file() and is_image_file(file_path.name):
            total_files += 1
            try:
                image_info = get_id_string_and_time_sec(file_path.name)
                source_id = image_info['id_string']
                source_images[source_id].append(file_path)
                processed_files += 1
            except (IndexError, KeyError, ValueError):
                logger.warning(f"Skipping malformed filename: {file_path.name}")
                malformed_files.append(file_path)
                continue
    
    logger.info(f"Found {total_files} image files")
    logger.info(f"Successfully processed {processed_files} images")
    logger.info(f"Found {len(malformed_files)} malformed filenames")
    logger.info(f"Identified {len(source_images)} unique sources")
    
    return source_images


def print_distribution_report(source_images: Dict[str, List[Path]], max_per_source: int) -> None:
    """Print a detailed report of the current and target distribution."""
    print("\n" + "="*60)
    print("SOURCE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    print(f"{'Source ID':<25} | {'Available':<9} | {'Selected':<8} | {'Action'}")
    print("-" * 60)
    
    total_available = 0
    total_selected = 0
    sources_with_excess = 0
    
    for source_id in sorted(source_images.keys()):
        available_count = len(source_images[source_id])
        selected_count = min(available_count, max_per_source)
        action = "Copy all" if available_count <= max_per_source else f"Copy {selected_count}/{available_count}"
        
        print(f"{source_id:<25} | {available_count:<9} | {selected_count:<8} | {action}")
        
        total_available += available_count
        total_selected += selected_count
        if available_count > max_per_source:
            sources_with_excess += 1
    
    print("-" * 60)
    print(f"{'TOTAL':<25} | {total_available:<9} | {total_selected:<8} | Copy {total_selected}/{total_available}")
    print(f"\nSources with excess: {sources_with_excess}/{len(source_images)}")
    print(f"Images to copy: {total_selected}")


def harmonize_sources(source_images: Dict[str, List[Path]],
                     max_per_source: int,
                     output_dir: Path,
                     dry_run: bool = False,
                     seed: int = None) -> None:
    """
    Harmonize the source distribution by randomly selecting max_per_source images per source
    and copying them to the output directory.
    
    Args:
        source_images: Dictionary mapping source_id to list of image paths
        max_per_source: Maximum number of images to select per source
        output_dir: Directory to copy selected images to
        dry_run: If True, only show what would be done
        seed: Random seed for reproducible results
    """
    if seed is not None:
        random.seed(seed)
        logger.info(f"Using random seed: {seed}")
    
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    
    total_copied = 0
    total_available = 0
    
    for source_id, image_list in source_images.items():
        available_count = len(image_list)
        total_available += available_count
        
        if available_count <= max_per_source:
            # Copy all images for this source
            images_to_copy = image_list
            logger.info(f"Source {source_id}: copying all {available_count} images")
        else:
            # Randomly select images to copy
            images_to_copy = random.sample(image_list, max_per_source)
            logger.info(f"Source {source_id}: copying {len(images_to_copy)} out of {available_count} images")
        
        total_copied += len(images_to_copy)
        
        # Copy selected images
        for image_path in images_to_copy:
            if dry_run:
                logger.info(f"  Would copy: {image_path.name}")
            else:
                try:
                    output_path = output_dir / image_path.name
                    
                    # Handle filename conflicts in output directory
                    counter = 1
                    while output_path.exists():
                        stem = image_path.stem
                        suffix = image_path.suffix
                        output_path = output_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    shutil.copy2(image_path, output_path)
                    logger.debug(f"  Copied: {image_path.name} -> {output_path.name}")
                        
                except Exception as e:
                    logger.error(f"  Error copying {image_path.name}: {e}")
                    total_copied -= 1  # Adjust count for failed copy
    
    print(f"\n{'DRY RUN ' if dry_run else ''}HARMONIZATION SUMMARY:")
    print(f"  Total images available: {total_available}")
    print(f"  Images {'would be ' if dry_run else ''}copied: {total_copied}")
    print(f"  Output directory: {output_dir}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Harmonize image source distribution by randomly selecting X images per source and copying them to a new directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Copy max 100 images per source to output directory
  python harmonize_source_distribution.py /path/to/input --output-dir /path/to/output --max-per-source 100
  
  # Preview what would be done (dry run)
  python harmonize_source_distribution.py /path/to/input --output-dir /path/to/output --max-per-source 100 --dry-run
  
  # Use a specific random seed for reproducible results
  python harmonize_source_distribution.py /path/to/input --output-dir /path/to/output --max-per-source 100 --seed 42
        """
    )
    
    parser.add_argument(
        'folder_path',
        help='Path to the input folder containing images to harmonize'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        required=True,
        help='Output directory where selected images will be copied'
    )
    
    parser.add_argument(
        '--max-per-source', '-m',
        type=int,
        required=True,
        help='Maximum number of images to select per source'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually moving/deleting files'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducible results'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce log output (only show warnings and errors)'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    
    # Validate input folder
    folder_path = Path(args.folder_path)
    if not folder_path.exists():
        logger.error(f"Error: Folder not found: {folder_path}")
        sys.exit(1)
    
    if not folder_path.is_dir():
        logger.error(f"Error: Path is not a directory: {folder_path}")
        sys.exit(1)
    
    # Validate max_per_source
    if args.max_per_source <= 0:
        logger.error("Error: max-per-source must be a positive integer")
        sys.exit(1)
    
    print("SOURCE DISTRIBUTION HARMONIZER")
    print("=" * 50)
    print(f"Input folder: {folder_path.absolute()}")
    print(f"Output folder: {args.output_dir.absolute()}")
    print(f"Max images per source: {args.max_per_source}")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be copied ***")
    
    # Analyze current distribution
    source_images = analyze_source_distribution(folder_path)
    
    if not source_images:
        logger.error("No valid images found in the specified folder")
        sys.exit(1)
    
    # Print distribution report
    print_distribution_report(source_images, args.max_per_source)
    
    # Ask for confirmation if not in dry run mode
    if not args.dry_run:
        print(f"\nThis will copy selected images to: {args.output_dir.absolute()}")
        confirm = input("Do you want to continue? (y/N): ").lower().strip()
        if confirm not in ['y', 'yes']:
            print("Operation cancelled.")
            sys.exit(0)
    
    # Harmonize the distribution
    harmonize_sources(
        source_images=source_images,
        max_per_source=args.max_per_source,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        seed=args.seed
    )
    
    print("\nHarmonization complete!")


if __name__ == "__main__":
    main()