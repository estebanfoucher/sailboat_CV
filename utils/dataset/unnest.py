#!/usr/bin/env python3
"""
Image Flattening Script

This script recursively collects all images from specified input directories
and copies them to a single flat output directory.

Configure the input paths and output directory below.
"""

import os
import shutil
from pathlib import Path
from typing import List, Set
import argparse


# =============================================================================
# CONFIGURATION - Modify these paths as needed
# =============================================================================

# Input directories to search for images (can be relative or absolute paths)
INPUT_DIRECTORIES = [
    'to_flatten'
]

# Output directory where all images will be copied (will be created if it doesn't exist)
OUTPUT_DIRECTORY = "flattened_images"

# Supported image file extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

# =============================================================================


def get_image_files(directory: Path) -> List[Path]:
    """
    Recursively find all image files in the given directory.
    
    Args:
        directory: Path to search for images
        
    Returns:
        List of Path objects for all found image files
    """
    image_files = []
    
    if not directory.exists():
        print(f"Warning: Directory {directory} does not exist, skipping...")
        return image_files
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_files.append(file_path)
    
    return image_files


def handle_filename_conflict(output_path: Path, original_path: Path) -> Path:
    """
    Handle filename conflicts by adding a suffix to the filename.
    
    Args:
        output_path: The intended output path
        original_path: The original file path (for getting parent directory info)
        
    Returns:
        A unique output path that doesn't conflict with existing files
    """
    if not output_path.exists():
        return output_path
    
    # Get the parent directory name to use as a prefix
    parent_name = original_path.parent.name
    stem = output_path.stem
    suffix = output_path.suffix
    counter = 1
    
    # Try with parent directory name first
    new_name = f"{parent_name}_{stem}{suffix}"
    new_path = output_path.parent / new_name
    
    # If still conflicts, add numbers
    while new_path.exists():
        new_name = f"{parent_name}_{stem}_{counter}{suffix}"
        new_path = output_path.parent / new_name
        counter += 1
    
    return new_path


def flatten_images(input_dirs: List[str], output_dir: str, dry_run: bool = False) -> None:
    """
    Flatten all images from input directories into a single output directory.
    
    Args:
        input_dirs: List of input directory paths
        output_dir: Output directory path
        dry_run: If True, only show what would be done without actually copying files
    """
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path.absolute()}")
    else:
        print(f"DRY RUN - Would create output directory: {output_path.absolute()}")
    
    total_files = 0
    copied_files = 0
    skipped_files = 0
    
    # Process each input directory
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        print(f"\nProcessing directory: {input_path.absolute()}")
        
        # Get all image files from this directory
        image_files = get_image_files(input_path)
        print(f"Found {len(image_files)} image files")
        
        total_files += len(image_files)
        
        # Copy each image file
        for image_file in image_files:
            try:
                # Determine output filename
                output_file_path = output_path / image_file.name
                
                # Handle filename conflicts
                if output_file_path.exists() or any(
                    f.name == image_file.name for f in output_path.glob('*') if f.is_file()
                ):
                    output_file_path = handle_filename_conflict(output_file_path, image_file)
                
                if dry_run:
                    print(f"  Would copy: {image_file} -> {output_file_path.name}")
                else:
                    # Copy the file
                    shutil.copy2(image_file, output_file_path)
                    print(f"  Copied: {image_file.name} -> {output_file_path.name}")
                
                copied_files += 1
                
            except Exception as e:
                print(f"  Error copying {image_file}: {e}")
                skipped_files += 1
    
    # Print summary
    print(f"\n{'DRY RUN ' if dry_run else ''}Summary:")
    print(f"  Total files found: {total_files}")
    print(f"  Files {'would be ' if dry_run else ''}copied: {copied_files}")
    if skipped_files > 0:
        print(f"  Files skipped due to errors: {skipped_files}")
    
    if not dry_run:
        print(f"\nAll images have been flattened to: {output_path.absolute()}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Flatten images from multiple directories into a single directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python flatten.py                           # Use configured paths
  python flatten.py --dry-run                 # Preview what would be done
  python flatten.py --input data/raw --output flat_images
  python flatten.py --input dir1 dir2 dir3 --output combined
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        nargs='*',
        default=INPUT_DIRECTORIES,
        help='Input directories to search for images (default: configured paths)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default=OUTPUT_DIRECTORY,
        help='Output directory for flattened images (default: configured path)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually copying files'
    )
    
    parser.add_argument(
        '--list-extensions',
        action='store_true',
        help='List supported image file extensions and exit'
    )
    
    args = parser.parse_args()
    
    if args.list_extensions:
        print("Supported image file extensions:")
        for ext in sorted(SUPPORTED_EXTENSIONS):
            print(f"  {ext}")
        return
    
    print("Image Flattening Script")
    print("=" * 50)
    print(f"Input directories: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be copied ***")
    
    flatten_images(args.input, args.output, args.dry_run)


if __name__ == "__main__":
    main()