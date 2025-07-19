#!/usr/bin/env python3
"""
Dataset Processing Orchestration Script

This script orchestrates the complete dataset processing pipeline:
1. Create backup of original data with '_saved' suffix
2. Organize raw data into images/ and videos/ subdirectories
3. Rename files with UUID and folder prefix (in-place)
4. Optionally downsample videos

Usage:
    python utils/dataset/process_dataset.py /path/to/dataset
    python utils/dataset/process_dataset.py /path/to/dataset --skip-downsample
    python utils/dataset/process_dataset.py /path/to/dataset --fps 25 --resolution 720x1280
"""

import os
import sys
import shutil
import argparse
import datetime
import subprocess
from pathlib import Path

# Add utils directory to Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.dirname(script_dir)
sys.path.insert(0, utils_dir)

# Import the processing functions
from raw_dataset.organize_raw_data import organize_raw_data
from raw_dataset.rename_raw_data import rename_in_place


class DatasetProcessor:
    """Main class for orchestrating dataset processing."""
    
    def __init__(self, input_dir, skip_downsample=False, fps=None, resolution=None, force_backup=False, verbose=False):
        self.input_dir = os.path.abspath(input_dir)
        self.skip_downsample = skip_downsample
        self.fps = fps
        self.resolution = resolution
        self.force_backup = force_backup
        self.verbose = verbose
        
        # Create log file
        self.log_file = os.path.join(self.input_dir, 'dataset_processing.log')
        
        # Backup directory path
        self.backup_dir = self.input_dir + '_saved'
        
    def log(self, message, print_also=True):
        """Log message to file and optionally print to console."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        if print_also:
            print(f"[{timestamp}] {message}")
    
    def create_backup(self):
        """Create backup copy with '_saved' suffix."""
        self.log("=== STEP 0: Creating Backup ===")
        
        if os.path.exists(self.backup_dir):
            if not self.force_backup:
                raise FileExistsError(
                    f"Backup directory already exists: {self.backup_dir}\n"
                    f"Use --force-backup to overwrite or remove the existing backup."
                )
            else:
                self.log(f"Removing existing backup: {self.backup_dir}")
                shutil.rmtree(self.backup_dir)
        
        self.log(f"Creating backup: {self.input_dir} -> {self.backup_dir}")
        
        try:
            shutil.copytree(self.input_dir, self.backup_dir)
            
            # Verify backup
            if not os.path.exists(self.backup_dir):
                raise Exception("Backup creation failed - directory not found")
            
            self.log("Backup created successfully")
            return True
            
        except Exception as e:
            self.log(f"ERROR: Failed to create backup: {e}")
            raise
    
    def run_organize_step(self):
        """Execute organize raw data step."""
        self.log("=== STEP 1: Organizing Raw Data ===")
        
        try:
            # Count files before organizing
            files_before = []
            for item in os.listdir(self.input_dir):
                item_path = os.path.join(self.input_dir, item)
                if os.path.isfile(item_path):
                    files_before.append(item)
            
            self.log(f"Files found in root directory: {len(files_before)}")
            
            # Run organize function
            organize_raw_data(self.input_dir)
            
            # Check results
            images_dir = os.path.join(self.input_dir, 'images')
            videos_dir = os.path.join(self.input_dir, 'videos')
            
            image_count = len(os.listdir(images_dir)) if os.path.exists(images_dir) else 0
            video_count = len(os.listdir(videos_dir)) if os.path.exists(videos_dir) else 0
            
            self.log(f"Organization complete - Images: {image_count}, Videos: {video_count}")
            return True
            
        except Exception as e:
            self.log(f"ERROR: Organization step failed: {e}")
            raise
    
    def run_rename_step(self):
        """Execute rename files step."""
        self.log("=== STEP 2: Renaming Files ===")
        
        try:
            renamed_count = rename_in_place(self.input_dir)
            self.log(f"Renaming complete - {renamed_count} files renamed")
            return True
            
        except Exception as e:
            self.log(f"ERROR: Rename step failed: {e}")
            raise
    
    def run_downsample_step(self):
        """Execute video downsampling step."""
        self.log("=== STEP 3: Downsampling Videos ===")
        
        videos_dir = os.path.join(self.input_dir, 'videos')
        
        if not os.path.exists(videos_dir):
            self.log("No videos directory found - skipping downsample step")
            return True
        
        video_files = [f for f in os.listdir(videos_dir) 
                      if os.path.isfile(os.path.join(videos_dir, f))]
        
        if not video_files:
            self.log("No video files found - skipping downsample step")
            return True
        
        self.log(f"Found {len(video_files)} video files to process")
        
        try:
            # Build downsample command
            downsample_script = os.path.abspath(os.path.join(utils_dir, '..', 'video', 'downsample.py'))
            cmd = [sys.executable, downsample_script, videos_dir]
            
            if self.fps:
                cmd.extend(['--fps', str(self.fps)])
            if self.resolution:
                cmd.extend(['--resolution', self.resolution])
            
            self.log(f"Executing: {' '.join(cmd)}")
            
            # Run downsample script
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                self.log("Video downsampling completed successfully")
                if self.verbose and result.stdout:
                    self.log(f"Downsample output: {result.stdout}")
                return True
            else:
                error_msg = result.stderr or "Unknown error"
                self.log(f"ERROR: Video downsampling failed: {error_msg}")
                raise Exception(f"Downsample script failed: {error_msg}")
            
        except Exception as e:
            self.log(f"ERROR: Downsample step failed: {e}")
            raise
    
    def rollback(self):
        """Rollback changes by restoring from backup."""
        self.log("=== ROLLBACK: Restoring from backup ===")
        
        if not os.path.exists(self.backup_dir):
            self.log("ERROR: Cannot rollback - backup directory not found")
            return False
        
        try:
            # Remove corrupted directory
            if os.path.exists(self.input_dir):
                shutil.rmtree(self.input_dir)
            
            # Restore from backup
            shutil.move(self.backup_dir, self.input_dir)
            
            self.log("Rollback completed successfully")
            return True
            
        except Exception as e:
            self.log(f"ERROR: Rollback failed: {e}")
            return False
    
    def process(self):
        """Execute the complete processing pipeline."""
        self.log("=== DATASET PROCESSING STARTED ===")
        self.log(f"Input directory: {self.input_dir}")
        self.log(f"Skip downsample: {self.skip_downsample}")
        if not self.skip_downsample:
            self.log(f"FPS: {self.fps or 'default'}")
            self.log(f"Resolution: {self.resolution or 'default'}")
        
        try:
            # Step 0: Create backup
            self.create_backup()
            
            # Step 1: Organize raw data
            self.run_organize_step()
            
            # Step 2: Rename files
            self.run_rename_step()
            
            # Step 3: Downsample videos (optional)
            if not self.skip_downsample:
                self.run_downsample_step()
            else:
                self.log("=== STEP 3: Skipping Video Downsampling ===")
            
            self.log("=== DATASET PROCESSING COMPLETED SUCCESSFULLY ===")
            self.log(f"Original data backed up to: {self.backup_dir}")
            return True
            
        except Exception as e:
            self.log(f"=== PROCESSING FAILED: {e} ===")
            
            # Attempt rollback
            self.log("Attempting to rollback changes...")
            if self.rollback():
                self.log("Changes have been rolled back. Original data restored.")
            else:
                self.log("ERROR: Rollback failed. Check backup manually.")
            
            return False


def parse_resolution(resolution_str):
    """Parse resolution string in format 'HxW' or 'H,W'."""
    if not resolution_str:
        return None
    
    # Try different separators
    for sep in ['x', 'X', ',', ':']:
        if sep in resolution_str:
            parts = resolution_str.split(sep)
            if len(parts) == 2:
                try:
                    height, width = int(parts[0]), int(parts[1])
                    return f"{height}x{width}"
                except ValueError:
                    continue
    
    raise ValueError(f"Invalid resolution format: {resolution_str}. Use HxW (e.g., 720x1280)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process dataset through organize, rename, and downsample steps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage (all steps):
    python utils/dataset/process_dataset.py /path/to/dataset

  Skip video downsampling:
    python utils/dataset/process_dataset.py /path/to/dataset --skip-downsample

  Custom downsampling settings:
    python utils/dataset/process_dataset.py /path/to/dataset --fps 25 --resolution 720x1280

  Force overwrite existing backup:
    python utils/dataset/process_dataset.py /path/to/dataset --force-backup
        """
    )
    
    parser.add_argument('input_dir', help='Path to the dataset directory to process')
    parser.add_argument('--downsample', action='store_true',
                       help='Enable the video downsampling step (default: off)')
    parser.add_argument('--fps', type=int, 
                       help='Target FPS for video downsampling (e.g., 25)')
    parser.add_argument('--resolution', type=str, 
                       help='Target resolution for downsampling as HxW (e.g., 720x1280)')
    parser.add_argument('--force-backup', action='store_true',
                       help='Overwrite existing backup directory if it exists')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        return 1
    
    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Input path is not a directory: {args.input_dir}")
        return 1
    
    # Parse resolution if provided
    resolution = None
    if args.resolution:
        try:
            resolution = parse_resolution(args.resolution)
        except ValueError as e:
            print(f"ERROR: {e}")
            return 1
    
    # Validate FPS and resolution requirements for downsample
    if args.downsample and not args.fps and not args.resolution:
        print("WARNING: No FPS or resolution specified for downsampling.")
        print("Videos will be processed with default settings from downsample script.")
        print("Omit --downsample to skip video processing entirely.")
    
    # Create processor and run
    processor = DatasetProcessor(
        input_dir=args.input_dir,
        skip_downsample=not args.downsample,
        fps=args.fps,
        resolution=resolution,
        force_backup=args.force_backup,
        verbose=args.verbose
    )
    
    success = processor.process()
    
    if success:
        print("\n✅ Dataset processing completed successfully!")
        print(f"📁 Processed data: {args.input_dir}")
        print(f"💾 Backup saved to: {args.input_dir}_saved")
        print(f"📋 Log file: {os.path.join(args.input_dir, 'dataset_processing.log')}")
        return 0
    else:
        print("\n❌ Dataset processing failed!")
        print(f"📋 Check log file: {os.path.join(args.input_dir, 'dataset_processing.log')}")
        return 1


if __name__ == '__main__':
    sys.exit(main())