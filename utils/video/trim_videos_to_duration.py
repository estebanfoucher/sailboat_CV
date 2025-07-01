#!/usr/bin/env python3
"""
Trim Videos to Duration Script

This script takes a folder of videos and trims them all to a specified maximum duration,
generating a detailed report before execution.

All videos longer than the specified duration will be trimmed from the beginning.
Videos shorter than the duration will be copied unchanged.

Usage:
    python trim_videos_to_duration.py /path/to/videos --max-duration 60 --output-dir /path/to/output
    python trim_videos_to_duration.py /path/to/videos --max-duration 120 --output-dir /path/to/output --dry-run
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger

# Add the project root to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.video.video_utils import (
    find_video_files, 
    get_comprehensive_video_info,
    validate_video_files,
    is_video_file
)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def trim_video_to_duration(input_path: Path, output_path: Path, max_duration: float) -> bool:
    """
    Trim a video to maximum duration using ffmpeg.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        max_duration: Maximum duration in seconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-t", str(max_duration),  # Duration from start
            "-c", "copy",  # Copy streams without re-encoding for speed
            "-y",  # Overwrite output file without asking
            str(output_path)
        ]
        
        logger.debug(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.debug(f"Successfully trimmed: {input_path.name} -> {output_path.name}")
            return True
        else:
            logger.error(f"FFmpeg error for {input_path.name}: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error trimming {input_path.name}: {e}")
        return False


def copy_video(input_path: Path, output_path: Path) -> bool:
    """
    Copy a video file unchanged.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(input_path, output_path)
        logger.debug(f"Copied unchanged: {input_path.name} -> {output_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Error copying {input_path.name}: {e}")
        return False


def analyze_videos_for_trimming(video_paths: List[Path], max_duration: float) -> Dict:
    """
    Analyze videos and categorize them for trimming operations.
    
    Args:
        video_paths: List of video file paths
        max_duration: Maximum duration in seconds
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "videos_to_trim": [],
        "videos_to_copy": [],
        "videos_with_errors": [],
        "total_videos": len(video_paths),
        "total_input_duration": 0,
        "total_output_duration": 0,
        "total_input_size_mb": 0,
        "estimated_output_size_mb": 0,
        "time_saved_seconds": 0
    }
    
    for video_path in video_paths:
        try:
            info = get_comprehensive_video_info(str(video_path))
            duration = info.get("duration_seconds", 0)
            size_mb = info.get("size_mb", 0)
            
            analysis["total_input_duration"] += duration
            analysis["total_input_size_mb"] += size_mb
            
            video_data = {
                "path": video_path,
                "filename": video_path.name,
                "current_duration": duration,
                "size_mb": size_mb,
                "resolution": info.get("resolution", "unknown"),
                "fps": info.get("fps", 0)
            }
            
            if duration > max_duration:
                # Video needs trimming
                video_data["output_duration"] = max_duration
                video_data["time_saved"] = duration - max_duration
                video_data["action"] = "trim"
                
                # Estimate output size (proportional to duration)
                size_ratio = max_duration / duration
                estimated_size = size_mb * size_ratio
                video_data["estimated_output_size_mb"] = estimated_size
                
                analysis["videos_to_trim"].append(video_data)
                analysis["total_output_duration"] += max_duration
                analysis["estimated_output_size_mb"] += estimated_size
                analysis["time_saved_seconds"] += video_data["time_saved"]
                
            else:
                # Video is short enough, copy unchanged
                video_data["output_duration"] = duration
                video_data["time_saved"] = 0
                video_data["action"] = "copy"
                video_data["estimated_output_size_mb"] = size_mb
                
                analysis["videos_to_copy"].append(video_data)
                analysis["total_output_duration"] += duration
                analysis["estimated_output_size_mb"] += size_mb
                
        except Exception as e:
            analysis["videos_with_errors"].append({
                "path": video_path,
                "filename": video_path.name,
                "error": str(e)
            })
            logger.warning(f"Could not analyze {video_path.name}: {e}")
    
    return analysis


def print_analysis_report(analysis: Dict, max_duration: float) -> None:
    """Print a detailed analysis report."""
    print("\n" + "="*80)
    print("VIDEO TRIMMING ANALYSIS REPORT")
    print("="*80)
    
    print(f"\nTarget maximum duration: {format_duration(max_duration)}")
    print(f"Total videos found: {analysis['total_videos']}")
    
    # Summary statistics
    videos_to_trim = len(analysis['videos_to_trim'])
    videos_to_copy = len(analysis['videos_to_copy'])
    videos_with_errors = len(analysis['videos_with_errors'])
    
    print(f"\nPROCESSING SUMMARY:")
    print(f"  Videos to trim: {videos_to_trim}")
    print(f"  Videos to copy unchanged: {videos_to_copy}")
    print(f"  Videos with errors: {videos_with_errors}")
    
    if videos_with_errors > 0:
        print(f"\nERROR DETAILS:")
        for video in analysis['videos_with_errors']:
            print(f"  ❌ {video['filename']}: {video['error']}")
    
    # Duration analysis
    input_duration = analysis['total_input_duration']
    output_duration = analysis['total_output_duration']
    time_saved = analysis['time_saved_seconds']
    
    print(f"\nDURATION ANALYSIS:")
    print(f"  Total input duration: {format_duration(input_duration)}")
    print(f"  Total output duration: {format_duration(output_duration)}")
    print(f"  Time saved: {format_duration(time_saved)} ({time_saved/input_duration*100:.1f}%)")
    
    # Size analysis
    input_size = analysis['total_input_size_mb']
    estimated_output_size = analysis['estimated_output_size_mb']
    size_saved = input_size - estimated_output_size
    
    print(f"\nSIZE ANALYSIS:")
    print(f"  Total input size: {input_size:.1f} MB ({input_size/1024:.1f} GB)")
    print(f"  Estimated output size: {estimated_output_size:.1f} MB ({estimated_output_size/1024:.1f} GB)")
    print(f"  Estimated size saved: {size_saved:.1f} MB ({size_saved/input_size*100:.1f}%)")
    
    # Detailed breakdown
    if videos_to_trim > 0:
        print(f"\nVIDEOS TO TRIM ({videos_to_trim}):")
        print(f"{'Filename':<35} | {'Current':<8} | {'Target':<8} | {'Saved':<8} | {'Size (MB)':<10}")
        print("-" * 80)
        
        for video in analysis['videos_to_trim']:
            current_dur = format_duration(video['current_duration'])
            target_dur = format_duration(video['output_duration'])
            saved_dur = format_duration(video['time_saved'])
            size = f"{video['size_mb']:.1f}"
            
            filename = video['filename']
            if len(filename) > 34:
                filename = filename[:31] + "..."
            
            print(f"{filename:<35} | {current_dur:<8} | {target_dur:<8} | {saved_dur:<8} | {size:<10}")
    
    if videos_to_copy > 0:
        print(f"\nVIDEOS TO COPY UNCHANGED ({videos_to_copy}):")
        print(f"{'Filename':<35} | {'Duration':<8} | {'Size (MB)':<10}")
        print("-" * 55)
        
        for video in analysis['videos_to_copy']:
            duration = format_duration(video['current_duration'])
            size = f"{video['size_mb']:.1f}"
            
            filename = video['filename']
            if len(filename) > 34:
                filename = filename[:31] + "..."
            
            print(f"{filename:<35} | {duration:<8} | {size:<10}")


def process_videos(analysis: Dict, output_dir: Path, dry_run: bool = False) -> Dict:
    """
    Process videos according to the analysis.
    
    Args:
        analysis: Analysis results from analyze_videos_for_trimming
        output_dir: Output directory for processed videos
        dry_run: If True, only show what would be done
        
    Returns:
        Dictionary with processing results
    """
    results = {
        "successful_trims": 0,
        "successful_copies": 0,
        "failed_operations": 0,
        "failed_files": []
    }
    
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    
    # Process videos that need trimming
    for video_data in analysis['videos_to_trim']:
        input_path = video_data['path']
        output_path = output_dir / input_path.name
        max_duration = video_data['output_duration']
        
        if dry_run:
            print(f"  Would trim: {input_path.name} to {format_duration(max_duration)}")
        else:
            logger.info(f"Trimming: {input_path.name} to {format_duration(max_duration)}")
            if trim_video_to_duration(input_path, output_path, max_duration):
                results["successful_trims"] += 1
            else:
                results["failed_operations"] += 1
                results["failed_files"].append(input_path.name)
    
    # Process videos that will be copied unchanged
    for video_data in analysis['videos_to_copy']:
        input_path = video_data['path']
        output_path = output_dir / input_path.name
        
        if dry_run:
            print(f"  Would copy: {input_path.name} (unchanged)")
        else:
            logger.info(f"Copying unchanged: {input_path.name}")
            if copy_video(input_path, output_path):
                results["successful_copies"] += 1
            else:
                results["failed_operations"] += 1
                results["failed_files"].append(input_path.name)
    
    return results


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Trim videos in a folder to a maximum duration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Trim all videos to 60 seconds maximum
  python trim_videos_to_duration.py /path/to/videos --max-duration 60 --output-dir /path/to/output
  
  # Preview what would be done (dry run)
  python trim_videos_to_duration.py /path/to/videos --max-duration 120 --output-dir /path/to/output --dry-run
  
  # Process recursively and trim to 2 minutes
  python trim_videos_to_duration.py /path/to/videos --max-duration 120 --output-dir /path/to/output --recursive
        """
    )
    
    parser.add_argument(
        'input_dir',
        help='Directory containing video files to process'
    )
    
    parser.add_argument(
        '--max-duration', '-d',
        type=float,
        required=True,
        help='Maximum duration in seconds (videos longer than this will be trimmed)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        required=True,
        help='Output directory for processed videos'
    )
    
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Search for videos recursively in subdirectories'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually processing videos'
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
    
    # Validate inputs
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    if args.max_duration <= 0:
        logger.error("Maximum duration must be positive")
        sys.exit(1)
    
    print("VIDEO TRIMMING TOOL")
    print("=" * 50)
    print(f"Input directory: {input_dir.absolute()}")
    print(f"Output directory: {args.output_dir.absolute()}")
    print(f"Maximum duration: {format_duration(args.max_duration)}")
    print(f"Recursive search: {'Yes' if args.recursive else 'No'}")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No videos will be processed ***")
    
    # Find video files
    logger.info("Scanning for video files...")
    video_files = find_video_files(input_dir, recursive=args.recursive)
    
    if not video_files:
        logger.error(f"No video files found in {input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Validate video files
    logger.info("Validating video files...")
    validation_results = validate_video_files(video_files)
    
    if validation_results['total_valid'] == 0:
        logger.error("No valid video files found")
        sys.exit(1)
    
    if validation_results['total_invalid'] > 0:
        logger.warning(f"Found {validation_results['total_invalid']} invalid/missing video files")
    
    # Use only valid videos for processing
    valid_video_paths = [Path(path) for path in validation_results['valid_videos']]
    
    # Analyze videos for trimming
    logger.info("Analyzing videos...")
    analysis = analyze_videos_for_trimming(valid_video_paths, args.max_duration)
    
    # Print detailed report
    print_analysis_report(analysis, args.max_duration)
    
    # Ask for confirmation if not in dry run mode
    if not args.dry_run and (analysis['videos_to_trim'] or analysis['videos_to_copy']):
        print(f"\nThis will process {len(analysis['videos_to_trim']) + len(analysis['videos_to_copy'])} videos")
        print(f"Output directory: {args.output_dir.absolute()}")
        confirm = input("\nDo you want to continue? (y/N): ").lower().strip()
        if confirm not in ['y', 'yes']:
            print("Operation cancelled.")
            sys.exit(0)
    
    # Process videos
    if analysis['videos_to_trim'] or analysis['videos_to_copy']:
        logger.info("Processing videos...")
        results = process_videos(analysis, args.output_dir, args.dry_run)
        
        # Print processing summary
        print(f"\n{'DRY RUN ' if args.dry_run else ''}PROCESSING SUMMARY:")
        print(f"  Videos trimmed: {results['successful_trims']}")
        print(f"  Videos copied unchanged: {results['successful_copies']}")
        
        if results['failed_operations'] > 0:
            print(f"  Failed operations: {results['failed_operations']}")
            print("  Failed files:")
            for filename in results['failed_files']:
                print(f"    - {filename}")
        
        if not args.dry_run:
            total_success = results['successful_trims'] + results['successful_copies']
            print(f"\nProcessing complete! {total_success} videos processed successfully.")
            print(f"Output directory: {args.output_dir.absolute()}")
    else:
        print("\nNo videos need processing.")


if __name__ == "__main__":
    main()