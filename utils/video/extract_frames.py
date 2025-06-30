#!/usr/bin/env python3
"""
Universal Frame Extractor
Extract frames from video files with enhanced batch processing capabilities
"""

import os
import sys
import json
import cv2
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from tqdm import tqdm
import argparse
import time

# Import existing video utilities
sys.path.append(str(Path(__file__).parent.parent))
from video.video_utils import is_video_file, get_video_fps, get_video_duration, VIDEO_EXTENSIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('frame_extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FrameExtractionState:
    """Track processing state for resume capability"""
    video_path: str
    total_frames: int
    extracted_frames: int
    last_processed_frame: int
    start_time: str
    status: str  # 'in_progress', 'completed', 'failed'
    error_message: Optional[str] = None

@dataclass
class ExtractionStats:
    """Statistics for frame extraction operations"""
    total_videos_found: int = 0
    total_videos_processed: int = 0
    total_frames_extracted: int = 0
    total_processing_time: float = 0.0
    failed_videos: int = 0
    skipped_videos: int = 0
    average_fps_processed: float = 0.0
    disk_space_used_mb: float = 0.0

class FrameExtractor:
    """Universal frame extractor with enhanced batch processing"""
    
    def __init__(self, output_dir: str = "./frames", max_workers: int = 4, 
                 create_subdirs: bool = True, resume_enabled: bool = True):
        """
        Initialize the frame extractor
        
        Args:
            output_dir: Root directory for extracted frames
            max_workers: Number of parallel workers for batch processing
            create_subdirs: Create subdirectories mirroring input structure
            resume_enabled: Enable resume capability for interrupted processing
        """
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.create_subdirs = create_subdirs
        self.resume_enabled = resume_enabled
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State management for resume capability
        self.state_dir = self.output_dir / ".extraction_state"
        if self.resume_enabled:
            self.state_dir.mkdir(exist_ok=True)
        
        # Statistics tracking
        self.stats = ExtractionStats()
        
        logger.info(f"Frame extractor initialized: {self.output_dir}")
        logger.info(f"Workers: {self.max_workers}, Subdirs: {self.create_subdirs}, Resume: {self.resume_enabled}")

    def find_video_files(self, directory: Union[str, Path], recursive: bool = True) -> List[Path]:
        """
        Find all video files in directory tree
        
        Args:
            directory: Directory to search
            recursive: Search recursively in subdirectories
            
        Returns:
            List of video file paths
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        video_files = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and is_video_file(str(file_path)):
                video_files.append(file_path)
        
        # Sort for consistent processing order
        video_files.sort()
        self.stats.total_videos_found = len(video_files)
        
        logger.info(f"Found {len(video_files)} video files in {directory}")
        return video_files

    def get_video_info(self, video_path: Union[str, Path]) -> Dict:
        """
        Get comprehensive video information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            # Use OpenCV for frame count and basic info
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Get duration from video_utils if available
            try:
                duration = get_video_duration(str(video_path))
            except:
                duration = frame_count / fps if fps > 0 else 0
            
            return {
                "path": str(video_path),
                "filename": video_path.name,
                "size_bytes": video_path.stat().st_size,
                "fps": fps,
                "frame_count": frame_count,
                "duration_seconds": duration,
                "resolution": f"{width}x{height}",
                "width": width,
                "height": height
            }
            
        except Exception as e:
            logger.error(f"Error getting video info for {video_path}: {e}")
            raise

    def _get_output_path(self, video_path: Path, relative_to: Optional[Path] = None) -> Path:
        """
        Get organized output path for video frames
        
        Args:
            video_path: Source video file path
            relative_to: Base directory for relative path calculation
            
        Returns:
            Output directory path for this video's frames
        """
        if self.create_subdirs and relative_to:
            # Create subdirectory structure mirroring input
            relative_path = video_path.relative_to(relative_to)
            video_subdir = relative_path.parent / video_path.stem
            output_path = self.output_dir / video_subdir
        else:
            # Simple subdirectory with video name
            output_path = self.output_dir / video_path.stem
        
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def _save_extraction_state(self, state: FrameExtractionState):
        """Save extraction state for resume capability"""
        if not self.resume_enabled:
            return
        
        state_file = self.state_dir / f"{Path(state.video_path).stem}.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(state), f, indent=2)

    def _load_extraction_state(self, video_path: Path) -> Optional[FrameExtractionState]:
        """Load extraction state for resume capability"""
        if not self.resume_enabled:
            return None
        
        state_file = self.state_dir / f"{video_path.stem}.json"
        if not state_file.exists():
            return None
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return FrameExtractionState(**data)
        except Exception as e:
            logger.warning(f"Could not load state for {video_path}: {e}")
            return None

    def _should_skip_video(self, video_path: Path, output_path: Path) -> bool:
        """Check if video should be skipped (already processed)"""
        if not self.resume_enabled:
            return False
        
        state = self._load_extraction_state(video_path)
        if state and state.status == 'completed':
            logger.info(f"Skipping completed video: {video_path.name}")
            return True
        
        return False

    def extract_frames(self, video_path: Union[str, Path], step: int = 1, 
                      max_frames: Optional[int] = None, start_time: float = 0, 
                      end_time: Optional[float] = None, output_path: Optional[Path] = None) -> int:
        """
        Extract frames from a single video
        
        Args:
            video_path: Path to video file
            step: Extract every Nth frame (default=1 for every frame)
            max_frames: Maximum number of frames to extract
            start_time: Start time in seconds
            end_time: End time in seconds (None for end of video)
            output_path: Custom output path (None for auto-generated)
            
        Returns:
            Number of frames extracted
        """
        video_path = Path(video_path)
        start_extraction_time = time.time()
        
        try:
            # Get video information
            video_info = self.get_video_info(video_path)
            fps = video_info["fps"]
            total_frames = video_info["frame_count"]
            duration = video_info["duration_seconds"]
            
            # Determine output path
            if output_path is None:
                output_path = self._get_output_path(video_path)
            
            # Check if we should skip this video
            if self._should_skip_video(video_path, output_path):
                self.stats.skipped_videos += 1
                return 0
            
            # Calculate frame range
            start_frame = int(start_time * fps) if start_time > 0 else 0
            if end_time:
                end_frame = min(int(end_time * fps), total_frames)
            else:
                end_frame = total_frames
            
            # Generate frame indices with step
            frame_indices = list(range(start_frame, end_frame, step))
            if max_frames:
                frame_indices = frame_indices[:max_frames]
            
            if not frame_indices:
                logger.warning(f"No frames to extract for {video_path.name}")
                return 0
            
            logger.info(f"Extracting {len(frame_indices)} frames from {video_path.name} (step={step})")
            
            # Initialize extraction state
            state = FrameExtractionState(
                video_path=str(video_path),
                total_frames=len(frame_indices),
                extracted_frames=0,
                last_processed_frame=start_frame,
                start_time=datetime.now().isoformat(),
                status='in_progress'
            )
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            
            extracted_count = 0
            frame_size_total = 0
            
            # Extract frames with progress bar
            desc = f"Extracting {video_path.name}"
            with tqdm(total=len(frame_indices), desc=desc, leave=False) as pbar:
                for i, frame_idx in enumerate(frame_indices):
                    # Set frame position
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if not ret:
                        logger.warning(f"Could not read frame {frame_idx} from {video_path.name}")
                        break
                    
                    # Generate frame filename
                    timestamp = frame_idx / fps
                    frame_filename = f"frame_{video_path.stem}_{timestamp:.2f}s.jpg"
                    frame_path = output_path / frame_filename
                    
                    # Save frame
                    success = cv2.imwrite(str(frame_path), frame)
                    if success:
                        extracted_count += 1
                        frame_size_total += frame_path.stat().st_size
                        
                        # Update state periodically
                        state.extracted_frames = extracted_count
                        state.last_processed_frame = frame_idx
                        if i % 50 == 0:  # Save state every 50 frames
                            self._save_extraction_state(state)
                    else:
                        logger.warning(f"Failed to save frame {frame_filename}")
                    
                    pbar.update(1)
            
            cap.release()
            
            # Update final state
            state.extracted_frames = extracted_count
            state.status = 'completed'
            self._save_extraction_state(state)
            
            # Update statistics
            extraction_time = time.time() - start_extraction_time
            self.stats.total_videos_processed += 1
            self.stats.total_frames_extracted += extracted_count
            self.stats.total_processing_time += extraction_time
            self.stats.disk_space_used_mb += frame_size_total / (1024 * 1024)
            
            logger.info(f"✅ Extracted {extracted_count} frames from {video_path.name} in {extraction_time:.1f}s")
            return extracted_count
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            # Update state with error
            if 'state' in locals():
                state.status = 'failed'
                state.error_message = str(e)
                self._save_extraction_state(state)
            
            self.stats.failed_videos += 1
            return 0

    def extract_from_videos(self, video_paths: List[Path], step: int = 1, 
                           max_frames: Optional[int] = None, start_time: float = 0,
                           end_time: Optional[float] = None) -> Dict:
        """
        Extract frames from multiple videos with parallel processing
        
        Args:
            video_paths: List of video file paths
            step: Extract every Nth frame
            max_frames: Maximum frames per video
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Dictionary with processing results
        """
        if not video_paths:
            logger.warning("No video paths provided")
            return {"success": [], "errors": [], "total_frames": 0}
        
        results = {"success": [], "errors": [], "total_frames": 0}
        batch_start_time = time.time()
        
        logger.info(f"Starting batch extraction: {len(video_paths)} videos, {self.max_workers} workers")
        
        # Process videos in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_video = {
                executor.submit(
                    self.extract_frames, video_path, step, max_frames, start_time, end_time
                ): video_path for video_path in video_paths
            }
            
            # Process completed tasks
            with tqdm(total=len(video_paths), desc="Processing videos") as pbar:
                for future in as_completed(future_to_video):
                    video_path = future_to_video[future]
                    try:
                        frames_extracted = future.result()
                        if frames_extracted > 0:
                            results["success"].append({
                                "video_path": str(video_path),
                                "frames_extracted": frames_extracted
                            })
                            results["total_frames"] += frames_extracted
                        else:
                            results["errors"].append({
                                "video_path": str(video_path),
                                "error": "No frames extracted"
                            })
                    except Exception as e:
                        results["errors"].append({
                            "video_path": str(video_path),
                            "error": str(e)
                        })
                        logger.error(f"Failed to process {video_path}: {e}")
                    
                    pbar.update(1)
        
        # Calculate final statistics
        total_time = time.time() - batch_start_time
        if self.stats.total_processing_time > 0:
            self.stats.average_fps_processed = self.stats.total_frames_extracted / self.stats.total_processing_time
        
        logger.info(f"✅ Batch extraction completed in {total_time:.1f}s")
        logger.info(f"📊 Success: {len(results['success'])}, Errors: {len(results['errors'])}")
        logger.info(f"🖼️  Total frames: {results['total_frames']}")
        
        return results

    def extract_from_directory(self, video_dir: Union[str, Path], step: int = 1,
                              max_frames: Optional[int] = None, start_time: float = 0,
                              end_time: Optional[float] = None, recursive: bool = True) -> Dict:
        """
        Extract frames from all videos in a directory
        
        Args:
            video_dir: Directory containing videos
            step: Extract every Nth frame
            max_frames: Maximum frames per video
            start_time: Start time in seconds
            end_time: End time in seconds
            recursive: Search recursively in subdirectories
            
        Returns:
            Dictionary with processing results
        """
        video_dir = Path(video_dir)
        logger.info(f"Starting directory extraction: {video_dir}")
        
        # Find all video files
        video_files = self.find_video_files(video_dir, recursive)
        if not video_files:
            logger.warning(f"No video files found in {video_dir}")
            return {"success": [], "errors": [], "total_frames": 0}
        
        # Set relative path for organized output
        if self.create_subdirs:
            for video_path in video_files:
                # Pre-calculate output paths to ensure directory structure
                self._get_output_path(video_path, video_dir)
        
        # Process all videos
        return self.extract_from_videos(video_files, step, max_frames, start_time, end_time)

    def resume_extraction(self, video_dir: Union[str, Path]) -> Dict:
        """
        Resume interrupted extraction from directory
        
        Args:
            video_dir: Directory that was being processed
            
        Returns:
            Dictionary with resume results
        """
        if not self.resume_enabled:
            logger.warning("Resume capability is disabled")
            return {"success": [], "errors": [], "total_frames": 0}
        
        video_dir = Path(video_dir)
        logger.info(f"Resuming extraction from {video_dir}")
        
        # Find videos that need processing
        video_files = self.find_video_files(video_dir, recursive=True)
        incomplete_videos = []
        
        for video_path in video_files:
            state = self._load_extraction_state(video_path)
            if not state or state.status in ['in_progress', 'failed']:
                incomplete_videos.append(video_path)
        
        logger.info(f"Found {len(incomplete_videos)} videos to resume/retry")
        
        if incomplete_videos:
            return self.extract_from_videos(incomplete_videos)
        else:
            logger.info("No videos need resuming")
            return {"success": [], "errors": [], "total_frames": 0}

    def get_extraction_stats(self) -> Dict:
        """Get comprehensive extraction statistics"""
        stats_dict = asdict(self.stats)
        stats_dict.update({
            "success_rate": (
                self.stats.total_videos_processed / max(self.stats.total_videos_found, 1) * 100
            ),
            "average_frames_per_video": (
                self.stats.total_frames_extracted / max(self.stats.total_videos_processed, 1)
            ),
            "disk_space_used_gb": self.stats.disk_space_used_mb / 1024
        })
        return stats_dict

    def cleanup_incomplete_frames(self):
        """Remove frames from incomplete extractions"""
        if not self.resume_enabled:
            return
        
        cleaned_count = 0
        for state_file in self.state_dir.glob("*.json"):
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                if state_data.get('status') == 'failed':
                    video_path = Path(state_data['video_path'])
                    output_path = self._get_output_path(video_path)
                    
                    if output_path.exists():
                        import shutil
                        shutil.rmtree(output_path)
                        cleaned_count += 1
                        logger.info(f"Cleaned incomplete frames: {output_path}")
                    
                    # Remove state file
                    state_file.unlink()
                    
            except Exception as e:
                logger.warning(f"Error cleaning {state_file}: {e}")
        
        logger.info(f"Cleaned {cleaned_count} incomplete extractions")


def main():
    """Command-line interface for frame extraction"""
    parser = argparse.ArgumentParser(description="Universal Frame Extractor")
    
    # Input options
    parser.add_argument("--video", "-v", help="Single video file to process")
    parser.add_argument("--directory", "-d", help="Directory containing videos")
    parser.add_argument("--recursive", "-r", action="store_true", default=True, 
                       help="Search directories recursively")
    
    # Processing options
    parser.add_argument("--step", "-s", type=int, default=1, 
                       help="Extract every Nth frame (default: 1)")
    parser.add_argument("--max-frames", "-m", type=int, 
                       help="Maximum frames per video")
    parser.add_argument("--start-time", type=float, default=0, 
                       help="Start time in seconds")
    parser.add_argument("--end-time", type=float, 
                       help="End time in seconds")
    
    # Output options
    parser.add_argument("--output", "-o", default="./frames", 
                       help="Output directory for frames")
    parser.add_argument("--no-subdirs", action="store_true", 
                       help="Don't create subdirectories")
    
    # Performance options
    parser.add_argument("--workers", "-w", type=int, default=4, 
                       help="Number of parallel workers")
    parser.add_argument("--no-resume", action="store_true", 
                       help="Disable resume capability")
    
    # Utility options
    parser.add_argument("--resume", help="Resume processing from directory")
    parser.add_argument("--stats", action="store_true", 
                       help="Show extraction statistics")
    parser.add_argument("--cleanup", action="store_true", 
                       help="Clean up incomplete extractions")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = FrameExtractor(
        output_dir=args.output,
        max_workers=args.workers,
        create_subdirs=not args.no_subdirs,
        resume_enabled=not args.no_resume
    )
    
    # Handle utility operations
    if args.cleanup:
        extractor.cleanup_incomplete_frames()
        return
    
    if args.stats:
        stats = extractor.get_extraction_stats()
        print("\n📊 EXTRACTION STATISTICS")
        print("=" * 50)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        return
    
    # Handle resume operation
    if args.resume:
        results = extractor.resume_extraction(args.resume)
    
    # Handle processing operations
    elif args.video:
        # Single video processing
        frames_extracted = extractor.extract_frames(
            video_path=args.video,
            step=args.step,
            max_frames=args.max_frames,
            start_time=args.start_time,
            end_time=args.end_time
        )
        results = {
            "success": [{"video_path": args.video, "frames_extracted": frames_extracted}] if frames_extracted > 0 else [],
            "errors": [] if frames_extracted > 0 else [{"video_path": args.video, "error": "No frames extracted"}],
            "total_frames": frames_extracted
        }
    
    elif args.directory:
        # Directory processing
        results = extractor.extract_from_directory(
            video_dir=args.directory,
            step=args.step,
            max_frames=args.max_frames,
            start_time=args.start_time,
            end_time=args.end_time,
            recursive=args.recursive
        )
    
    else:
        logger.error("No input specified. Use --video or --directory")
        return
    
    # Display results
    print("\n🎯 EXTRACTION RESULTS")
    print("=" * 50)
    print(f"✅ Successful videos: {len(results['success'])}")
    print(f"❌ Failed videos: {len(results['errors'])}")
    print(f"🖼️  Total frames extracted: {results['total_frames']}")
    
    # Show statistics
    stats = extractor.get_extraction_stats()
    print(f"📊 Success rate: {stats['success_rate']:.1f}%")
    print(f"⏱️  Processing time: {stats['total_processing_time']:.1f}s")
    print(f"💾 Disk space used: {stats['disk_space_used_gb']:.2f} GB")
    
    # Show errors if any
    if results['errors']:
        print("\n❌ ERRORS:")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  - {Path(error['video_path']).name}: {error['error']}")
        if len(results['errors']) > 5:
            print(f"  ... and {len(results['errors']) - 5} more errors")


if __name__ == "__main__":
    main()