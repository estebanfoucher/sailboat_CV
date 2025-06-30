#!/usr/bin/env python3
"""
Example workflow demonstrating the new separated video processing components
"""

import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.youtube.download_videos import YouTubeDownloader
from utils.video.extract_frames import FrameExtractor
from utils.video.video_utils import find_video_files, validate_video_files, calculate_total_frames_for_batch

def complete_video_processing_workflow():
    """Demonstrate complete workflow from YouTube URLs to extracted frames"""
    
    print("🎬 Complete Video Processing Workflow")
    print("=" * 50)
    
    # Example YouTube URLs (replace with real URLs)
    example_urls = [
        "https://www.youtube.com/watch?v=EXAMPLE1",
        "https://www.youtube.com/watch?v=EXAMPLE2"
    ]
    
    # Step 1: Download YouTube videos
    print("\n📥 Step 1: Downloading YouTube videos...")
    downloader = YouTubeDownloader(output_dir="./demo_downloads", max_workers=2)
    
    # In real usage, you would use actual URLs:
    # download_results = downloader.download_urls(example_urls, quality="720p")
    print("   (Skipping actual download for demo)")
    
    # Step 2: Find and validate existing video files
    print("\n🔍 Step 2: Finding and validating video files...")
    # For demo, let's look for videos in the data directory
    video_files = find_video_files("./data/videos", recursive=True) if Path("./data/videos").exists() else []
    
    if video_files:
        print(f"   Found {len(video_files)} video files")
        
        # Validate videos
        validation = validate_video_files(video_files[:3])  # Validate first 3 for demo
        print(f"   ✅ Valid: {validation['total_valid']}")
        print(f"   ❌ Invalid: {validation['total_invalid']}")
        
        # Calculate batch statistics
        batch_stats = calculate_total_frames_for_batch(video_files[:3], step=30)
        print(f"   📊 Total frames to extract (step=30): {batch_stats['total_frames_to_extract']}")
        print(f"   ⏱️  Estimated processing time: {batch_stats['estimated_extraction_time_minutes']:.1f} minutes")
    else:
        print("   No video files found for demonstration")
        print("   💡 To test with real videos, place some video files in ./data/videos/")
        return
    
    # Step 3: Extract frames with enhanced batch processing
    print("\n🖼️  Step 3: Extracting frames with batch processing...")
    extractor = FrameExtractor(
        output_dir="./demo_frames",
        max_workers=2,  # Lower for demo
        create_subdirs=True,
        resume_enabled=True
    )
    
    # Extract frames from the first video only for demo
    if video_files:
        first_video = video_files[0]
        print(f"   Processing: {first_video.name}")
        
        frames_extracted = extractor.extract_frames(
            video_path=first_video,
            step=30,  # Every 30th frame
            max_frames=10  # Limit to 10 frames for demo
        )
        
        print(f"   ✅ Extracted {frames_extracted} frames")
        
        # Show statistics
        stats = extractor.get_extraction_stats()
        print(f"   📊 Processing time: {stats['total_processing_time']:.2f}s")
        print(f"   💾 Disk space used: {stats['disk_space_used_mb']:.2f} MB")
    
    print("\n🎯 Workflow Complete!")
    print("   📁 Downloaded videos: ./demo_downloads/videos/")
    print("   🖼️  Extracted frames: ./demo_frames/")

def demonstrate_individual_components():
    """Demonstrate individual component usage"""
    
    print("\n🔧 Individual Component Demonstrations")
    print("=" * 50)
    
    # YouTube Downloader Demo
    print("\n📥 YouTube Downloader Features:")
    downloader = YouTubeDownloader(output_dir="./demo_downloads")
    
    # Show available methods
    print("   • Download single video: downloader.download_video(url, quality='720p')")
    print("   • Download multiple: downloader.download_urls(urls)")
    print("   • Download from file: downloader.download_from_file('urls.txt')")
    print("   • Get statistics: downloader.get_stats()")
    print("   • List downloads: downloader.list_downloaded_videos()")
    print("   • Cleanup failed: downloader.cleanup_failed_downloads()")
    
    # Frame Extractor Demo
    print("\n🖼️  Frame Extractor Features:")
    extractor = FrameExtractor(output_dir="./demo_frames")
    
    print("   • Extract from video: extractor.extract_frames(video_path, step=1)")
    print("   • Process directory: extractor.extract_from_directory(video_dir)")
    print("   • Resume processing: extractor.resume_extraction(video_dir)")
    print("   • Get statistics: extractor.get_extraction_stats()")
    print("   • Cleanup incomplete: extractor.cleanup_incomplete_frames()")
    
    # Parameter examples
    print("\n⚙️  Key Parameters:")
    print("   • step=1  : Extract every frame")
    print("   • step=30 : Extract every 30th frame")
    print("   • quality='720p' : YouTube download quality")
    print("   • max_frames=100 : Limit frames per video")
    print("   • recursive=True : Process subdirectories")

def show_cli_examples():
    """Show command-line interface examples"""
    
    print("\n💻 Command-Line Examples")
    print("=" * 50)
    
    print("\n📥 YouTube Downloads:")
    print("   # Single video")
    print("   python utils/youtube/download_videos.py --urls 'https://youtube.com/watch?v=ID'")
    print("   ")
    print("   # Multiple videos from file")
    print("   python utils/youtube/download_videos.py --file urls.txt --quality 720p")
    print("   ")
    print("   # With authentication")
    print("   python utils/youtube/download_videos.py --file urls.txt --cookies cookies.txt")
    
    print("\n🖼️  Frame Extraction:")
    print("   # Extract every frame from single video")
    print("   python utils/video/extract_frames.py --video video.mp4 --step 1")
    print("   ")
    print("   # Process directory with every 30th frame")
    print("   python utils/video/extract_frames.py --directory ./videos --step 30 --recursive")
    print("   ")
    print("   # Extract with time limits")
    print("   python utils/video/extract_frames.py --video video.mp4 --step 5 --start-time 60 --end-time 180")
    print("   ")
    print("   # Resume interrupted processing")
    print("   python utils/video/extract_frames.py --resume ./videos")

def main():
    """Main demonstration function"""
    print("🎬 Video Processing Refactoring Demonstration")
    print("=" * 60)
    print("This script demonstrates the new separated video processing components")
    print("that replace the original monolithic frame_extract.py")
    
    # Run demonstrations
    demonstrate_individual_components()
    show_cli_examples()
    complete_video_processing_workflow()
    
    print("\n📖 For detailed documentation, see:")
    print("   • README_video_refactoring.md - Complete usage guide")
    print("   • video_refactoring_plan.md - Architecture plan")
    
    print("\n🚀 New Components Created:")
    print("   ✅ utils/youtube/download_videos.py - YouTube video downloader")
    print("   ✅ utils/video/extract_frames.py - Universal frame extractor")
    print("   ✅ utils/video/video_utils.py - Enhanced with batch processing utilities")
    
    print("\n💡 Key Improvements:")
    print("   • Separated concerns (downloading vs extraction)")
    print("   • Enhanced batch processing with parallel workers")
    print("   • Resume capability for interrupted processing")
    print("   • Organized output with subdirectory mirroring")
    print("   • Step parameter (default=1, every frame)")
    print("   • Comprehensive error handling and statistics")
    print("   • Works with any video format, not just YouTube")

if __name__ == "__main__":
    main()