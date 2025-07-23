#!/usr/bin/env python3
"""
YouTube Video Downloader
Download videos from YouTube URLs with metadata extraction and batch processing
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import yt_dlp
from tqdm import tqdm
import argparse
import cv2

# Add project root to Python path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from utils.youtube.youtube_utils import get_video_id_from_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_downloader.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    """Structure for video metadata"""
    video_id: str
    title: str
    description: str
    duration: int
    upload_date: str
    uploader: str
    view_count: int
    like_count: int
    url: str
    file_size: int
    resolution: str
    fps: float
    format: str
    download_date: str
    file_path: str

@dataclass
class DownloadStats:
    """Statistics for download operations"""
    total_urls_processed: int = 0
    successful_downloads: int = 0
    failed_downloads: int = 0
    skipped_downloads: int = 0
    total_file_size_mb: float = 0.0
    total_download_time: float = 0.0
    average_download_speed_mbps: float = 0.0

class YouTubeDownloader:
    """YouTube video downloader with batch processing and metadata management"""
    
    def __init__(self, output_dir: str = "./videos", max_workers: int = 2):
        """
        Initialize YouTube downloader
        
        Args:
            output_dir: Directory for downloaded videos
            max_workers: Number of parallel download workers (keep low for YouTube)
        """
        self.output_dir = Path(output_dir)
        self.max_workers = min(max_workers, 2)  # YouTube rate limiting
        
        # Create directory structure
        self.videos_dir = self.output_dir / "videos"
        self.metadata_dir = self.output_dir / "metadata"
        self.logs_dir = self.output_dir / "logs"
        
        for directory in [self.videos_dir, self.metadata_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = DownloadStats()
        
        logger.info(f"YouTube downloader initialized: {self.output_dir}")
        logger.info(f"Max workers: {self.max_workers}")

    def _is_video_downloaded(self, video_id: str) -> bool:
        """Check if video is already downloaded"""
        video_files = list(self.videos_dir.glob(f"{video_id}.*"))
        return any(f.suffix in ['.mp4', '.webm', '.mkv', '.avi'] for f in video_files)

    def _save_metadata(self, metadata: VideoMetadata):
        """Save video metadata to JSON file"""
        metadata_file = self.metadata_dir / f"{metadata.video_id}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)

    def _load_metadata(self, video_id: str) -> Optional[Dict]:
        """Load video metadata from JSON file"""
        metadata_file = self.metadata_dir / f"{video_id}.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata for {video_id}: {e}")
        return None

    def _get_video_file_info(self, video_path: Path) -> Dict:
        """Get technical information from downloaded video file"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {"fps": 0, "resolution": "unknown"}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            return {
                "fps": fps,
                "resolution": f"{width}x{height}",
                "width": width,
                "height": height
            }
        except Exception as e:
            logger.warning(f"Could not analyze video file {video_path}: {e}")
            return {"fps": 0, "resolution": "unknown"}

    def download_video(self, url: str, quality: str = "1080p", 
                      cookies: Optional[str] = None, force_redownload: bool = False, max_duration: int = 600) -> Optional[Dict]:
        """
        Download a single video from YouTube
        
        Args:
            url: YouTube video URL
            quality: Video quality preference (360p, 480p, 720p, 1080p, 2k, 4k)
            cookies: Path to cookies file for authentication
            force_redownload: Force redownload even if video exists
            
        Returns:
            Dictionary with video metadata or None if failed
        """
        import time
        start_time = time.time()
        
        try:
            # Map quality to height
            quality_map = {
                "360p": 360, "480p": 480, "720p": 720, "1080p": 1080,
                "2k": 1440, "4k": 2160
            }
            height = quality_map.get(quality, 1080)
            
            # Configure yt-dlp options
            ydl_opts = {
                'format': f'bestvideo[height<={height}]+bestaudio/best[height<={height}]',
                'outtmpl': str(self.videos_dir / '%(id)s.%(ext)s'),
                'writeinfojson': False,
                'writethumbnail': True,
                'writesubtitles': False,
                'writeautomaticsub': False,
                'ignoreerrors': False,
            }
            
            if cookies:
                ydl_opts['cookiefile'] = cookies
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract metadata first
                logger.info(f"Extracting metadata: {url}")
                info = ydl.extract_info(url, download=False)
                video_id = info['id']
                duration = info.get('duration', 0)
                if duration > max_duration:
                    logger.info(f"Skipping {video_id}: duration {duration}s exceeds max_duration {max_duration}s")
                    self.stats.skipped_downloads += 1
                    self.stats.total_urls_processed += 1
                    return None
                
                # Check if already downloaded
                if not force_redownload and self._is_video_downloaded(video_id):
                    logger.info(f"Video {video_id} already downloaded, skipping")
                    existing_metadata = self._load_metadata(video_id)
                    if existing_metadata:
                        self.stats.skipped_downloads += 1
                        return existing_metadata
                
                # Download the video
                logger.info(f"Downloading: {info.get('title', 'Unknown Title')} ({quality})")
                ydl.download([url])
                
                # Find downloaded video file
                video_files = list(self.videos_dir.glob(f"{video_id}.*"))
                video_file = next(
                    (f for f in video_files if f.suffix in ['.mp4', '.webm', '.mkv', '.avi']), 
                    None
                )
                
                if not video_file:
                    logger.error(f"Downloaded video file not found for {video_id}")
                    self.stats.failed_downloads += 1
                    return None
                
                # Get technical video information
                video_info = self._get_video_file_info(video_file)
                file_size = video_file.stat().st_size
                
                # Create metadata object
                metadata = VideoMetadata(
                    video_id=video_id,
                    title=info.get('title', ''),
                    description=info.get('description', '')[:1000] if info.get('description') else '',
                    duration=info.get('duration', 0),
                    upload_date=info.get('upload_date', ''),
                    uploader=info.get('uploader', ''),
                    view_count=info.get('view_count', 0),
                    like_count=info.get('like_count', 0),
                    url=url,
                    file_size=file_size,
                    resolution=video_info["resolution"],
                    fps=video_info["fps"],
                    format=video_file.suffix[1:],
                    download_date=datetime.now().isoformat(),
                    file_path=str(video_file)
                )
                
                # Save metadata
                self._save_metadata(metadata)
                
                # Update statistics
                download_time = time.time() - start_time
                self.stats.successful_downloads += 1
                self.stats.total_file_size_mb += file_size / (1024 * 1024)
                self.stats.total_download_time += download_time
                
                logger.info(f"✅ Downloaded: {video_id} ({metadata.title}) - {file_size/(1024*1024):.1f}MB in {download_time:.1f}s")
                return asdict(metadata)
                
        except Exception as e:
            self.stats.failed_downloads += 1
            logger.error(f"Error downloading {url}: {e}")
            return None
        finally:
            self.stats.total_urls_processed += 1

    def download_urls(self, urls: List[str], quality: str = "720p", 
                     cookies: Optional[str] = None, force_redownload: bool = False, max_duration: int = 600) -> Dict:
        """
        Download multiple videos from YouTube URLs with parallel processing
        
        Args:
            urls: List of YouTube URLs
            quality: Video quality preference
            cookies: Path to cookies file
            force_redownload: Force redownload existing videos
            
        Returns:
            Dictionary with download results
        """
        if not urls:
            logger.warning("No URLs provided for download")
            return {"success": [], "errors": [], "total_size_mb": 0}
        
        results = {"success": [], "errors": [], "total_size_mb": 0}
        
        logger.info(f"Starting batch download: {len(urls)} URLs with {self.max_workers} workers")
        
        # Use ThreadPoolExecutor for parallel downloads (limited for YouTube)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit download tasks
            future_to_url = {
                executor.submit(
                    self.download_video, url, quality, cookies, force_redownload, max_duration
                ): url for url in urls
            }
            
            # Process completed downloads
            with tqdm(total=len(urls), desc="Downloading videos") as pbar:
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        metadata = future.result()
                        if metadata:
                            results["success"].append({
                                "url": url,
                                "video_id": metadata["video_id"],
                                "title": metadata["title"],
                                "file_size_mb": metadata["file_size"] / (1024 * 1024),
                                "file_path": metadata["file_path"]
                            })
                            results["total_size_mb"] += metadata["file_size"] / (1024 * 1024)
                        else:
                            results["errors"].append({
                                "url": url,
                                "error": "Download failed"
                            })
                    except Exception as e:
                        results["errors"].append({
                            "url": url,
                            "error": str(e)
                        })
                    
                    pbar.update(1)
        
        # Calculate final statistics
        if self.stats.total_download_time > 0:
            self.stats.average_download_speed_mbps = (
                self.stats.total_file_size_mb / self.stats.total_download_time
            )
        
        # Save batch report
        report = {
            "download_session": {
                "timestamp": datetime.now().isoformat(),
                "total_urls": len(urls),
                "successful": len(results["success"]),
                "failed": len(results["errors"]),
                "quality": quality,
                "total_size_mb": results["total_size_mb"]
            },
            "results": results,
            "statistics": asdict(self.stats)
        }
        
        report_file = self.logs_dir / f"download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Batch download completed:")
        logger.info(f"   Success: {len(results['success'])}, Errors: {len(results['errors'])}")
        logger.info(f"   Total size: {results['total_size_mb']:.1f} MB")
        logger.info(f"   Report saved: {report_file}")
        
        return results

    def download_from_file(self, file_path: Union[str, Path], quality: str = "720p",
                          cookies: Optional[str] = None, force_redownload: bool = False, max_duration: int = 600) -> Dict:
        """
        Download videos from URLs listed in a text file
        
        Args:
            file_path: Path to file containing URLs (one per line)
            quality: Video quality preference
            cookies: Path to cookies file
            force_redownload: Force redownload existing videos
            
        Returns:
            Dictionary with download results
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"URL file not found: {file_path}")
        
        logger.info(f"Loading URLs from file: {file_path}")
        
        # Read URLs from file
        urls = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip empty lines and comments
                        url = line.split(';')[0].strip()
                        if 'youtube.com' in url or 'youtu.be' in url:
                            urls.append(url)
                        else:
                            logger.warning(f"Line {line_num}: Not a YouTube URL: {url}")
        except Exception as e:
            logger.error(f"Error reading URL file {file_path}: {e}")
            raise
        
        if not urls:
            logger.warning(f"No valid YouTube URLs found in {file_path}")
            return {"success": [], "errors": [], "total_size_mb": 0}
        
        logger.info(f"Found {len(urls)} valid YouTube URLs")
        return self.download_urls(urls, quality, cookies, force_redownload, max_duration)

    def get_stats(self) -> Dict:
        """Get comprehensive download statistics"""
        stats_dict = asdict(self.stats)
        
        # Calculate additional metrics
        if self.stats.total_urls_processed > 0:
            stats_dict["success_rate"] = (
                self.stats.successful_downloads / self.stats.total_urls_processed * 100
            )
        else:
            stats_dict["success_rate"] = 0
        
        # Average file size
        if self.stats.successful_downloads > 0:
            stats_dict["average_file_size_mb"] = (
                self.stats.total_file_size_mb / self.stats.successful_downloads
            )
        else:
            stats_dict["average_file_size_mb"] = 0
        
        # Convert to GB
        stats_dict["total_file_size_gb"] = self.stats.total_file_size_mb / 1024
        
        return stats_dict

    def cleanup_failed_downloads(self) -> int:
        """Remove incomplete or corrupted download files"""
        cleaned_count = 0
        
        # Check for video files without metadata
        for video_file in self.videos_dir.glob("*.*"):
            if video_file.suffix in ['.mp4', '.webm', '.mkv', '.avi']:
                video_id = video_file.stem
                metadata_file = self.metadata_dir / f"{video_id}.json"
                
                if not metadata_file.exists():
                    logger.info(f"Removing orphaned video file: {video_file.name}")
                    video_file.unlink()
                    cleaned_count += 1
                    
                    # Also remove thumbnail if exists
                    thumbnail_file = self.videos_dir / f"{video_id}.webp"
                    if thumbnail_file.exists():
                        thumbnail_file.unlink()
        
        # Check for zero-size files
        for video_file in self.videos_dir.glob("*.*"):
            if video_file.stat().st_size == 0:
                logger.info(f"Removing zero-size file: {video_file.name}")
                video_file.unlink()
                cleaned_count += 1
        
        logger.info(f"Cleanup completed: {cleaned_count} files removed")
        return cleaned_count

    def list_downloaded_videos(self) -> List[Dict]:
        """List all downloaded videos with metadata"""
        videos = []
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Check if video file still exists
                video_path = Path(metadata.get('file_path', ''))
                if video_path.exists():
                    videos.append(metadata)
                    
            except Exception as e:
                logger.warning(f"Error reading metadata {metadata_file}: {e}")
        
        # Sort by download date
        videos.sort(key=lambda x: x.get('download_date', ''), reverse=True)
        return videos


def main():
    """Command-line interface for YouTube video downloading"""
    parser = argparse.ArgumentParser(description="YouTube Video Downloader")
    
    # Input options
    parser.add_argument("--file", "-f", help="File containing YouTube URLs (one per line)")
    
    # Download options
    parser.add_argument("--quality", "-q", default="1080p", 
                       choices=["360p", "480p", "720p", "1080p", "2k", "4k"],
                       help="Video quality preference")
    parser.add_argument("--cookies", help="Path to cookies.txt file for authentication")
    parser.add_argument("--force", action="store_true", 
                       help="Force redownload existing videos")
    parser.add_argument("--max_duration", type=int, default=600, help="Maximum allowed video duration in seconds (default: 600)")
    
    # Output options
    parser.add_argument("--output", "-o", default="./downloaded_data",
                       help="Output directory for videos")
    parser.add_argument("--workers", "-w", type=int, default=2,
                       help="Number of parallel download workers (max 2 recommended)")
    
    # Utility options
    parser.add_argument("--stats", action="store_true", help="Show download statistics")
    parser.add_argument("--list", action="store_true", help="List downloaded videos")
    parser.add_argument("--cleanup", action="store_true", help="Clean up failed downloads")
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = YouTubeDownloader(
        output_dir=args.output,
        max_workers=min(args.workers, 2)  # Respect YouTube rate limits
    )
    
    # Handle utility operations
    if args.cleanup:
        cleaned = downloader.cleanup_failed_downloads()
        print(f"Cleaned up {cleaned} incomplete download files")
        return
    
    if args.stats:
        stats = downloader.get_stats()
        print("\n📊 DOWNLOAD STATISTICS")
        print("=" * 50)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        return
    
    if args.list:
        videos = downloader.list_downloaded_videos()
        print(f"\n📹 DOWNLOADED VIDEOS ({len(videos)} total)")
        print("=" * 80)
        for video in videos[:10]:  # Show last 10
            size_mb = video.get('file_size', 0) / (1024 * 1024)
            print(f"• {video.get('title', 'Unknown')[:50]}...")
            print(f"  ID: {video.get('video_id', 'N/A')} | Size: {size_mb:.1f}MB | "
                  f"Quality: {video.get('resolution', 'N/A')}")
        if len(videos) > 10:
            print(f"... and {len(videos) - 10} more videos")
        return
    
    # Handle download operations
    if args.file:
        # Download from file
        results = downloader.download_from_file(
            file_path=args.file,
            quality=args.quality,
            cookies=args.cookies,
            force_redownload=args.force,
            max_duration=args.max_duration
        )
    else:
        logger.error("No URL file specified. Use --file")
        return
    
    # Display results
    print("\n🎯 DOWNLOAD RESULTS")
    print("=" * 50)
    print(f"✅ Successful downloads: {len(results['success'])}")
    print(f"❌ Failed downloads: {len(results['errors'])}")
    print(f"📦 Total size: {results['total_size_mb']:.1f} MB")
    
    # Show download statistics
    stats = downloader.get_stats()
    print(f"📊 Success rate: {stats['success_rate']:.1f}%")
    print(f"⏱️  Total download time: {stats['total_download_time']:.1f}s")
    if stats['average_download_speed_mbps'] > 0:
        print(f"🚀 Average speed: {stats['average_download_speed_mbps']:.2f} MB/s")
    
    # Show errors if any
    if results['errors']:
        print("\n❌ ERRORS:")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  - {error['url']}: {error['error']}")
        if len(results['errors']) > 5:
            print(f"  ... and {len(results['errors']) - 5} more errors")


if __name__ == "__main__":
    main()