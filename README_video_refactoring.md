# Video Processing Refactoring

This document describes the refactored video processing system that separates YouTube downloading from frame extraction for better modularity and reusability.

## 🎯 Overview

The original [`utils/youtube/frame_extract.py`](utils/youtube/frame_extract.py) has been split into two focused components:

1. **YouTube Video Downloader** ([`utils/youtube/download_videos.py`](utils/youtube/download_videos.py)) - Download videos from YouTube
2. **Universal Frame Extractor** ([`utils/video/extract_frames.py`](utils/video/extract_frames.py)) - Extract frames from any video file

## 📁 New Architecture

```
utils/
├── youtube/
│   ├── download_videos.py          # YouTube-specific downloading
│   └── frame_extract.py            # Original (deprecated)
└── video/
    ├── video_utils.py              # Enhanced utilities  
    ├── extract_frames.py           # Universal frame extraction
    └── downsample.py               # Existing downsampling tools
```

## 🚀 Quick Start

### 1. Download YouTube Videos

```bash
# Download single video
python utils/youtube/download_videos.py --urls "https://www.youtube.com/watch?v=VIDEO_ID" --quality 720p

# Download from file
python utils/youtube/download_videos.py --file urls.txt --quality 1080p --output ./downloads

# Download with authentication
python utils/youtube/download_videos.py --file urls.txt --cookies cookies.txt
```

### 2. Extract Frames from Videos

```bash
# Extract every frame from single video
python utils/video/extract_frames.py --video video.mp4 --step 1 --output ./frames

# Extract every 30th frame from directory
python utils/video/extract_frames.py --directory ./videos --step 30 --recursive

# Extract with time constraints
python utils/video/extract_frames.py --video video.mp4 --step 5 --start-time 60 --end-time 180
```

### 3. Combined Workflow

```bash
# Step 1: Download videos
python utils/youtube/download_videos.py --file sailing_urls.txt --output ./raw_videos

# Step 2: Extract frames
python utils/video/extract_frames.py --directory ./raw_videos/videos --step 30 --output ./dataset_frames
```

## 📖 Detailed Usage

### YouTube Downloader

#### Basic Usage

```python
from utils.youtube.download_videos import YouTubeDownloader

# Initialize downloader
downloader = YouTubeDownloader(output_dir="./downloads", max_workers=2)

# Download single video
metadata = downloader.download_video("https://youtube.com/watch?v=VIDEO_ID", quality="720p")

# Download multiple videos
urls = ["url1", "url2", "url3"]
results = downloader.download_urls(urls, quality="1080p")

# Download from file
results = downloader.download_from_file("urls.txt", quality="720p")
```

#### Advanced Features

```python
# Get download statistics
stats = downloader.get_stats()
print(f"Success rate: {stats['success_rate']:.1f}%")
print(f"Total size: {stats['total_file_size_gb']:.2f} GB")

# List downloaded videos
videos = downloader.list_downloaded_videos()

# Cleanup failed downloads
cleaned = downloader.cleanup_failed_downloads()
```

### Frame Extractor

#### Basic Usage

```python
from utils.video.extract_frames import FrameExtractor

# Initialize extractor
extractor = FrameExtractor(output_dir="./frames", max_workers=4)

# Extract from single video
frames_count = extractor.extract_frames("video.mp4", step=1)  # Every frame
frames_count = extractor.extract_frames("video.mp4", step=30)  # Every 30th frame

# Extract from directory
results = extractor.extract_from_directory("./videos", step=10, recursive=True)
```

#### Enhanced Batch Processing

```python
# Process with time constraints
results = extractor.extract_from_directory(
    video_dir="./videos",
    step=5,
    start_time=60,    # Start at 1 minute
    end_time=300,     # End at 5 minutes
    max_frames=100    # Maximum 100 frames per video
)

# Resume interrupted processing
results = extractor.resume_extraction("./videos")

# Get comprehensive statistics
stats = extractor.get_extraction_stats()
print(f"Total frames: {stats['total_frames_extracted']}")
print(f"Processing time: {stats['total_processing_time']:.1f}s")
print(f"Success rate: {stats['success_rate']:.1f}%")
```

#### Directory Structure Management

The frame extractor automatically creates organized output:

```
Input:
videos/
├── sailing/
│   ├── race1.mp4
│   └── race2.mp4
└── training/
    └── basic.mp4

Output with create_subdirs=True:
frames/
├── sailing/
│   ├── race1/
│   │   ├── frame_race1_0.00s.jpg
│   │   ├── frame_race1_1.00s.jpg
│   │   └── ...
│   └── race2/
│       └── ...
└── training/
    └── basic/
        └── ...
```

## 🎛️ Parameter Reference

### Step Parameter

The `step` parameter controls frame extraction frequency:
- `step=1`: Extract every frame
- `step=2`: Extract every 2nd frame  
- `step=30`: Extract every 30th frame
- `step=N`: Extract every Nth frame

### Quality Settings (YouTube)

- `360p`: Low quality, small files
- `480p`: Standard definition
- `720p`: HD (recommended default)
- `1080p`: Full HD
- `2k`: 1440p
- `4k`: 2160p (large files)

## 🔧 CLI Reference

### YouTube Downloader CLI

```bash
python utils/youtube/download_videos.py [OPTIONS]

Options:
  --urls, -u TEXT          YouTube URLs to download
  --file, -f PATH          File with URLs (one per line)
  --quality, -q CHOICE     Video quality [360p|480p|720p|1080p|2k|4k]
  --output, -o PATH        Output directory
  --workers, -w INTEGER    Parallel workers (max 2 recommended)
  --cookies PATH           Cookies file for authentication
  --force                  Force redownload existing videos
  --stats                  Show download statistics
  --list                   List downloaded videos
  --cleanup                Clean up failed downloads
```

### Frame Extractor CLI

```bash
python utils/video/extract_frames.py [OPTIONS]

Options:
  --video, -v PATH         Single video file to process
  --directory, -d PATH     Directory containing videos
  --recursive, -r          Search directories recursively
  --step, -s INTEGER       Extract every Nth frame (default: 1)
  --max-frames, -m INTEGER Maximum frames per video
  --start-time FLOAT       Start time in seconds
  --end-time FLOAT         End time in seconds
  --output, -o PATH        Output directory for frames
  --no-subdirs             Don't create subdirectories
  --workers, -w INTEGER    Number of parallel workers
  --no-resume              Disable resume capability
  --resume PATH            Resume processing from directory
  --stats                  Show extraction statistics
  --cleanup                Clean up incomplete extractions
```

## 🔄 Resume Capability

The frame extractor includes robust resume functionality:

```bash
# Start processing (may be interrupted)
python utils/video/extract_frames.py --directory ./videos --step 30

# Resume from where it left off
python utils/video/extract_frames.py --resume ./videos

# Clean up incomplete extractions
python utils/video/extract_frames.py --cleanup
```

The system tracks:
- Which videos have been completed
- Which frames have been extracted
- Processing status and errors
- Ability to resume at frame level

## 📊 Performance Optimization

### YouTube Downloads
- Maximum 2 workers recommended (YouTube rate limiting)
- Use cookies for private/age-restricted videos
- Monitor success rate and adjust quality if needed

### Frame Extraction
- Adjust workers based on CPU cores (typically 2-8)
- Use `step` parameter to reduce processing time
- Enable resume for long-running jobs
- Monitor disk space (frames can be large)

### Disk Space Estimation

```python
from utils.video.video_utils import estimate_frame_extraction_size

# Estimate space needed
video_files = ["video1.mp4", "video2.mp4"]
estimate = estimate_frame_extraction_size(video_files, step=30, quality=85)
print(f"Estimated size: {estimate['estimated_size_gb']:.2f} GB")
```

## 🛠️ Advanced Features

### Batch Analysis

```python
from utils.video.video_utils import (
    find_video_files, 
    validate_video_files,
    calculate_total_frames_for_batch
)

# Find all videos in directory
video_files = find_video_files("./videos", recursive=True)

# Validate videos before processing
validation = validate_video_files(video_files)
print(f"Valid videos: {validation['total_valid']}")

# Calculate total frames for planning
stats = calculate_total_frames_for_batch(video_files, step=30)
print(f"Total frames to extract: {stats['total_frames_to_extract']}")
print(f"Estimated time: {stats['estimated_extraction_time_minutes']:.1f} minutes")
```

### Error Handling

Both components include comprehensive error handling:
- Network errors for YouTube downloads
- Corrupted video file detection
- Disk space monitoring
- Graceful failure with detailed logging

## 🔄 Migration from Original

If you were using the original [`frame_extract.py`](utils/youtube/frame_extract.py):

### Before (Original)
```python
from utils.youtube.frame_extract import YouTubeFrameExtractor

extractor = YouTubeFrameExtractor("./output")
results = extractor.process_urls(urls, frame_interval=30)
```

### After (New Components)
```python
# Step 1: Download
from utils.youtube.download_videos import YouTubeDownloader
downloader = YouTubeDownloader("./downloads")
download_results = downloader.download_urls(urls)

# Step 2: Extract frames  
from utils.video.extract_frames import FrameExtractor
extractor = FrameExtractor("./frames")
frame_results = extractor.extract_from_directory("./downloads/videos", step=30)
```

### Parameter Mapping
- `frame_interval=30` → `step=30`
- `max_frames` → `max_frames` (unchanged)
- `quality="720p"` → `quality="720p"` (unchanged)

## 🐛 Troubleshooting

### Common Issues

1. **YouTube Download Fails**
   ```bash
   # Try with cookies for authentication
   python utils/youtube/download_videos.py --urls URL --cookies cookies.txt
   
   # Check if URL is valid
   python utils/youtube/download_videos.py --urls URL --quality 480p
   ```

2. **Frame Extraction Slow**
   ```bash
   # Reduce workers if system is overloaded
   python utils/video/extract_frames.py --directory ./videos --workers 2
   
   # Increase step to extract fewer frames
   python utils/video/extract_frames.py --directory ./videos --step 60
   ```

3. **Disk Space Issues**
   ```python
   # Check space before processing
   from utils.video.video_utils import estimate_frame_extraction_size
   estimate = estimate_frame_extraction_size(video_files, step=30)
   print(f"Need {estimate['estimated_size_gb']:.1f} GB")
   ```

4. **Resume Not Working**
   ```bash
   # Clean up corrupted state and restart
   python utils/video/extract_frames.py --cleanup
   python utils/video/extract_frames.py --directory ./videos --step 30
   ```

### Log Files

Check log files for detailed error information:
- `youtube_downloader.log` - Download issues
- `frame_extraction.log` - Extraction issues

## 📈 Performance Benchmarks

Typical performance on modern hardware:

### YouTube Downloads
- 720p video: ~2-5 MB/s download speed
- Concurrent limit: 2 workers (YouTube rate limiting)
- Metadata extraction: ~1-2 seconds per video

### Frame Extraction
- Processing speed: ~5-15 fps (depends on resolution)
- Step=30: ~10x faster than step=1
- Parallel processing: Near-linear scaling up to CPU cores

### Disk Usage
- 1080p frame (JPEG 85%): ~200-500 KB per frame
- 720p frame (JPEG 85%): ~100-300 KB per frame
- Step=30: ~2GB per hour of 1080p video

## 🎯 Best Practices

1. **Plan disk space** before large extractions
2. **Use appropriate step values** (30 for most datasets)
3. **Enable resume** for long-running jobs
4. **Validate videos** before processing
5. **Monitor logs** for errors
6. **Use quality settings** appropriate for your use case
7. **Organize output** with subdirectories for large datasets

## 📝 Example Workflows

### Complete Dataset Creation
```bash
#!/bin/bash

# 1. Download videos
python utils/youtube/download_videos.py \
    --file dataset_urls.txt \
    --quality 720p \
    --output ./raw_videos

# 2. Validate downloads
python utils/youtube/download_videos.py --stats

# 3. Extract frames for training dataset
python utils/video/extract_frames.py \
    --directory ./raw_videos/videos \
    --step 30 \
    --max-frames 200 \
    --output ./dataset_frames \
    --workers 4

# 4. Check extraction results
python utils/video/extract_frames.py --stats
```

### Selective Processing
```bash
# Extract only key moments (first 5 minutes)
python utils/video/extract_frames.py \
    --directory ./videos \
    --step 10 \
    --start-time 0 \
    --end-time 300 \
    --output ./key_moments

# High-frequency sampling for specific video
python utils/video/extract_frames.py \
    --video important_video.mp4 \
    --step 1 \
    --max-frames 1000 \
    --output ./detailed_analysis
```

This refactored system provides much more flexibility, better performance, and cleaner separation of concerns while maintaining all the functionality of the original implementation.