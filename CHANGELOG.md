# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2025-06-24

### Added

#### 🎥 Data Collection Pipeline
- **YouTube Video Frame Extraction**: Automated frame extraction from YouTube videos with configurable quality settings (4K support)
  - [`utils/youtube_utils/frame_extract.py`](utils/youtube_utils/frame_extract.py) - Main extraction utility
  - [`utils/youtube_utils/urls.txt`](utils/youtube_utils/urls.txt) - URL management system
  - Cookie-based authentication support for accessing restricted content
  - [`cmd/extract_youtube_frames.sh`](cmd/extract_youtube_frames.sh) - Batch processing script
- **Intelligent Frame Filtering**: YOLO-powered filtering to keep only frames containing pennons
  - [`utils/filter_youtube_photos_by_yolo.py`](utils/filter_youtube_photos_by_yolo.py) - Automated filtering with confidence thresholds
  - Configurable model score thresholds for detection filtering
  - Progress tracking with detailed subfolder statistics
- **Video Processing Utilities**: 
  - [`utils/cut_video.py`](utils/cut_video.py) - Video segmentation and processing tools

#### 🏷️ Dataset Management System
- **Advanced Dataset Splitting**: Intelligent train/validation/test splitting with comprehensive validation
  - [`utils/split_dataset.py`](utils/split_dataset.py) - Configurable ratio splitting (80/10/10 default)
  - Automatic YOLO `data.yaml` generation with proper path configuration
  - Missing label detection and reporting
  - Reproducible splits with random state control
  - Split summary generation in JSON format
- **Multi-Version Dataset Support**: 
  - Four dataset iterations: [`pennon-label-yolo-01`](data/datasets/pennon-label-yolo-01/) through [`pennon-label-yolo-04`](data/datasets/pennon-label-yolo-04/)
  - Dataset versioning with backup and rollback capabilities
  - Comprehensive metadata tracking with [`notes.json`](data/datasets/pennon-label-yolo-01/notes.json)
- **Dataset Quality Assurance**:
  - [`utils/yolo_dataset_validator.py`](utils/yolo_dataset_validator.py) - Dataset integrity validation
  - [`utils/rename_yolo_labels.py`](utils/rename_yolo_labels.py) - Label file management and renaming
  - [`utils/merge_yolo_folders.py`](utils/merge_yolo_folders.py) - Dataset consolidation utilities
  - Duplicate image management with backup systems

#### 🤖 YOLO Model Integration
- **Custom YOLO Model Training**: Specialized pennon detection model with iterative improvements
  - Support for multiple model versions (`custom-01.pt`, `custom-02.pt`, `custom-03.pt`)
  - Single class detection focused on sailing pennons
  - Model performance tracking across training iterations
- **Comprehensive Detection & Tracking Pipeline**:
  - [`utils/detect_with_model.py`](utils/detect_with_model.py) - Object detection inference
  - [`utils/detect_and_track.py`](utils/detect_and_track.py) - Multi-object tracking capabilities
  - [`utils/track_with_model.py`](utils/track_with_model.py) - Advanced tracking utilities
  - ByteTrack integration for robust object tracking
- **Model Deployment & Management**:
  - [`utils/deploy_model.py`](utils/deploy_model.py) - Production deployment utilities
  - Configurable confidence thresholds and inference parameters
  - Batch processing capabilities for large datasets

#### 🐳 Docker Infrastructure
- **Production-Ready YOLO Container**: 
  - [`docker/yolo/Dockerfile`](docker/yolo/Dockerfile) - PyTorch 2.1.2 with CUDA 12.1 support
  - Optimized build process with layer caching
  - Pre-installed YOLO models for immediate inference
  - [`docker/yolo/docker-compose.yml`](docker/yolo/docker-compose.yml) - Service orchestration
- **Label Studio Integration**: 
  - [`docker/label_studio/`](docker/label_studio/) - Complete annotation environment
  - Persistent data storage with volume mounts
  - Export/import capabilities for YOLO format
- **Simplified Operations**:
  - [`cmd/start_yolo`](cmd/start_yolo) - One-command YOLO service startup
  - [`cmd/start_label_studio`](cmd/start_label_studio) - One-command annotation environment
  - Automated service discovery and networking

#### 🏷️ Label Studio Integration
- **ML Backend for Auto-Annotation**:
  - [`docker/yolo/cli.py`](docker/yolo/cli.py) - Label Studio ML backend integration
  - Automated prediction generation for new images
  - Batch processing support for large annotation tasks
  - Model version tracking and prediction scoring
- **Seamless Annotation Workflow**:
  - Direct YOLO format export from Label Studio projects
  - API integration for programmatic task management
  - Project configuration management with label schemas
  - Task preparation utilities for efficient annotation

#### ☁️ Cloud Storage Integration
- **AWS S3 Synchronization**:
  - [`utils/sync_folder_to_s3.py`](utils/sync_folder_to_s3.py) - Bidirectional dataset synchronization
  - Automated backup and restore capabilities
  - Large dataset management in cloud storage
- **Cloud-Based Data Management**:
  - [`utils/download_images_from_s3_and_label_json.py`](utils/download_images_from_s3_and_label_json.py) - Automated data retrieval with labeling
  - Distributed dataset storage and access
  - Cloud-native ML pipeline support

#### 📊 Tracking & Analysis
- **Advanced Object Tracking**:
  - ByteTrack integration via [`trackers/bytetrack.yaml`](trackers/bytetrack.yaml)
  - Multi-object tracking across video sequences
  - Persistent identity management for sailing objects
- **Results Management & Visualization**:
  - Organized results storage in [`runs/detect/`](runs/detect/) hierarchy
  - Training run tracking and comparison
  - Model performance analysis and visualization
  - Clip generation and video result processing in [`results/clips/`](results/clips/)

#### ⚙️ Configuration Management
- **Environment-Based Configuration**:
  - [`utils/config.py`](utils/config.py) - Centralized configuration management
  - `.env` file support for sensitive parameters
  - Configurable dataset versions, model paths, and processing parameters
- **Flexible Parameter Management**:
  - Environment variable support for all major components
  - Docker environment integration
  - Development vs. production configuration separation

### Technical Specifications

- **Primary Object Class**: Pennon (sailing flag) detection and tracking
- **Model Architecture**: YOLOv8 with custom training on sailing imagery
- **Dataset Statistics**: 308 total images in latest iteration (pennon-label-yolo-03)
  - Training: 246 images (80%)
  - Validation: 62 images (20%)
  - Single class: `pennon`
- **Infrastructure**: Docker-based deployment with GPU acceleration support
- **Annotation Platform**: Label Studio with ML-assisted labeling
- **Cloud Integration**: AWS S3 for dataset storage and management

### Project Structure

```
sailboat_CV/
├── cmd/                     # Command scripts and utilities
├── data/                    # Datasets and training data
│   ├── datasets/           # Raw labeled datasets
│   └── splitted_datasets/  # Train/val/test splits
├── docker/                 # Docker containers and services
│   ├── label_studio/      # Annotation environment
│   └── yolo/              # ML inference backend
├── results/               # Model outputs and analysis
├── runs/                  # Training and detection results
├── trackers/              # Object tracking configurations
└── utils/                 # Core utilities and scripts
```

[Unreleased]: https://github.com/yourusername/sailboat_CV/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/sailboat_CV/releases/tag/v1.0.0