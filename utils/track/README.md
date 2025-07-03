# Tracking Utilities (`utils/track`)

This module provides utilities for running YOLOv8 + ByteTrack tracking on videos, with customizable class mapping and colored rendering.

## Modules
- `track.py`: Tracking logic, class mapping, detection processing.
- `video_io.py`: Video reading/writing utilities.
- `render.py`: Drawing functions for tracks and overlays.
- `pipeline.py`: High-level pipeline to run detection, tracking, and rendering.

## Installation
Install dependencies:
```bash
pip install opencv-python ultralytics bytetrack pyyaml
```

## Usage
Prepare a YAML config file (see example below), then run:
```bash
python pipeline.py --config path/to/config.yml
```

## Example `config.yml`
```yaml
input_video: path/to/input.mp4
output_video: path/to/output.mp4
yolo_model_path: path/to/model.pt

class_info:
  0:
    name: pennon_attached
    color: [255, 0, 0]
  1:
    name: pennon_detached
    color: [0, 255, 0]
  2:
    name: pennon_leech
    color: [0, 0, 255]

base_class_mapping:
  0: 0
  1: 0
  2: 2

bytetrack_params:
  frame_rate: 30
  track_thresh: 0.5
  track_buffer: 30
  match_thresh: 0.8
```
- Colors are in BGR format for OpenCV.

## Notes
- Rendering uses the original detector class for color and label.
- Extend or modify the pipeline as needed for your use case. 