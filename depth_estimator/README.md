# Depth Estimator

A Python package for performing stereo triangulation using SuperGlue matches and stereo camera calibration.

## Overview

This package provides tools to:
1. Load SuperGlue match outputs from `.npz` files
2. Load stereo camera calibration parameters
3. Perform triangulation to compute 3D positions of matched keypoints

## Components

### MatchLoader
Loads SuperGlue match outputs and provides matched keypoint pairs.

```python
from depth_estimator.match_loader import MatchLoader

# Load matches from SuperGlue output
loader = MatchLoader("path/to/matches.npz")

# Get matched pairs with confidence threshold
matched_pairs = loader.get_matched_pairs(confidence_threshold=0.5)

# Get statistics
stats = loader.get_match_statistics()
```

### StereoCalibration
Loads and manages stereo camera calibration parameters from JSON files.

```python
from depth_estimator.triangulation import StereoCalibration

# Load calibration
cal = StereoCalibration("path/to/calibration.json")

# Access calibration parameters
K1, K2 = cal.get_camera_matrices()
R = cal.get_rotation_matrix()
t = cal.get_translation_vector()
```

### Triangulator
Performs triangulation of matched keypoints using stereo calibration.

```python
from depth_estimator.triangulation import Triangulator

# Create triangulator
triangulator = Triangulator(stereo_cal)

# Triangulate from matched pairs
points_3d = triangulator.triangulate_from_matches(matched_pairs)

# Or triangulate directly from MatchLoader
points_3d, confidence_scores = triangulator.triangulate_from_match_loader(
    match_loader, confidence_threshold=0.5
)
```

## Usage Example

See `example_usage.py` for a complete example using the test files:

```bash
python depth_estimator/example_usage.py
```

This will generate point cloud files in multiple formats:
- **PLY**: Standard point cloud format, readable by most 3D software
- **XYZ**: Simple text format with X Y Z coordinates per line
- **NPZ**: NumPy format with metadata and confidence scores

## Point Cloud Utilities

The `pointcloud_utils.py` module provides utilities for working with generated point clouds:

```python
from depth_estimator.pointcloud_utils import load_ply, load_xyz, load_npz_with_metadata

# Load point clouds
points_3d, colors = load_ply("pointcloud_thresh_0.0.ply")
points_3d = load_xyz("pointcloud_thresh_0.0.xyz")
data = load_npz_with_metadata("pointcloud_thresh_0.0.npz")

# Get point cloud information
from depth_estimator.pointcloud_utils import print_point_cloud_summary
print_point_cloud_summary("pointcloud_thresh_0.0.ply")
```

## Input Data Format

### SuperGlue Matches (.npz)
The `.npz` file should contain:
- `keypoints0`: Camera 1 keypoints (N, 2)
- `keypoints1`: Camera 2 keypoints (N, 2)  
- `matches`: Match indices (-1 for no match)
- `match_confidence`: Confidence scores

### Stereo Calibration (JSON)
The calibration file should contain:
- Camera intrinsic matrices (`camera_matrix1`, `camera_matrix2`)
- Distortion coefficients (`dist_coeffs1`, `dist_coeffs2`)
- Rotation matrix and translation vector
- Essential and fundamental matrices

## Output

The triangulation outputs 3D points in the coordinate system of camera 1, where:
- X: Right direction
- Y: Down direction  
- Z: Forward direction (depth)

## Dependencies

- numpy
- opencv-python
- matplotlib (for visualization)

Install with:
```bash
pip install numpy opencv-python matplotlib
```
