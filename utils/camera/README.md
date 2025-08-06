# Camera Calibration

This module provides camera calibration functionality using OpenCV's `calibrateCamera` method.

## Files

- `calibrate_camera.py` - Main calibration script
- `checkerboard_specs_example.json` - Example checkerboard specifications
- `README.md` - This file

## Usage

### 1. Prepare Checkerboard Images

Take multiple photos of a checkerboard pattern from different angles and distances. The checkerboard should be:
- Flat and undistorted
- Well-lit with good contrast
- Visible in the entire frame
- Taken from various angles (0-45 degrees)

### 2. Create Checkerboard Specifications

Create a JSON or YAML file with your checkerboard specifications:

```json
{
  "inner_corners_x": 9,
  "inner_corners_y": 6,
  "square_size_mm": 25.0
}
```

**Note**: Use the number of **inner corners**, not the total squares. For a 10x7 checkerboard, you have 9x6 inner corners.

### 3. Run Calibration

```bash
python calibrate_camera.py /path/to/images /path/to/specs.json --output calibration_results.json
```

### 4. Optional: Test Calibration

```bash
python calibrate_camera.py /path/to/images /path/to/specs.json --test-image test_image.jpg
```

## Output

The script generates:
- `calibration_results.json` - Camera intrinsics and distortion coefficients
- `calibration_results_undistorted.jpg` - Undistorted test image (if --test-image provided)

## Output Format

```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "distortion_coefficients": [k1, k2, p1, p2, k3],
  "calibration_error": 0.1234,
  "image_width": 1920,
  "image_height": 1080
}
```

## Requirements

- At least 5 successful checkerboard detections
- Images in JPG or PNG format
- Checkerboard pattern clearly visible in images

## Tips

- Use a high-quality, printed checkerboard
- Ensure good lighting and contrast
- Take images from various angles and distances
- Avoid motion blur
- Include some images with the checkerboard at the edges of the frame 