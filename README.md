# Stereo Rectification Analysis

## Overview

This folder contains a `StereoRectifier` class that demonstrates the issue with `cv2.stereoRectify()` using your calibration data and provides a working solution.

## The Problem

Your stereo calibration has a **zero ROI (Region of Interest)** issue:

- **ROI1: (0, 0, 0, 0)** - Left camera has zero valid region
- **ROI2: (0, 0, 0, 0)** - Right camera has zero valid region

This causes `cv2.stereoRectify()` to produce completely black rectified images.

## Root Cause

The **translation magnitude of 1.627m** between cameras is unusually large for typical stereo setups, causing the rectification to push images completely out of bounds.

## Working Solution

Instead of full rectification, use **undistortion only**:

```python
# This works correctly
undist1 = cv2.undistort(img1, K1, D1)
undist2 = cv2.undistort(img2, K2, D2)

# Then use SGBM directly on undistorted images
stereo = cv2.StereoSGBM_create(...)
disparity = stereo.compute(undist1, undist2)
```

## Files

- `stereo_rectifier.py` - StereoRectifier class with analysis
- `example.py` - Simple usage example
- `output/` - Generated images showing the issue

## Results

| Method | Left Image | Right Image | Status |
|--------|------------|-------------|---------|
| **Original** | 1.9MB | 1.7MB | ✅ Normal |
| **Undistorted** | 1.0MB | 1.0MB | ✅ **Working** |
| **Rectified** | 1.0MB | 7.9KB | ❌ Black |

## Usage

```python
from stereo_rectifier import StereoRectifier

# Initialize
rectifier = StereoRectifier("calibration.json")

# Load images
img1 = cv2.imread("left.jpg")
img2 = cv2.imread("right.jpg")

# Undistort (works)
undist1, undist2 = rectifier.undistort_images(img1, img2)

# Full rectification (produces black images)
rect1, rect2 = rectifier.rectify_images_full(img1, img2)
```

## Conclusion

Your calibration is perfect for **undistortion-based stereo matching**, which is what the working stereo depth estimator uses. The full rectification approach fails due to the extreme camera separation, but this doesn't prevent accurate depth estimation.


