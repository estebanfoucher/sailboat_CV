# SparseDC Direct Inference

This directory contains the migrated SparseDC inference pipeline with SuperGlue + triangulation integration.

## Files

- `sparsedc_direct_inference.py` - Main inference script with aspect-ratio preserving resize
- `sparsedc_results/` - Output directory containing inference results
- `README_inference.md` - This file

## Usage

### From Root Directory (Recommended)
```bash
python run_sparsedc_inference.py \
    --image stereo_calibration/captured_image_pairs/pair_0/capture_1754654952_camera1_20250808_140912_350.jpg \
    --matches SuperGluePretrainedNetwork/outputs/capture_1754654952_camera1_20250808_140912_350_capture_1754654952_camera2_20250808_140912_351_matches.npz \
    --calibration stereo_calibration/captured_image_pairs/calibration.json \
    --output SparseDC/sparsedc_results \
    --model kitti.ckpt \
    --resolution 512,1024
```

### From SparseDC Directory
```bash
cd SparseDC
python sparsedc_direct_inference.py \
    --image ../stereo_calibration/captured_image_pairs/pair_0/capture_1754654952_camera1_20250808_140912_350.jpg \
    --matches ../SuperGluePretrainedNetwork/outputs/capture_1754654952_camera1_20250808_140912_350_capture_1754654952_camera2_20250808_140912_351_matches.npz \
    --calibration ../stereo_calibration/captured_image_pairs/calibration.json \
    --output sparsedc_results \
    --model ../kitti.ckpt \
    --resolution 512,1024
```

## Key Features

### ✅ Aspect-Ratio Preserving Resize
- Maintains image aspect ratio during resizing
- Applies same transformation to sparse depth coordinates
- 100% point preservation during coordinate transformation

### ✅ Enhanced Visualization
- Scatter plot visualization for sparse depth points
- Original vs transformed coordinate comparison
- Point preservation statistics

### ✅ Debug Information
- Detailed transformation logging
- Coordinate tracking through pipeline
- Point preservation metrics

## Output Structure

```
sparsedc_results/
├── capture_1754654952_camera1_20250808_140912_350_depth_prediction.npy
├── capture_1754654952_camera1_20250808_140912_350_sparse_depth.npy
├── capture_1754654952_camera1_20250808_140912_350_visualization.png
└── debug/
    └── sparse_depth.npy
```

## Technical Details

### Coordinate Transformation
- **Original**: `(1080, 1920)` → **Scaled**: `(512, 910)` → **Padded**: `(512, 1024)`
- **Scale factor**: `0.4741` (maintains aspect ratio)
- **Padding**: `57 pixels` on left and right sides

### Point Preservation
- **30/30 points preserved** (100% retention)
- Exact depth values maintained
- Correct spatial relationships preserved
