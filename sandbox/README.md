# Tracker Overlayer System

## Overview

This tracker overlayer system maintains ID consistency in object tracking by comparing current detections against a reference snapshot. It's designed to handle common tracking issues like ID swaps, false positives, and missing detections.

## Core Concept

The system works by:

1. **Establishing a reference snapshot** of object IDs and their 2D positions when tracking is correct
2. **Comparing current arrangements** to this reference arrangement  
3. **Identifying outliers** by measuring how much each object has deviated from its expected position relative to the group
4. **Using an entropy-based metric** on deviation distances to set a data-driven threshold for outlier rejection
5. **Reassigning IDs** to valid (non-outlier) objects by matching them to the reference order using the Hungarian algorithm

## Files Structure

```
sandbox/
├── tracker_overlayer.py          # Main matching logic
├── test_tracker_overlayer.py     # Unit tests
├── visualize_tests.py            # Visual test examples
├── demo_visual_links.py          # Focused demonstration of missing detections and false positives
├── README.md                     # This file
├── tracker_overlayer_tests.png   # Visual test results (8 scenarios)
├── threshold_sensitivity_analysis.png  # Threshold analysis
└── focused_demo_missing_false_positives.png  # Focused 3-panel demo
```

## Main Components

### TrackerOverlayer Class

The main class with key methods:

- `set_reference_snapshot()`: Establish the reference point configuration
- `compute_deviation_distances()`: Calculate how far each point has moved from reference
- `compute_entropy_threshold()`: Calculate adaptive threshold using Shannon entropy
- `identify_outliers()`: Separate valid points from outliers
- `reassign_ids_hungarian()`: Optimal ID reassignment using Hungarian algorithm
- `process_frame()`: Complete pipeline to process a frame

### Test Transformations

The system is tested independently against these core tracking issues:

**Baseline Tests:**
- **Small Noise**: Random displacement of points
- **Translation**: Global shift of the entire point cloud

**Independent Issue Testing:**
- **Missing Detections**: Some tracked objects disappear (tested with single and multiple missing points)
- **False Positives**: Extra outlier points appear (tested with single and multiple outliers)

**Combined Scenarios:**
- **Mixed Issues**: Realistic combinations of missing detections and false positives
- **Stress Tests**: Heavy scenarios with many missing points and multiple outliers simultaneously

## Usage Example

```python
import numpy as np
from tracker_overlayer import TrackerOverlayer, generate_test_points

# Generate reference points
reference_points = generate_test_points(n_points=10, bounds=(0, 100, 0, 100))

# Initialize tracker
tracker = TrackerOverlayer(entropy_threshold_factor=2.0, max_distance_threshold=50.0)
tracker.set_reference_snapshot(reference_points)

# Process a frame with some noise and missing points
current_points = {
    0: np.array([12.0, 11.0]),  # Slightly moved
    1: np.array([22.0, 19.0]),  # Slightly moved  
    3: np.array([41.0, 26.0]),  # Slightly moved (ID 2 missing)
    99: np.array([200.0, 200.0])  # Outlier
}

# Process the frame
corrected_points, outlier_ids, id_mapping = tracker.process_frame(current_points)

print(f"Corrected points: {corrected_points}")
print(f"Outliers: {outlier_ids}")
print(f"ID mapping: {id_mapping}")
```

## Running the Tests

### Unit Tests
```bash
cd sandbox
python test_tracker_overlayer.py
```

### Visual Tests
```bash
cd sandbox
python visualize_tests.py
```

This will generate:
- `tracker_overlayer_tests.png`: Visual comparison of different transformation scenarios
- `threshold_sensitivity_analysis.png`: Analysis of how entropy threshold affects outlier detection

### Demo Visualization
```bash
cd sandbox
python demo_visual_links.py
```

This creates a focused 3-panel demonstration showing:
- `focused_demo_missing_false_positives.png`: Independent testing of missing detections, false positives, and mixed scenarios

## Visual Test Results

The visual tests show comprehensive matching analysis with visual links:

### Point Types:
- **Light Blue Circles (R0, R1, ...)**: Reference Layout (ground truth positions)
- **Light Coral X's (D0, D1, ...)**: Valid Detections
- **Red X's**: Outliers/False Positives  
- **Dark Blue Circles**: Missing Detections (reference points with no corresponding detection)

### Visual Links:
- **Blue Dashed Lines (---)**: Ground Truth Links (where each detection ID should ideally match)
- **Green Solid Lines (——)**: Predicted Matches (what our algorithm actually decided)

### Interpretation:
- When blue dashed and green solid lines overlap: **Correct match** ✅
- When they differ: **Incorrect match** ❌ (ID swap or misassignment)
- Missing blue dashed lines: **Missing detection** (reference point has no detection)
- Green lines to nowhere: **False positive** (detection matched to wrong reference)

### Metrics Displayed:
- **Matches**: Number of predictions made by algorithm
- **Outliers**: Number of detections rejected as false positives
- **Missing**: Number of reference points with no detection
- **Correct**: Number of predictions that match ground truth
- **Accuracy**: Percentage of correct predictions

## Key Features

1. **Entropy-based thresholding**: Adapts to different noise levels automatically
2. **Hungarian algorithm**: Ensures optimal ID reassignment with minimal total distance
3. **Robust outlier detection**: Handles false positives and tracking failures
4. **Missing detection handling**: Gracefully handles disappeared objects
5. **Comprehensive testing**: Unit tests and visual validation

## Parameters

- `entropy_threshold_factor`: Controls sensitivity to outliers (higher = more permissive)
- `max_distance_threshold`: Hard limit on maximum allowed deviation
- `noise_std`: Standard deviation for test noise
- `outlier_distance`: How far to move outlier points in tests

## Performance Notes

The system is designed for real-time performance:
- O(n²) complexity for distance matrix computation
- O(n³) Hungarian algorithm for ID reassignment  
- Suitable for typical tracking scenarios with 10-50 objects 