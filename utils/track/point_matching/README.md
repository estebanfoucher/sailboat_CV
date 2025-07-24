# 2D Point Matching Pipeline

Lightweight pipeline for matching detected points to reference layout.

## Usage

```python
from utils.track.point_matching import PointMatcher

matcher = PointMatcher()
matches, transform = matcher.match(reference_points, detected_points, use_tps=False)
```

## Pipeline

1. **Align**: RANSAC + Kabsch → rigid transformation
2. **Match**: Hungarian algorithm → assign points  
3. **Refine**: Optional TPS → non-rigid deformation

## Config

- `ransac_iterations=100`: Fixed iterations for embedded systems
- `use_tps=False`: Enable/disable TPS refinement
- TPS is completely optional and removable

## Output

- `matches`: List of (det_idx, ref_idx, cost) tuples
- `transform`: (R, t) rigid transformation matrix

## Test

```bash
python utils/track/point_matching/example.py