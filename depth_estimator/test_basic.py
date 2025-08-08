#!/usr/bin/env python3
"""
Basic test of the depth estimator package without matplotlib.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_estimator.match_loader import MatchLoader
from depth_estimator.triangulation import StereoCalibration, Triangulator


def test_match_loader():
    """Test the MatchLoader class."""
    print("Testing MatchLoader...")
    
    npz_path = "../SuperGluePretrainedNetwork/outputs/capture_1754654952_camera1_20250808_140912_350_capture_1754654952_camera2_20250808_140912_351_matches.npz"
    
    try:
        loader = MatchLoader(npz_path)
        print("✓ Successfully loaded matches")
        
        # Test statistics
        stats = loader.get_match_statistics()
        print(f"✓ Match statistics: {stats['valid_matches']} valid matches out of {stats['total_keypoints']} keypoints")
        
        # Test getting matched pairs
        pairs = loader.get_matched_pairs(confidence_threshold=0.0)
        print(f"✓ Got {len(pairs)} matched pairs")
        
        # Test numpy arrays
        points1, points2 = loader.get_matched_pairs_numpy(confidence_threshold=0.0)
        print(f"✓ Got numpy arrays: {points1.shape}, {points2.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ MatchLoader test failed: {e}")
        return False


def test_stereo_calibration():
    """Test the StereoCalibration class."""
    print("\nTesting StereoCalibration...")
    
    calibration_path = "../stereo_calibration/captured_image_pairs/calibration.json"
    
    try:
        cal = StereoCalibration(calibration_path)
        print("✓ Successfully loaded calibration")
        
        # Test getting calibration parameters
        K1, K2 = cal.get_camera_matrices()
        R = cal.get_rotation_matrix()
        t = cal.get_translation_vector()
        
        print(f"✓ Camera matrices: {K1.shape}, {K2.shape}")
        print(f"✓ Rotation matrix: {R.shape}")
        print(f"✓ Translation vector: {t.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ StereoCalibration test failed: {e}")
        return False


def test_triangulation():
    """Test the Triangulator class."""
    print("\nTesting Triangulator...")
    
    try:
        # Load calibration and matches
        calibration_path = "../stereo_calibration/captured_image_pairs/calibration.json"
        npz_path = "../SuperGluePretrainedNetwork/outputs/capture_1754654952_camera1_20250808_140912_350_capture_1754654952_camera2_20250808_140912_351_matches.npz"
        
        cal = StereoCalibration(calibration_path)
        loader = MatchLoader(npz_path)
        triangulator = Triangulator(cal)
        
        print("✓ Successfully created triangulator")
        
        # Test triangulation
        points_3d, confidence_scores = triangulator.triangulate_from_match_loader(
            loader, confidence_threshold=0.0, undistort=False  # Skip undistortion for now
        )
        
        print(f"✓ Triangulated {len(points_3d)} 3D points")
        if len(points_3d) > 0:
            print(f"✓ Depth range: {points_3d[:, 2].min():.3f} to {points_3d[:, 2].max():.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Triangulator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Running basic tests for depth estimator package...\n")
    
    tests = [
        test_match_loader,
        test_stereo_calibration,
        test_triangulation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
