#!/usr/bin/env python3
"""
Example usage of the depth estimator package.
This script demonstrates how to load SuperGlue matches and perform triangulation.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_estimator.match_loader import MatchLoader
from depth_estimator.triangulation import StereoCalibration, Triangulator


def save_ply(points_3d, colors=None, filename="pointcloud.ply"):
    """
    Save 3D points as PLY file.
    
    Args:
        points_3d: Array of shape (N, 3) with 3D points
        colors: Optional array of shape (N, 3) with RGB colors (0-255)
        filename: Output filename
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write points
        for i in range(len(points_3d)):
            x, y, z = points_3d[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    
    print(f"  Saved PLY file: {filename}")


def save_xyz(points_3d, filename="pointcloud.xyz"):
    """
    Save 3D points as XYZ file.
    
    Args:
        points_3d: Array of shape (N, 3) with 3D points
        filename: Output filename
    """
    with open(filename, 'w') as f:
        for point in points_3d:
            x, y, z = point
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    
    print(f"  Saved XYZ file: {filename}")


def save_npz_with_metadata(points_3d, confidence_scores, matched_pairs, filename="pointcloud.npz"):
    """
    Save 3D points with metadata as NPZ file.
    
    Args:
        points_3d: Array of shape (N, 3) with 3D points
        confidence_scores: Array of confidence scores
        matched_pairs: Array of matched keypoint pairs
        filename: Output filename
    """
    np.savez(filename,
             points_3d=points_3d,
             confidence_scores=confidence_scores,
             matched_pairs=matched_pairs,
             metadata={
                 'num_points': len(points_3d),
                 'mean_confidence': np.mean(confidence_scores),
                 'depth_range': [points_3d[:, 2].min(), points_3d[:, 2].max()],
                 'format': 'depth_estimator_v1.0'
             })
    
    print(f"  Saved NPZ file: {filename}")


def main():
    """Main example function."""
    
    # File paths
    npz_path = "../SuperGluePretrainedNetwork/outputs/capture_1754654952_camera1_20250808_140912_350_capture_1754654952_camera2_20250808_140912_351_matches.npz"
    calibration_path = "../stereo_calibration/captured_image_pairs/calibration.json"
    
    print("Loading SuperGlue matches...")
    # Load matches
    match_loader = MatchLoader(npz_path)
    
    # Print match statistics
    stats = match_loader.get_match_statistics()
    print(f"Match Statistics:")
    print(f"  Total keypoints: {stats['total_keypoints']}")
    print(f"  Valid matches: {stats['valid_matches']}")
    print(f"  Match ratio: {stats['match_ratio']:.3f}")
    print(f"  Average confidence: {stats['average_confidence']:.3f}")
    
    print("\nLoading stereo calibration...")
    # Load stereo calibration
    stereo_cal = StereoCalibration(calibration_path)
    
    # Print calibration info
    K1, K2 = stereo_cal.get_camera_matrices()
    R = stereo_cal.get_rotation_matrix()
    t = stereo_cal.get_translation_vector()
    print(f"Camera 1 focal length: {K1[0,0]:.1f}, {K1[1,1]:.1f}")
    print(f"Camera 2 focal length: {K2[0,0]:.1f}, {K2[1,1]:.1f}")
    print(f"Baseline: {np.linalg.norm(t):.3f} units")
    
    print("\nCreating triangulator...")
    # Create triangulator
    triangulator = Triangulator(stereo_cal)
    
    print("\nPerforming triangulation...")
    # Perform triangulation with different confidence thresholds
    confidence_thresholds = [0.0, 0.5, 0.8]
    
    for threshold in confidence_thresholds:
        print(f"\nConfidence threshold: {threshold}")
        
        # Get matched pairs
        matched_pairs = match_loader.get_matched_pairs(confidence_threshold=threshold)
        print(f"  Number of matched pairs: {len(matched_pairs)}")
        
        if len(matched_pairs) > 0:
            # Triangulate
            points_3d, confidence_scores = triangulator.triangulate_from_match_loader(
                match_loader, confidence_threshold=threshold, undistort=False
            )
            
            print(f"  3D points shape: {points_3d.shape}")
            print(f"  Depth range: {points_3d[:, 2].min():.3f} to {points_3d[:, 2].max():.3f}")
            print(f"  Mean depth: {points_3d[:, 2].mean():.3f}")
            
            # Filter by reasonable depth range
            depth_mask = triangulator.filter_points_by_depth(points_3d, min_depth=0.1, max_depth=50.0)
            filtered_points = points_3d[depth_mask]
            filtered_confidence = confidence_scores[depth_mask]
            filtered_pairs = np.array(matched_pairs)[depth_mask]
            
            print(f"  Points after depth filtering: {len(filtered_points)}")
            
            # Save results for the highest confidence threshold
            if threshold == 0.8 and len(filtered_points) > 0:
                # Save in multiple formats
                base_filename = f"pointcloud_thresh_{threshold}"
                
                # Save as PLY (standard point cloud format)
                save_ply(filtered_points, filename=f"{base_filename}.ply")
                
                # Save as XYZ (simple text format)
                save_xyz(filtered_points, filename=f"{base_filename}.xyz")
                
                # Save as NPZ with metadata
                save_npz_with_metadata(filtered_points, filtered_confidence, filtered_pairs, 
                                     filename=f"{base_filename}.npz")
                
                # Print some sample 3D points
                print(f"  Sample 3D points (first 5):")
                for i in range(min(5, len(filtered_points))):
                    x, y, z = filtered_points[i]
                    conf = filtered_confidence[i]
                    print(f"    Point {i}: ({x:.3f}, {y:.3f}, {z:.3f}) - confidence: {conf:.3f}")
            
            # Also save for threshold 0.0 (all matches)
            elif threshold == 0.0 and len(filtered_points) > 0:
                base_filename = f"pointcloud_thresh_{threshold}"
                
                # Save as PLY
                save_ply(filtered_points, filename=f"{base_filename}.ply")
                
                # Save as XYZ
                save_xyz(filtered_points, filename=f"{base_filename}.xyz")
                
                # Save as NPZ with metadata
                save_npz_with_metadata(filtered_points, filtered_confidence, filtered_pairs, 
                                     filename=f"{base_filename}.npz")
    
    print("\nExample completed successfully!")
    print("\nGenerated files:")
    print("  - pointcloud_thresh_0.0.ply (PLY format - all matches)")
    print("  - pointcloud_thresh_0.0.xyz (XYZ format - all matches)")
    print("  - pointcloud_thresh_0.0.npz (NPZ format with metadata)")
    print("  - pointcloud_thresh_0.8.* (if high confidence matches exist)")


if __name__ == "__main__":
    main()
