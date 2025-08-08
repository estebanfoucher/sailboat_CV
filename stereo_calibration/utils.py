import cv2
import numpy as np
import json
from typing import Dict, Any, Tuple, List


def _load_intrinsics(intrinsics_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera intrinsics from JSON file."""
    with open(intrinsics_path, 'r') as f:
        data = json.load(f)
    
    camera_matrix = np.array(data['camera_matrix'], dtype=np.float32)
    dist_coeffs = np.array(data['distortion_coefficients'], dtype=np.float32)
    
    return camera_matrix, dist_coeffs


def calibrate_stereo(points_3d: np.ndarray, points_2d_1: np.ndarray, points_2d_2: np.ndarray,
                    intrinsics1_path: str, intrinsics2_path: str, image_size: Tuple[int, int]) -> Dict[str, Any]:
    """
    Perform stereo calibration using provided correspondences.
    
    Args:
        points_3d: 3D points in world coordinates
        points_2d_1: 2D points from first camera
        points_2d_2: 2D points from second camera
        intrinsics1_path: Path to first camera intrinsics JSON
        intrinsics2_path: Path to second camera intrinsics JSON
        image_size: Image size as (width, height)
        
    Returns:
        Dictionary with stereo calibration results
    """
    # Load camera intrinsics
    camera_matrix1, dist_coeffs1 = _load_intrinsics(intrinsics1_path)
    camera_matrix2, dist_coeffs2 = _load_intrinsics(intrinsics2_path)
    
    print(f"Using {len(points_3d)} 3D-2D correspondences for calibration")
    
    # Perform stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC  # Use provided intrinsics
    
    ret, camera_matrix1_cal, dist_coeffs1_cal, camera_matrix2_cal, dist_coeffs2_cal, \
    R, T, E, F = cv2.stereoCalibrate(
        [points_3d], [points_2d_1], [points_2d_2],
        camera_matrix1, dist_coeffs1,
        camera_matrix2, dist_coeffs2,
        image_size,  # image size (width, height)
        flags=flags
    )
    
    # Calculate reprojection error
    reprojection_error = ret
    
    # Create results dictionary
    results = {
        'success': True,
        'reprojection_error': float(reprojection_error),
        'num_correspondences': len(points_3d),
        'camera_matrix1': camera_matrix1_cal.tolist(),
        'camera_matrix2': camera_matrix2_cal.tolist(),
        'dist_coeffs1': dist_coeffs1_cal.tolist(),
        'dist_coeffs2': dist_coeffs2_cal.tolist(),
        'rotation_matrix': R.tolist(),
        'translation_vector': T.tolist(),
        'essential_matrix': E.tolist(),
        'fundamental_matrix': F.tolist(),
        'image_size': image_size  # (width, height)
    }
    
    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save calibration results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def print_summary(results: Dict[str, Any]):
    """Print a summary of the calibration results."""
    print("\n" + "="*50)
    print("STEREO CALIBRATION RESULTS")
    print("="*50)
    print(f"Success: {results['success']}")
    print(f"Reprojection Error: {results['reprojection_error']:.6f}")
    print(f"Number of Correspondences: {results['num_correspondences']}")
    
    # Extract translation and rotation info
    T = np.array(results['translation_vector'])
    R = np.array(results['rotation_matrix'])
    
    # Calculate baseline distance
    baseline = np.linalg.norm(T)
    print(f"Baseline Distance: {baseline:.6f} meters")
    
    # Convert rotation matrix to Euler angles
    euler_angles = _rotation_matrix_to_euler_angles(R)
    print(f"Rotation (roll, pitch, yaw): ({euler_angles[0]:.2f}°, {euler_angles[1]:.2f}°, {euler_angles[2]:.2f}°)")
    
    print(f"Translation: ({T[0,0]:.6f}, {T[1,0]:.6f}, {T[2,0]:.6f}) meters")
    print("="*50)


def _rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees."""
    # Extract Euler angles from rotation matrix
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return np.degrees([roll, pitch, yaw])


def calibrate_stereo_many(
    object_points_list: List[np.ndarray],
    image_points1_list: List[np.ndarray],
    image_points2_list: List[np.ndarray],
    intrinsics1_path: str,
    intrinsics2_path: str,
    image_size: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Perform stereo calibration using multiple pairs of correspondences.
    
    Args:
        object_points_list: List of 3D points for each pair
        image_points1_list: List of 2D points from first camera for each pair
        image_points2_list: List of 2D points from second camera for each pair
        intrinsics1_path: Path to first camera intrinsics JSON
        intrinsics2_path: Path to second camera intrinsics JSON
        image_size: Image size as (width, height)
        
    Returns:
        Dictionary with stereo calibration results
    """
    camera_matrix1, dist_coeffs1 = _load_intrinsics(intrinsics1_path)
    camera_matrix2, dist_coeffs2 = _load_intrinsics(intrinsics2_path)

    total_points = int(sum(len(op) for op in object_points_list))
    print(f"Using {total_points} 3D-2D correspondences from {len(object_points_list)} pair(s)")

    flags = cv2.CALIB_FIX_INTRINSIC

    ret, camera_matrix1_cal, dist_coeffs1_cal, camera_matrix2_cal, dist_coeffs2_cal, \
    R, T, E, F = cv2.stereoCalibrate(
        object_points_list,
        image_points1_list,
        image_points2_list,
        camera_matrix1, dist_coeffs1,
        camera_matrix2, dist_coeffs2,
        image_size,
        flags=flags
    )

    results = {
        'success': True,
        'reprojection_error': float(ret),
        'num_correspondences': total_points,
        'num_pairs': len(object_points_list),
        'camera_matrix1': camera_matrix1_cal.tolist(),
        'camera_matrix2': camera_matrix2_cal.tolist(),
        'dist_coeffs1': dist_coeffs1_cal.tolist(),
        'dist_coeffs2': dist_coeffs2_cal.tolist(),
        'rotation_matrix': R.tolist(),
        'translation_vector': T.tolist(),
        'essential_matrix': E.tolist(),
        'fundamental_matrix': F.tolist(),
        'image_size': image_size
    }
    return results
