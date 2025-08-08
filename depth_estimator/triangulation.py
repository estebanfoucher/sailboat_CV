import numpy as np
import json
from typing import List, Tuple, Optional, Dict, Any
from .match_loader import MatchLoader


class StereoCalibration:
    """Loads and manages stereo camera calibration parameters."""
    
    def __init__(self, calibration_path: str):
        """
        Initialize stereo calibration from JSON file.
        
        Args:
            calibration_path: Path to calibration.json file
        """
        self.calibration_path = calibration_path
        self.calibration_data = None
        self._load_calibration()
    
    def _load_calibration(self):
        """Load calibration data from JSON file."""
        try:
            with open(self.calibration_path, 'r') as f:
                self.calibration_data = json.load(f)
            
            if not self.calibration_data.get('success', False):
                raise ValueError("Calibration was not successful")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load calibration file {self.calibration_path}: {e}")
    
    def get_camera_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera intrinsic matrices.
        
        Returns:
            Tuple of (K1, K2) camera matrices
        """
        K1 = np.array(self.calibration_data['camera_matrix1'], dtype=np.float64)
        K2 = np.array(self.calibration_data['camera_matrix2'], dtype=np.float64)
        return K1, K2
    
    def get_distortion_coeffs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get distortion coefficients.
        
        Returns:
            Tuple of (dist1, dist2) distortion coefficients
        """
        dist1 = np.array(self.calibration_data['dist_coeffs1'], dtype=np.float64).flatten()
        dist2 = np.array(self.calibration_data['dist_coeffs2'], dtype=np.float64).flatten()
        return dist1, dist2
    
    def get_rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix from camera 1 to camera 2."""
        return np.array(self.calibration_data['rotation_matrix'], dtype=np.float64)
    
    def get_translation_vector(self) -> np.ndarray:
        """Get translation vector from camera 1 to camera 2."""
        return np.array(self.calibration_data['translation_vector'], dtype=np.float64).flatten()
    
    def get_essential_matrix(self) -> np.ndarray:
        """Get essential matrix."""
        return np.array(self.calibration_data['essential_matrix'], dtype=np.float64)
    
    def get_fundamental_matrix(self) -> np.ndarray:
        """Get fundamental matrix."""
        return np.array(self.calibration_data['fundamental_matrix'], dtype=np.float64)
    
    def get_image_size(self) -> Tuple[int, int]:
        """Get image dimensions."""
        size = self.calibration_data['image_size']
        return size[0], size[1]  # width, height


class Triangulator:
    """Performs triangulation of matched keypoints using stereo calibration."""
    
    def __init__(self, stereo_calibration: StereoCalibration):
        """
        Initialize triangulator with stereo calibration.
        
        Args:
            stereo_calibration: StereoCalibration object
        """
        self.calibration = stereo_calibration
        self.K1, self.K2 = stereo_calibration.get_camera_matrices()
        self.dist1, self.dist2 = stereo_calibration.get_distortion_coeffs()
        self.R = stereo_calibration.get_rotation_matrix()
        self.t = stereo_calibration.get_translation_vector()
        
        # Create projection matrices
        self.P1 = self.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])  # Camera 1 projection matrix
        self.P2 = self.K2 @ np.hstack([self.R, self.t.reshape(3, 1)])  # Camera 2 projection matrix
    
    def undistort_points(self, points: np.ndarray, camera_idx: int) -> np.ndarray:
        """
        Undistort points using camera distortion coefficients.
        
        Args:
            points: Array of shape (N, 2) with (x, y) coordinates
            camera_idx: 1 for camera 1, 2 for camera 2
            
        Returns:
            Undistorted points
        """
        if camera_idx == 1:
            K = self.K1
            dist = self.dist1
        elif camera_idx == 2:
            K = self.K2
            dist = self.dist2
        else:
            raise ValueError("camera_idx must be 1 or 2")
        
        # Convert to homogeneous coordinates
        points_homog = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # Undistort
        undistorted = cv2.undistortPoints(points.reshape(-1, 1, 2), K, dist, P=K)
        return undistorted.reshape(-1, 2)
    
    def triangulate_points(self, points1: np.ndarray, points2: np.ndarray, 
                          undistort: bool = True) -> np.ndarray:
        """
        Triangulate 3D points from matched keypoints.
        
        Args:
            points1: Camera 1 keypoints, shape (N, 2)
            points2: Camera 2 keypoints, shape (N, 2)
            undistort: Whether to undistort points before triangulation
            
        Returns:
            3D points in camera 1 coordinate system, shape (N, 3)
        """
        if points1.shape != points2.shape or points1.shape[1] != 2:
            raise ValueError("points1 and points2 must have shape (N, 2)")
        
        # Undistort points if requested
        if undistort and cv2 is not None:
            points1_undist = self.undistort_points(points1, 1)
            points2_undist = self.undistort_points(points2, 2)
        else:
            points1_undist = points1
            points2_undist = points2
        
        # Triangulate using OpenCV
        points_3d_homog = cv2.triangulatePoints(self.P1, self.P2, 
                                               points1_undist.T, points2_undist.T)
        
        # Convert from homogeneous to 3D coordinates
        points_3d = points_3d_homog[:3] / points_3d_homog[3]
        return points_3d.T
    
    def triangulate_from_matches(self, matched_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                                undistort: bool = True) -> np.ndarray:
        """
        Triangulate 3D points from matched pairs.
        
        Args:
            matched_pairs: List of (camera1_point, camera2_point) tuples
            undistort: Whether to undistort points before triangulation
            
        Returns:
            3D points in camera 1 coordinate system, shape (N, 3)
        """
        if not matched_pairs:
            return np.empty((0, 3))
        
        # Convert to numpy arrays
        points1 = np.array([pair[0] for pair in matched_pairs])
        points2 = np.array([pair[1] for pair in matched_pairs])
        
        return self.triangulate_points(points1, points2, undistort)
    
    def triangulate_from_match_loader(self, match_loader: MatchLoader, 
                                     confidence_threshold: float = 0.0,
                                     undistort: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Triangulate 3D points from a MatchLoader object.
        
        Args:
            match_loader: MatchLoader object
            confidence_threshold: Minimum confidence for matches
            undistort: Whether to undistort points before triangulation
            
        Returns:
            Tuple of (points_3d, confidence_scores)
                - points_3d: 3D points in camera 1 coordinate system, shape (N, 3)
                - confidence_scores: Confidence scores for each point, shape (N,)
        """
        # Get matched pairs and confidence scores
        points1, points2 = match_loader.get_matched_pairs_numpy(confidence_threshold)
        confidence_scores = match_loader.get_confidence_scores(confidence_threshold)
        
        if len(points1) == 0:
            return np.empty((0, 3)), np.empty((0,))
        
        # Triangulate
        points_3d = self.triangulate_points(points1, points2, undistort)
        
        return points_3d, confidence_scores
    
    def filter_points_by_depth(self, points_3d: np.ndarray, 
                              min_depth: float = 0.1, 
                              max_depth: float = 100.0) -> np.ndarray:
        """
        Filter 3D points by depth range.
        
        Args:
            points_3d: 3D points, shape (N, 3)
            min_depth: Minimum depth (Z coordinate)
            max_depth: Maximum depth (Z coordinate)
            
        Returns:
            Boolean mask of valid points
        """
        depths = points_3d[:, 2]  # Z coordinate is depth
        return (depths >= min_depth) & (depths <= max_depth)


# Import cv2 at the top level to avoid issues
try:
    import cv2
except ImportError:
    print("Warning: OpenCV (cv2) not found. Install with: pip install opencv-python")
    cv2 = None
