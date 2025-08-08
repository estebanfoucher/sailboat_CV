import cv2
import numpy as np
import json
import yaml
import argparse
import os
from typing import Dict, Any, List, Tuple
import pupil_apriltags as apriltag


class StereoTagDetector:
    """Detect April tags in stereo images and extract correspondences."""
    
    def __init__(self, config_path: str = "config.yml"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.detector = self._create_detector()
        
        # Define 3D points of the April tag in tag coordinate system
        tag_size = self.config['apriltag']['tag_size_meters']
        half_size = tag_size / 2.0
        self.tag_3d_points = np.array([
            [-half_size, -half_size, 0],  # Bottom-left
            [ half_size, -half_size, 0],  # Bottom-right  
            [ half_size,  half_size, 0],  # Top-right
            [-half_size,  half_size, 0]   # Top-left
        ], dtype=np.float32)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Error: Config file {config_path} not found.")
            raise
    
    def _create_detector(self):
        """Create AprilTag detector with configured parameters."""
        tag_config = self.config.get('apriltag', {})
        
        return apriltag.Detector(
            families=tag_config.get('tag_family', 'tag36h11'),
            nthreads=4,
            quad_decimate=tag_config.get('decimation', 1.0),
            quad_sigma=tag_config.get('blur', 0.0),
            refine_edges=int(tag_config.get('refine_edges', True)),
            decode_sharpening=0.25,
            debug=0
        )
    
    def detect_tags(self, frame: np.ndarray) -> List:
        """Detect AprilTags in the given frame."""
        # Convert to grayscale for detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Detect tags
        detections = self.detector.detect(gray)
        
        # Filter detections by decision margin
        min_margin = self.config.get('apriltag', {}).get('min_decision_margin', 10.0)
        filtered_detections = [
            detection for detection in detections 
            if detection.decision_margin >= min_margin
        ]
        
        # Filter by specific tag ID if specified
        target_tag_id = self.config.get('apriltag', {}).get('target_tag_id')
        if target_tag_id is not None:
            filtered_detections = [
                detection for detection in filtered_detections
                if detection.tag_id == target_tag_id
            ]
        
        return filtered_detections
    
    def find_matching_tags(self, detections1: List, detections2: List) -> List[Tuple]:
        """Find matching tags between two camera views."""
        matches = []
        
        for det1 in detections1:
            for det2 in detections2:
                if det1.tag_id == det2.tag_id:
                    matches.append((det1, det2))
                    break
        
        return matches
    
    def extract_correspondences(self, matches: List[Tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract 2D-2D correspondences and 3D points from matching tags."""
        points_3d = []
        points_2d_1 = []
        points_2d_2 = []
        
        for det1, det2 in matches:
            # Add 3D points (same for both cameras since they're in world coordinates)
            points_3d.append(self.tag_3d_points)
            
            # Add 2D points from both cameras
            points_2d_1.append(det1.corners.astype(np.float32))
            points_2d_2.append(det2.corners.astype(np.float32))
        
        if not points_3d:
            return None, None, None
        
        # Stack all points
        points_3d = np.vstack(points_3d)
        points_2d_1 = np.vstack(points_2d_1)
        points_2d_2 = np.vstack(points_2d_2)
        
        return points_3d, points_2d_1, points_2d_2
    
    def get_correspondences(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Take two stereo images and return the extracted correspondences.
        
        Args:
            img1: First camera image as numpy array
            img2: Second camera image as numpy array
            
        Returns:
            Tuple of (points_3d, points_2d_1, points_2d_2) correspondences
        """
        if img1 is None or img2 is None:
            raise ValueError("Images cannot be None")
        
        # Detect tags in both images
        detections1 = self.detect_tags(img1)
        detections2 = self.detect_tags(img2)
        
        print(f"Detected {len(detections1)} tags in camera 1")
        print(f"Detected {len(detections2)} tags in camera 2")
        
        # Find matching tags
        matches = self.find_matching_tags(detections1, detections2)
        print(f"Found {len(matches)} matching tags")
        
        if len(matches) < 1:
            raise ValueError("Need at least 1 matching tag for stereo calibration")
        
        # Extract correspondences
        points_3d, points_2d_1, points_2d_2 = self.extract_correspondences(matches)
        
        if points_3d is None:
            raise ValueError("Failed to extract correspondences")
        
        print(f"Using {len(points_3d)} 3D-2D correspondences")
        
        return points_3d, points_2d_1, points_2d_2



