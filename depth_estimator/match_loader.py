import numpy as np
from typing import Tuple, List, Optional


class MatchLoader:
    """Loads SuperGlue match outputs from .npz files and provides matched keypoint pairs."""
    
    def __init__(self, npz_path: str):
        """
        Initialize the match loader with a SuperGlue output .npz file.
        
        Args:
            npz_path: Path to the .npz file containing SuperGlue matches
        """
        self.npz_path = npz_path
        self.data = None
        self._load_data()
    
    def _load_data(self):
        """Load the .npz file data."""
        try:
            self.data = np.load(self.npz_path)
            required_keys = ['keypoints0', 'keypoints1', 'matches', 'match_confidence']
            for key in required_keys:
                if key not in self.data:
                    raise ValueError(f"Missing required key '{key}' in npz file")
        except Exception as e:
            raise RuntimeError(f"Failed to load npz file {self.npz_path}: {e}")
    
    def get_matched_pairs(self, confidence_threshold: float = 0.0) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Get matched keypoint pairs as tuples of (x, y) coordinates.
        
        Args:
            confidence_threshold: Minimum confidence score for matches (0.0 to 1.0)
            
        Returns:
            List of matched pairs: [(camera1_point, camera2_point), ...]
            Each point is a tuple of (x, y) coordinates
        """
        if self.data is None:
            raise RuntimeError("Data not loaded")
        
        # Get valid matches (matches > -1 means there's a match)
        valid_mask = (self.data['matches'] > -1) & (self.data['match_confidence'] >= confidence_threshold)
        
        # Get matched keypoints
        kpts0 = self.data['keypoints0'][valid_mask]  # Camera 1 keypoints
        kpts1 = self.data['keypoints1'][self.data['matches'][valid_mask]]  # Camera 2 keypoints
        
        # Convert to list of tuples
        matched_pairs = []
        for i in range(len(kpts0)):
            point1 = (float(kpts0[i][0]), float(kpts0[i][1]))  # (x, y) for camera 1
            point2 = (float(kpts1[i][0]), float(kpts1[i][1]))  # (x, y) for camera 2
            matched_pairs.append((point1, point2))
        
        return matched_pairs
    
    def get_matched_pairs_numpy(self, confidence_threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get matched keypoint pairs as numpy arrays.
        
        Args:
            confidence_threshold: Minimum confidence score for matches (0.0 to 1.0)
            
        Returns:
            Tuple of (camera1_points, camera2_points) as numpy arrays
            Each array has shape (N, 2) where N is number of matches
        """
        if self.data is None:
            raise RuntimeError("Data not loaded")
        
        # Get valid matches
        valid_mask = (self.data['matches'] > -1) & (self.data['match_confidence'] >= confidence_threshold)
        
        # Get matched keypoints
        kpts0 = self.data['keypoints0'][valid_mask]  # Camera 1 keypoints
        kpts1 = self.data['keypoints1'][self.data['matches'][valid_mask]]  # Camera 2 keypoints
        
        return kpts0, kpts1
    
    def get_confidence_scores(self, confidence_threshold: float = 0.0) -> np.ndarray:
        """
        Get confidence scores for matched pairs.
        
        Args:
            confidence_threshold: Minimum confidence score for matches (0.0 to 1.0)
            
        Returns:
            Array of confidence scores for matched pairs
        """
        if self.data is None:
            raise RuntimeError("Data not loaded")
        
        valid_mask = (self.data['matches'] > -1) & (self.data['match_confidence'] >= confidence_threshold)
        return self.data['match_confidence'][valid_mask]
    
    def get_match_statistics(self) -> dict:
        """
        Get statistics about the matches.
        
        Returns:
            Dictionary with match statistics
        """
        if self.data is None:
            raise RuntimeError("Data not loaded")
        
        total_keypoints = len(self.data['keypoints0'])
        valid_matches = np.sum(self.data['matches'] > -1)
        avg_confidence = np.mean(self.data['match_confidence'][self.data['matches'] > -1])
        
        return {
            'total_keypoints': total_keypoints,
            'valid_matches': valid_matches,
            'match_ratio': valid_matches / total_keypoints if total_keypoints > 0 else 0,
            'average_confidence': avg_confidence
        }
