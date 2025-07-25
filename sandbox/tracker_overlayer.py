import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional
import logging

class TrackerOverlayer:
    """
    Tracker overlayer that maintains ID consistency by comparing against a reference snapshot.
    
    Core idea:
    1. Establish reference snapshot of object IDs and 2D positions when tracking is correct
    2. Compare current arrangement to reference arrangement  
    3. Identify outliers using deviation distances and entropy-based threshold
    4. Reassign IDs to valid objects by matching to reference order
    """
    
    def __init__(self, entropy_threshold_factor: float = 2.0, max_distance_threshold: float = 50.0):
        """
        Initialize tracker overlayer.
        
        Args:
            entropy_threshold_factor: Multiplier for entropy-based outlier threshold
            max_distance_threshold: Maximum allowed distance for valid matches
        """
        self.reference_points: Optional[Dict[int, np.ndarray]] = None
        self.entropy_threshold_factor = entropy_threshold_factor
        self.max_distance_threshold = max_distance_threshold
        
    def set_reference_snapshot(self, points: Dict[int, np.ndarray]) -> None:
        """
        Set the reference snapshot of object IDs and their 2D positions.
        
        Args:
            points: Dictionary mapping object IDs to 2D positions [x, y]
        """
        self.reference_points = {id_: np.array(pos) for id_, pos in points.items()}
        logging.info(f"Reference snapshot set with {len(points)} points")
        
    def compute_deviation_distances(self, current_points: Dict[int, np.ndarray]) -> Dict[int, float]:
        """
        Compute deviation distances for each current point from its reference position.
        
        Args:
            current_points: Dictionary mapping current IDs to 2D positions
            
        Returns:
            Dictionary mapping IDs to their deviation distances
        """
        if self.reference_points is None:
            raise ValueError("Reference snapshot not set")
            
        deviations = {}
        for id_, current_pos in current_points.items():
            if id_ in self.reference_points:
                ref_pos = self.reference_points[id_]
                deviation = np.linalg.norm(current_pos - ref_pos)
                deviations[id_] = deviation
            else:
                # New ID not in reference - mark as large deviation
                deviations[id_] = float('inf')
                
        return deviations
    
    def compute_entropy_threshold(self, deviations: List[float]) -> float:
        """
        Compute entropy-based threshold for outlier detection.
        
        Args:
            deviations: List of deviation distances
            
        Returns:
            Threshold value for outlier detection
        """
        if len(deviations) == 0:
            return 0.0
            
        # Remove infinite values for entropy calculation
        finite_deviations = [d for d in deviations if np.isfinite(d)]
        
        if len(finite_deviations) == 0:
            return 0.0
            
        # Compute histogram for entropy calculation
        hist, bin_edges = np.histogram(finite_deviations, bins=min(10, len(finite_deviations)))
        hist = hist + 1e-10  # Add small value to avoid log(0)
        probs = hist / np.sum(hist)
        
        # Calculate Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))
        
        # Use entropy to set adaptive threshold
        mean_deviation = np.mean(finite_deviations)
        std_deviation = np.std(finite_deviations)
        
        # Higher entropy (more spread) -> higher threshold
        threshold = mean_deviation + self.entropy_threshold_factor * std_deviation * (1 + entropy)
        
        return min(threshold, self.max_distance_threshold)
    
    def identify_outliers(self, current_points: Dict[int, np.ndarray]) -> Tuple[List[int], List[int]]:
        """
        Identify outlier IDs based on deviation distances and entropy threshold.
        
        Args:
            current_points: Dictionary mapping current IDs to 2D positions
            
        Returns:
            Tuple of (valid_ids, outlier_ids)
        """
        deviations = self.compute_deviation_distances(current_points)
        deviation_values = list(deviations.values())
        threshold = self.compute_entropy_threshold(deviation_values)
        
        valid_ids = []
        outlier_ids = []
        
        for id_, deviation in deviations.items():
            if deviation <= threshold and np.isfinite(deviation):
                valid_ids.append(id_)
            else:
                outlier_ids.append(id_)
                
        return valid_ids, outlier_ids
    
    def reassign_ids_hungarian(self, valid_points: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        Reassign IDs to valid points using Hungarian algorithm for optimal matching.
        
        Args:
            valid_points: Dictionary of valid points (non-outliers)
            
        Returns:
            Dictionary mapping old IDs to new IDs
        """
        if self.reference_points is None:
            raise ValueError("Reference snapshot not set")
            
        if len(valid_points) == 0:
            return {}
            
        # Create cost matrix between valid points and reference points
        valid_ids = list(valid_points.keys())
        ref_ids = list(self.reference_points.keys())
        
        valid_positions = np.array([valid_points[id_] for id_ in valid_ids])
        ref_positions = np.array([self.reference_points[id_] for id_ in ref_ids])
        
        # Compute distance matrix
        cost_matrix = cdist(valid_positions, ref_positions)
        
        # Solve assignment problem
        valid_indices, ref_indices = linear_sum_assignment(cost_matrix)
        
        # Create mapping from old ID to new ID
        id_mapping = {}
        for i, j in zip(valid_indices, ref_indices):
            old_id = valid_ids[i]
            new_id = ref_ids[j]
            id_mapping[old_id] = new_id
            
        return id_mapping
    
    def process_frame(self, current_points: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], List[int], Dict[int, int]]:
        """
        Process a frame: identify outliers and reassign IDs.
        
        Args:
            current_points: Dictionary mapping current IDs to 2D positions
            
        Returns:
            Tuple of (corrected_points, outlier_ids, id_mapping)
        """
        if self.reference_points is None:
            raise ValueError("Reference snapshot not set")
            
        # Identify outliers
        valid_ids, outlier_ids = self.identify_outliers(current_points)
        
        # Get valid points
        valid_points = {id_: current_points[id_] for id_ in valid_ids}
        
        # Reassign IDs
        id_mapping = self.reassign_ids_hungarian(valid_points)
        
        # Create corrected points with reassigned IDs
        corrected_points = {}
        for old_id, new_id in id_mapping.items():
            corrected_points[new_id] = current_points[old_id]
            
        return corrected_points, outlier_ids, id_mapping


def generate_test_points(n_points: int = 10, bounds: Tuple[float, float, float, float] = (0, 100, 0, 100)) -> Dict[int, np.ndarray]:
    """
    Generate random test points within specified bounds.
    
    Args:
        n_points: Number of points to generate
        bounds: (x_min, x_max, y_min, y_max) bounds for point generation
        
    Returns:
        Dictionary mapping IDs to 2D positions
    """
    points = {}
    x_min, x_max, y_min, y_max = bounds
    
    for i in range(n_points):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        points[i] = np.array([x, y])
        
    return points


def apply_transformations(reference_points: Dict[int, np.ndarray], 
                         noise_std: float = 2.0,
                         translation: Tuple[float, float] = (0.0, 0.0),
                         missing_ids: List[int] = None,
                         outlier_ids: List[int] = None,
                         outlier_distance: float = 30.0) -> Dict[int, np.ndarray]:
    """
    Apply various transformations to reference points for testing.
    
    Args:
        reference_points: Original reference points
        noise_std: Standard deviation of Gaussian noise to add
        translation: (dx, dy) translation to apply
        missing_ids: List of IDs to remove (missing detections)
        outlier_ids: List of IDs to turn into outliers
        outlier_distance: Distance to move outlier points
        
    Returns:
        Transformed points dictionary
    """
    if missing_ids is None:
        missing_ids = []
    if outlier_ids is None:
        outlier_ids = []
        
    transformed_points = {}
    dx, dy = translation
    
    for id_, pos in reference_points.items():
        if id_ in missing_ids:
            continue  # Skip missing points
            
        # Apply noise and translation
        noise = np.random.normal(0, noise_std, size=2)
        new_pos = pos + np.array([dx, dy]) + noise
        
        # Apply outlier transformation
        if id_ in outlier_ids:
            # Move point far away in random direction
            angle = np.random.uniform(0, 2 * np.pi)
            outlier_offset = outlier_distance * np.array([np.cos(angle), np.sin(angle)])
            new_pos = pos + outlier_offset
            
        transformed_points[id_] = new_pos
        
    return transformed_points 