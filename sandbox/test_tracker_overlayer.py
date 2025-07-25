import unittest
import numpy as np
from typing import Dict
from tracker_overlayer import TrackerOverlayer, generate_test_points, apply_transformations


class TestTrackerOverlayer(unittest.TestCase):
    """Unit tests for TrackerOverlayer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = TrackerOverlayer(entropy_threshold_factor=2.0, max_distance_threshold=50.0)
        
        # Create a simple reference set of points
        self.reference_points = {
            0: np.array([10.0, 10.0]),
            1: np.array([20.0, 15.0]),
            2: np.array([30.0, 20.0]),
            3: np.array([40.0, 25.0]),
            4: np.array([50.0, 30.0])
        }
        
    def test_set_reference_snapshot(self):
        """Test setting reference snapshot."""
        self.tracker.set_reference_snapshot(self.reference_points)
        self.assertIsNotNone(self.tracker.reference_points)
        self.assertEqual(len(self.tracker.reference_points), 5)
        
        # Check that points are correctly stored as numpy arrays
        for id_, pos in self.reference_points.items():
            np.testing.assert_array_equal(self.tracker.reference_points[id_], pos)
            
    def test_compute_deviation_distances(self):
        """Test deviation distance computation."""
        self.tracker.set_reference_snapshot(self.reference_points)
        
        # Test with identical points (zero deviation)
        identical_points = self.reference_points.copy()
        deviations = self.tracker.compute_deviation_distances(identical_points)
        
        for id_ in self.reference_points.keys():
            self.assertAlmostEqual(deviations[id_], 0.0, places=6)
            
        # Test with slightly moved points
        moved_points = {
            0: np.array([11.0, 11.0]),  # moved by sqrt(2) ≈ 1.414
            1: np.array([20.0, 15.0]),  # unchanged
            2: np.array([35.0, 20.0]),  # moved by 5.0
        }
        
        deviations = self.tracker.compute_deviation_distances(moved_points)
        self.assertAlmostEqual(deviations[0], np.sqrt(2), places=3)
        self.assertAlmostEqual(deviations[1], 0.0, places=6)
        self.assertAlmostEqual(deviations[2], 5.0, places=6)
        
        # Test with new ID not in reference
        new_id_points = {0: np.array([10.0, 10.0]), 99: np.array([100.0, 100.0])}
        deviations = self.tracker.compute_deviation_distances(new_id_points)
        self.assertAlmostEqual(deviations[0], 0.0, places=6)
        self.assertEqual(deviations[99], float('inf'))
        
    def test_compute_entropy_threshold(self):
        """Test entropy-based threshold computation."""
        # Test with empty list
        threshold = self.tracker.compute_entropy_threshold([])
        self.assertEqual(threshold, 0.0)
        
        # Test with single value
        threshold = self.tracker.compute_entropy_threshold([5.0])
        self.assertGreater(threshold, 0.0)
        
        # Test with multiple values
        deviations = [1.0, 2.0, 3.0, 4.0, 50.0]  # Last one is outlier
        threshold = self.tracker.compute_entropy_threshold(deviations)
        self.assertGreater(threshold, 4.0)  # Should be higher than normal values
        self.assertLessEqual(threshold, self.tracker.max_distance_threshold)
        
        # Test with infinite values
        deviations_with_inf = [1.0, 2.0, float('inf'), 3.0]
        threshold = self.tracker.compute_entropy_threshold(deviations_with_inf)
        self.assertGreater(threshold, 0.0)
        self.assertTrue(np.isfinite(threshold))
        
    def test_identify_outliers(self):
        """Test outlier identification."""
        self.tracker.set_reference_snapshot(self.reference_points)
        
        # Test with normal points (small deviations)
        normal_points = {
            0: np.array([11.0, 11.0]),  # Small deviation
            1: np.array([21.0, 16.0]),  # Small deviation
            2: np.array([29.0, 19.0]),  # Small deviation
        }
        
        valid_ids, outlier_ids = self.tracker.identify_outliers(normal_points)
        self.assertEqual(len(outlier_ids), 0)  # No outliers expected
        self.assertEqual(len(valid_ids), 3)
        
        # Test with one clear outlier
        points_with_outlier = {
            0: np.array([11.0, 11.0]),   # Small deviation
            1: np.array([21.0, 16.0]),   # Small deviation
            2: np.array([100.0, 100.0]), # Clear outlier
        }
        
        valid_ids, outlier_ids = self.tracker.identify_outliers(points_with_outlier)
        self.assertIn(2, outlier_ids)  # Point 2 should be identified as outlier
        self.assertIn(0, valid_ids)
        self.assertIn(1, valid_ids)
        
    def test_reassign_ids_hungarian(self):
        """Test ID reassignment using Hungarian algorithm."""
        self.tracker.set_reference_snapshot(self.reference_points)
        
        # Test with swapped IDs (but same positions)
        swapped_points = {
            1: np.array([10.0, 10.0]),  # Should match to ID 0
            0: np.array([20.0, 15.0]),  # Should match to ID 1
            2: np.array([30.0, 20.0]),  # Should stay as ID 2
        }
        
        id_mapping = self.tracker.reassign_ids_hungarian(swapped_points)
        
        # Check that points are mapped to their closest reference positions
        self.assertEqual(id_mapping[1], 0)  # Point at (10,10) maps to ID 0
        self.assertEqual(id_mapping[0], 1)  # Point at (20,15) maps to ID 1
        self.assertEqual(id_mapping[2], 2)  # Point at (30,20) stays ID 2
        
    def test_process_frame_complete(self):
        """Test complete frame processing pipeline."""
        self.tracker.set_reference_snapshot(self.reference_points)
        
        # Create test frame with various issues
        current_points = {
            1: np.array([10.5, 10.5]),   # Slightly moved, should map to ID 0
            0: np.array([20.5, 15.5]),   # Slightly moved, should map to ID 1  
            2: np.array([30.5, 20.5]),   # Slightly moved, should stay ID 2
            5: np.array([200.0, 200.0]), # Outlier, should be rejected
        }
        
        corrected_points, outlier_ids, id_mapping = self.tracker.process_frame(current_points)
        
        # Check that outlier was identified
        self.assertIn(5, outlier_ids)
        
        # Check that valid points were reassigned correctly
        self.assertIn(0, corrected_points)  # Should have ID 0
        self.assertIn(1, corrected_points)  # Should have ID 1
        self.assertIn(2, corrected_points)  # Should have ID 2
        
        # Check positions are approximately correct
        np.testing.assert_array_almost_equal(corrected_points[0], np.array([10.5, 10.5]), decimal=1)
        
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test without setting reference
        current_points = {0: np.array([1.0, 1.0])}
        
        with self.assertRaises(ValueError):
            self.tracker.compute_deviation_distances(current_points)
            
        with self.assertRaises(ValueError):
            self.tracker.reassign_ids_hungarian(current_points)
            
        with self.assertRaises(ValueError):
            self.tracker.process_frame(current_points)


class TestUtilityFunctions(unittest.TestCase):
    """Unit tests for utility functions."""
    
    def test_generate_test_points(self):
        """Test test point generation."""
        points = generate_test_points(n_points=5, bounds=(0, 10, 0, 10))
        
        self.assertEqual(len(points), 5)
        
        # Check all points are within bounds
        for id_, pos in points.items():
            self.assertGreaterEqual(pos[0], 0)
            self.assertLessEqual(pos[0], 10)
            self.assertGreaterEqual(pos[1], 0)
            self.assertLessEqual(pos[1], 10)
            
        # Check IDs are sequential
        expected_ids = set(range(5))
        actual_ids = set(points.keys())
        self.assertEqual(expected_ids, actual_ids)
        
    def test_apply_transformations(self):
        """Test transformation application."""
        reference_points = {
            0: np.array([10.0, 10.0]),
            1: np.array([20.0, 20.0]),
            2: np.array([30.0, 30.0]),
        }
        
        # Test noise only
        np.random.seed(42)  # For reproducible tests
        transformed = apply_transformations(
            reference_points, 
            noise_std=1.0,
            translation=(0, 0),
            missing_ids=[],
            outlier_ids=[]
        )
        
        self.assertEqual(len(transformed), 3)
        # Points should be close to original but not identical due to noise
        for id_ in reference_points.keys():
            distance = np.linalg.norm(transformed[id_] - reference_points[id_])
            self.assertLess(distance, 5.0)  # Should be close
            
        # Test translation
        transformed = apply_transformations(
            reference_points,
            noise_std=0.0,  # No noise
            translation=(5.0, -3.0),
            missing_ids=[],
            outlier_ids=[]
        )
        
        for id_, original_pos in reference_points.items():
            expected_pos = original_pos + np.array([5.0, -3.0])
            np.testing.assert_array_almost_equal(transformed[id_], expected_pos, decimal=6)
            
        # Test missing points
        transformed = apply_transformations(
            reference_points,
            noise_std=0.0,
            translation=(0, 0),
            missing_ids=[1],
            outlier_ids=[]
        )
        
        self.assertEqual(len(transformed), 2)  # One point should be missing
        self.assertNotIn(1, transformed)
        self.assertIn(0, transformed)
        self.assertIn(2, transformed)
        
        # Test outliers
        transformed = apply_transformations(
            reference_points,
            noise_std=0.0,
            translation=(0, 0),
            missing_ids=[],
            outlier_ids=[2],
            outlier_distance=50.0
        )
        
        # Outlier should be far from original position
        outlier_distance = np.linalg.norm(transformed[2] - reference_points[2])
        self.assertGreater(outlier_distance, 40.0)  # Should be far away


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2) 