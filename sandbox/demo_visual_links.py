import matplotlib.pyplot as plt
import numpy as np
from tracker_overlayer import TrackerOverlayer, generate_test_points, apply_transformations
from visualize_tests import plot_point_comparison

def create_focused_demos():
    """Create focused demonstrations of missing detections and false positives."""
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Generate a simple set of reference points (6 points for clarity)
    reference_points = {
        0: np.array([20.0, 20.0]),
        1: np.array([40.0, 20.0]), 
        2: np.array([60.0, 20.0]),
        3: np.array([20.0, 40.0]),
        4: np.array([40.0, 40.0]),
        5: np.array([60.0, 40.0])
    }
    
    # Initialize tracker
    tracker = TrackerOverlayer(entropy_threshold_factor=2.0, max_distance_threshold=30.0)
    tracker.set_reference_snapshot(reference_points)
    
    # Create 3 focused test scenarios
    scenarios = [
        ("Missing Detections", {
            0: np.array([21.0, 21.0]),   # Correct match
            1: np.array([41.0, 21.0]),   # Correct match  
            2: np.array([61.0, 19.0]),   # Correct match
            # ID 3, 4, 5 are missing (missing detections)
        }),
        
        ("False Positives", {
            0: np.array([21.0, 21.0]),   # Correct match
            1: np.array([41.0, 21.0]),   # Correct match
            2: np.array([61.0, 19.0]),   # Correct match
            3: np.array([21.0, 39.0]),   # Correct match
            4: np.array([41.0, 39.0]),   # Correct match
            5: np.array([59.0, 41.0]),   # Correct match
            99: np.array([80.0, 60.0]),  # False positive
            98: np.array([10.0, 60.0]),  # False positive
        }),
        
        ("Mixed: Missing + False Positives", {
            0: np.array([21.0, 21.0]),   # Correct match
            2: np.array([61.0, 19.0]),   # Correct match
            4: np.array([41.0, 39.0]),   # Correct match
            # IDs 1, 3, 5 are missing
            97: np.array([80.0, 60.0]),  # False positive
            98: np.array([10.0, 60.0]),  # False positive
        })
    ]
    
    # Create 3-panel visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (scenario_name, current_points) in enumerate(scenarios):
        ax = axes[idx]
        
        # Process through tracker
        corrected_points, outlier_ids, id_mapping = tracker.process_frame(current_points)
        
        plot_point_comparison(reference_points, current_points, corrected_points, 
                            outlier_ids, id_mapping, scenario_name, ax)
        
        # Print analysis
        print(f"\n=== {scenario_name} ===")
        print(f"Reference points: {len(reference_points)}")
        print(f"Current detections: {len(current_points)}")
        print(f"Valid detections: {len(current_points) - len(outlier_ids)}")
        print(f"Outliers detected: {len(outlier_ids)}")
        print(f"Missing detections: {len(reference_points) - len(id_mapping)}")
        
        # Calculate accuracy
        correct_matches = sum(1 for curr_id, ref_id in id_mapping.items() 
                             if curr_id == ref_id and curr_id not in outlier_ids)
        total_valid = len([id_ for id_ in current_points.keys() if id_ not in outlier_ids])
        accuracy = (correct_matches / total_valid * 100) if total_valid > 0 else 0
        print(f"Accuracy: {correct_matches}/{total_valid} = {accuracy:.1f}%")
    
    # Add comprehensive explanation
    explanation = """
FOCUSED TEST SCENARIOS

Left Panel - Missing Detections:
• Only detections D0, D1, D2 are present (IDs 3, 4, 5 are missing)
• Blue dashed lines show where missing detections should connect
• Dark blue circles highlight reference points with no detection
• All present detections correctly match (100% accuracy for detected points)

Middle Panel - False Positives:  
• All reference points have correct detections (D0-D5)
• Additional false detections D98, D99 are added
• Red X's show outliers identified and rejected by algorithm
• Green solid lines only connect to valid reference points

Right Panel - Mixed Scenario:
• Some detections missing (D1, D3, D5) 
• Some false positives present (D97, D98)
• Shows realistic tracking scenario with both types of errors
• Algorithm must simultaneously handle missing data and reject outliers

Key Insight: The algorithm independently handles missing detections (by not requiring matches) 
and false positives (by rejecting outliers), making it robust to both error types.
"""
    
    plt.figtext(0.02, 0.02, explanation, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.45)  # Make room for explanation
    plt.savefig('focused_demo_missing_false_positives.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("Creating focused demonstrations of missing detections and false positives...")
    create_focused_demos()
    print("\nDemo completed! Check 'focused_demo_missing_false_positives.png' for the visual explanation.") 