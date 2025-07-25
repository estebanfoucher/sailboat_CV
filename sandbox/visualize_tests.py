import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from tracker_overlayer import TrackerOverlayer, generate_test_points, apply_transformations


def plot_point_comparison(reference_points: Dict[int, np.ndarray],
                         current_points: Dict[int, np.ndarray],
                         corrected_points: Dict[int, np.ndarray],
                         outlier_ids: List[int],
                         id_mapping: Dict[int, int],
                         title: str,
                         ax: plt.Axes) -> None:
    """
    Plot comparison between reference, current, and corrected points with matching links.
    
    Args:
        reference_points: Original reference points
        current_points: Current detected points
        corrected_points: Points after correction
        outlier_ids: IDs identified as outliers
        id_mapping: Mapping from current IDs to corrected IDs
        title: Plot title
        ax: Matplotlib axes to plot on
    """
    # Plot reference points (ground truth layout)
    ref_ids = list(reference_points.keys())
    ref_x = [reference_points[id_][0] for id_ in ref_ids]
    ref_y = [reference_points[id_][1] for id_ in ref_ids]
    
    ax.scatter(ref_x, ref_y, c='lightblue', s=100, marker='o', alpha=0.8, 
               label='Reference Layout', edgecolors='blue', linewidth=2)
    
    # Add reference ID labels
    for id_ in ref_ids:
        ax.annotate(f'R{id_}', reference_points[id_], xytext=(5, 8), 
                   textcoords='offset points', fontsize=9, color='blue', weight='bold')
    
    # Plot current points (detections)
    curr_ids = list(current_points.keys())
    
    # Separate outliers and valid points
    valid_curr_ids = [id_ for id_ in curr_ids if id_ not in outlier_ids]
    outlier_curr_ids = [id_ for id_ in curr_ids if id_ in outlier_ids]
    
    # Plot valid current points
    if valid_curr_ids:
        valid_x = [current_points[id_][0] for id_ in valid_curr_ids]
        valid_y = [current_points[id_][1] for id_ in valid_curr_ids]
        ax.scatter(valid_x, valid_y, c='lightcoral', s=80, marker='x', alpha=0.8,
                   label='Valid Detections', linewidth=3)
    
    # Plot outlier current points  
    if outlier_curr_ids:
        outlier_x = [current_points[id_][0] for id_ in outlier_curr_ids]
        outlier_y = [current_points[id_][1] for id_ in outlier_curr_ids]
        ax.scatter(outlier_x, outlier_y, c='red', s=100, marker='X', alpha=0.9,
                   label='Outliers/False Positives', linewidth=2)
    
    # Add current detection ID labels
    for id_ in curr_ids:
        color = 'red' if id_ in outlier_ids else 'darkred'
        ax.annotate(f'D{id_}', current_points[id_], xytext=(-15, -8), 
                   textcoords='offset points', fontsize=9, color=color, weight='bold')
    
    # Draw ground truth links (blue dashed lines)
    # Ground truth: each detection ID should ideally match to the reference point with same ID
    ground_truth_links_drawn = False
    for curr_id in valid_curr_ids:
        if curr_id in reference_points:
            # This is a ground truth correspondence (detection should match its own ID in reference)
            ref_pos = reference_points[curr_id]
            curr_pos = current_points[curr_id]
            ax.plot([curr_pos[0], ref_pos[0]], [curr_pos[1], ref_pos[1]], 
                   'b--', linewidth=2, alpha=0.7, 
                   label='Ground Truth Links' if not ground_truth_links_drawn else "")
            ground_truth_links_drawn = True
    
    # Draw predicted matching links (green solid lines)
    predicted_links_drawn = False
    for curr_id, assigned_ref_id in id_mapping.items():
        if curr_id not in outlier_ids and assigned_ref_id in reference_points:
            curr_pos = current_points[curr_id]
            ref_pos = reference_points[assigned_ref_id]
            ax.plot([curr_pos[0], ref_pos[0]], [curr_pos[1], ref_pos[1]], 
                   'g-', linewidth=3, alpha=0.8,
                   label='Predicted Matches' if not predicted_links_drawn else "")
            predicted_links_drawn = True
    
    # Highlight unmatched reference points (missing detections)
    matched_ref_ids = set(id_mapping.values())
    unmatched_ref_ids = [id_ for id_ in ref_ids if id_ not in matched_ref_ids]
    
    if unmatched_ref_ids:
        unmatched_x = [reference_points[id_][0] for id_ in unmatched_ref_ids]
        unmatched_y = [reference_points[id_][1] for id_ in unmatched_ref_ids]
        ax.scatter(unmatched_x, unmatched_y, c='blue', s=120, marker='o', 
                   label='Missing Detections', edgecolors='darkblue', linewidth=3, alpha=0.9)
    
    # Calculate matching accuracy
    correct_matches = 0
    total_valid = len(valid_curr_ids)
    
    for curr_id in valid_curr_ids:
        if curr_id in id_mapping and curr_id in reference_points:
            # Check if the prediction matches ground truth
            predicted_ref_id = id_mapping[curr_id]
            if predicted_ref_id == curr_id:  # Correct match
                correct_matches += 1
    
    accuracy = (correct_matches / total_valid * 100) if total_valid > 0 else 0
    
    # Create comprehensive title with metrics
    missing_count = len(unmatched_ref_ids)
    metrics_title = f"{title}\n{len(id_mapping)} matches | {len(outlier_curr_ids)} outliers | {missing_count} missing | {correct_matches}/{total_valid} correct | Accuracy: {accuracy:.1f}%"
    
    ax.set_title(metrics_title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')


def create_test_scenarios() -> List[Tuple[str, Dict]]:
    """
    Create different test scenarios with various transformations.
    Focus on independent testing of misses and false positives, then combinations.
    
    Returns:
        List of (scenario_name, transformation_params) tuples
    """
    scenarios = [
        ("Baseline: Small Noise Only", {
            'noise_std': 2.0,
            'translation': (0.0, 0.0),
            'missing_ids': [],
            'outlier_ids': [],
            'outlier_distance': 30.0
        }),
        
        ("Baseline: Translation + Noise", {
            'noise_std': 1.5,
            'translation': (8.0, -5.0),
            'missing_ids': [],
            'outlier_ids': [],
            'outlier_distance': 30.0
        }),
        
        ("Missing: Single Detection", {
            'noise_std': 1.0,
            'translation': (1.0, 1.0),
            'missing_ids': [5],  # Remove one point
            'outlier_ids': [],
            'outlier_distance': 30.0
        }),
        
        ("Missing: Multiple Detections", {
            'noise_std': 1.0,
            'translation': (2.0, 1.0),
            'missing_ids': [2, 6, 8],  # Remove three points
            'outlier_ids': [],
            'outlier_distance': 30.0
        }),
        
        ("False Positive: Single Outlier", {
            'noise_std': 1.0,
            'translation': (1.0, -1.0),
            'missing_ids': [],
            'outlier_ids': [3],  # One outlier
            'outlier_distance': 40.0
        }),
        
        ("False Positive: Multiple Outliers", {
            'noise_std': 1.0,
            'translation': (0.5, 0.5),
            'missing_ids': [],
            'outlier_ids': [1, 7],  # Two outliers
            'outlier_distance': 35.0
        }),
        
        ("Mixed: Missing + False Positive", {
            'noise_std': 1.5,
            'translation': (2.0, 2.0),
            'missing_ids': [4, 9],  # Two missing
            'outlier_ids': [2, 6],  # Two outliers
            'outlier_distance': 38.0
        }),
        
        ("Complex: Heavy Missing + Outliers", {
            'noise_std': 2.0,
            'translation': (3.0, -2.0),
            'missing_ids': [0, 3, 7, 8],  # Four missing
            'outlier_ids': [1, 5, 9],     # Three outliers  
            'outlier_distance': 42.0
        })
    ]
    
    return scenarios



def run_visual_tests():
    """Run visual tests showing different transformation scenarios."""
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Generate reference points (10 points in a 100x100 area)
    reference_points = generate_test_points(n_points=10, bounds=(10, 90, 10, 90))
    
    # Initialize tracker
    tracker = TrackerOverlayer(entropy_threshold_factor=1.5, max_distance_threshold=30.0)
    tracker.set_reference_snapshot(reference_points)
    
    # Get test scenarios
    scenarios = create_test_scenarios()
    
    # Create figure with subplots
    n_scenarios = len(scenarios)
    n_cols = 4  # Better layout for 8 scenarios
    n_rows = (n_scenarios + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (scenario_name, params) in enumerate(scenarios):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        print(f"\n=== {scenario_name} ===")
        
        # Apply transformations
        current_points = apply_transformations(reference_points, **params)
        
        # Process frame through tracker
        try:
            corrected_points, outlier_ids, id_mapping = tracker.process_frame(current_points)
            
            print(f"Original points: {len(reference_points)}")
            print(f"Current points: {len(current_points)}")
            print(f"Corrected points: {len(corrected_points)}")
            print(f"Outliers: {outlier_ids}")
            print(f"ID mapping: {id_mapping}")
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            corrected_points = {}
            outlier_ids = []
        
        # Create visualization
        plot_point_comparison(reference_points, current_points, corrected_points, 
                            outlier_ids, id_mapping, scenario_name, ax)
    
    # Hide unused subplots
    for idx in range(n_scenarios, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('tracker_overlayer_tests.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    # Test entropy threshold computation with different noise levels
    print("\nEntropy threshold analysis:")
    for noise_level in [0.5, 1.0, 2.0, 5.0]:
        test_points = apply_transformations(reference_points, noise_std=noise_level)
        deviations = tracker.compute_deviation_distances(test_points)
        threshold = tracker.compute_entropy_threshold(list(deviations.values()))
        mean_dev = np.mean([d for d in deviations.values() if np.isfinite(d)])
        print(f"  Noise std {noise_level:3.1f}: threshold={threshold:5.2f}, mean_deviation={mean_dev:5.2f}")


def analyze_threshold_sensitivity():
    """Analyze how entropy threshold affects outlier detection."""
    
    np.random.seed(42)
    reference_points = generate_test_points(n_points=8, bounds=(20, 80, 20, 80))
    
    # Create test case with known outlier
    current_points = apply_transformations(
        reference_points,
        noise_std=2.0,
        outlier_ids=[3],
        outlier_distance=40.0
    )
    
    # Test different entropy thresholds
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, threshold_factor in enumerate(thresholds):
        tracker = TrackerOverlayer(entropy_threshold_factor=threshold_factor, max_distance_threshold=50.0)
        tracker.set_reference_snapshot(reference_points)
        
        corrected_points, outlier_ids, id_mapping = tracker.process_frame(current_points)
        
        ax = axes[idx]
        plot_point_comparison(reference_points, current_points, corrected_points, 
                            outlier_ids, id_mapping, f'Entropy Factor: {threshold_factor}', ax)
        
        # Print threshold value
        deviations = tracker.compute_deviation_distances(current_points)
        computed_threshold = tracker.compute_entropy_threshold(list(deviations.values()))
        ax.text(0.02, 0.98, f'Threshold: {computed_threshold:.2f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('threshold_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("Running tracker overlayer visual tests...")
    run_visual_tests()
    
    print("\nRunning threshold sensitivity analysis...")
    analyze_threshold_sensitivity()
    
    print("\nVisual tests completed! Check the generated PNG files.") 