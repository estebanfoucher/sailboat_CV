import numpy as np
import matplotlib.pyplot as plt
from matcher import PointMatcher

def create_deformation_grid(bounds, grid_size=10):
    """Create a regular grid for deformation visualization"""
    x_min, x_max, y_min, y_max = bounds
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])

def apply_transform_to_grid(grid, R, t):
    """Apply rigid transformation to grid"""
    return (grid @ R.T) + t

def apply_tps_to_grid(grid, tps_plugin, control_points, target_points):
    """Apply TPS transformation to grid"""
    try:
        tps_params = tps_plugin._fit_tps(control_points, target_points)
        return tps_plugin._apply_tps(grid, control_points, target_points, tps_params)
    except:
        return grid

def generate_translation_test_data(n_ref=10, n_det=8, noise=0.05, outliers=2):
    """Generate test data with pure translation (no rotation)"""
    np.random.seed(42)
    ref = np.random.rand(n_ref, 2) * 6 + 2  # 2-8 range, compact layout
    
    # Transform subset of reference points with pure translation
    indices = np.random.choice(n_ref, n_det - outliers, replace=False)
    ref_subset = ref[indices]
    
    # Apply pure translation
    translation = np.array([1.5, 0.8])
    det_clean = ref_subset + translation
    
    # Add measurement noise
    det_clean += np.random.normal(0, noise, det_clean.shape)
    
    # Add outliers
    outlier_center = ref.mean(axis=0)
    outliers_pts = outlier_center + np.random.normal(0, 3, (outliers, 2))
    det = np.vstack([det_clean, outliers_pts])
    
    # Ground truth correspondences (det_idx -> ref_idx for non-outliers)
    ground_truth = [(i, indices[i]) for i in range(len(indices))]
    
    return ref, det, ground_truth

def generate_rotation_test_data(n_ref=10, n_det=8, noise=0.05, rotation_angle=15, outliers=2):
    """Generate test data with pure rotation around center (no translation)"""
    np.random.seed(42)
    ref = np.random.rand(n_ref, 2) * 6 + 2  # 2-8 range, compact layout
    
    # Transform subset of reference points with pure rotation
    indices = np.random.choice(n_ref, n_det - outliers, replace=False)
    ref_subset = ref[indices]
    
    # Rotation around center of reference layout
    center = ref.mean(axis=0)
    angle_rad = np.radians(rotation_angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])
    
    # Apply pure rotation around center
    centered = ref_subset - center
    rotated = centered @ R.T
    det_clean = rotated + center
    
    # Add measurement noise
    det_clean += np.random.normal(0, noise, det_clean.shape)
    
    # Add outliers
    outlier_center = ref.mean(axis=0)
    outliers_pts = outlier_center + np.random.normal(0, 3, (outliers, 2))
    det = np.vstack([det_clean, outliers_pts])
    
    # Ground truth correspondences (det_idx -> ref_idx for non-outliers)
    ground_truth = [(i, indices[i]) for i in range(len(indices))]
    
    return ref, det, ground_truth

def visualize_results(ref, det, matches, ground_truth=None, transform=None, title=""):
    """Enhanced visualization showing structure, assignments, and deformation"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    ax1, ax2, ax3 = axes
    
    # Left: Reference structure + raw detections
    ax1.scatter(ref[:, 0], ref[:, 1], c='blue', s=80, marker='o',
               label='Reference Layout', edgecolors='darkblue', linewidth=2)
    
    # Show reference structure (connect points in order)
    ref_hull = ref[np.argsort(np.arctan2(ref[:, 1] - ref.mean(axis=0)[1],
                                        ref[:, 0] - ref.mean(axis=0)[0]))]
    for i in range(len(ref_hull)):
        next_i = (i + 1) % len(ref_hull)
        ax1.plot([ref_hull[i, 0], ref_hull[next_i, 0]],
                [ref_hull[i, 1], ref_hull[next_i, 1]], 'b--', alpha=0.3, linewidth=1)
    
    ax1.scatter(det[:, 0], det[:, 1], c='red', s=60, marker='x',
               label='Raw Detections', linewidth=3)
    
    # Number the points for clarity
    for i, (x, y) in enumerate(ref):
        ax1.annotate(f'R{i}', (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='blue', weight='bold')
    for i, (x, y) in enumerate(det):
        ax1.annotate(f'D{i}', (x, y), xytext=(5, -15), textcoords='offset points',
                    fontsize=8, color='red', weight='bold')
    
    ax1.set_title('Input: Reference Layout + Detections', fontsize=12, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Right: Assignment results
    ax2.scatter(ref[:, 0], ref[:, 1], c='lightblue', s=80, marker='o',
               alpha=0.6, edgecolors='blue', linewidth=1)
    ax2.scatter(det[:, 0], det[:, 1], c='lightcoral', s=60, marker='x',
               alpha=0.6, linewidth=2)
    
    # Show ground truth matches first (if available) with dashed blue lines
    if ground_truth is not None:
        for det_idx, ref_idx in ground_truth:
            if det_idx < len(det) and ref_idx < len(ref):
                ax2.plot([det[det_idx, 0], ref[ref_idx, 0]],
                        [det[det_idx, 1], ref[ref_idx, 1]], 'b--',
                        linewidth=2, alpha=0.6, label='Ground Truth' if det_idx == ground_truth[0][0] else "")
    
    # Show predicted matches with thick green lines
    matched_refs = set()
    matched_dets = set()
    for det_idx, ref_idx, cost in matches:
        ax2.plot([det[det_idx, 0], ref[ref_idx, 0]],
                [det[det_idx, 1], ref[ref_idx, 1]], 'g-',
                linewidth=3, alpha=0.8, label='Predicted' if det_idx == matches[0][0] else "")
        matched_refs.add(ref_idx)
        matched_dets.add(det_idx)
    
    # Highlight unmatched points
    unmatched_ref_idx = [i for i in range(len(ref)) if i not in matched_refs]
    unmatched_det_idx = [i for i in range(len(det)) if i not in matched_dets]
    
    if unmatched_ref_idx:
        ax2.scatter(ref[unmatched_ref_idx, 0], ref[unmatched_ref_idx, 1],
                   c='blue', s=100, marker='o', label='Unmatched Ref')
    if unmatched_det_idx:
        ax2.scatter(det[unmatched_det_idx, 0], det[unmatched_det_idx, 1],
                   c='red', s=80, marker='X', label='Outliers/False Pos')
    
    # Calculate accuracy if ground truth is available
    accuracy_text = ""
    if ground_truth is not None:
        gt_set = set(ground_truth)
        pred_set = set((det_idx, ref_idx) for det_idx, ref_idx, _ in matches)
        correct_matches = len(gt_set & pred_set)
        total_gt = len(ground_truth)
        accuracy = correct_matches / total_gt * 100 if total_gt > 0 else 0
        accuracy_text = f" | Accuracy: {accuracy:.1f}%"
    
    ax2.set_title(f'{title}\n{len(matches)} matches | {len(unmatched_ref_idx)} missing | {len(unmatched_det_idx)} outliers{accuracy_text}',
                 fontsize=12, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Third panel: Deformation flow visualization
    if transform is not None:
        R, t = transform
        
        # Create deformation grid
        all_points = np.vstack([ref, det])
        bounds = [all_points[:, 0].min()-2, all_points[:, 0].max()+2,
                 all_points[:, 1].min()-2, all_points[:, 1].max()+2]
        grid = create_deformation_grid(bounds, grid_size=8)
        
        # Apply rigid transformation
        grid_transformed = apply_transform_to_grid(grid, R, t)
        
        # Show original grid (light gray)
        grid_2d = grid.reshape(8, 8, 2)
        for i in range(8):
            ax3.plot(grid_2d[i, :, 0], grid_2d[i, :, 1], 'lightgray', alpha=0.5, linewidth=1)
            ax3.plot(grid_2d[:, i, 0], grid_2d[:, i, 1], 'lightgray', alpha=0.5, linewidth=1)
        
        # Show transformed grid (blue for rigid)
        grid_trans_2d = grid_transformed.reshape(8, 8, 2)
        for i in range(8):
            ax3.plot(grid_trans_2d[i, :, 0], grid_trans_2d[i, :, 1], 'blue', alpha=0.7, linewidth=1.5)
            ax3.plot(grid_trans_2d[:, i, 0], grid_trans_2d[:, i, 1], 'blue', alpha=0.7, linewidth=1.5)
        
        # Show flow vectors
        step = 2  # Subsample for clarity
        for i in range(0, len(grid), step):
            ax3.arrow(grid[i, 0], grid[i, 1],
                     grid_transformed[i, 0] - grid[i, 0],
                     grid_transformed[i, 1] - grid[i, 1],
                     head_width=0.2, head_length=0.1, fc='red', ec='red', alpha=0.6)
        
        # If TPS was used, show additional deformation
        if 'TPS' in title and len(matches) >= 4:
            try:
                from tps_plugin import TPSPlugin
                tps = TPSPlugin()
                matched_det = det[[m[0] for m in matches]]
                matched_ref = ref[[m[1] for m in matches]]
                grid_tps = apply_tps_to_grid(grid_transformed, tps, matched_det, matched_ref)
                
                # Show TPS grid (green)
                grid_tps_2d = grid_tps.reshape(8, 8, 2)
                for i in range(8):
                    ax3.plot(grid_tps_2d[i, :, 0], grid_tps_2d[i, :, 1], 'green', alpha=0.8, linewidth=2)
                    ax3.plot(grid_tps_2d[:, i, 0], grid_tps_2d[:, i, 1], 'green', alpha=0.8, linewidth=2)
            except:
                pass
        
        ax3.scatter(ref[:, 0], ref[:, 1], c='blue', s=50, marker='o', alpha=0.8, label='Reference')
        ax3.scatter(det[:, 0], det[:, 1], c='red', s=40, marker='x', alpha=0.8, label='Detected')
        
        # Add rotation angle info
        angle_deg = np.degrees(np.arccos(np.clip(R[0, 0], -1, 1)))
        ax3.set_title(f'Deformation Flow\nRotation: {angle_deg:.1f}°', fontsize=12, weight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
    else:
        ax3.text(0.5, 0.5, 'No transformation\navailable', ha='center', va='center',
                transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Deformation Flow', fontsize=12, weight='bold')
    
    plt.tight_layout()
    
    # Save plot instead of showing (for headless environments)
    save_path = f"{title.lower().replace(' ', '_').replace('+', '_')}_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.close()

def run_example():
    """Run two simple examples: translation only vs rotation only"""
    matcher = PointMatcher()
    
    print("=== EXAMPLE 1: PURE TRANSLATION ===")
    ref1, det1, gt1 = generate_translation_test_data()
    
    matches_trans, transform_trans = matcher.match(ref1, det1, use_tps=False)
    print(f"Translation matches: {len(matches_trans)}")
    visualize_results(ref1, det1, matches_trans, gt1, transform_trans, "Pure Translation")
    
    print("\n=== EXAMPLE 2: PURE ROTATION ===")
    ref2, det2, gt2 = generate_rotation_test_data()
    
    matches_rot, transform_rot = matcher.match(ref2, det2, use_tps=False)
    print(f"Rotation matches: {len(matches_rot)}")
    visualize_results(ref2, det2, matches_rot, gt2, transform_rot, "Pure Rotation")

if __name__ == "__main__":
    run_example()