import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class PointMatcher:
    def __init__(self, ransac_iterations=100, min_inliers=3, inlier_threshold=2.0):
        self.ransac_iterations = ransac_iterations
        self.min_inliers = min_inliers
        self.inlier_threshold = inlier_threshold
        
    def match(self, reference_points, detected_points, use_tps=False):
        """Main pipeline: align -> match -> optionally refine"""
        ref = np.array(reference_points)
        det = np.array(detected_points)
        
        # Step 1: RANSAC + Kabsch alignment
        R, t, inlier_mask = self._ransac_align(det, ref)
        
        # Step 2: Apply transformation and compute costs
        det_aligned = (det @ R.T) + t
        cost_matrix = cdist(det_aligned, ref)
        
        # Step 3: Hungarian assignment
        assignments = self._assign(cost_matrix)
        
        # Step 4: Consensus-based outlier rejection
        final_matches = self._reject_outliers(assignments, cost_matrix, inlier_mask)
        
        # Step 5: Optional TPS refinement
        if use_tps and len(final_matches) >= 4:
            try:
                from tps_plugin import TPSPlugin
                tps = TPSPlugin()
                final_matches = tps.refine(final_matches, ref, det)
            except ImportError:
                pass
                
        return final_matches, (R, t)
    
    def _ransac_align(self, det, ref):
        """RANSAC + Kabsch for robust rigid alignment"""
        best_inliers = 0
        best_R, best_t = np.eye(2), np.zeros(2)
        best_mask = np.zeros(len(det), dtype=bool)
        
        for _ in range(self.ransac_iterations):
            if len(det) < self.min_inliers:
                break
                
            # Sample random points
            sample_idx = np.random.choice(len(det), min(self.min_inliers, len(det)), replace=False)
            det_sample = det[sample_idx]
            
            # Find closest reference points for sample
            dists = cdist(det_sample, ref)
            ref_idx = dists.argmin(axis=1)
            ref_sample = ref[ref_idx]
            
            # Kabsch algorithm
            R, t = self._kabsch(det_sample, ref_sample)
            
            # Count inliers
            det_transformed = (det @ R.T) + t
            dists_all = cdist(det_transformed, ref).min(axis=1)
            inlier_mask = dists_all < self.inlier_threshold
            inlier_count = inlier_mask.sum()
            
            if inlier_count > best_inliers:
                best_inliers = inlier_count
                best_R, best_t = R, t
                best_mask = inlier_mask
                
        return best_R, best_t, best_mask
    
    def _kabsch(self, P, Q):
        """Kabsch algorithm for optimal rigid transformation"""
        P_centered = P - P.mean(axis=0)
        Q_centered = Q - Q.mean(axis=0)
        
        H = P_centered.T @ Q_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
            
        t = Q.mean(axis=0) - (P.mean(axis=0) @ R.T)
        return R, t
    
    def _assign(self, cost_matrix):
        """Hungarian assignment with rectangular matrix support"""
        if cost_matrix.size == 0:
            return []
            
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        return list(zip(row_idx, col_idx))
    
    def _reject_outliers(self, assignments, cost_matrix, inlier_mask):
        """Consensus-based outlier rejection using RANSAC inliers"""
        valid_assignments = []
        
        for det_idx, ref_idx in assignments:
            # Accept if detection was RANSAC inlier
            if det_idx < len(inlier_mask) and inlier_mask[det_idx]:
                cost = cost_matrix[det_idx, ref_idx]
                valid_assignments.append((det_idx, ref_idx, cost))
                
        return valid_assignments