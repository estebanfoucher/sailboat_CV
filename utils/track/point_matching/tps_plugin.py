import numpy as np
from scipy.spatial.distance import cdist

class TPSPlugin:
    def __init__(self, regularization=0.1):
        self.regularization = regularization
        
    def refine(self, matches, reference_points, detected_points):
        """Apply TPS refinement to improve matches"""
        if len(matches) < 4:
            return matches
            
        ref = np.array(reference_points)
        det = np.array(detected_points)
        
        # Extract matched point pairs
        det_matched = det[[m[0] for m in matches]]
        ref_matched = ref[[m[1] for m in matches]]
        
        # Fit TPS transformation
        tps_params = self._fit_tps(det_matched, ref_matched)
        
        # Transform all detected points
        det_warped = self._apply_tps(det, det_matched, ref_matched, tps_params)
        
        # Re-assign with warped points
        cost_matrix = cdist(det_warped, ref)
        from scipy.optimize import linear_sum_assignment
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        
        # Return refined matches
        refined_matches = []
        for r, c in zip(row_idx, col_idx):
            cost = cost_matrix[r, c]
            refined_matches.append((r, c, cost))
            
        return refined_matches
    
    def _fit_tps(self, source, target):
        """Fit TPS transformation parameters"""
        n = len(source)
        
        # Compute distance matrices
        K = self._tps_kernel(source, source)
        P = np.column_stack([np.ones(n), source])
        
        # Build system matrix
        A = np.block([[K + self.regularization * np.eye(n), P],
                      [P.T, np.zeros((3, 3))]])
        
        # Solve for x and y coordinates separately
        w_x = np.linalg.solve(A, np.concatenate([target[:, 0], np.zeros(3)]))
        w_y = np.linalg.solve(A, np.concatenate([target[:, 1], np.zeros(3)]))
        
        return {
            'w': np.column_stack([w_x[:n], w_y[:n]]),
            'a': np.column_stack([w_x[n:], w_y[n:]]),
            'control_points': source
        }
    
    def _apply_tps(self, points, source, target, params):
        """Apply TPS transformation to points"""
        K = self._tps_kernel(points, params['control_points'])
        P = np.column_stack([np.ones(len(points)), points])
        
        warped = K @ params['w'] + P @ params['a']
        return warped
    
    def _tps_kernel(self, p1, p2):
        """TPS radial basis function kernel"""
        dists = cdist(p1, p2)
        with np.errstate(divide='ignore', invalid='ignore'):
            K = dists**2 * np.log(dists)
            K[dists == 0] = 0
        return K