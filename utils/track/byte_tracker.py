# Pure Python ByteTrack implementation (adapted from Yolov5_DeepSort_Pytorch)
# Uses scipy.optimize.linear_sum_assignment for assignment (no lap required)
# MIT License

import numpy as np
from scipy.optimize import linear_sum_assignment
from loguru import logger

class STrack:
    def __init__(self, tlwh, score, cls):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = float(score)
        self.cls = int(cls)
        self.track_id = None
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.end_frame = 0
        self.current_frame_class = self.cls  # Store the detector class for the current frame

    def activate(self, frame_id, track_id):
        self.is_activated = True
        self.track_id = track_id
        self.start_frame = frame_id
        self.frame_id = frame_id

    def update(self, tlwh, score, frame_id, current_frame_class=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = float(score)
        self.frame_id = frame_id
        if current_frame_class is not None:
            self.current_frame_class = int(current_frame_class)

    def to_xyxy(self):
        x, y, w, h = self.tlwh
        return [x, y, x + w, y + h]

class ByteTracker:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.next_id = 1

    def update(self, detections):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Convert detections to STrack objects
        det_stracks = []
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            score = det['confidence']
            cls = det['class_id'] if 'class_id' in det else det['original_class_id']
            det_cls = det['original_class_id']  # Always the detector class for this frame
            if score >= self.track_thresh:
                s = STrack([x1, y1, w, h], score, cls)
                s.current_frame_class = det_cls
                det_stracks.append(s)

        # Remove lost tracks that have been lost for too long
        new_lost_stracks = []
        for t in self.lost_stracks:
            if self.frame_id - t.frame_id > self.track_buffer:
                removed_stracks.append(t)
            else:
                new_lost_stracks.append(t)
        self.lost_stracks = new_lost_stracks

        # Association with currently tracked tracks
        matches, u_track, u_detection = self._associate(self.tracked_stracks, det_stracks)


        # Update matched tracks
        for t_idx, d_idx in matches:
            track = self.tracked_stracks[t_idx]
            det = det_stracks[d_idx]
            track.update(det.tlwh, det.score, self.frame_id, current_frame_class=det.current_frame_class)
            activated_stracks.append(track)

        # Unmatched tracks become lost
        for idx in u_track:
            track = self.tracked_stracks[idx]
            track.end_frame = self.frame_id
            self.lost_stracks.append(track)

        # Try to match unmatched detections to lost tracks (within buffer window)
        lost_candidates = [t for t in self.lost_stracks if self.frame_id - t.frame_id <= self.track_buffer]
        lost_det_sublist = [det_stracks[i] for i in u_detection]
        matches_lost, u_lost, u_detection_new = self._associate(lost_candidates, lost_det_sublist)
        # Collect lost tracks to reactivate and indices to remove after processing
        reactivated_lost_tracks = []
        lost_tracks_to_remove = []
        for l_idx, d_idx in matches_lost:
            if d_idx < len(u_detection):
                lost_track = lost_candidates[l_idx]
                det = det_stracks[u_detection[d_idx]]
                lost_track.update(det.tlwh, det.score, self.frame_id, current_frame_class=det.current_frame_class)
                lost_track.is_activated = True
                reactivated_lost_tracks.append(lost_track)
                lost_tracks_to_remove.append(lost_track)
            else:
                logger.error(f"[BYTETracker] d_idx {d_idx} out of range for u_detection (len={len(u_detection)}). matches_lost={matches_lost}, u_detection={u_detection}, lost_candidates={len(lost_candidates)}, det_stracks={len(det_stracks)}")
        # After processing all matches, update lists
        for lost_track in lost_tracks_to_remove:
            if lost_track in self.lost_stracks:
                self.lost_stracks.remove(lost_track)
        activated_stracks.extend(reactivated_lost_tracks)
        # Update unmatched detections after lost matching
        matched_lost_det_indices = [d for _, d in matches_lost]
        unmatched_detection_final = [idx for j, idx in enumerate(u_detection) if j not in matched_lost_det_indices]

        # Activate new tracks for remaining unmatched detections
        for idx in unmatched_detection_final:
            det = det_stracks[idx]
            det.activate(self.frame_id, self.next_id)
            self.next_id += 1
            activated_stracks.append(det)

        self.tracked_stracks = activated_stracks

        # Remove lost tracks that have exceeded the buffer
        removed_lost_tracks = []
        for t in self.lost_stracks:
            if self.frame_id - t.frame_id > self.track_buffer:
                logger.debug(f"[BYTETracker] Removing lost track id={t.track_id} (lost for {self.frame_id - t.frame_id} frames, buffer={self.track_buffer})")
                removed_lost_tracks.append(t)
        for t in removed_lost_tracks:
            self.lost_stracks.remove(t)
            removed_stracks.append(t)

        # Prepare output
        output_tracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                continue
            output_tracks.append({
                'bbox': track.to_xyxy(),
                'track_id': track.track_id,
                'original_class_id': track.current_frame_class,  # Use per-frame detector class
                'confidence': track.score
            })
        return output_tracks

    def _associate(self, tracks, detections):
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(detections):
                iou = self._iou(t.tlwh, d.tlwh)
                iou_matrix[i, j] = iou
                cost_matrix[i, j] = 1 - iou
                logger.debug(f"[BYTETracker] IoU(track {i}, det {j}) = {iou:.4f}")
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(range(len(detections)))
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.match_thresh:
                matches.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_detections.remove(c)
        # Always log assignment details
        logger.debug(f"[BYTETracker] cost_matrix=\n{cost_matrix}")
        logger.debug(f"[BYTETracker] assignments={list(zip(row_ind, col_ind))}")
        logger.debug(f"[BYTETracker] filtered matches={matches}")
        logger.debug(f"[BYTETracker] unmatched_tracks={unmatched_tracks}, unmatched_detections={unmatched_detections}")
        return matches, unmatched_tracks, unmatched_detections

    def _tlwh_to_xyxy(self, tlwh):
        x, y, w, h = tlwh
        return [x, y, x + w, y + h]

    def _iou(self, tlwh1, tlwh2):
        box1 = self._tlwh_to_xyxy(tlwh1)
        box2 = self._tlwh_to_xyxy(tlwh2)
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2
        xa = max(x1, x1g)
        ya = max(y1, y1g)
        xb = min(x2, x2g)
        yb = min(y2, y2g)
        inter = max(0, xb - xa) * max(0, yb - ya)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

def _test_iou():
    tracker = BYTETracker()
    # Identical boxes
    box1 = [10, 10, 20, 20]  # x, y, w, h
    box2 = [10, 10, 20, 20]
    iou = tracker._iou(box1, box2)
    print(f"IoU identical: {iou:.4f} (expected 1.0)")

    # Partial overlap
    box3 = [10, 10, 20, 20]
    box4 = [20, 20, 20, 20]  # Overlaps at (20,20)-(30,30)
    iou = tracker._iou(box3, box4)
    print(f"IoU partial: {iou:.4f} (expected ~0.1429)")

    # No overlap
    box5 = [0, 0, 10, 10]
    box6 = [20, 20, 10, 10]
    iou = tracker._iou(box5, box6)
    print(f"IoU none: {iou:.4f} (expected 0.0)")

if __name__ == "__main__":
    _test_iou() 