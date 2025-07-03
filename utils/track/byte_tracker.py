# Pure Python ByteTrack implementation (adapted from Yolov5_DeepSort_Pytorch)
# Uses scipy.optimize.linear_sum_assignment for assignment (no lap required)
# MIT License

import numpy as np
from scipy.optimize import linear_sum_assignment

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

    def activate(self, frame_id, track_id):
        self.is_activated = True
        self.track_id = track_id
        self.start_frame = frame_id
        self.frame_id = frame_id

    def update(self, tlwh, score, frame_id):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = float(score)
        self.frame_id = frame_id

    def to_xyxy(self):
        x, y, w, h = self.tlwh
        return [x, y, x + w, y + h]

class BYTETracker:
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
            if score >= self.track_thresh:
                det_stracks.append(STrack([x1, y1, w, h], score, cls))

        # Association (simple IoU + Hungarian)
        matches, u_track, u_detection = self._associate(self.tracked_stracks, det_stracks)

        # Update matched tracks
        for t_idx, d_idx in matches:
            track = self.tracked_stracks[t_idx]
            det = det_stracks[d_idx]
            track.update(det.tlwh, det.score, self.frame_id)
            activated_stracks.append(track)

        # Activate new tracks
        for idx in u_detection:
            det = det_stracks[idx]
            det.activate(self.frame_id, self.next_id)
            self.next_id += 1
            activated_stracks.append(det)

        self.tracked_stracks = activated_stracks

        # Prepare output
        output_tracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                continue
            output_tracks.append({
                'bbox': track.to_xyxy(),
                'track_id': track.track_id,
                'original_class_id': track.cls,
                'confidence': track.score
            })
        return output_tracks

    def _associate(self, tracks, detections):
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(detections):
                cost_matrix[i, j] = 1 - self._iou(t.tlwh, d.tlwh)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches, u_track, u_detection = [], [], []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < (1 - self.match_thresh):
                matches.append((r, c))
            else:
                u_track.append(r)
                u_detection.append(c)
        u_track += [i for i in range(len(tracks)) if i not in row_ind]
        u_detection += [i for i in range(len(detections)) if i not in col_ind]
        return matches, u_track, u_detection

    def _iou(self, tlwh1, tlwh2):
        x1, y1, w1, h1 = tlwh1
        x2, y2, w2, h2 = tlwh2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        inter = max(0, xb - xa) * max(0, yb - ya)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0 