import numpy as np
from typing import Dict, List, Any, Optional, Union

class BaseClassMapper:
    """
    Maps specific object-attribute classes to their base classes for tracking.
    This ensures ByteTrack only considers the base object type, not attributes.
    """
    
    def __init__(self):
        # Define your base class mapping for YOLO classes 0, 1, 2
        # Format: {yolo_class_id: base_class_for_tracking}
        self.class_mapping = {
            0: 0,   # YOLO class 0 -> base class 0 for tracking
            1: 0,   # YOLO class 1 -> base class 0 for tracking (same as class 0)
            2: 2,   # YOLO class 2 -> base class 2 for tracking (separate)
        }
        
        # Reverse mapping for convenience
        self.base_to_specific = {}
        for specific, base in self.class_mapping.items():
            if base not in self.base_to_specific:
                self.base_to_specific[base] = []
            self.base_to_specific[base].append(specific)
    
    def get_base_class(self, class_id: int) -> int:
        """Get the base class for a specific class ID."""
        return self.class_mapping.get(class_id, class_id)
    
    def get_specific_classes(self, base_class_id: int) -> List[int]:
        """Get all specific classes that map to a base class."""
        return self.base_to_specific.get(base_class_id, [base_class_id])
    
    def is_same_base_class(self, class_id1: int, class_id2: int) -> bool:
        """Check if two class IDs belong to the same base class."""
        return self.get_base_class(class_id1) == self.get_base_class(class_id2)


class ByteTrackWrapper:
    """
    Wrapper around ByteTrack that uses base class mapping for consistent tracking.
    """
    
    def __init__(self, bytetrack_instance, base_mapper: BaseClassMapper):
        self.bytetrack = bytetrack_instance
        self.base_mapper = base_mapper
        
    def preprocess_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess detections to use base classes for tracking.
        
        Args:
            detections: List of detection dictionaries with keys:
                       ['bbox', 'confidence', 'class_id', 'original_class_id']
        """
        processed_detections = []
        
        for det in detections:
            processed_det = det.copy()
            
            # Store original class ID for later reference
            processed_det['original_class_id'] = det['class_id']
            
            # Replace class_id with base class for tracking
            processed_det['class_id'] = self.base_mapper.get_base_class(det['class_id'])
            
            processed_detections.append(processed_det)
        
        return processed_detections
    
    def postprocess_tracks(self, tracks: List[Any]) -> List[Any]:
        """
        Postprocess tracks to restore original class information if needed.
        You can customize this based on your specific needs.
        """
        # This is where you might want to restore original class info
        # or add additional metadata to tracks
        return tracks
    
    def update(self, detections: List[Dict[str, Any]], *args, **kwargs):
        """
        Update tracker with preprocessed detections.
        """
        # Preprocess detections to use base classes
        processed_detections = self.preprocess_detections(detections)
        
        # Update ByteTrack with base class detections
        tracks = self.bytetrack.update(processed_detections, *args, **kwargs)
        
        # Postprocess if needed
        return self.postprocess_tracks(tracks)


class DetectionProcessor:
    """
    Helper class to process raw detections and prepare them for tracking.
    """
    
    def __init__(self, base_mapper: BaseClassMapper):
        self.base_mapper = base_mapper
    
    def process_detections(self, 
                          bboxes: np.ndarray, 
                          confidences: np.ndarray, 
                          class_ids: np.ndarray) -> List[Dict[str, Any]]:
        """
        Convert raw detection arrays to structured format.
        
        Args:
            bboxes: (N, 4) array of bounding boxes [x1, y1, x2, y2]
            confidences: (N,) array of confidence scores
            class_ids: (N,) array of class IDs
        """
        detections = []
        
        for i in range(len(bboxes)):
            detection = {
                'bbox': bboxes[i],
                'confidence': confidences[i],
                'class_id': int(class_ids[i]),
                'original_class_id': int(class_ids[i])
            }
            detections.append(detection)
        
        return detections

