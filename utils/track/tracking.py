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


# Example usage and configuration
def create_custom_mapper() -> BaseClassMapper:
    """
    Create a custom base class mapper for your specific use case.
    Modify this function to match your class hierarchy.
    """
    mapper = BaseClassMapper()
    
    # Remove the complex example function since we have a simpler YOLO-specific case
def simple_yolo_integration_example():
    """
    Simple example showing how to integrate with your YOLO model.
    """
    # Initialize components
    mapper = BaseClassMapper()
    detection_processor = DetectionProcessor(mapper)
    
    # Your YOLO model
    # model = YOLO('your_custom_model.pt')
    
    # In your video processing loop:
    # results = model(frame)
    # boxes = results[0].boxes
    # 
    # if boxes is not None:
    #     bboxes = boxes.xyxy.cpu().numpy()
    #     confidences = boxes.conf.cpu().numpy()  
    #     class_ids = boxes.cls.cpu().numpy()
    #     
    #     # Process with base class mapping
    #     detections = detection_processor.process_detections(bboxes, confidences, class_ids)
    #     
    #     # Now classes 0 and 1 will both be tracked as class 0
    #     # Class 2 will be tracked as class 2
    #     
    #     # Update your ByteTrack tracker
    #     # tracks = tracker_wrapper.update(detections)
    
    pass
    
    return mapper


# Example integration with YOLO and ByteTrack
def setup_yolo_tracking_pipeline():
    """
    Setup for YOLO + ByteTrack pipeline with base class mapping.
    """
    # Create base class mapper
    base_mapper = BaseClassMapper()
    
    # Initialize your ByteTrack instance
    # bytetrack = BYTETracker(frame_rate=30, track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    
    # Create wrapper
    # tracker_wrapper = ByteTrackWrapper(bytetrack, base_mapper)
    
    # Create detection processor
    detection_processor = DetectionProcessor(base_mapper)
    
    return base_mapper, detection_processor  # , tracker_wrapper


# Example of how to use with YOLO detections
def yolo_tracking_example():
    """
    Example of how to integrate this with YOLO detections.
    """
    import torch
    from ultralytics import YOLO
    
    # Load your custom YOLO model
    model = YOLO('your_custom_model.pt')
    
    # Setup tracking pipeline
    base_mapper, detection_processor = setup_yolo_tracking_pipeline()
    
    # Example frame processing
    def process_frame(frame):
        # Get YOLO detections
        results = model(frame)
        
        # Extract detection data
        boxes = results[0].boxes
        if boxes is not None:
            # Get bounding boxes, confidences, and class IDs
            bboxes = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            
            # Process detections with base class mapping
            detections = detection_processor.process_detections(bboxes, confidences, class_ids)
            
            # Now detections have:
            # - class_id: mapped to base class (0 for both YOLO classes 0&1, 2 for YOLO class 2)
            # - original_class_id: original YOLO class (0, 1, or 2)
            
            # Update tracker
            # tracks = tracker_wrapper.update(detections)
            
            # Print example of mapping
            for det in detections:
                print(f"Original YOLO class: {det['original_class_id']} -> Tracking class: {det['class_id']}")
            
            return detections
        
        return []
    
    return process_frame


# Quick usage example for your YOLO classes 0, 1, 2
if __name__ == "__main__":
    # Test the base class mapping for your specific case
    mapper = BaseClassMapper()
    
    # Test mapping functionality
    print("Testing YOLO class mapping:")
    print(f"YOLO class 0 -> tracking class: {mapper.get_base_class(0)}")
    print(f"YOLO class 1 -> tracking class: {mapper.get_base_class(1)}")
    print(f"YOLO class 2 -> tracking class: {mapper.get_base_class(2)}")
    
    print(f"\nSame base class check:")
    print(f"YOLO class 0 and 1 (should be True): {mapper.is_same_base_class(0, 1)}")
    print(f"YOLO class 0 and 2 (should be False): {mapper.is_same_base_class(0, 2)}")
    print(f"YOLO class 1 and 2 (should be False): {mapper.is_same_base_class(1, 2)}")
    
    # Example detection processing
    print(f"\nExample detection processing:")
    detection_processor = DetectionProcessor(mapper)
    
    # Simulate YOLO detections
    import numpy as np
    bboxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400], [500, 500, 600, 600]])
    confidences = np.array([0.9, 0.8, 0.7])
    class_ids = np.array([0, 1, 2])  # Your YOLO classes
    
    detections = detection_processor.process_detections(bboxes, confidences, class_ids)
    
    for i, det in enumerate(detections):
        print(f"Detection {i+1}: YOLO class {det['original_class_id']} -> tracking class {det['class_id']}")