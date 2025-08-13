'''
Combines pose detection (MediaPipe) and object detection (YOLO) in a single pass
'''

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import time
import logging
from pose_detector_2d import PoseDetector2D
from object_detector_2d import ObjectDetector2D

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Single 2D keypoint with confidence
@dataclass
class Keypoint2D:
    x: float #Normalized x coordinate (0-1)
    y: float #Normalized y coordinate (0-1)
    confidence: float #Visibility/confidence (0-1)
    name: str #Keypoint name

#2D pose detection result for a single frame
@dataclass
class Pose2D:
    keypoints: List[Keypoint2D]
    detection_confidence: float #Overall pose detection confidence
    
    def get_keypoint_by_name(self, name: str) -> Optional[Keypoint2D]:
        for kp in self.keypoints:
            if kp.name == name:
                return kp
        return None
    
    #Keypoint by MediaPipe index
    def get_keypoint_by_index(self, index: int) -> Optional[Keypoint2D]:
        if 0 <= index < len(self.keypoints):
            return self.keypoints[index]
        return None

#2D object detection result
@dataclass
class Object2D:
    class_name: str
    bbox: Tuple[float, float, float, float] #(x1, y1, x2, y2) normalized
    center: Tuple[float, float] #(cx, cy) normalized
    confidence: float
    object_id: Optional[int] = None #Assigned by tracker
    
    #Normalized width
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    #Normalized height
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]
    
    #Normalized area
    @property
    def area(self) -> float:
        return self.width * self.height

#Final combined frame
@dataclass
class Unified2DFrame:
    frame_index: int
    timestamp: float
    pose_2d: Optional[Pose2D]
    objects_2d: List[Object2D]
    frame_confidence: float #Combined confidence score
    processing_time: float = 0.0 #Time taken to process this frame
    
    #Validate if frame has detections
    def has_detections(self) -> bool:
        return self.pose_2d is not None or len(self.objects_2d) > 0
    
    def get_detection_summary(self) -> Dict[str, Any]:
        return {
            'has_pose': self.pose_2d is not None,
            'pose_confidence': self.pose_2d.detection_confidence if self.pose_2d else 0.0,
            'num_objects': len(self.objects_2d),
            'object_classes': list(set(obj.class_name for obj in self.objects_2d)),
            'frame_confidence': self.frame_confidence
        }

class UnifiedDetector:
    
    def __init__(self,
                 pose_confidence: float = 0.5, #Minimum confidence for pose detection
                 pose_tracking_confidence: float = 0.5, #Minimum confidence for pose tracking
                 object_confidence: float = 0.25, #Minimum confidence for object detection
                 enable_yolo: bool = True,
                 yolo_model_size: str = 'n', #YOLO model size ('n', 's', 'm', 'l', 'x')
                 sports_objects_only: bool = True,
                 verbose: bool = False): #Enable verbose logging
        
        self.enable_yolo = enable_yolo
        self.verbose = verbose
        
        #Initialize detectors using composition
        self.pose_detector = PoseDetector2D(
            min_detection_confidence=pose_confidence,
            min_tracking_confidence=pose_tracking_confidence,
            verbose=verbose
        )
        
        self.object_detector = None
        if enable_yolo:
            self.object_detector = ObjectDetector2D(
                model_size=yolo_model_size,
                confidence_threshold=object_confidence,
                sports_objects_only=sports_objects_only,
                verbose=verbose
            )
        
        #Statistics
        self.stats = {
            'frames_processed': 0,
            'poses_detected': 0,
            'objects_detected': 0,
            'total_processing_time': 0.0
        }
    

    def process_frame(self, 
                     frame: np.ndarray, #BGR input frame
                     frame_index: int, #Frame index of the video
                     timestamp: float,
                     context: Optional[Dict[str, Any]] = None #Extra context about the sport analized for YOLO
                     ) -> Unified2DFrame:
        
        start_time = time.time()
        
        #Convert from BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #Detect pose (MediaPipe)
        pose_2d = self.pose_detector.detect(frame_rgb, frame_index, timestamp)
        
        #Detect objects (YOLO)
        objects_2d = []
        if self.object_detector:
            objects_2d = self.object_detector.detect(frame, frame_index, timestamp, context)
        
        #Calculate combined confidence
        frame_confidence = self._calculate_frame_confidence(pose_2d, objects_2d)
        
        #Update statistics
        self.stats['frames_processed'] += 1
        if pose_2d:
            self.stats['poses_detected'] += 1
        self.stats['objects_detected'] += len(objects_2d)
        
        processing_time = time.time() - start_time
        self.stats['total_processing_time'] += processing_time
        
        if self.verbose and frame_index % 30 == 0:
            logger.info(f"Frame {frame_index}: Pose={pose_2d is not None}, "
                       f"Objects={len(objects_2d)}, Time={processing_time:.3f}s")
        
        return Unified2DFrame(
            frame_index=frame_index,
            timestamp=timestamp,
            pose_2d=pose_2d,
            objects_2d=objects_2d,
            frame_confidence=frame_confidence,
            processing_time=processing_time
        )
    
    #Calculate combined confidence score for the frame
    def _calculate_frame_confidence(self, 
                                   pose: Optional[Pose2D], 
                                   objects: List[Object2D]) -> float:
        
        confidences = []
        
        if pose:
            confidences.append(pose.detection_confidence * 1.5)  #Weight pose higher
        
        for obj in objects:
            confidences.append(obj.confidence)
        
        if not confidences:
            return 0.0
        
        return min(1.0, np.mean(confidences))
    
    
    def process_batch(self, 
                     frames: List[np.ndarray],
                     start_index: int,
                     fps: float) -> List[Unified2DFrame]:
        results = []
        
        for i, frame in enumerate(frames):
            frame_index = start_index + i
            timestamp = frame_index / fps
            result = self.process_frame(frame, frame_index, timestamp)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        avg_time = (self.stats['total_processing_time'] / self.stats['frames_processed'] if self.stats['frames_processed'] > 0 else 0)
        
        stats = {
            'frames_processed': self.stats['frames_processed'],
            'poses_detected': self.stats['poses_detected'],
            'pose_detection_rate': (self.stats['poses_detected'] / self.stats['frames_processed'] 
                                   if self.stats['frames_processed'] > 0 else 0),
            'total_objects_detected': self.stats['objects_detected'],
            'avg_objects_per_frame': (self.stats['objects_detected'] / self.stats['frames_processed']
                                     if self.stats['frames_processed'] > 0 else 0),
            'total_processing_time': self.stats['total_processing_time'],
            'avg_processing_time': avg_time,
            'estimated_fps': 1.0 / avg_time if avg_time > 0 else 0
        }
        
        #Add pose and object stats as well
        stats['pose_detector_stats'] = self.pose_detector.get_statistics()
        if self.object_detector:
            stats['object_detector_stats'] = self.object_detector.get_statistics()
        
        return stats
    
    def cleanup(self):
        self.pose_detector.cleanup()
        if self.object_detector:
            self.object_detector.cleanup()
        
        logger.info("UnifiedDetector cleaned up")
        logger.info(f"Final statistics: {self.get_statistics()}")