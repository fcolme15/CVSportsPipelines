'''
2D Object Detector Component
YOLO-based object detection for the unified detector.
'''

from ultralytics import YOLO
import numpy as np
from typing import List, Optional, Dict, Any
import logging

from unified_detector import Object2D

logger = logging.getLogger(__name__)


class ObjectDetector2D:
    
    #Sports-relevant COCO classes
    SPORTS_CLASSES = {
        0: 'person',
        32: 'sports_ball',
        37: 'baseball_bat',
        38: 'baseball_glove',
        39: 'skateboard',
        40: 'surfboard',
        41: 'tennis_racket',
        42: 'bottle',
    }
    
    def __init__(self,
                 model_size: str = 'n', #YOLO model size ('n', 's', 'm', 'l', 'x')
                 confidence_threshold: float = 0.25, #Minimum detection confidence
                 sports_objects_only: bool = True, #Filter only sports-relevant objects
                 verbose: bool = False): #Enable verbose logging
        
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.sports_objects_only = sports_objects_only
        self.verbose = verbose
        
        #Initialize YOLO
        self.model = None
        self.class_names = {}
        self._init_yolo()
        
        #Statistics
        self.frames_processed = 0
        self.total_objects_detected = 0
        self.objects_by_class = {}
    
    def _init_yolo(self):
        try:
            model_name = f"yolov8{self.model_size}.pt"
            logger.info(f"Loading YOLO model: {model_name}")
            
            self.model = YOLO(model_name)
            
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            else:
                self.class_names = self.SPORTS_CLASSES
            
            logger.info(f"ObjectDetector2D initialized with {len(self.class_names)} classes")
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO: {e}")
            raise
    
    #Process and Detect objects in a single frame
    def detect(self,
               frame: np.ndarray, #Frame in BGR format
               frame_index: int,
               timestamp: float,
               context: Optional[Dict[str, Any]] = None #Sport type context
               ) -> List[Object2D]:
        
        self.frames_processed += 1
        
        if self.model is None:
            return []
        
        try:
            #Run YOLO detection
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            detected_objects = []
            
            for result in results: #One result for each frame given
                if result.boxes is None:
                    continue
                
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    #Get detection data
                    #boxes.xyxyn -> Normalized bounding boxes 
                    #boxes.conf -> Confidence scores
                    #boxes.cls -> Class IDs
                    #boxes.xyxy -> Pixel bounding boxes
                    #.cpu() -> Convert from GPU to CPU memory, .numpyy() -> convert to regular Python number
                    bbox = boxes.xyxyn[i].cpu().numpy() #Normalized coords
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    #Filter for sports objects
                    if self.sports_objects_only and class_id not in self.SPORTS_CLASSES:
                        continue
                    
                    #Get class name
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    #Refine sports ball classification
                    if class_name == 'sports_ball' and context:
                        class_name = self._refine_ball_classification(
                            bbox, context.get('sport_type', 'general'))
                    
                    #Calculate center
                    x1, y1, x2, y2 = bbox
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    obj = Object2D(
                        class_name=class_name,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        center=center,
                        confidence=confidence
                    )
                    
                    detected_objects.append(obj)
                    
                    # Update statistics
                    self.total_objects_detected += 1
                    if class_name not in self.objects_by_class:
                        self.objects_by_class[class_name] = 0
                    self.objects_by_class[class_name] += 1
            
            return detected_objects
            
        except Exception as e:
            if self.verbose:
                logger.error(f"Object detection error at frame {frame_index}: {e}")
            return []
    
    #Adjust class classification based on the given sports context
    def _refine_ball_classification(self, bbox: np.ndarray, sport_type: str) -> str:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        aspect_ratio = width / height if height > 0 else 1.0
        
        sport_type = sport_type.lower()
        
        #Sport-specific classification
        if 'soccer' in sport_type:
            return 'soccer_ball'
        elif 'basketball' in sport_type:
            return 'basketball'
        elif 'tennis' in sport_type:
            return 'tennis_ball'
        elif 'american_football' in sport_type and (aspect_ratio > 1.3 or aspect_ratio < 0.77):
            return 'american_football'
        
        #Size-based heuristics
        if area > 0.01:
            return 'basketball'
        elif area > 0.005:
            return 'soccer_ball'
        elif area < 0.002:
            return 'tennis_ball'
        
        return 'sports_ball'
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'frames_processed': self.frames_processed,
            'total_objects_detected': self.total_objects_detected,
            'avg_objects_per_frame': (self.total_objects_detected / self.frames_processed 
                                     if self.frames_processed > 0 else 0),
            'objects_by_class': dict(self.objects_by_class),
            'model_size': self.model_size
        }
    
    def cleanup(self):
        if self.model:
            del self.model
            self.model = None
            
        logger.info(f"ObjectDetector2D cleaned up. Detected {self.total_objects_detected} objects")