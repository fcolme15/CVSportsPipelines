import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from ultralytics import YOLO
import time


@dataclass
class DetectObject:
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float] #x1, y1, x2, y2 (normalized 0-1)
    center_x: float #Normalized center x
    center_y: float #Normalized center y
    width: float #Normalized width
    height: float #Normalized height

@dataclass
class ObjectFrame:
    frame_index: int
    timestamp: float
    detected_objects: List[DetectObject]
    detection_confidence: float

class ObjectDetector:
    
    def __init__(self, model_size='n', confidence_threshold=0.25, sports_objects_only=True):
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.sports_objects_only = sports_objects_only

        #Sports relevant COCO class names
        self.sports_classes = {
            32: 'sports_ball',  #Generic sports ball
            0: 'person',        #Keep for reference/validation
            37: 'baseball_bat',
            38: 'baseball_glove',
            39: 'skateboard',
            40: 'surfboard',
            41: 'tennis_racket',
            42: 'bottle',       #Water bottles common in sports
            73: 'laptop',
            74: 'mouse',
            75: 'remote',
            76: 'keyboard',
            77: 'cell_phone'    #Common in sports videos
        }
        
        #Custom mapping
        self.enhanced_sports_mapping = {
            'sports_ball': ['soccer_ball', 'basketball', 'american_football', 'tennis_ball', 'volleyball'],
            'person': ['player', 'athlete', 'coach'],
            'bottle': ['water_bottle', 'sports_drink']
        }

        self.model = None
        self.class_names = []

    def initialize_model(self) -> bool:
        
        try:
            model_name = f"yolov8{self.model_size}.pt"
            print(f"Loading YOLO model: {model_name}...")
            
            self.model = YOLO(model_name)
            
            # Get class names - fix the logic here
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = self.model.names
                print(f"YOLO model loaded successfully")
                print(f"    Available classes: {len(self.class_names)}")
                print(f"    Sports classes: {len(self.sports_classes)}")
                return True
            else:
                # Fallback COCO class names
                self.class_names = {
                    0: 'person', 32: 'sports_ball', 37: 'baseball_bat',
                    38: 'baseball_glove', 39: 'skateboard', 40: 'surfboard',
                    41: 'tennis_racket', 42: 'bottle'
                }
                print(f"YOLO model loaded successfully (using fallback class names)")
                print(f"    Available classes: {len(self.class_names)}")
                print(f"    Sports classes: {len(self.sports_classes)}")
                return True
                
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            self.model = None
            return False

    def process_frame(self, frame: np.ndarray, frame_index: int, timestamp: float) -> Optional[ObjectFrame]:

        if self.model is None:
            print('Model not initialized')
            return None
        
        try: 
            #Run YOLO detection
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)

            detected_objects = []
            total_confidence = 0.0

            #Process detections
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes

                    for i in range(len(boxes)):
                        #Get detection data
                        bbox = boxes.xyxyn[i].cpu().numpy() #Normalized coordinates
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())

                        #Get class names
                        class_name = self.class_names.get(class_id, f'class_{class_id}')

                        #Filter for sports objects is enabled
                        if self.sports_objects_only and class_id not in self.sports_classes:
                            continue

                        #Calculate the center and dimensions
                        x1, y1, x2, y2 = bbox
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1

                        detected_object = DetectObject(
                            class_name = class_name, 
                            confidence = confidence, 
                            bbox = (float(x1), float(y1), float(x2), float(y2)),
                            center_x = float(center_x), 
                            center_y = float(center_y),
                            width = float(width),
                            height = float(height)
                        )

                        detected_objects.append(detected_object)
                        total_confidence += confidence

            #Calculate overall detection confidence
            detected_confidence = total_confidence / len(detected_objects) if detected_objects else 0.0

            return ObjectFrame(
                frame_index = frame_index,
                timestamp = timestamp,
                detected_objects = detected_objects,
                detection_confidence = detected_confidence
            )
        
        except Exception as e: 
            print(f'Error processing frame {frame_index}: {e}')
            return None
        

    def enhance_sports_detection(self, object_frame: ObjectFrame, context_info: Dict = None) -> ObjectFrame:

        enhanced_objects = []

        for obj in object_frame.detected_objects: 
            enhanced_object = obj

            #Ball refinement
            if obj.class_name == 'sports_ball':
                #Use size and context to determine the ball type
                ball_area = obj.width * obj.height

                if context_info and 'sport_type' in context_info:
                    sport = context_info['sport_type'].lower()
                    if 'soccer' in sport:
                        enhanced_object.class_name = 'soccer_ball'
                    elif 'american_football' in sport or 'nfl' in sport:
                        enhanced_object.class_name = 'american_football'
                    elif 'basketball' in sport:
                        enhanced_object.class_name = 'basketball'
                    elif 'tennis' in sport:
                        enhanced_object.class_name = 'tennis_ball'
                
                #Size based heuristics
                elif ball_area > 0.01: #Large
                    enhanced_object.class_name = 'basketball'
                elif ball_area > 0.008: #Large oval
                    enhanced_object.class_name = 'american_football'
                elif ball_area > 0.005: #Medium
                    enhanced_object.class_name = 'soccer_ball'
                else: #Small
                    enhanced_object.class_name = 'tennis_ball'

            enhanced_objects.append(enhanced_object)

        return ObjectFrame(
            frame_index = object_frame.frame_index,
            timestamp = object_frame.timestamp,
            detected_objects = enhanced_objects,
            detection_confidence = object_frame.detection_confidence
        )
    
    def track_objects_across_frames(self, object_frames: List[ObjectFrame]) -> List[ObjectFrame]:

        if len(object_frames) < 2: 
            return object_frames
        
        tracked_frames = []

        for i, current_frame in enumerate(object_frames):
            #Ignore the current frame
            if i == 0:
                tracked_frames.append(current_frame)
                continue

            previous_frame = tracked_frames[i-1]
            tracked_objects = []

            for current_obj in current_frame.detected_objects:
                best_match = None
                best_distance = float('inf')

                #Find closest object of same class in previous frame
                for prev_obj in previous_frame.detected_objects:
                    if prev_obj.class_name == current_obj.class_name:
                        #Calcualte distance between centers
                        distance = np.sqrt(
                            (current_obj.center_x - prev_obj.center_x) ** 2 + (current_obj.center_y - prev_obj.center_y) ** 2
                        )

                        if distance < best_distance and distance < 0.1: #Max tracking distance
                            best_distance = distance
                            best_match = prev_obj
                        
                #Use tracking to smooth the confidence if object was tracked
                if best_match:
                    #Smooth confidence with previous frame
                    smoothed_confidence = (current_obj.confidence + best_match.confidence) / 2
                    current_obj.confidence = smoothed_confidence
                
                tracked_objects.append(current_obj )

            tracked_frame = ObjectFrame(
                frame_index = current_frame.frame_index,
                timestamp = current_frame.timestamp,
                detected_objects = tracked_objects,
                detection_confidence = current_frame.detection_confidence
            )

            tracked_frames.append(tracked_frame)

        return tracked_frames
    
    def get_detection_statistics(self, object_frames: List[ObjectFrame]) -> Dict[str, Any]:

        if not object_frames:
            return {
                "error": "No object frames provided",
                "total_frames": 0,
                "frames_with_detections": 0,
                "object_counts": {},
                "average_confidence": {},
                "detection_rate": {}
            }
        
        stats = {
            'total_frames': len(object_frames),
            'frames_with_detections': 0,
            'object_counts': {},
            'average_confidence': {},
            'detection_rate': {},
            'temporal_consistency': {}
        }

        all_objects = []
        frames_with_objects = 0

        for frame in object_frames:
            if frame.detected_objects:
                frames_with_objects += 1
            
            for obj in frame.detected_objects:
                all_objects.append(obj)

                #Count objects by class
                if obj.class_name not in stats['object_counts']:
                    stats['object_counts'][obj.class_name] = 0
                stats['object_counts'][obj.class_name] += 1

                #Track confidence by class
                if obj.class_name not in stats['average_confidence']:
                    stats['average_confidence'][obj.class_name].append(obj.confidence)
                stats['average_confidence'][obj.class_name].append(obj.confidence)
        
        stats['frames_with_detections'] = frames_with_objects

        #Calculate averages
        for class_name, confidences in stats['average_confidence'].items():
            stats['average_confidence'][class_name] = np.mean(confidences)
            stats['detection_rate'][class_name] = len(confidences) / len(object_frames)

        return stats
    
    def cleanup(self):

        if self.model:
            del self.model
            self.model = None




