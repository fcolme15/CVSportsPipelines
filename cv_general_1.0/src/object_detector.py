import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from ultralytics import YOLO
import time


@dataclass
class DetectedObject:
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
    detected_objects: List[DetectedObject]
    detection_confidence: float

#YOLO object detection for sports equipment
class ObjectDetector:

    def __init__(self, 
                 model_size='n',  # 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
                 confidence_threshold=0.25,
                 sports_objects_only=True,
                 light_smoothing=True): #Light temporal smoothing
        
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.sports_objects_only = sports_objects_only
        self.light_smoothing = light_smoothing
        
        # Sports-relevant COCO class names
        self.sports_classes = {
            32: 'sports_ball', #Generic sports ball
            0: 'person', #For reference/validation
            37: 'baseball_bat',
            38: 'baseball_glove',
            39: 'skateboard',
            40: 'surfboard',
            41: 'tennis_racket',
            42: 'bottle', #Water bottles common in sports
            73: 'laptop', #For analysis/coaching setups
            74: 'mouse',
            75: 'remote',
            76: 'keyboard',
            77: 'cell_phone' #Common in sports videos
        }
        
        #Enhanced sports detection
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
            
            #Get class names
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            else:
                #Fallback COCO class names
                self.class_names = {
                    0: 'person', 32: 'sports_ball', 37: 'baseball_bat',
                    38: 'baseball_glove', 39: 'skateboard', 40: 'surfboard',
                    41: 'tennis_racket', 42: 'bottle'
                }
            
            print(f"YOLO model loaded successfully")
            print(f"    Available classes: {len(self.class_names)}")
            print(f"    Sports classes: {len(self.sports_classes)}")
            
            return True
            
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            return False
    
    #Detect objects in a single frame
    def process_frame(self, frame: np.ndarray, frame_index: int, timestamp: float, context_info: dict) -> Optional[ObjectFrame]:
        
        if self.model is None:
            print("Model not initialized. Call initialize_model() first.")
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
                        #boxes.xyxyn - Normalized bounding boxes 
                        #boxes.conf - Confidence scores
                        #boxes.cls - Class IDs
                        #boxes.xyxy - Pixel bounding boxes (alternative)
                        #.cpu() -> Convert from GPU to CPU memory, .numpyy() -> convert to regular Python number
                        bbox = boxes.xyxyn[i].cpu().numpy() #Normalized coordinates
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        #Get class name
                        class_name = self.class_names.get(class_id, f"class_{class_id}")
                        
                        #Filter for sports objects if enabled
                        if self.sports_objects_only and class_id not in self.sports_classes:
                            continue
                        
                        #Calculate center and dimensions
                        x1, y1, x2, y2 = bbox
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1

                        if class_name == 'sports_ball':
                            class_name = self._enhance_sports_detection(width, height, context_info)

                        detected_obj = DetectedObject(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            center_x=float(center_x),
                            center_y=float(center_y),
                            width=float(width),
                            height=float(height)
                        )
                        
                        detected_objects.append(detected_obj)
                        total_confidence += confidence
            
            #Calculate overall detection confidence
            detection_confidence = total_confidence / len(detected_objects) if detected_objects else 0.0
            
            return ObjectFrame(
                frame_index=frame_index,
                timestamp=timestamp,
                detected_objects=detected_objects,
                detection_confidence=detection_confidence
            )
            
        except Exception as e:
            print(f"Error processing frame {frame_index}: {e}")
            return None
    
    #Enhance the detection with sports specific logic
    def _enhance_sports_detection(self, width:float, height:float, context_info: dict) -> str:
        
        #Use size and context to determine specific ball type
        ball_area = width * height
        
        if context_info and 'sport_type' in context_info:
            sport = context_info['sport_type'].lower()
            if 'soccer' in sport:
                return 'soccer_ball'
            elif 'basketball' in sport:
                return 'basketball'
            elif 'american_football' in sport or 'nfl' in sport:
                return 'american_football'
            elif 'tennis' in sport:
                return 'tennis_ball'
        
        #Size-based heuristics
        elif ball_area > 0.01: #Large ball
            return 'basketball'
        elif ball_area > 0.008: #Large oval ball
            return 'american_football'  
        elif ball_area > 0.005: #Medium ball  
            return 'soccer_ball'
        else: #Small ball
            return 'tennis_ball'
            
    #Object tracking across frames applying temporal smoothing to positions and confidence values
    def track_and_smooth_objects_across_frames(self, object_frames: List[ObjectFrame]) -> List[ObjectFrame]:
        
        if len(object_frames) < 2:
            return object_frames
        
        print(f"Tracking and smoothing objects across {len(object_frames)} frames.")
        
        #Build object tracks by class name
        object_tracks = {}
        for frame_idx, frame in enumerate(object_frames):
            for obj in frame.detected_objects:
                if obj.class_name not in object_tracks:
                    object_tracks[obj.class_name] = []
                #Append Every frame pertaining to the class
                object_tracks[obj.class_name].append((frame_idx, obj))
        
        #Apply smoothing to each track
        smoothing_window = 3
        for class_name, track in object_tracks.items():
            if len(track) < smoothing_window:
                continue #Need minimum frames for smoothing
            
            #Extract data arrays for smoothing
            positions = []
            sizes = []
            confidences = []
            
            for frame_idx, obj in track:
                positions.append([obj.center_x, obj.center_y])
                sizes.append([obj.width, obj.height])
                confidences.append(obj.confidence)
            
            positions = np.array(positions)
            sizes = np.array(sizes)
            confidences = np.array(confidences)
            
            #Apply smoothing
            smoothed_positions = self._smooth_array(positions, smoothing_window)
            smoothed_sizes = self._smooth_array(sizes, smoothing_window)
            smoothed_confidences = self._smooth_array(confidences.reshape(-1, 1), smoothing_window).flatten()
            
            #Update objects with smoothed values
            for i, (frame_idx, obj) in enumerate(track):
                if i < len(smoothed_positions):
                    obj.center_x = float(smoothed_positions[i][0])
                    obj.center_y = float(smoothed_positions[i][1])
                    obj.width = float(smoothed_sizes[i][0])
                    obj.height = float(smoothed_sizes[i][1])
                    obj.confidence = float(smoothed_confidences[i])
                    
                    #Recalculate bounding box from smoothed center and size
                    x1 = obj.center_x - obj.width / 2
                    y1 = obj.center_y - obj.height / 2
                    x2 = obj.center_x + obj.width / 2
                    y2 = obj.center_y + obj.height / 2
                    obj.bbox = (x1, y1, x2, y2)

        return object_frames
    
    #Apply simple moving average smoothing to array data
    def _smooth_array(self, data: np.ndarray, window_size: int) -> np.ndarray: 
        
        if len(data) < window_size:
            return data
        
        smoothed = np.zeros_like(data)
        half_window = window_size // 2
        
        for i in range(len(data)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(data), i + half_window + 1)
            smoothed[i] = np.mean(data[start_idx:end_idx], axis=0)
        
        return smoothed
    
    #Get statistics about object detection across the frames
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
            "total_frames": len(object_frames),
            "frames_with_detections": 0,
            "object_counts": {},
            "average_confidence": {},
            "detection_rate": {},
            "temporal_consistency": {}
        }
        
        all_objects = []
        frames_with_objects = 0
        
        for frame in object_frames:
            if frame.detected_objects:
                frames_with_objects += 1
            
            for obj in frame.detected_objects:
                all_objects.append(obj)
                
                #Count objects by class
                if obj.class_name not in stats["object_counts"]:
                    stats["object_counts"][obj.class_name] = 0
                stats["object_counts"][obj.class_name] += 1
                
                #Track confidence by class
                if obj.class_name not in stats["average_confidence"]:
                    stats["average_confidence"][obj.class_name] = []
                stats["average_confidence"][obj.class_name].append(obj.confidence)
        
        stats["frames_with_detections"] = frames_with_objects
        
        #Calculate averages
        for class_name, confidences in stats["average_confidence"].items():
            stats["average_confidence"][class_name] = np.mean(confidences)
            stats["detection_rate"][class_name] = len(confidences) / len(object_frames)
        
        return stats
    
    def cleanup(self):
        if self.model:
            del self.model
            self.model = None
        print("ObjectDetector cleaned up")