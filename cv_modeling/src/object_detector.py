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
    def process_frame(self, frame: np.ndarray, frame_index: int, timestamp: float) -> Optional[ObjectFrame]:
        
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
                        bbox = boxes.xyxyn[i].cpu().numpy()  #Normalized coordinates
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
    def enhance_sports_detection(self, object_frame: ObjectFrame, context_info: Dict = None) -> ObjectFrame:
        
        enhanced_objects = []
        
        for obj in object_frame.detected_objects:
            enhanced_obj = obj
            
            #Sports ball refinement
            if obj.class_name == 'sports_ball':
                #Use size and context to determine specific ball type
                ball_area = obj.width * obj.height
                
                if context_info and 'sport_type' in context_info:
                    sport = context_info['sport_type'].lower()
                    if 'soccer' in sport:
                        enhanced_obj.class_name = 'soccer_ball'
                    elif 'basketball' in sport:
                        enhanced_obj.class_name = 'basketball'
                    elif 'american_football' in sport or 'nfl' in sport:
                        enhanced_obj.class_name = 'american_football'
                    elif 'tennis' in sport:
                        enhanced_obj.class_name = 'tennis_ball'
                
                #Size-based heuristics
                elif ball_area > 0.01: #Large ball
                    enhanced_obj.class_name = 'basketball'
                elif ball_area > 0.008: #Large oval ball
                    enhanced_obj.class_name = 'american_football'  
                elif ball_area > 0.005: #Medium ball  
                    enhanced_obj.class_name = 'soccer_ball'
                else: #Small ball
                    enhanced_obj.class_name = 'tennis_ball'
            
            enhanced_objects.append(enhanced_obj)
        
        return ObjectFrame(
            frame_index=object_frame.frame_index,
            timestamp=object_frame.timestamp,
            detected_objects=enhanced_objects,
            detection_confidence=object_frame.detection_confidence
        )
    
    #Simple object tracking across frames for consistency
    def track_objects_across_frames(self, object_frames: List[ObjectFrame]) -> List[ObjectFrame]:
        
        if len(object_frames) < 2:
            return object_frames
        
        print(f"Applying object tracking across {len(object_frames)} frames.")
        
        #Apply light smoothing first if enabled
        if self.light_smoothing and len(object_frames) >= 3:
            object_frames = self._apply_light_smoothing(object_frames)
        
        tracked_frames = []
        
        for i, current_frame in enumerate(object_frames):
            #Ignore the first frame from tracking
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
                        #Calculate distance between centers
                        distance = np.sqrt(
                            (current_obj.center_x - prev_obj.center_x)**2 + 
                            (current_obj.center_y - prev_obj.center_y)**2
                        )
                        
                        if distance < best_distance and distance < 0.1:  #Max tracking distance
                            best_distance = distance
                            best_match = prev_obj
                
                #Use tracking to smooth confidence if object was tracked
                if best_match:
                    #Smooth confidence with previous frame
                    smoothed_confidence = (current_obj.confidence + best_match.confidence) / 2
                    current_obj.confidence = smoothed_confidence
                
                tracked_objects.append(current_obj)
            
            tracked_frame = ObjectFrame(
                frame_index=current_frame.frame_index,
                timestamp=current_frame.timestamp,
                detected_objects=tracked_objects,
                detection_confidence=current_frame.detection_confidence
            )
            
            tracked_frames.append(tracked_frame)
        
        print(f"Object tracking complete")
        return tracked_frames
    
    #Apply light temporal smoothing to remove YOLO detection jitter
    def _apply_light_smoothing(self, object_frames: List[ObjectFrame]) -> List[ObjectFrame]:
        
        print(f"Applying light YOLO smoothing to {len(object_frames)} frames.")
        
        #Group objects by class name for individual smoothing
        object_tracks = {}
        
        #Build tracks for each object class
        for frame_idx, frame in enumerate(object_frames):
            for obj in frame.detected_objects:
                if obj.class_name not in object_tracks:
                    object_tracks[obj.class_name] = []
                object_tracks[obj.class_name].append((frame_idx, obj))
        
        #Apply light smoothing to each track
        smoothing_window = 3  #Small window for light smoothing
        
        for class_name, track in object_tracks.items():
            if len(track) < smoothing_window:
                continue  #Need minimum frames for smoothing
            
            #Extract positions and sizes
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
            
            #Apply light smoothing with small window
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
                    
                    #Recalculate bounding box from center and size
                    x1 = obj.center_x - obj.width / 2
                    y1 = obj.center_y - obj.height / 2
                    x2 = obj.center_x + obj.width / 2
                    y2 = obj.center_y + obj.height / 2
                    obj.bbox = (x1, y1, x2, y2)
        
        print(f"Light YOLO smoothing complete")
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
        print("🧹 ObjectDetector cleaned up")