'''
2D Pose Detector Component
MediaPipe pose detection for the unified detector.
'''

import mediapipe as mp
import numpy as np
from typing import Optional, List, Dict, Any
import logging

from unified_detector import Pose2D, Keypoint2D

logger = logging.getLogger(__name__)

#2D Pose detection using MediaPipe.
class PoseDetector2D:
    #MediaPipe keypoint names (33 keypoints)
    KEYPOINT_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]
    
    '''
    Key joints for confidence calculation
    11 → left_shoulder, 12 → right_shoulder, 13 → left_elbow, 14 → right_elbow
    15 → left_wrist, 16 → right_wrist, 23 → left_hip, 24 → right_hip
    25 → left_knee, 26 → right_knee, 27 → left_ankle, 28 → right_ankle
    '''
    KEY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    
    def __init__(self,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 verbose: bool = False):
       
        self.min_detection_confidence = min_detection_confidence #
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity #0, 1, or 2 (higher = more accurate but slower)
        self.smooth_landmarks = smooth_landmarks
        self.verbose = verbose #Enable verbose logging

        #Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        #Statistics
        self.frames_processed = 0
        self.poses_detected = 0
        
        logger.info(f"PoseDetector2D initialized (complexity={model_complexity})")
    
    #Detect pose in a single frame. Returns Pose2D object or None if no pose detected
    def detect(self, 
               frame_rgb: np.ndarray,
               frame_index: int,
               timestamp: float) -> Optional[Pose2D]:
        
        self.frames_processed += 1
        
        try:
            results = self.pose.process(frame_rgb)
            
            if not results.pose_landmarks:
                return None
            
            #Extract keypoints
            keypoints = []
            landmarks = results.pose_landmarks.landmark
            
            for idx, landmark in enumerate(landmarks):
                keypoint = Keypoint2D(
                    x=landmark.x,
                    y=landmark.y,
                    confidence=landmark.visibility,
                    name=self.KEYPOINT_NAMES[idx] if idx < len(self.KEYPOINT_NAMES) else f"keypoint_{idx}"
                )
                keypoints.append(keypoint)
            
            #Calculate overall confidence using key joints
            #TODO: add weights on some joints dependent on sports value
            key_confidences = [keypoints[i].confidence for i in self.KEY_JOINTS if i < len(keypoints)]
            detection_confidence = np.mean(key_confidences) if key_confidences else 0.0
            
            self.poses_detected += 1
            
            return Pose2D(
                keypoints=keypoints,
                detection_confidence=detection_confidence
            )
            
        except Exception as e:
            if self.verbose:
                logger.error(f"Pose detection error at frame {frame_index}: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'frames_processed': self.frames_processed,
            'poses_detected': self.poses_detected,
            'detection_rate': self.poses_detected / self.frames_processed if self.frames_processed > 0 else 0,
            'model_complexity': self.model_complexity
        }
    
    def cleanup(self):
        if self.pose:
            self.pose.close()
        logger.info(f"PoseDetector2D cleaned up. Detected {self.poses_detected}/{self.frames_processed} poses")