import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

#Pose keypoint coords and confidence level
@dataclass
class PoseKeyPoint:
    x: float
    y: float
    z: float
    visibility: float
    presence: float

#All pose data for a frame
@dataclass 
class PoseFrame:
    frame_index: int
    timestamp: float
    keypoints: List[PoseKeyPoint]
    detection_confidence: float

class PoseDetector:
    def __init__ (self, min_detection_confidence=.7, min_tracking_confidence=.5):
        #Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity = 1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
            )
        
        #Keypoint names reference
        self.keypoint_names = [
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

        #Critical points indices
        self.sports_keypoints = {
            'wrists': [15, 16], #left, right
            'ankles': [27, 28], #left, right
            'arms': [11, 12, 13, 14, 15, 16], #shoulders, elbows, wrists
            'legs': [23, 24, 25, 26, 27, 28] #hips, knees, ankles
        }


    #Process a single frame and extract pose keypoints
    def process_frame(self, frame: np.ndarray, frame_index: int, timestamp: float) -> Optional[PoseFrame]:
        
        results = self.pose.process(frame)
        
        if not results.pose_landmarks:
            print(f"No pose detected in frame {frame_index}")
            return None
        
        #Get keypoints
        keypoints = []
        landmarks = results.pose_landmarks.landmark

        for i, landmark in enumerate(landmarks):
            keypoint = PoseKeyPoint(
                x = landmark.x,
                y = landmark.y,
                z = landmark.z,
                visibility = landmark.visibility,
                presence = getattr(landmark, 'presence', 1.0)
            )

            keypoints.append(keypoint)

        #Calculate the detection confidence(avg of key points)
        sports_confidence = self._calculate_sports_confidence(keypoints)

        return PoseFrame(
            frame_index = frame_index,
            timestamp = timestamp, 
            keypoints = keypoints,
            detection_confidence = sports_confidence
        )

    def _calculate_sports_confidence(self, keypoints: List[PoseKeyPoint]) -> float:

        critical_indices = (
            self.sports_keypoints['wrists'] +
            self.sports_keypoints['ankles'] + 
            self.sports_keypoints['arms']
        )

        confidence = []

        for idx in critical_indices:
            if idx < len(keypoints):
                confidence.append(keypoints[idx].visibility)

        return np.mean(confidence) if confidence else 0.0

    def get_keypoint_name(self, index: int) -> str:
        if 0 <= index < len(self.keypoint_names):
            return self.keypoint_names[index]
        return f'unknown index: {index}'
    
    def cleanup(self):
        if self.pose:
            self.pose.close()