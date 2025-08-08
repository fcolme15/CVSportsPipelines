import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from src.pose_detector import PoseFrame, PoseKeyPoint

#3D keypoint with enhanced depth info
@dataclass
class Keypoint3D:
    x:float
    y: float
    z:float
    confidence: float
    name: str

#Complete 3D pose for single frame
@dataclass
class Pose3D: 
    frame_index: int
    timestamp: float
    keypoints_3d: List[Keypoint3D]
    world_scale: float #Scale factor applied

class Coordinate3DGenerator:
    def __init__ (self, reference_height=1.75, depth_scale_factor=0.5, coordinate_system='right_handed'):
        self.reference_height = reference_height
        self.depth_scale_factor = depth_scale_factor
        self.coordinate_system = coordinate_system

        #MediaPipe keypoint names for reference
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

        #Key point measurements for scaling
        self.body_proportions = {
            'shoulder_width': 0.25, #25% of height
            'torso_length': 0.30, #30% of height  
            'upper_leg': 0.25, #25% of height
            'lower_leg': 0.25, #25% of height
            'arm_span': 1.0 #Approximately equal to height
        }

    #Convert the entire sequence of 2D poses to 3D
    def convert_pose_sequence_to_3d(self, pose_frames: List[PoseFrame]) -> List[Pose3D]:
        
        if not pose_frames:
            return []
        
        #Calculate world scale based on first frame
        world_scale = self._calculate_world_scale(pose_frames[0])
        print(f'    Calculated world scale: {world_scale} meters/unit')

        poses_3d = []

        for pose_frame in pose_frames:
            pose_3d = self._convert_single_pose_to_3d(pose_frame, world_scale)
            poses_3d.append(pose_3d)
        
        return poses_3d
        
    #Calculate scale factor to convert normalized coordinates to real-world meters
    #Use the normalized torso length to get  scaling factor
    def _calculate_world_scale(self, first_frame: PoseFrame) -> float:

        #Use shoulder to hip distance as a reference
        left_shoulder = first_frame.keypoints[11]
        right_shoulder = first_frame.keypoints[12]
        left_hip = first_frame.keypoints[23]
        right_hip = first_frame.keypoints[24]

        #Calculate torso length in normalized coordinates 
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        torso_length_normalized = abs(shoulder_center_y - hip_center_y)

        if torso_length_normalized > 0:
            #Expected torso length in meters
            expected_torso_meters = self.reference_height * self.body_proportions['torso_length']

            #Scale factor(meters per normalized unit)
            scale_factor = expected_torso_meters / torso_length_normalized

            return scale_factor

        #Fall back scale if <= 0
        return self.reference_height

    #Convert single 2D pose to 3D coordinates
    def _convert_single_pose_to_3d (self, pose_frame: PoseFrame, world_scale: float) -> Pose3D:
        keypoints_3d = []

        for i, keypoint_2d in enumerate(pose_frame.keypoints):
            #Convert normalized coordinates to world coordinates
            world_x = (keypoint_2d.x - 0.5) * world_scale #Center at origin
            world_y = (0.5 - keypoint_2d.y) * world_scale #Flip y(screen vs world coords)

            #Enhanced MediaPipe's relative Z with our scaling
            world_z = keypoint_2d.z * world_scale * self.depth_scale_factor

            #Apply biomechanical contraints for more realistic depth
            world_z = self._apply_depth_constraints(i, world_z, keypoints_3d)

            keypoint_3d = Keypoint3D(
                x=world_x,
                y=world_y,
                z=world_z,
                confidence=keypoint_2d.visibility,
                name= self.keypoint_names[i] if i < len(self.keypoint_names) else f'point_{i}'
            )

            keypoints_3d.append(keypoint_3d)
        
        return Pose3D(
            frame_index=pose_frame.frame_index,
            timestamp=pose_frame.timestamp,
            keypoints_3d=keypoints_3d,
            world_scale=world_scale
        )
    
    #Convert single 2D pose to 
    def _apply_depth_constraints(self, keypoint_index: int, z_coord: float, existing_keypoints: List[Keypoint3D]) -> float:
        #TODO
        #For now, just ensure reasonable depth variation
        #Future enhancement: add bone length constraints, joint angle limits

        #Limit extreme depth values
        max_depth = self.reference_height * 0.3 #30% of person height
        z_constrained = np.clip(z_coord, -max_depth, max_depth)

        return z_constrained
    
    #Get list of bone connections for skeleton visualization
    def get_bone_connections (self) -> List[Tuple[int, int]]:
        
        #Define skeleton structure as pairs of keypoint indices
        bones = [
            #Face
            (0, 1), (1, 2), (2, 3), #Nose to left eye
            (0, 4), (4, 5), (5, 6), #Nose to right eye
            (2, 7), (5, 8), #Eyes to ears
            (9, 10), #Mouth
        
            #Torso
            (11, 12), #Shoulders
            (11, 23), (12, 24), #Shoulders to hips
            (23, 24), #Hips
            
            #Left arm
            (11, 13), (13, 15), #Shoulder to elbow to wrist
            (15, 17), (15, 19), (15, 21), #Wrist to hand points
            
            #Right arm  
            (12, 14), (14, 16), #Shoulder to elbow to wrist
            (16, 18), (16, 20), (16, 22), #Wrist to hand points
            
            #Left leg
            (23, 25), (25, 27), #Hip to knee to ankle
            (27, 29), (27, 31), #Ankle to foot points
            
            #Right leg
            (24, 26), (26, 28), #Hip to knee to ankle  
            (28, 30), (28, 32), #Ankle to foot points
        ]
        
        return bones
    
    #Analyze 3D movement for quality metrics
    def analyze_movement_quality(self, poses_3d: List[Pose3D]) -> Dict:
        if len(poses_3d) < 2:
            return {'error': 'Need atleast 2 poses for analysis'}

        #Calculate movement metrics
        ankle_movement = self._analyze_keypoint_movement(poses_3d, 28, 'right_ankle')
        wrist_movement = self._analyze_keypoint_movement(poses_3d, 16, 'right_wrist')

        return {
            'total_frames': len(poses_3d),
            'duration_seconds': poses_3d[-1].timestamp - poses_3d[0].timestamp,
            'ankle_movement_range': ankle_movement, 
            'wrist_movement_range': wrist_movement,
            'world_scale': poses_3d[0].world_scale
        }
    
    #Analyze movement range for specific keypoint
    def _analyze_keypoint_movement(self, poses_3d: List[Pose3D], keypoint_idx: int, name: str) -> Dict:

        positions = []
        for pose in poses_3d:
            if keypoint_idx < len(pose.keypoints_3d):
                kp = pose.keypoints_3d[keypoint_idx]
                positions.append([kp.x, kp.y, kp.z])

        if not positions:
            return {'error': f'No data for {name}'}
        
        positions = np.array(positions)

        return {
            'name': name, 
            'x_range': float(np.max(positions[:, 0]) - np.min(positions[:, 0])),
            'y_range': float(np.max(positions[:, 1]) - np.min(positions[:, 1])),
            'z_range': float(np.max(positions[:, 2]) - np.min(positions[:, 2])),
            'total_distance': float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))),
        }

