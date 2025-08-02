import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import gzip
import time
from src.coordinate_3d_generator import Pose3D, Keypoint3D

@dataclass
class ModelMetadata:
    drill_name: str
    duration_seconds: float
    fps: float
    total_frames: int
    world_scale: float
    creation_timestamp: float
    file_version: str = '1.0'
    coordinate_system: str = 'right_handed'

#Single bone connections in the skeleton
@dataclass 
class SkeletonBone:
    start_keypoint: int
    end_keypoint: int
    name: str
    bone_group: str #torso, left/right arm, left/right leg, face

#A single keyframe containing all joint positions
@dataclass 
class ModelKeyframe:
    timestamp: float
    frame_index: int
    keypoints: List[Dict[str, Any]]

#Complete 3D model structure for export
@dataclass
class Model3D: 
    metadata: ModelMetadata
    skeleton_structure: List[SkeletonBone]
    keyframes: List[ModelKeyframe]
    movement_analysis: Dict[str, Any]

class ModelCreator:
    def __init__(self, coordinate_precision=3, compress_output=True, optimize_for_web=True):
        self.coordinate_precision = coordinate_precision
        self.compress_output = compress_output
        self.optimize_for_web = optimize_for_web

        #Bone definitions with groupings for rendering control
        self.bone_definitions = [
            #Face connections
            SkeletonBone(0, 1, "nose_to_left_eye_inner", "face"),
            SkeletonBone(1, 2, "left_eye_inner_to_eye", "face"),
            SkeletonBone(2, 3, "left_eye_to_outer", "face"),
            SkeletonBone(0, 4, "nose_to_right_eye_inner", "face"),
            SkeletonBone(4, 5, "right_eye_inner_to_eye", "face"),
            SkeletonBone(5, 6, "right_eye_to_outer", "face"),
            SkeletonBone(2, 7, "left_eye_to_ear", "face"),
            SkeletonBone(5, 8, "right_eye_to_ear", "face"),
            SkeletonBone(9, 10, "mouth_connection", "face"),
            
            #Torso structure
            SkeletonBone(11, 12, "shoulder_line", "torso"),
            SkeletonBone(11, 23, "left_shoulder_to_hip", "torso"),
            SkeletonBone(12, 24, "right_shoulder_to_hip", "torso"),
            SkeletonBone(23, 24, "hip_line", "torso"),
            
            #Left arm
            SkeletonBone(11, 13, "left_shoulder_to_elbow", "left_arm"),
            SkeletonBone(13, 15, "left_elbow_to_wrist", "left_arm"),
            SkeletonBone(15, 17, "left_wrist_to_pinky", "left_arm"),
            SkeletonBone(15, 19, "left_wrist_to_index", "left_arm"),
            SkeletonBone(15, 21, "left_wrist_to_thumb", "left_arm"),
            
            #Right arm
            SkeletonBone(12, 14, "right_shoulder_to_elbow", "right_arm"),
            SkeletonBone(14, 16, "right_elbow_to_wrist", "right_arm"),
            SkeletonBone(16, 18, "right_wrist_to_pinky", "right_arm"),
            SkeletonBone(16, 20, "right_wrist_to_index", "right_arm"),
            SkeletonBone(16, 22, "right_wrist_to_thumb", "right_arm"),
            
            #Left leg
            SkeletonBone(23, 25, "left_hip_to_knee", "left_leg"),
            SkeletonBone(25, 27, "left_knee_to_ankle", "left_leg"),
            SkeletonBone(27, 29, "left_ankle_to_heel", "left_leg"),
            SkeletonBone(27, 31, "left_ankle_to_foot_index", "left_leg"),
            
            #Right leg
            SkeletonBone(24, 26, "right_hip_to_knee", "right_leg"),
            SkeletonBone(26, 28, "right_knee_to_ankle", "right_leg"),
            SkeletonBone(28, 30, "right_ankle_to_heel", "right_leg"),
            SkeletonBone(28, 32, "right_ankle_to_foot_index", "right_leg"),
        ]

    #Create complete 3D model from pose sequence
    def create_3d_model(self, poses_3d: List[Pose3D], drill_name:str = 'Unknown Drill'):
        if not poses_3d:
            raise ValueError(f'No 3D Model for {drill_name} with {len(poses_3d)} frames')
        
        print('Creating the 3d Model')

        #Extract the metadata 
        duration = poses_3d[-1].timestamp - poses_3d[0].timestamp
        fps = len(poses_3d) / duration if duration > 0 else 0

        metadata = ModelMetadata(
            drill_name=drill_name, 
            duration_seconds=duration,
            fps=fps,
            total_frames=len(poses_3d),
            world_scale=poses_3d[0].world_scale,
            creation_timestamp=time.time()
        )

        #Convert the poses to key frames
        keyframes = self._create_keyframes(poses_3d)

        #Calcualte movement analysis
        movement_analysis = self._analyze_movement_patterns(poses_3d)

        model_3d = Model3D (
            metadata=metadata,
            skeleton_structure=self.bone_definitions,
            keyframes=keyframes,
            movement_analysis=movement_analysis
        )

        print('Model created succesfully')
        print(f'    {len(keyframes)} keyframes')
        print(f'    {len(self.bone_definitions)} bone connections')
        print(f'    {duration:.2f} seconds duration')

        return model_3d
    
    #Convert the poses into individual optimized keyframes
    def _create_keyframes(self, poses_3d: List[Pose3D]) -> List[ModelKeyframe]:

        keyframes = []

        for pose in poses_3d:
            #Optimize keypoint data for the web
            optimized_keypoints = []

            for kp in pose.keypoints_3d:
                keypoint_data = {
                    'x': round(kp.x, self.coordinate_precision),
                    'y': round(kp.y, self.coordinate_precision),
                    'z': round(kp.z, self.coordinate_precision),
                    'c': round(kp.confidence, 2),
                    'n': kp.name if not self.optimize_for_web else None 
                }

                #Remove None values to reduce file size (if optimizing for web)
                if self.optimize_for_web:
                    keypoint_data = {k: v for k, v in keypoint_data.items() if v is not None}

                optimized_keypoints.append(keypoint_data)

            keyframe = ModelKeyframe(
                timestamp=round(pose.timestamp, 3),
                frame_index=pose.frame_index,
                keypoints=optimized_keypoints
            )

            keyframes.append(keyframe)

        return keyframes
    
    #Analyze movement patterns for metadata
    def _analyze_movement_patterns(self, poses_3d: List[Pose3D]) -> Dict[str, Any]:

        if len(poses_3d) < 2:
            return {'errpr': 'Insufficient frames for analysis'}

        #Key sports keypoints
        key_points = {
            "right_ankle": 28,
            "left_ankle": 27,
            "right_wrist": 16,
            "left_wrist": 15,
            "right_knee": 26,
            "left_knee": 25
        }

        movement_data = {}

        for name, idx in key_points.items():
            positions = []
            for pose in poses_3d:
                if idx < len(pose.keypoints_3d):
                    kp = pose.keypoints_3d[idx]
                    positions.append([kp.x, kp.y, kp.z])
                    
            if positions:
                positions = np.array(positions)

                movement_data[name] = {
                    "range_x": float(np.ptp(positions[:, 0])), 
                    "range_y": float(np.ptp(positions[:, 1])),
                    "range_z": float(np.ptp(positions[:, 2])),
                    "total_distance": float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))),
                    "avg_velocity": float(np.mean(np.linalg.norm(np.diff(positions, axis=0), axis=1))) if len(positions) > 1 else 0.0
                }

        #Overall movement summary
        all_positions = []
        for pose in poses_3d: 
            for kp in pose.keypoints_3d:
                all_positions.append([kp.x, kp.y, kp.z])

        if all_positions:
            all_positions = np.array(all_positions)
            movement_data['overall'] = {
                "bounding_box": {
                    "min_x": float(np.min(all_positions[:, 0])),
                    "max_x": float(np.max(all_positions[:, 0])),
                    "min_y": float(np.min(all_positions[:, 1])),
                    "max_y": float(np.max(all_positions[:, 1])),
                    "min_z": float(np.min(all_positions[:, 2])),
                    "max_z": float(np.max(all_positions[:, 2]))
                },
                "center_of_mass": {
                    "x": float(np.mean(all_positions[:, 0])),
                    "y": float(np.mean(all_positions[:, 1])),
                    "z": float(np.mean(all_positions[:, 2]))
                }
            }
        
        return movement_data
    
    #Export the 3D model to JSON with optimal compression
    def export_to_json(self, model_3d: Model3D, output_path: str) -> Dict[str, Any]:

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exists_ok=True)

        #Convert to dictionary
        model_dict = self._model_to_dict(model_3d)

        #Create JSON string
        json_string = json.dumps(model_dict, separators=(',', ':') if self.optimize_for_web else None)

        uncompressed_size = len(json_string.encode('utf-8'))

        if self.compress_output:
            #Create & save compressed version
            compressed_path = output_path.with_suffix(output_path.suffix + '.gz')
            with gzip.open(compressed_path, 'wt', encoding='utf-8') as f:
                f.write(json_string)

            compressed_size = compressed_path.stat().st_size

            print(f"Compressed model saved: {compressed_path}")
            print(f"    Original size: {uncompressed_size:,} bytes ({uncompressed_size/1024:.1f} KB)")
            print(f"    Compressed size: {compressed_size:,} bytes ({compressed_size/1024:.1f} KB)")
            print(f"    Compression ratio: {(1-compressed_size/uncompressed_size)*100:.1f}%")

            export_info = {
                'compressed_path': str(compressed_path), 
                'uncompressed_size': uncompressed_size,
                'compressed_size': compressed_size,
                'compression_ratio': (1-compressed_size/uncompressed_size)
            }

        else: 
            #Save the uncompressed version
            with open(output_path, 'w', econding='utf-8') as f:
                json.dump(model_dict, f, indent=2 if not self.optimize_for_web else None)
            
            print(f"Model saved: {output_path}")
            print(f"    File size: {uncompressed_size:,} bytes ({uncompressed_size/1024:.1f} KB)")

            export_info = {
                'output_path': str(output_path),
                'file_size': uncompressed_size
            }

        return export_info

    #Comvert the dateclass to dictionaries
    def _model_to_dict(self, model_3d: Model3D) -> Dict[str, Any]:

        model_dict = {
            "metadata": asdict(model_3d.metadata),
            "skeleton_structure": [asdict(bone) for bone in model_3d.skeleton_structure],
            "keyframes": [asdict(keyframe) for keyframe in model_3d.keyframes],
            "movement_analysis": model_3d.movement_analysis
        }
    
        return model_dict
    
    #Get detailed statistics about the 3D model
    def get_model_statistics(self, model_3d: Model3D) -> Dict[str, Any]:
        
        stats = {
            "basic_info": {
                "drill_name": model_3d.metadata.drill_name,
                "duration": model_3d.metadata.duration_seconds,
                "fps": model_3d.metadata.fps,
                "total_frames": model_3d.metadata.total_frames,
                "world_scale": model_3d.metadata.world_scale
            },
            "structure": {
                "total_keypoints_per_frame": len(model_3d.keyframes[0].keypoints) if model_3d.keyframes else 0,
                "total_bones": len(model_3d.skeleton_structure),
                "bone_groups": {}
            },
            "data_size": {
                "total_keyframes": len(model_3d.keyframes),
                "estimated_json_size_kb": len(json.dumps(self._model_to_dict(model_3d))) / 1024
            }
        }

        #Count bones in group
        for bone in model_3d.skeleton_structure:
            group = bone.bone_group
            if group not in stats['structure']['bone_groups']:
                stats['structure']['bone_groups'][group] = 0
            stats['structure']['bone_groups'][group]
        
        return stats
    

    







