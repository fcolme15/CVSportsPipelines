import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import time

from src.pose_detector import PoseFrame
from src.object_detector import ObjectFrame, DetectedObject
from src.coordinate_3d_generator import Pose3D, Keypoint3D

#3D object with world coordinates
@dataclass
class Object3D:
    class_name: str
    confidence: float
    center_x: float #World X coordinate
    center_y: float #World Y coordinate
    center_z: float #World Z coordinate (estimated)
    width: float #World width
    height: float #World height
    depth: float #World depth (estimated)
    bbox_2d: Tuple[float, float, float, float] #Original 2D bounding box

#Combined human pose and object data for a single frame
@dataclass
class FusedFrame:
    frame_index: int
    timestamp: float
    human_pose_3d: Optional[Pose3D]
    objects_3d: List[Object3D]
    fusion_confidence: float #Quality of pose-object alignment
    frame_quality: str #"excellent", "good", "fair", "poor"

#Complete fused sequence with metadata
@dataclass
class FusedSequence:
    frames: List[FusedFrame]
    metadata: Dict[str, Any]
    human_analysis: Dict[str, Any]
    object_analysis: Dict[str, Any]
    interaction_analysis: Dict[str, Any] #Human-object interactions

#Combine human pose and object detectino data into a unified 3D model
class DataFusion:

    def __init__(self, object_depth_estimation='simple', interaction_threshold=0.3, context_smoothing=True):
        self.object_depth_estimation = object_depth_estimation
        self.interaction_threshold = interaction_threshold
        self.context_smoothing = context_smoothing

        #Object size estimates in meters (for depth calculation)
        self.object_size_estimates = {
            'soccer_ball': {'diameter': 0.22, 'depth': 0.22},
            'basketball': {'diameter': 0.24, 'depth': 0.24},
            'american_football': {'length': 0.28, 'width': 0.17, 'depth': 0.17},
            'tennis_ball': {'diameter': 0.067, 'depth': 0.067},
            'volleyball': {'diameter': 0.21, 'depth': 0.21},
            'water_bottle': {'height': 0.25, 'diameter': 0.065, 'depth': 0.065},
            'sports_ball': {'diameter': 0.22, 'depth': 0.22}  #Default
        }

    #Fusion sequences of pose and object data
    def fuse_sequences(self, poses_3d: List[Pose3D], object_frames: List[ObjectFrame], drill_name: str='Unknown Drill') -> FusedSequence:
        
        print(f'Fusion sequences: {len(poses_3d)} poses + {len(object_frames)} object frames')

        #Align frames by timestamp
        aligned_frames = self._align_frames_by_timestamp(poses_3d, object_frames)

        #Create fused frames 
        fused_frames = []
        for pose_3d, object_frame in aligned_frames:
            fused_frame = self._create_fused_frame(pose_3d, object_frame)
            fused_frames.append(fused_frame)

        #Apply context aware smoothing if enabled
        if self.context_smoothing and len(fused_frames) > 2:
            fused_frames = self._apply_context_aware_smoothing(fused_frames)

        #Generate comprehensive analysis
        metadata = self._generate_metadata(fused_frames, drill_name)
        human_analysis = self._analyze_human_movement(fused_frames)
        object_analysis = self._analyze_object_movement(fused_frames)
        interaction_analysis = self._analyze_interactions(fused_frames)

        fused_sequence = FusedSequence(
            frames = fused_frames,
            metadata = metadata,
            human_analysis = human_analysis,
            object_analysis = object_analysis,
            interaction_analysis = interaction_analysis
        )

        print(f"Fusion complete: {len(fused_frames)} fused frames")
        print(f"    Human poses: {sum(1 for f in fused_frames if f.human_pose_3d)}")
        print(f"    Object detections: {sum(len(f.objects_3d) for f in fused_frames)}")
        print(f"    Interactions detected: {len(interaction_analysis.get('interactions', []))}")

        return fused_sequence
    
    #Align pose and object frames by timestamp
    def _align_frames_by_timestamp(self, poses_3d: List[Pose3D], object_frames: List[ObjectFrame]) -> List[Tuple[Optional[Pose3D], Optional[ObjectFrame]]]:

        aligned = []
        pose_idx = 0
        obj_idx = 0

        while pose_idx < len(poses_3d) or obj_idx < len(object_frames):
            pose = poses_3d[pose_idx] if pose_idx < len(poses_3d) else None
            obj_frame = object_frames[obj_idx] if obj_idx < len(object_frames) else None
            
            if pose is None:
                #No more poses, use remaining object frames
                aligned.append((None, obj_frame))
                obj_idx += 1
            elif obj_frame is None:
                #No more object frames, use remaining poses
                aligned.append((pose, None))
                pose_idx += 1
            else:
                #Both available, align by timestamp
                pose_time = pose.timestamp
                obj_time = obj_frame.timestamp
                
                if abs(pose_time - obj_time) < 0.02:  #Within 20ms - consider synchronized
                    aligned.append((pose, obj_frame))
                    pose_idx += 1
                    obj_idx += 1
                elif pose_time < obj_time:
                    aligned.append((pose, None))
                    pose_idx += 1
                else:
                    aligned.append((None, obj_frame))
                    obj_idx += 1
        
        return aligned
    
    #Create a single fused frame from pose and object data
    def _create_fused_frame(self, pose_3d: Optional[Pose3D], object_frame: Optional[ObjectFrame]) -> FusedFrame:
        
        #Determine frame properties
        if pose_3d and object_frame:
            frame_index = pose_3d.frame_index
            timestamp = pose_3d.timestamp
            world_scale = pose_3d.world_scale
        elif pose_3d:
            frame_index = pose_3d.frame_index
            timestamp = pose_3d.timestamp
            world_scale = pose_3d.world_scale
        elif object_frame:
            frame_index = object_frame.frame_index
            timestamp = object_frame.timestamp
            world_scale = 1.0  #Default scale
        else:
            raise ValueError("Both pose_3d and object_frame cannot be None")
        
        #Convert 2D objects to 3D
        objects_3d = []
        if object_frame:
            for obj_2d in object_frame.detected_objects:
                obj_3d = self._convert_object_to_3d(obj_2d, world_scale, pose_3d)
                objects_3d.append(obj_3d)
        
        #Calculate fusion confidence
        fusion_confidence = self._calculate_fusion_confidence(pose_3d, object_frame)
        
        #Determine frame quality
        frame_quality = self._assess_frame_quality(pose_3d, object_frame, fusion_confidence)
        
        return FusedFrame(
            frame_index=frame_index,
            timestamp=timestamp,
            human_pose_3d=pose_3d,
            objects_3d=objects_3d,
            fusion_confidence=fusion_confidence,
            frame_quality=frame_quality
        )
    
    #Convert 2D detected object to 3D world coordinates
    def _convert_object_to_3d(self, obj_2d: DetectedObject, world_scale: float, pose_3d: Optional[Pose3D] = None) -> Object3D:
        
        #Convert normalized 2D coordinates to world coordinates
        center_x_world = (obj_2d.center_x - 0.5) * world_scale
        center_y_world = (0.5 - obj_2d.center_y) * world_scale  # Flip Y axis
        
        #Estimate depth (Z coordinate)
        if self.object_depth_estimation == 'simple':
            #Simple depth estimation based on object size and known dimensions
            center_z_world = self._estimate_simple_depth(obj_2d, world_scale)
        else:
            #Advanced depth could use human pose context
            center_z_world = self._estimate_advanced_depth(obj_2d, world_scale, pose_3d)
        
        #Convert 2D size to world dimensions
        width_world = obj_2d.width * world_scale
        height_world = obj_2d.height * world_scale
        
        #Estimate depth dimension
        depth_world = self._estimate_object_depth_dimension(obj_2d, world_scale)
        
        return Object3D(
            class_name=obj_2d.class_name,
            confidence=obj_2d.confidence,
            center_x=center_x_world,
            center_y=center_y_world,
            center_z=center_z_world,
            width=width_world,
            height=height_world,
            depth=depth_world,
            bbox_2d=obj_2d.bbox
        )
    
    #Simple depth estimation based on object size
    def _estimate_simple_depth(self, obj_2d: DetectedObject, world_scale: float) -> float:
        
        #Get expected object size
        size_info = self.object_size_estimates.get(obj_2d.class_name, self.object_size_estimates['sports_ball'])
        
        if 'diameter' in size_info:
            expected_size = size_info['diameter']
        elif 'length' in size_info:
            expected_size = size_info['length']
        else:
            expected_size = 0.22  #Default soccer ball size
        
        #Estimate depth based on apparent size vs expected size
        apparent_size = max(obj_2d.width, obj_2d.height) * world_scale
        
        if apparent_size > 0:
            #Closer objects appear larger, farther objects appear smaller
            relative_distance = expected_size / apparent_size
            depth_estimate = relative_distance * world_scale * 0.1  # Scale factor
        else:
            depth_estimate = 0.0
        
        return depth_estimate
    
    #Advanced depth estimation using human pose context
    def _estimate_advanced_depth(self, obj_2d: DetectedObject, world_scale: float, pose_3d: Optional[Pose3D]) -> float:
        
        #Start with simple depth
        simple_depth = self._estimate_simple_depth(obj_2d, world_scale)
        
        if pose_3d is None:
            return simple_depth
        
        #Use human pose to refine depth estimate
        #For example, if ball is near player's feet, depth should be similar
        #This is a simplified version - could be much more sophisticated
        
        #Find closest human keypoint to object center
        min_distance = float('inf')
        closest_keypoint_z = simple_depth
        
        for keypoint in pose_3d.keypoints_3d:
            distance = np.sqrt((keypoint.x - (obj_2d.center_x - 0.5) * world_scale)**2 + 
                              (keypoint.y - (0.5 - obj_2d.center_y) * world_scale)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_keypoint_z = keypoint.z
        
        #Blend simple depth with contextual depth
        if min_distance < 0.5:  # If object is close to human
            blended_depth = 0.7 * closest_keypoint_z + 0.3 * simple_depth
        else:
            blended_depth = simple_depth
        
        return blended_depth
    
    #Estimate the depth dimensino of the object
    def _estimate_object_depth_dimension(self, obj_2d: DetectedObject, world_scale: float) -> float: 
        
        size_info = self.object_size_estimates.get(obj_2d.class_name, self.object_size_estimates['sports_ball'])
        
        if 'depth' in size_info:
            return size_info['depth']
        elif 'diameter' in size_info:
            return size_info['diameter']
        else:
            #Estimate based on 2D dimensions
            return min(obj_2d.width, obj_2d.height) * world_scale
    
    #Calcualte confidene in the pose object fusion quality
    def _calculate_fusion_confidence(self, pose_3d: Optional[Pose3D], object_frame: Optional[ObjectFrame]) -> float: 
        
        if pose_3d is None and object_frame is None:
            return 0.0
        elif pose_3d is None:
            return object_frame.detection_confidence * 0.7  #Object only
        elif object_frame is None:
            return pose_3d.keypoints_3d[0].confidence * 0.8  #Pose only (use first keypoint as reference)
        else:
            #Both available - combine confidences
            pose_conf = np.mean([kp.confidence for kp in pose_3d.keypoints_3d])
            obj_conf = object_frame.detection_confidence
            return (pose_conf + obj_conf) / 2
    
    #Assess overall frame quality for filtering/priority
    def _assess_frame_quality(self, pose_3d: Optional[Pose3D], object_frame: Optional[ObjectFrame], fusion_confidence: float) -> str:
        
        if fusion_confidence > 0.8:
            return "excellent"
        elif fusion_confidence > 0.6:
            return "good"
        elif fusion_confidence > 0.4:
            return "fair"
        else:
            return "poor"
    
    #Apply context aware smoothing using human pose data to refine object positions
    def _apply_context_aware_smoothing(self, fused_frames: List[FusedFrame]) -> List[FusedFrame]:
        
        print(f"Applying context-aware smoothing to {len(fused_frames)} frames...")
        
        #Group objects by class name for tracking
        object_tracks = {}
        
        for frame in fused_frames:
            for obj in frame.objects_3d:
                if obj.class_name not in object_tracks:
                    object_tracks[obj.class_name] = []
                object_tracks[obj.class_name].append((frame.frame_index, obj, frame.human_pose_3d))
        
        #Apply context-aware smoothing to each object track
        for class_name, track in object_tracks.items():
            if len(track) < 5:  #Need more frames for context-aware smoothing
                continue
            
            #Extract positions and human context
            positions = []
            human_contexts = []
            
            for frame_idx, obj, human_pose in track:
                positions.append([obj.center_x, obj.center_y, obj.center_z])
                
                #Get relevant human keypoints for context
                if human_pose and self._is_sports_ball(class_name):
                    context = self._extract_human_context(human_pose, class_name)
                    human_contexts.append(context)
                else:
                    human_contexts.append(None)
            
            positions = np.array(positions)
            
            #Apply context-informed smoothing
            smoothed_positions = self._context_informed_smoothing(
                positions, human_contexts, class_name
            )
            
            #Update object positions
            for i, (frame_idx, obj, human_pose) in enumerate(track):
                if i < len(smoothed_positions):
                    obj.center_x = smoothed_positions[i][0]
                    obj.center_y = smoothed_positions[i][1]
                    obj.center_z = smoothed_positions[i][2]
        
        print(f"Context-aware smoothing complete")
        return fused_frames
    
    #Check if object is a sports ball taht should follow human movement
    def _is_sports_ball(self, class_name: str) -> bool:
        return 'ball' in class_name.lower()
    
    #Extract relevant human keypoints for object context
    def _extract_human_context(self, human_pose: Pose3D, object_class: str) -> Dict[str, Any]:
        
        context = {}
        
        if 'ball' in object_class.lower():
            #For balls, focus on feet and hands
            key_indices = {
                'right_ankle': 28,
                'left_ankle': 27,
                'right_wrist': 16,
                'left_wrist': 15,
                'right_foot_index': 32,
                'left_foot_index': 31
            }
            
            for name, idx in key_indices.items():
                if idx < len(human_pose.keypoints_3d):
                    kp = human_pose.keypoints_3d[idx]
                    context[name] = {
                        'position': [kp.x, kp.y, kp.z],
                        'confidence': kp.confidence
                    }
        
        return context
    
    #Apply smoothing that considers human movement context
    def _context_informed_smoothing(self, positions: np.ndarray, human_contexts: List[Optional[Dict]], object_class: str) -> np.ndarray:
        
        if len(positions) < 5:
            return positions
        
        smoothed = np.zeros_like(positions)
        window_size = 5  #Larger window for context-aware smoothing
        
        for i in range(len(positions)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(positions), i + window_size // 2 + 1)
            
            #Standard smoothing window
            window_positions = positions[start_idx:end_idx]
            base_smooth = np.mean(window_positions, axis=0)
            
            #Apply human context influence if available
            if human_contexts[i] and self._is_sports_ball(object_class):
                context_adjusted = self._apply_human_context_influence(
                    base_smooth, human_contexts[i], object_class
                )
                smoothed[i] = context_adjusted
            else:
                smoothed[i] = base_smooth
        
        return smoothed
    
    #Adjust object position based on human movement context
    def _apply_human_context_influence(self, object_position: np.ndarray, human_context: Dict, object_class: str) -> np.ndarray: 
        
        #Find the closest high-confidence human keypoint
        min_distance = float('inf')
        closest_keypoint_pos = None
        
        for keypoint_name, data in human_context.items():
            if data['confidence'] > 0.7:  #High confidence threshold
                kp_pos = np.array(data['position'])
                distance = np.linalg.norm(object_position - kp_pos)
                
                if distance < min_distance and distance < 0.4:  #Within reasonable range
                    min_distance = distance
                    closest_keypoint_pos = kp_pos
        
        #If ball is close to high-confidence human keypoint, bias smoothing toward it
        if closest_keypoint_pos is not None and min_distance < 0.2:
            #Blend object position with human keypoint influence
            influence_strength = 0.3  # 30% influence from human context
            
            #Direction from keypoint to object
            direction = object_position - closest_keypoint_pos
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 0:
                #Maintain relative position but bias toward human keypoint
                adjusted_position = (
                    (1 - influence_strength) * object_position + 
                    influence_strength * (closest_keypoint_pos + direction * 0.8)
                )
                return adjusted_position
        
        return object_position
    
    #Generate metadata for the fused sequence
    def _generate_metadata(self, fused_frames: List[FusedFrame], drill_name: str) -> Dict[str, Any]: 
        
        return {
            'drill_name': drill_name,
            'total_frames': len(fused_frames),
            'duration_seconds': fused_frames[-1].timestamp - fused_frames[0].timestamp if fused_frames else 0.0,
            'creation_timestamp': time.time(),
            'fusion_settings': {
                'object_depth_estimation': self.object_depth_estimation,
                'interaction_threshold': self.interaction_threshold,
                'context_smoothing': self.context_smoothing
            },
            'quality_distribution': {
                'excellent': sum(1 for f in fused_frames if f.frame_quality == 'excellent'),
                'good': sum(1 for f in fused_frames if f.frame_quality == 'good'),
                'fair': sum(1 for f in fused_frames if f.frame_quality == 'fair'),
                'poor': sum(1 for f in fused_frames if f.frame_quality == 'poor')
            }
        }
    
    #Analyze human movement patterns 
    def _analyze_human_movement(self, fused_frames: List[FusedFrame]) -> Dict[str, Any]: 
        
        poses_with_data = [f.human_pose_3d for f in fused_frames if f.human_pose_3d]
        
        if not poses_with_data:
            return {"error": "No human pose data available"}
        
        #Analyze key body parts
        analysis = {}
        key_points = {'right_ankle': 28, 'left_ankle': 27, 'right_wrist': 16, 'left_wrist': 15}
        
        for name, idx in key_points.items():
            positions = []
            for pose in poses_with_data:
                if idx < len(pose.keypoints_3d):
                    kp = pose.keypoints_3d[idx]
                    positions.append([kp.x, kp.y, kp.z])
            
            if positions:
                positions = np.array(positions)
                analysis[name] = {
                    'range_x': float(np.ptp(positions[:, 0])),
                    'range_y': float(np.ptp(positions[:, 1])),
                    'range_z': float(np.ptp(positions[:, 2])),
                    'avg_velocity': float(np.mean(np.linalg.norm(np.diff(positions, axis=0), axis=1))) if len(positions) > 1 else 0.0
                }
        
        return analysis
    
    #Analyze object movement patterns 
    def _analyze_object_movement(self, fused_frames: List[FusedFrame]) -> Dict[str, Any]: 
        
        object_analysis = {}
        
        #Group objects by class
        object_tracks = {}
        for frame in fused_frames:
            for obj in frame.objects_3d:
                if obj.class_name not in object_tracks:
                    object_tracks[obj.class_name] = []
                object_tracks[obj.class_name].append(obj)
        
        #Analyze each object type
        for class_name, objects in object_tracks.items():
            positions = np.array([[obj.center_x, obj.center_y, obj.center_z] for obj in objects])
            
            if len(positions) > 0:
                object_analysis[class_name] = {
                    'total_detections': len(objects),
                    'avg_confidence': float(np.mean([obj.confidence for obj in objects])),
                    'range_x': float(np.ptp(positions[:, 0])),
                    'range_y': float(np.ptp(positions[:, 1])),
                    'range_z': float(np.ptp(positions[:, 2])),
                    'avg_velocity': float(np.mean(np.linalg.norm(np.diff(positions, axis=0), axis=1))) if len(positions) > 1 else 0.0
                }
        
        return object_analysis
    
    #Analyze human object interactions
    def _analyze_interactions(self, fused_frames: List[FusedFrame]) -> Dict[str, Any]:
        
        interactions = []
        
        for frame in fused_frames:
            if not frame.human_pose_3d or not frame.objects_3d:
                continue
            
            #Check distance between human keypoints and objects
            for obj in frame.objects_3d:
                for i, keypoint in enumerate(frame.human_pose_3d.keypoints_3d):
                    distance = np.sqrt(
                        (keypoint.x - obj.center_x)**2 + 
                        (keypoint.y - obj.center_y)**2 + 
                        (keypoint.z - obj.center_z)**2
                    )
                    
                    if distance < self.interaction_threshold:
                        interactions.append({
                            'frame_index': frame.frame_index,
                            'timestamp': frame.timestamp,
                            'keypoint_name': keypoint.name,
                            'keypoint_index': i,
                            'object_class': obj.class_name,
                            'distance': float(distance),
                            'interaction_type': self._classify_interaction(keypoint.name, obj.class_name)
                        })
        
        return {
            'interactions': interactions,
            'total_interactions': len(interactions),
            'interaction_summary': self._summarize_interactions(interactions)
        }
    
    #Classify the type of interaction
    def _classify_interaction(self, keypoint_name: str, object_class: str) -> str:
        
        if 'ankle' in keypoint_name or 'foot' in keypoint_name:
            if 'ball' in object_class:
                return 'foot_ball_contact'
            else:
                return 'foot_object_contact'
        elif 'wrist' in keypoint_name or 'hand' in keypoint_name:
            return 'hand_object_contact'
        elif 'knee' in keypoint_name:
            if 'ball' in object_class:
                return 'knee_ball_contact'
            else:
                return 'knee_object_contact'
        else:
            return 'general_proximity'
    
    #Summarize interaction patterns
    def _summarize_interactions(self, interactions: List[Dict]) -> Dict[str, Any]: 
        
        if not interactions:
            return {}
        
        summary = {}
        
        #Count by interaction type
        interaction_types = {}
        for interaction in interactions:
            int_type = interaction['interaction_type']
            if int_type not in interaction_types:
                interaction_types[int_type] = 0
            interaction_types[int_type] += 1
        
        summary['interaction_types'] = interaction_types
        
        #Most active keypoints
        keypoint_activity = {}
        for interaction in interactions:
            kp_name = interaction['keypoint_name']
            if kp_name not in keypoint_activity:
                keypoint_activity[kp_name] = 0
            keypoint_activity[kp_name] += 1
        
        summary['most_active_keypoints'] = dict(sorted(keypoint_activity.items(), 
                                                      key=lambda x: x[1], reverse=True)[:5])
        
        return summary

