import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import gzip
import time

from src.model_creator import ModelCreator, ModelMetadata, SkeletonBone, ModelKeyframe
from src.data_fusion import FusedSequence, FusedFrame, Object3D

#Object position and properties for a single keyframe
@dataclass
class ObjectKeyframe:
    timestamp: float
    frame_index: int
    objects: List[Dict[str, Any]]  #Optimized object data

#Single human object interation event
@dataclass
class InteractionEvent:
    timestamp: float
    frame_index: int
    keypoint_name: str
    keypoint_index: int
    object_class: str
    distance: float
    interaction_type: str

#Extended metadata for fused models
@dataclass
class FusionMetadata:
    drill_name: str
    duration_seconds: float
    fps: float
    total_frames: int
    world_scale: float
    creation_timestamp: float
    file_version: str = "1.1"  #Upgraded for fusion
    coordinate_system: str = "right_handed"
    
    #Fusion-specific metadata
    fusion_settings: Dict[str, Any] = None
    quality_distribution: Dict[str, int] = None
    object_types_detected: List[str] = None
    total_interactions: int = 0

#Complete fused 3D model structure for export
@dataclass
class FusedModel3D:
    metadata: FusionMetadata
    skeleton_structure: List[SkeletonBone]
    human_keyframes: List[ModelKeyframe] #Human pose animation
    object_keyframes: List[ObjectKeyframe] #Object animation
    interaction_events: List[InteractionEvent] #Human-object interactions
    human_analysis: Dict[str, Any]
    object_analysis: Dict[str, Any]
    interaction_analysis: Dict[str, Any]

#Model creator that fuses human and object data
class ModelCreatorFusion(ModelCreator):
    
    def __init__(self, 
                 coordinate_precision=3,
                 compress_output=True,
                 optimize_for_web=True,
                 include_interaction_events=True):
        
        #Initialize parent class
        super().__init__(coordinate_precision, compress_output, optimize_for_web)
        
        self.include_interaction_events = include_interaction_events
        
        #Object size estimates for web visualization hints
        self.object_render_hints = {
            'soccer_ball': {'radius': 0.11, 'color': '#FFFFFF', 'type': 'sphere'},
            'basketball': {'radius': 0.12, 'color': '#FF8C00', 'type': 'sphere'},
            'american_football': {'length': 0.28, 'width': 0.17, 'color': '#8B4513', 'type': 'ellipsoid'},
            'tennis_ball': {'radius': 0.034, 'color': '#CCFF00', 'type': 'sphere'},
            'volleyball': {'radius': 0.105, 'color': '#FFFF00', 'type': 'sphere'},
            'water_bottle': {'height': 0.25, 'radius': 0.033, 'color': '#87CEEB', 'type': 'cylinder'},
            'sports_ball': {'radius': 0.11, 'color': '#FFFFFF', 'type': 'sphere'}  # Default
        }
    
    #Create complete fused 3D model for fusion sequence
    def create_fused_3d_model(self, fused_sequence: FusedSequence) -> FusedModel3D:
        
        if not fused_sequence.frames:
            raise ValueError("No fused frames provided for model creation")
        
        print(f"Creating fused 3D model with {len(fused_sequence.frames)} frames...")
        
        #Extract metadata
        duration = fused_sequence.frames[-1].timestamp - fused_sequence.frames[0].timestamp
        fps = len(fused_sequence.frames) / duration if duration > 0 else 0
        
        #Get world scale from first frame with human pose
        world_scale = 1.0
        for frame in fused_sequence.frames:
            if frame.human_pose_3d:
                world_scale = frame.human_pose_3d.world_scale
                break
        
        #Create fusion metadata
        fusion_metadata = FusionMetadata(
            drill_name=fused_sequence.metadata['drill_name'],
            duration_seconds=duration,
            fps=fps,
            total_frames=len(fused_sequence.frames),
            world_scale=world_scale,
            creation_timestamp=time.time(),
            fusion_settings=fused_sequence.metadata.get('fusion_settings', {}),
            quality_distribution=fused_sequence.metadata.get('quality_distribution', {}),
            object_types_detected=list(fused_sequence.object_analysis.keys()) if fused_sequence.object_analysis else [],
            total_interactions=fused_sequence.interaction_analysis.get('total_interactions', 0)
        )
        
        #Create human keyframes
        human_keyframes = self._create_human_keyframes(fused_sequence.frames)
        
        #Create object keyframes
        object_keyframes = self._create_object_keyframes(fused_sequence.frames)
        
        #Create interaction events
        interaction_events = []
        if self.include_interaction_events:
            interaction_events = self._create_interaction_events(fused_sequence.interaction_analysis)
        
        fused_model = FusedModel3D(
            metadata=fusion_metadata,
            skeleton_structure=self.bone_definitions,
            human_keyframes=human_keyframes,
            object_keyframes=object_keyframes,
            interaction_events=interaction_events,
            human_analysis=fused_sequence.human_analysis,
            object_analysis=fused_sequence.object_analysis,
            interaction_analysis=fused_sequence.interaction_analysis
        )
        
        print(f"Fused 3D model created successfully!")
        print(f"    Human keyframes: {len(human_keyframes)}")
        print(f"    Object keyframes: {len(object_keyframes)}")
        print(f"    Interaction events: {len(interaction_events)}")
        print(f"    Duration: {duration:.2f} seconds")
        
        return fused_model
    
    #Create human keyframs from fused frames
    def _create_human_keyframes(self, fused_frames: List[FusedFrame]) -> List[ModelKeyframe]:
        
        human_keyframes = []
        
        for frame in fused_frames:
            if frame.human_pose_3d is None:
                #Create empty keyframe for missing human data
                keyframe = ModelKeyframe(
                    timestamp=round(frame.timestamp, 3),
                    frame_index=frame.frame_index,
                    keypoints=[]
                )
            else:
                #Use existing keyframe creation logic from parent class
                optimized_keypoints = []
                
                for kp in frame.human_pose_3d.keypoints_3d:
                    keypoint_data = {
                        "x": round(kp.x, self.coordinate_precision),
                        "y": round(kp.y, self.coordinate_precision),
                        "z": round(kp.z, self.coordinate_precision),
                        "c": round(kp.confidence, 2),
                        "n": kp.name if not self.optimize_for_web else None
                    }
                    
                    #Remove None values for web optimization
                    if self.optimize_for_web:
                        keypoint_data = {k: v for k, v in keypoint_data.items() if v is not None}
                    
                    optimized_keypoints.append(keypoint_data)
                
                keyframe = ModelKeyframe(
                    timestamp=round(frame.timestamp, 3),
                    frame_index=frame.frame_index,
                    keypoints=optimized_keypoints
                )
            
            human_keyframes.append(keyframe)
        
        return human_keyframes
    
    #Create object keyframes from fused frames
    def _create_object_keyframes(self, fused_frames: List[FusedFrame]) -> List[ObjectKeyframe]:
        
        object_keyframes = []
        
        for frame in fused_frames:
            #Optimize object data for web
            optimized_objects = []
            
            for obj in frame.objects_3d:
                #Get render hints for this object type
                render_hints = self.object_render_hints.get(obj.class_name, self.object_render_hints['sports_ball'])
                
                object_data = {
                    "class": obj.class_name,
                    "conf": round(obj.confidence, 2),
                    "pos": [
                        round(obj.center_x, self.coordinate_precision),
                        round(obj.center_y, self.coordinate_precision),
                        round(obj.center_z, self.coordinate_precision)
                    ],
                    "size": [
                        round(obj.width, self.coordinate_precision),
                        round(obj.height, self.coordinate_precision),
                        round(obj.depth, self.coordinate_precision)
                    ]
                }
                
                #Add render hints for web visualization
                if not self.optimize_for_web:
                    object_data["render_hints"] = render_hints
                else:
                    #Compact render hints for web
                    object_data["r"] = render_hints.get('radius', 0.1)
                    object_data["t"] = render_hints.get('type', 'sphere')
                    object_data["col"] = render_hints.get('color', '#FFFFFF')
                
                optimized_objects.append(object_data)
            
            object_keyframe = ObjectKeyframe(
                timestamp=round(frame.timestamp, 3),
                frame_index=frame.frame_index,
                objects=optimized_objects
            )
            
            object_keyframes.append(object_keyframe)
        
        return object_keyframes
    
    #Create interaction events from analysis data
    def _create_interaction_events(self, interaction_analysis: Dict[str, Any]) -> List[InteractionEvent]:
        
        interaction_events = []
        interactions = interaction_analysis.get('interactions', [])
        
        for interaction in interactions:
            event = InteractionEvent(
                timestamp=round(interaction['timestamp'], 3),
                frame_index=interaction['frame_index'],
                keypoint_name=interaction['keypoint_name'],
                keypoint_index=interaction['keypoint_index'],
                object_class=interaction['object_class'],
                distance=round(interaction['distance'], 3),
                interaction_type=interaction['interaction_type']
            )
            interaction_events.append(event)
        
        return interaction_events
    
    #Export fused 3D model to JSON file with optional compression
    def export_fused_to_json(self, fused_model: FusedModel3D, output_path: str) -> Dict[str, Any]:
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        #Convert to dictionary
        model_dict = self._fused_model_to_dict(fused_model)
        
        #Create JSON string
        json_string = json.dumps(model_dict, separators=(',', ':') if self.optimize_for_web else None)
        
        #Calculate sizes
        uncompressed_size = len(json_string.encode('utf-8'))
        
        if self.compress_output:
            #Save compressed version
            compressed_path = output_path.with_suffix(output_path.suffix + '.gz')
            with gzip.open(compressed_path, 'wt', encoding='utf-8') as f:
                f.write(json_string)
            compressed_size = compressed_path.stat().st_size
            
            print(f"Fused model saved (compressed): {compressed_path}")
            print(f"    Original size: {uncompressed_size:,} bytes ({uncompressed_size/1024:.1f} KB)")
            print(f"    Compressed size: {compressed_size:,} bytes ({compressed_size/1024:.1f} KB)")
            print(f"    Compression ratio: {(1-compressed_size/uncompressed_size)*100:.1f}%")
            
            export_info = {
                "compressed_path": str(compressed_path),
                "uncompressed_size": uncompressed_size,
                "compressed_size": compressed_size,
                "compression_ratio": (1-compressed_size/uncompressed_size)
            }
        else:
            #Save uncompressed version
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(model_dict, f, indent=2 if not self.optimize_for_web else None)
            
            print(f"Fused model saved: {output_path}")
            print(f"    File size: {uncompressed_size:,} bytes ({uncompressed_size/1024:.1f} KB)")
            
            export_info = {
                "output_path": str(output_path),
                "file_size": uncompressed_size
            }
        
        return export_info
    
    #Convert FusedModel3D to dictionaryu for JSON serialization
    def _fused_model_to_dict(self, fused_model: FusedModel3D) -> Dict[str, Any]:
        
        model_dict = {
            "metadata": asdict(fused_model.metadata),
            "skeleton_structure": [asdict(bone) for bone in fused_model.skeleton_structure],
            "human_keyframes": [asdict(keyframe) for keyframe in fused_model.human_keyframes],
            "object_keyframes": [asdict(keyframe) for keyframe in fused_model.object_keyframes],
            "human_analysis": fused_model.human_analysis,
            "object_analysis": fused_model.object_analysis,
            "interaction_analysis": fused_model.interaction_analysis
        }
        
        #Add interaction events if included
        if self.include_interaction_events:
            model_dict["interaction_events"] = [asdict(event) for event in fused_model.interaction_events]
        
        return model_dict
    
    #Get detailed statistics about the fused 3D model
    def get_fused_model_statistics(self, fused_model: FusedModel3D) -> Dict[str, Any]:
        
        stats = {
            "basic_info": {
                "drill_name": fused_model.metadata.drill_name,
                "duration": fused_model.metadata.duration_seconds,
                "fps": fused_model.metadata.fps,
                "total_frames": fused_model.metadata.total_frames,
                "world_scale": fused_model.metadata.world_scale,
                "file_version": fused_model.metadata.file_version
            },
            "structure": {
                "human_keyframes": len(fused_model.human_keyframes),
                "object_keyframes": len(fused_model.object_keyframes),
                "interaction_events": len(fused_model.interaction_events),
                "total_bones": len(fused_model.skeleton_structure),
                "bone_groups": {}
            },
            "content_analysis": {
                "object_types": fused_model.metadata.object_types_detected,
                "total_interactions": fused_model.metadata.total_interactions,
                "quality_distribution": fused_model.metadata.quality_distribution,
                "fusion_settings": fused_model.metadata.fusion_settings
            },
            "data_size": {
                "estimated_json_size_kb": len(json.dumps(self._fused_model_to_dict(fused_model))) / 1024
            }
        }
        
        #Count bones by group
        for bone in fused_model.skeleton_structure:
            group = bone.bone_group
            if group not in stats["structure"]["bone_groups"]:
                stats["structure"]["bone_groups"][group] = 0
            stats["structure"]["bone_groups"][group] += 1
        
        return stats
    

    #Compare fused model with human only model for size analysis
    def compare_with_human_only_model(self, fused_model: FusedModel3D, human_only_size: int) -> Dict[str, Any]:
        
        fused_size = len(json.dumps(self._fused_model_to_dict(fused_model)))
        
        return {
            "human_only_size_kb": human_only_size / 1024,
            "fused_size_kb": fused_size / 1024,
            "size_increase_kb": (fused_size - human_only_size) / 1024,
            "size_increase_percent": ((fused_size / human_only_size) - 1) * 100 if human_only_size > 0 else 0,
            "additional_data": {
                "object_keyframes": len(fused_model.object_keyframes),
                "interaction_events": len(fused_model.interaction_events),
                "object_types": len(fused_model.metadata.object_types_detected)
            }
        }