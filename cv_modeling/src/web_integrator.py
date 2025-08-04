import json
import gzip
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import time

from src.model_creator_fusion import FusedModel3D
from src.motion_segmenter import SegmentedMotion

#Metadata optimized for web display
@dataclass
class WebDisplayMetadata: 
    title: str
    sport_type: str
    duration: float
    fps: float
    total_frames: int
    file_version: str
    creation_date: str
    
    # Display settings
    default_playback_speed: float = 1.0
    auto_loop: bool = True
    show_skeleton: bool = True
    show_objects: bool = True
    show_interactions: bool = True

#Timeline marker for web player
@dataclass
class WebTimelineMarker: 
    timestamp: float
    frame_index: int
    label: str
    marker_type: str  # "phase", "cycle", "event", "loop"
    color: str
    importance: str  # "high", "medium", "low"
    description: str

#Playback control configuration
@dataclass
class WebPlaybackControl: 
    loop_points: List[Dict[str, Any]]
    phase_controls: List[Dict[str, Any]]
    cycle_controls: List[Dict[str, Any]]
    speed_presets: List[float] = None
    timeline_markers: List[WebTimelineMarker] = None

#Three.js visualization configuration
@dataclass
class WebVisualizationConfig: 
    camera_position: List[float]
    camera_target: List[float]
    lighting_config: Dict[str, Any]
    skeleton_config: Dict[str, Any]
    object_config: Dict[str, Any]
    animation_config: Dict[str, Any]

#Complete web optimized sports model
@dataclass
class WebSportsModel:
    metadata: WebDisplayMetadata
    human_animation: Dict[str, Any]
    object_animation: Dict[str, Any]
    interaction_events: List[Dict[str, Any]]
    playback_controls: WebPlaybackControl
    visualization_config: WebVisualizationConfig
    coaching_data: Dict[str, Any]

#Integrates fusion and segmentation data into Three.js- optimized format
class WebIntegrator:
    
    def __init__(self, coordinate_precision=3, compress_output=True, include_coaching_data=True, optimize_for_mobile=False):
        
        self.coordinate_precision = coordinate_precision
        self.compress_output = compress_output
        self.include_coaching_data = include_coaching_data
        self.optimize_for_mobile = optimize_for_mobile
        
        #Web visualization defaults
        self.default_visualization_config = {
            'camera_position': [2.0, 1.5, 3.0],
            'camera_target': [0.0, 0.0, 0.0],
            'lighting_config': {
                'ambient_intensity': 0.4,
                'directional_intensity': 0.8,
                'directional_position': [1, 1, 1],
                'shadows': True
            },
            'skeleton_config': {
                'joint_radius': 0.03,
                'bone_thickness': 0.02,
                'joint_color': '#FFD700',
                'bone_color': '#87CEEB',
                'opacity': 0.9
            },
            'object_config': {
                'opacity': 0.8,
                'wireframe': False,
                'cast_shadows': True,
                'receive_shadows': True
            },
            'animation_config': {
                'interpolation': 'linear',
                'auto_start': False,
                'loop_mode': 'pingpong'
            }
        }
        
        #Timeline marker colors
        self.marker_colors = {
            'phase': {'setup': '#FFA500', 'execution': '#FF4500', 'recovery': '#32CD32', 'transition': '#87CEEB'},
            'cycle': '#9370DB',
            'event': {'ball_contact': '#FF0000', 'direction_change': '#FFD700', 'peak_moment': '#FF69B4'},
            'loop': '#00CED1'
        }

    #Create complete web optimized sports model
    def create_web_model(self, 
                        fused_model: FusedModel3D, 
                        segmented_motion: Optional[SegmentedMotion] = None,
                        sport_type: str = 'soccer') -> WebSportsModel: 
        
        print(f"Creating web-optimized model for {sport_type}...")
        
        #Create web metadata
        metadata = self._create_web_metadata(fused_model, sport_type)
        
        #Optimize human animation data
        human_animation = self._optimize_human_animation(fused_model.human_keyframes, fused_model.skeleton_structure)
        
        #Optimize object animation data  
        object_animation = self._optimize_object_animation(fused_model.object_keyframes)
        
        #Process interaction events
        interaction_events = self._process_interaction_events(fused_model.interaction_events)
        
        # Create playback controls
        playback_controls = self._create_playback_controls(fused_model, segmented_motion)
        
        #Create visualization config
        viz_config = self._create_visualization_config(sport_type, fused_model.metadata.world_scale)
        
        #Create coaching data
        coaching_data = {}
        if self.include_coaching_data:
            coaching_data = self._create_coaching_data(fused_model, segmented_motion)
        
        web_model = WebSportsModel(
            metadata=metadata,
            human_animation=human_animation,
            object_animation=object_animation,
            interaction_events=interaction_events,
            playback_controls=playback_controls,
            visualization_config=viz_config,
            coaching_data=coaching_data
        )
        
        print(f"Web model created successfully!")
        return web_model
    
    #Create web optimized metadata
    def _create_web_metadata(self, fused_model: FusedModel3D, sport_type: str) -> WebDisplayMetadata: 
        
        return WebDisplayMetadata(
            title=fused_model.metadata.drill_name,
            sport_type=sport_type,
            duration=fused_model.metadata.duration_seconds,
            fps=fused_model.metadata.fps,
            total_frames=fused_model.metadata.total_frames,
            file_version=f"web-{fused_model.metadata.file_version}",
            creation_date=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(fused_model.metadata.creation_timestamp)),
            default_playback_speed=1.0,
            auto_loop=True,
            show_skeleton=True,
            show_objects=True,
            show_interactions=True
        )
    
    #Optimized human animation data for Three.js
    def _optimize_human_animation(self, human_keyframes: List, skeleton_structure: List) -> Dict[str, Any]:
        
        #Convert to Three.js-friendly format
        animation_data = {
            'skeleton': {
                'bones': [],
                'bone_indices': {}
            },
            'keyframes': [],
            'duration': 0,
            'fps': 0
        }
        
        #Process skeleton structure
        for i, bone in enumerate(skeleton_structure):
            bone_data = {
                'id': i,
                'name': bone.name,
                'start_joint': bone.start_keypoint,
                'end_joint': bone.end_keypoint,
                'group': bone.bone_group
            }
            animation_data['skeleton']['bones'].append(bone_data)
            animation_data['skeleton']['bone_indices'][bone.name] = i
        
        #Process keyframes
        if human_keyframes:
            animation_data['duration'] = human_keyframes[-1].timestamp - human_keyframes[0].timestamp
            animation_data['fps'] = len(human_keyframes) / animation_data['duration'] if animation_data['duration'] > 0 else 0
            
            for keyframe in human_keyframes:
                frame_data = {
                    'time': round(keyframe.timestamp, 3),
                    'frame': keyframe.frame_index,
                    'joints': []
                }
                
                #Optimize joint data
                for joint in keyframe.keypoints:
                    if self.optimize_for_mobile:
                        #Reduced precision for mobile
                        joint_data = {
                            'pos': [
                                round(joint['x'], 2),
                                round(joint['y'], 2), 
                                round(joint['z'], 2)
                            ],
                            'conf': round(joint['c'], 1)
                        }
                    else:
                        joint_data = {
                            'position': [
                                round(joint['x'], self.coordinate_precision),
                                round(joint['y'], self.coordinate_precision),
                                round(joint['z'], self.coordinate_precision)
                            ],
                            'confidence': round(joint['c'], 2)
                        }
                    
                    frame_data['joints'].append(joint_data)
                
                animation_data['keyframes'].append(frame_data)
        
        return animation_data
    
    #Optimize object animatoin data for Three.js
    def _optimize_object_animation(self, object_keyframes: List) -> Dict[str, Any]: 
        
        animation_data = {
            'objects': {},
            'keyframes': [],
            'object_types': []
        }
        
        #Track unique objects
        object_registry = {}
        
        for keyframe in object_keyframes:
            frame_data = {
                'time': round(keyframe.timestamp, 3),
                'frame': keyframe.frame_index,
                'objects': []
            }
            
            for obj in keyframe.objects:
                obj_class = obj['class']
                
                #Register object type if new
                if obj_class not in object_registry:
                    object_registry[obj_class] = {
                        'id': len(object_registry),
                        'class': obj_class,
                        'render_config': {
                            'radius': obj.get('r', 0.1),
                            'type': obj.get('t', 'sphere'),
                            'color': obj.get('col', '#FFFFFF')
                        }
                    }
                
                #Optimize object data
                if self.optimize_for_mobile:
                    obj_data = {
                        'id': object_registry[obj_class]['id'],
                        'pos': [round(p, 2) for p in obj['pos']],
                        'conf': round(obj['conf'], 1)
                    }
                else:
                    obj_data = {
                        'object_id': object_registry[obj_class]['id'],
                        'position': [round(p, self.coordinate_precision) for p in obj['pos']],
                        'size': [round(s, self.coordinate_precision) for s in obj['size']],
                        'confidence': round(obj['conf'], 2)
                    }
                
                frame_data['objects'].append(obj_data)
            
            animation_data['keyframes'].append(frame_data)
        
        #Add object registry
        animation_data['objects'] = object_registry
        animation_data['object_types'] = list(object_registry.keys())
        
        return animation_data
    
    #Process interaction events for web display
    def _process_interaction_events(self, interaction_events: List) -> List[Dict[str, Any]]:
        
        web_events = []
        
        for event in interaction_events:
            web_event = {
                'timestamp': round(event.timestamp, 3),
                'frame': event.frame_index,
                'type': event.interaction_type,
                'keypoint': event.keypoint_name,
                'object': event.object_class,
                'distance': round(event.distance, 3),
                'display_config': {
                    'color': self.marker_colors['event'].get(event.interaction_type, '#FF0000'),
                    'size': 'large' if event.interaction_type == 'foot_ball_contact' else 'medium',
                    'duration': 0.5, #Display duration in seconds
                    'animation': 'pulse'
                }
            }
            web_events.append(web_event)
        
        return web_events