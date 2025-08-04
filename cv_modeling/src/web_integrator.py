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
        
        #Create coaching data as emtpty
        #TODO
        coaching_data = {}
        
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
    
    #Create playback controls
    def _create_playback_controls(self, fused_model: FusedModel3D, segmented_motion: Optional[SegmentedMotion]) -> WebPlaybackControl: 
        
        #Default loop points (full sequence)
        loop_points = [{
            'name': 'Full Sequence',
            'start_time': 0.0,
            'end_time': fused_model.metadata.duration_seconds,
            'start_frame': 0,
            'end_frame': fused_model.metadata.total_frames - 1,
            'quality': 1.0,
            'default': True
        }]
        
        phase_controls = []
        cycle_controls = []
        timeline_markers = []
        
        if segmented_motion:
            #Add segmented loop points
            for loop_rec in segmented_motion.loop_recommendations:
                if loop_rec.get('recommended', False):
                    loop_points.append({
                        'name': f"Loop {loop_rec['cycle_id']}",
                        'start_time': loop_rec['start_time'],
                        'end_time': loop_rec['end_time'],
                        'start_frame': loop_rec['start_frame'],
                        'end_frame': loop_rec['end_frame'],
                        'quality': loop_rec['loop_quality'],
                        'cycle_id': loop_rec['cycle_id']
                    })
            
            #Phase controls
            for phase in segmented_motion.overall_phases:
                phase_control = {
                    'name': phase.name,
                    'start_time': phase.start_time,
                    'end_time': phase.end_time,
                    'start_frame': phase.start_frame,
                    'end_frame': phase.end_frame,
                    'duration': phase.duration,
                    'type': phase.phase_type,
                    'intensity': phase.movement_intensity,
                    'body_parts': phase.primary_body_parts
                }
                phase_controls.append(phase_control)
                
                #Add phase markers
                marker = WebTimelineMarker(
                    timestamp=phase.start_time,
                    frame_index=phase.start_frame,
                    label=f"{phase.phase_type.title()} Phase",
                    marker_type="phase",
                    color=self.marker_colors['phase'].get(phase.phase_type, '#87CEEB'),
                    importance="medium",
                    description=f"{phase.phase_type.title()} phase ({phase.duration:.1f}s)"
                )
                timeline_markers.append(marker)
            
            #Cycle controls
            for cycle in segmented_motion.movement_cycles:
                cycle_control = {
                    'cycle_id': cycle.cycle_id,
                    'name': f"Cycle {cycle.cycle_id + 1}",
                    'start_time': cycle.start_time,
                    'end_time': cycle.end_time,
                    'start_frame': cycle.start_frame,
                    'end_frame': cycle.end_frame,
                    'duration': cycle.duration,
                    'quality': cycle.cycle_quality,
                    'phases': len(cycle.phases)
                }
                cycle_controls.append(cycle_control)
                
                #Add cycle markers
                marker = WebTimelineMarker(
                    timestamp=cycle.start_time,
                    frame_index=cycle.start_frame,
                    label=f"Cycle {cycle.cycle_id + 1}",
                    marker_type="cycle",
                    color=self.marker_colors['cycle'],
                    importance="high",
                    description=f"Movement cycle {cycle.cycle_id + 1} (Quality: {cycle.cycle_quality:.1f})"
                )
                timeline_markers.append(marker)
            
            #Add key event markers
            for event in segmented_motion.key_events:
                marker = WebTimelineMarker(
                    timestamp=event['timestamp'],
                    frame_index=event['frame_index'],
                    label=event['type'].replace('_', ' ').title(),
                    marker_type="event",
                    color=self.marker_colors['event'].get(event['type'], '#FFD700'),
                    importance=event.get('importance', 'medium'),
                    description=f"{event['type'].replace('_', ' ').title()} at {event['timestamp']:.2f}s"
                )
                timeline_markers.append(marker)
        
        return WebPlaybackControl(
            loop_points=loop_points,
            phase_controls=phase_controls,
            cycle_controls=cycle_controls,
            speed_presets=[0.25, 0.5, 1.0, 2.0, 4.0],
            timeline_markers=timeline_markers
        )
    

    #Create Three.js visualizations config
    def _create_visualization_config(self, sport_type: str, world_scale: float) -> WebVisualizationConfig:
        
        #Sport-specific camera positioning
        sport_cameras = {
            'soccer': {
                'position': [world_scale * 0.8, world_scale * 0.6, world_scale * 1.2],
                'target': [0, -world_scale * 0.2, 0]
            },
            'basketball': {
                'position': [world_scale * 0.6, world_scale * 0.8, world_scale * 1.0],
                'target': [0, world_scale * 0.1, 0]
            },
            'general': {
                'position': [world_scale * 0.7, world_scale * 0.5, world_scale * 1.1],
                'target': [0, 0, 0]
            }
        }
        
        camera_config = sport_cameras.get(sport_type, sport_cameras['general'])
        
        #Scale-aware configuration
        viz_config = self.default_visualization_config.copy()
        viz_config['camera_position'] = camera_config['position']
        viz_config['camera_target'] = camera_config['target']
        
        #Scale skeleton and object sizes
        viz_config['skeleton_config']['joint_radius'] = world_scale * 0.02
        viz_config['skeleton_config']['bone_thickness'] = world_scale * 0.015
        
        return WebVisualizationConfig(
            camera_position=viz_config['camera_position'],
            camera_target=viz_config['camera_target'],
            lighting_config=viz_config['lighting_config'],
            skeleton_config=viz_config['skeleton_config'],
            object_config=viz_config['object_config'],
            animation_config=viz_config['animation_config']
        )
    
    def export_web_model(self, web_model: WebSportsModel, output_path: str) -> Dict[str, Any]:
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        #Convert to dictionary
        model_dict = self._web_model_to_dict(web_model)

        #Convert numpy types to JSON-serializable types
        model_dict = self._convert_numpy_types(model_dict)
        
        #Create JSON string
        if self.optimize_for_mobile:
            # More aggressive compression for mobile
            json_string = json.dumps(model_dict, separators=(',', ':'))
        else:
            json_string = json.dumps(model_dict, separators=(',', ':'))
        
        #Calculate sizes
        uncompressed_size = len(json_string.encode('utf-8'))
        
        if self.compress_output:
            #Save compressed version
            compressed_path = output_path.with_suffix(output_path.suffix + '.gz')
            with gzip.open(compressed_path, 'wt', encoding='utf-8') as f:
                f.write(json_string)
            compressed_size = compressed_path.stat().st_size
            
            print(f"Web model exported (compressed): {compressed_path}")
            print(f"    Original size: {uncompressed_size:,} bytes ({uncompressed_size/1024:.1f} KB)")
            print(f"    Compressed size: {compressed_size:,} bytes ({compressed_size/1024:.1f} KB)")
            print(f"    Compression ratio: {(1-compressed_size/uncompressed_size)*100:.1f}%")
            
            export_info = {
                "compressed_path": str(compressed_path),
                "uncompressed_size": uncompressed_size,
                "compressed_size": compressed_size,
                "compression_ratio": (1-compressed_size/uncompressed_size),
                "web_optimized": True
            }
        else:
            #Save uncompressed version
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(model_dict, f, indent=2)
            
            print(f"Web model exported: {output_path}")
            print(f"    File size: {uncompressed_size:,} bytes ({uncompressed_size/1024:.1f} KB)")
            
            export_info = {
                "output_path": str(output_path),
                "file_size": uncompressed_size,
                "web_optimized": True
            }
        
        return export_info

    #Convert websportmode to a dictonary for json serialization
    def _web_model_to_dict(self, web_model: WebSportsModel) -> Dict[str, Any]: 
        
        model_dict = {
            "format": "three-js-sports-model",
            "version": "2.0",
            "metadata": asdict(web_model.metadata),
            "human_animation": web_model.human_animation,
            "object_animation": web_model.object_animation,
            "interaction_events": web_model.interaction_events,
            "playback_controls": {
                "loop_points": web_model.playback_controls.loop_points,
                "phase_controls": web_model.playback_controls.phase_controls,
                "cycle_controls": web_model.playback_controls.cycle_controls,
                "speed_presets": web_model.playback_controls.speed_presets,
                "timeline_markers": [asdict(marker) for marker in web_model.playback_controls.timeline_markers] if web_model.playback_controls.timeline_markers else []
            },
            "visualization_config": asdict(web_model.visualization_config),
        }
        
        #TODO
        # Add coaching data if included
        # if self.include_coaching_data and web_model.coaching_data:
        #     model_dict["coaching_data"] = web_model.coaching_data
        
        return model_dict
    
    #Convert numpy types to JSON serializable Python types
    def _convert_numpy_types(self, obj):
        """Convert numpy types to JSON-serializable Python types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
        

    #Create configuration for Three.js loader
    def create_three_js_loader_config(self, web_model: WebSportsModel) -> Dict[str, Any]:
        
        return {
            "loader_version": "1.0",
            "model_format": "three-js-sports-model",
            "loading_priority": {
                "skeleton": "high",
                "objects": "medium", 
                "interactions": "low",
                "coaching_data": "background"
            },
            "render_order": ["skeleton", "objects", "interactions"],
            "animation_settings": {
                "auto_start": web_model.metadata.auto_loop,
                "default_speed": web_model.metadata.default_playback_speed,
                "interpolation": "linear"
            },
            "camera_defaults": {
                "position": web_model.visualization_config.camera_position,
                "target": web_model.visualization_config.camera_target,
                "controls": "orbit"
            },
            "ui_elements": {
                "timeline": True,
                "phase_selector": len(web_model.playback_controls.phase_controls) > 0,
                "cycle_selector": len(web_model.playback_controls.cycle_controls) > 0,
                "speed_control": True,
                "loop_control": True
            }
        }