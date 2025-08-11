import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import scipy.signal
from scipy.spatial.distance import euclidean
from dtaidistance import dtw

from src.data_fusion import FusedSequence, FusedFrame

#Single phase of movement
@dataclass
class MotionPhase:
    name: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    phase_type: str #"setup", "execution", "recovery", "transition"
    key_events: List[Dict[str, Any]] #Important moments within phase
    movement_intensity: float #0-1 movement intensity during phase
    primary_body_parts: List[str] #Main body parts active in this phase

#Complete movement cycle
@dataclass
class MovementCycle:
    cycle_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    phases: List[MotionPhase]
    cycle_quality: float #0-1 quality score for this cycle
    peak_moments: List[Dict[str, Any]] #Key moments like ball contact
    loop_points: Tuple[int, int] #Best start/end for seamless looping

#Complete motion segmentation results
@dataclass
class SegmentedMotion:
    total_duration: float
    total_frames: int
    movement_cycles: List[MovementCycle]
    overall_phases: List[MotionPhase] #Global phases across all cycles
    key_events: List[Dict[str, Any]] #All important events
    loop_recommendations: List[Dict[str, Any]] #Best loop points
    coaching_insights: Dict[str, Any] #Analysis for coaching

#Segments motion into phases and cycles for coaching analysis
class MotionSegmenter:
    
    def __init__(self,
                 velocity_threshold=0.05, #m/frame minimum for "movement"
                 phase_min_duration=0.1, #Minimum phase duration in seconds
                 cycle_detection_method='velocity', #'velocity', 'dtw', 'periodicity'
                 smoothing_window=5): #Frames for velocity smoothing
        
        self.velocity_threshold = velocity_threshold
        self.phase_min_duration = phase_min_duration
        self.cycle_detection_method = cycle_detection_method
        self.smoothing_window = smoothing_window
        
        #Key body parts for different sports
        self.sports_key_points = {
            'soccer': {
                'primary': ['right_ankle', 'left_ankle'],
                'secondary': ['right_knee', 'left_knee'],
                'indices': [27, 28, 25, 26]  #left_ankle, right_ankle, left_knee, right_knee
            },
            'basketball': {
                'primary': ['right_wrist', 'left_wrist'],
                'secondary': ['right_elbow', 'left_elbow'],
                'indices': [15, 16, 13, 14]  #left_wrist, right_wrist, left_elbow, right_elbow
            },
            'general': {
                'primary': ['right_ankle', 'left_ankle', 'right_wrist', 'left_wrist'],
                'secondary': ['right_knee', 'left_knee'],
                'indices': [27, 28, 15, 16, 25, 26]
            }
        }
    
    #Segment motion into phases and cycles
    def segment_motion(self, fused_sequence: FusedSequence, sport_type: str = 'soccer') -> SegmentedMotion:
        
        if not fused_sequence.frames:
            raise ValueError("No frames provided for motion segmentation")
        
        print(f"Segmenting motion for {sport_type} analysis...")
        print(f"    Frames: {len(fused_sequence.frames)}")
        print(f"    Duration: {fused_sequence.frames[-1].timestamp - fused_sequence.frames[0].timestamp:.2f}s")
        
        #Extract movement data
        movement_data = self._extract_movement_data(fused_sequence.frames, sport_type)
        
        #Detect movement phases
        phases = self._detect_movement_phases(movement_data, fused_sequence.frames)
        
        #Detect movement cycles
        cycles = self._detect_movement_cycles(movement_data, fused_sequence.frames, phases)
        
        #Identify key events
        key_events = self._identify_key_events(fused_sequence, movement_data)
        
        #Find optimal loop points
        loop_recommendations = self._find_loop_points(movement_data, cycles)
    
        #Generate coaching insights
        coaching_insights = self._generate_coaching_insights(cycles, phases, key_events)
        
        total_duration = fused_sequence.frames[-1].timestamp - fused_sequence.frames[0].timestamp
        
        segmented_motion = SegmentedMotion(
            total_duration=total_duration,
            total_frames=len(fused_sequence.frames),
            movement_cycles=cycles,
            overall_phases=phases,
            key_events=key_events,
            loop_recommendations=loop_recommendations,
            coaching_insights=coaching_insights
        )
        
        print(f"Motion segmentation complete!")
        print(f"    Movement cycles: {len(cycles)}")
        print(f"    Movement phases: {len(phases)}")
        print(f"    Key events: {len(key_events)}")
        print(f"    Loop points: {len(loop_recommendations)}")
        
        return segmented_motion
    
    #Extract relevant movement data for analysis
    def _extract_movement_data(self, frames: List[FusedFrame], sport_type: str) -> Dict[str, Any]:
        
        sport_config = self.sports_key_points.get(sport_type, self.sports_key_points['general'])
        
        #Extract positions for key body parts
        positions = {}
        velocities = {}
        accelerations = {}
        timestamps = []
        
        #Initialize position arrays
        for name in sport_config['primary'] + sport_config['secondary']:
            positions[name] = []
        
        #Extract data frame by frame
        for frame in frames:
            timestamps.append(frame.timestamp)
            
            if frame.human_pose_3d:
                pose = frame.human_pose_3d
                
                #Map keypoint names to indices
                keypoint_mapping = {
                    'left_ankle': 27, 'right_ankle': 28,
                    'left_knee': 25, 'right_knee': 26,
                    'left_wrist': 15, 'right_wrist': 16,
                    'left_elbow': 13, 'right_elbow': 14
                }
                
                for name in positions.keys():
                    if name in keypoint_mapping:
                        idx = keypoint_mapping[name]
                        if idx < len(pose.keypoints_3d):
                            kp = pose.keypoints_3d[idx]
                            positions[name].append([kp.x, kp.y, kp.z])
                        else:
                            positions[name].append([0, 0, 0])
                    else:
                        positions[name].append([0, 0, 0])
            else:
                #No pose data for this frame
                for name in positions.keys():
                    positions[name].append([0, 0, 0])
        
        #Convert to numpy arrays
        for name in positions.keys():
            positions[name] = np.array(positions[name])
        
        timestamps = np.array(timestamps)
        
        #Calculate velocities and accelerations
        for name in positions.keys():
            pos = positions[name]
            
            if len(pos) > 1:
                #Velocity: change in position over time
                vel = np.zeros_like(pos)
                vel[1:] = np.diff(pos, axis=0) / np.diff(timestamps).reshape(-1, 1)
                vel[0] = vel[1]  #Set first frame velocity
                
                #Smooth velocities
                if len(vel) >= self.smoothing_window:
                    for axis in range(3):
                        vel[:, axis] = scipy.signal.savgol_filter(
                            vel[:, axis], 
                            window_length=min(self.smoothing_window, len(vel)),
                            polyorder=min(2, min(self.smoothing_window, len(vel)) - 1)
                        )
                
                velocities[name] = vel
                
                #Acceleration: change in velocity over time
                if len(vel) > 1:
                    acc = np.zeros_like(vel)
                    acc[1:] = np.diff(vel, axis=0) / np.diff(timestamps).reshape(-1, 1)
                    acc[0] = acc[1]  #Set first frame acceleration
                    accelerations[name] = acc
                else:
                    accelerations[name] = np.zeros_like(vel)
            else:
                velocities[name] = np.zeros_like(pos)
                accelerations[name] = np.zeros_like(pos)
        
        return {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'timestamps': timestamps,
            'sport_config': sport_config
        }
    
    #Detect distinct movement phases based on velocity patterns
    def _detect_movement_phases(self, movement_data: Dict, frames: List[FusedFrame]) -> List[MotionPhase]:
        
        timestamps = movement_data['timestamps']
        velocities = movement_data['velocities']
        sport_config = movement_data['sport_config']
        
        #Calculate overall movement intensity
        movement_intensity = []
        
        for i in range(len(timestamps)):
            frame_intensity = 0
            for name in sport_config['primary']:
                if name in velocities:
                    vel_magnitude = np.linalg.norm(velocities[name][i])
                    frame_intensity += vel_magnitude
            
            movement_intensity.append(frame_intensity / len(sport_config['primary']))
        
        movement_intensity = np.array(movement_intensity)
        
        #Smooth intensity
        if len(movement_intensity) >= self.smoothing_window:
            movement_intensity = scipy.signal.savgol_filter(
                movement_intensity,
                window_length=min(self.smoothing_window, len(movement_intensity)),
                polyorder=min(2, min(self.smoothing_window, len(movement_intensity)) - 1)
            )
        
        #Find peaks and valleys in movement intensity
        peaks, _ = scipy.signal.find_peaks(movement_intensity, height=self.velocity_threshold)
        valleys, _ = scipy.signal.find_peaks(-movement_intensity)
        
        #Combine and sort transition points
        transitions = sorted(list(peaks) + list(valleys))
        
        #Create phases between transitions
        phases = []
        phase_id = 0
        
        for i in range(len(transitions) - 1):
            start_idx = transitions[i]
            end_idx = transitions[i + 1]
            
            if end_idx - start_idx < 3: #Skip very short phases
                continue
            
            duration = timestamps[end_idx] - timestamps[start_idx]
            
            if duration < self.phase_min_duration: #Skip phases that are too short
                continue
            
            #Determine phase type based on movement intensity
            avg_intensity = np.mean(movement_intensity[start_idx:end_idx])
            
            if avg_intensity > np.percentile(movement_intensity, 75):
                phase_type = "execution"
                phase_name = f"High Activity Phase {phase_id + 1}"
            elif avg_intensity > np.percentile(movement_intensity, 25):
                phase_type = "transition"
                phase_name = f"Moderate Activity Phase {phase_id + 1}"
            else:
                phase_type = "setup"
                phase_name = f"Low Activity Phase {phase_id + 1}"
            
            # Identify key events within phase
            key_events = []
            if start_idx in peaks:
                key_events.append({
                    'type': 'movement_peak',
                    'timestamp': timestamps[start_idx],
                    'frame_index': start_idx,
                    'intensity': movement_intensity[start_idx]
                })
            
            #   Primary body parts for this phase
            primary_parts = []
            for name in sport_config['primary']:
                if name in velocities:
                    avg_vel = np.mean(np.linalg.norm(velocities[name][start_idx:end_idx], axis=1))
                    if avg_vel > self.velocity_threshold:
                        primary_parts.append(name)
            
            phase = MotionPhase(
                name=phase_name,
                start_frame=start_idx,
                end_frame=end_idx,
                start_time=timestamps[start_idx],
                end_time=timestamps[end_idx],
                duration=duration,
                phase_type=phase_type,
                key_events=key_events,
                movement_intensity=avg_intensity,
                primary_body_parts=primary_parts
            )
            
            phases.append(phase)
            phase_id += 1
        
        return phases
    
    #Detect repeating movement cycles
    def _detect_movement_cycles(self, movement_data: Dict, frames: List[FusedFrame], phases: List[MotionPhase]) -> List[MovementCycle]:
        
        if self.cycle_detection_method == 'velocity':
            return self._detect_cycles_by_velocity(movement_data, frames, phases)
        elif self.cycle_detection_method == 'dtw':
            return self._detect_cycles_by_dtw(movement_data, frames, phases)
        else:
            return self._detect_cycles_by_periodicity(movement_data, frames, phases)
    
    #Detect cycles based on velocity patterns
    def _detect_cycles_by_velocity(self, movement_data: Dict, frames: List[FusedFrame], phases: List[MotionPhase]) -> List[MovementCycle]: 
        
        timestamps = movement_data['timestamps']
        velocities = movement_data['velocities']
        sport_config = movement_data['sport_config']
        
        #Calculate combined velocity magnitude for primary body parts
        combined_velocity = np.zeros(len(timestamps))
        
        for name in sport_config['primary']:
            if name in velocities:
                vel_magnitudes = np.linalg.norm(velocities[name], axis=1)
                combined_velocity += vel_magnitudes
        
        combined_velocity /= len(sport_config['primary'])
        
        #Find velocity peaks that might indicate cycle starts/ends
        peaks, peak_properties = scipy.signal.find_peaks(
            combined_velocity, 
            height=self.velocity_threshold * 2,
            distance=10  #Minimum 10 frames between peaks
        )
        
        cycles = []
        cycle_id = 0
        
        #Create cycles between significant peaks
        for i in range(len(peaks) - 1):
            start_idx = peaks[i]
            end_idx = peaks[i + 1]
            
            duration = timestamps[end_idx] - timestamps[start_idx]
            
            if duration < 0.2: #Skip very short cycles
                continue
            
            #Find phases within this cycle
            cycle_phases = []
            for phase in phases:
                if (phase.start_frame >= start_idx and phase.end_frame <= end_idx):
                    cycle_phases.append(phase)
            
            #Calculate cycle quality based on movement smoothness and intensity
            cycle_velocities = combined_velocity[start_idx:end_idx]
            velocity_variance = np.var(cycle_velocities)
            avg_intensity = np.mean(cycle_velocities)
            
            #Quality score (0-1): balance of intensity and smoothness
            quality = min(1.0, avg_intensity / (self.velocity_threshold * 5)) * max(0, 1 - velocity_variance * 10)
            
            #Find peak moments within cycle
            cycle_peaks = [p for p in peaks if start_idx <= p <= end_idx]
            peak_moments = []
            
            for peak_idx in cycle_peaks:
                peak_moments.append({
                    'type': 'velocity_peak',
                    'timestamp': timestamps[peak_idx],
                    'frame_index': peak_idx,
                    'intensity': combined_velocity[peak_idx]
                })
            
            #Determine best loop points (slightly before/after peaks for smooth transitions)
            loop_start = max(0, start_idx - 2)
            loop_end = min(len(timestamps) - 1, end_idx + 2)
            
            cycle = MovementCycle(
                cycle_id=cycle_id,
                start_frame=start_idx,
                end_frame=end_idx,
                start_time=timestamps[start_idx],
                end_time=timestamps[end_idx],
                duration=duration,
                phases=cycle_phases,
                cycle_quality=quality,
                peak_moments=peak_moments,
                loop_points=(loop_start, loop_end)
            )
            
            cycles.append(cycle)
            cycle_id += 1
        
        return cycles
    
    #Detect cycles using Dynamic time warping
    def _detect_cycles_by_dtw(self, movement_data: Dict, frames: List[FusedFrame], phases: List[MotionPhase]) -> List[MovementCycle]:
        # TODO
        # This would implement more sophisticated cycle detection using DTW
        # For now, fall back to velocity method
        return self._detect_cycles_by_velocity(movement_data, frames, phases)
    
    #Detect cycles based on periodic patterns
    def _detect_cycles_by_periodicity(self, movement_data: Dict, frames: List[FusedFrame], phases: List[MotionPhase]) -> List[MovementCycle]:
        
        #Simple periodicity detection - could be enhanced with FFT analysis
        return self._detect_cycles_by_velocity(movement_data, frames, phases)
    
    #Identify key events like ball contact, directrion changes
    def _identify_key_events(self, fused_sequence: FusedSequence, movement_data: Dict) -> List[Dict[str, Any]]:
        
        key_events = []
        timestamps = movement_data['timestamps']
        velocities = movement_data['velocities']
        
        #Extract interaction events from fusion data
        interactions = fused_sequence.interaction_analysis.get('interactions', [])
        
        for interaction in interactions:
            if interaction['interaction_type'] == 'foot_ball_contact':
                key_events.append({
                    'type': 'ball_contact',
                    'timestamp': interaction['timestamp'],
                    'frame_index': interaction['frame_index'],
                    'body_part': interaction['keypoint_name'],
                    'object': interaction['object_class'],
                    'distance': interaction['distance'],
                    'importance': 'high'
                })
        
        #Detect direction changes
        for name in ['right_ankle', 'left_ankle']:
            if name in velocities:
                vel = velocities[name]
                
                #Find significant direction changes
                for i in range(1, len(vel) - 1):
                    #Calculate angle between consecutive velocity vectors
                    v1 = vel[i - 1]
                    v2 = vel[i + 1]
                    
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        #Normalize vectors
                        v1_norm = v1 / np.linalg.norm(v1)
                        v2_norm = v2 / np.linalg.norm(v2)
                        
                        #Calculate angle
                        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
                        angle = np.arccos(dot_product)
                        
                        #Significant direction change (>90 degrees)
                        if angle > np.pi / 2:
                            key_events.append({
                                'type': 'direction_change',
                                'timestamp': timestamps[i],
                                'frame_index': i,
                                'body_part': name,
                                'angle_change': np.degrees(angle),
                                'importance': 'medium'
                            })
        
        #Sort events by timestamp
        key_events.sort(key=lambda x: x['timestamp'])
        
        return key_events
    
    #Find optimal points for seamless video looping
    def _find_loop_points(self, movement_data: Dict, cycles: List[MovementCycle]) -> List[Dict[str, Any]]: 
        
        loop_recommendations = []
        timestamps = movement_data['timestamps']
        velocities = movement_data['velocities']
        
        if not cycles:
            return loop_recommendations
        
        #For each cycle, find the best loop points
        for cycle in cycles:
            start_idx = cycle.start_frame
            end_idx = cycle.end_frame
            
            #Calculate position and velocity similarity at start and end
            position_similarity = 0
            velocity_similarity = 0
            valid_comparisons = 0
            
            for name in movement_data['sport_config']['primary']:
                if name in movement_data['positions'] and name in velocities:
                    start_pos = movement_data['positions'][name][start_idx]
                    end_pos = movement_data['positions'][name][end_idx]
                    start_vel = velocities[name][start_idx]
                    end_vel = velocities[name][end_idx]
                    
                    #Position similarity (closer = better for looping)
                    pos_dist = euclidean(start_pos, end_pos)
                    pos_sim = max(0, 1 - pos_dist)  # Normalize to 0-1
                    
                    #Velocity similarity (similar direction and magnitude = smoother loop)
                    vel_sim = 0
                    if np.linalg.norm(start_vel) > 0 and np.linalg.norm(end_vel) > 0:
                        start_vel_norm = start_vel / np.linalg.norm(start_vel)
                        end_vel_norm = end_vel / np.linalg.norm(end_vel)
                        vel_sim = max(0, np.dot(start_vel_norm, end_vel_norm))
                    
                    position_similarity += pos_sim
                    velocity_similarity += vel_sim
                    valid_comparisons += 1
            
            if valid_comparisons > 0:
                position_similarity /= valid_comparisons
                velocity_similarity /= valid_comparisons
                
                #Overall loop quality (weighted combination)
                loop_quality = (position_similarity * 0.6) + (velocity_similarity * 0.4)
                
                if loop_quality > 0.3:  #Only recommend decent loops
                    loop_recommendations.append({
                        'cycle_id': cycle.cycle_id,
                        'start_frame': cycle.loop_points[0],
                        'end_frame': cycle.loop_points[1],
                        'start_time': timestamps[cycle.loop_points[0]],
                        'end_time': timestamps[cycle.loop_points[1]],
                        'loop_quality': loop_quality,
                        'position_similarity': position_similarity,
                        'velocity_similarity': velocity_similarity,
                        'recommended': loop_quality > 0.6
                    })
        
        #Sort by quality (best first)
        loop_recommendations.sort(key=lambda x: x['loop_quality'], reverse=True)
        
        return loop_recommendations
    
    #Generate insights for coaching analysis
    def _generate_coaching_insights(self, cycles: List[MovementCycle], phases: List[MotionPhase], key_events: List[Dict]) -> Dict[str, Any]:
        
        insights = {
            'movement_summary': {},
            'phase_analysis': {},
            'cycle_analysis': {},
            'key_moments': {},
            'recommendations': []
        }
        
        #Movement summary
        if cycles:
            avg_cycle_duration = np.mean([c.duration for c in cycles])
            avg_cycle_quality = np.mean([c.cycle_quality for c in cycles])
            
            insights['movement_summary'] = {
                'total_cycles': len(cycles),
                'avg_cycle_duration': avg_cycle_duration,
                'avg_cycle_quality': avg_cycle_quality,
                'movement_consistency': 'high' if avg_cycle_quality > 0.7 else 'moderate' if avg_cycle_quality > 0.4 else 'low'
            }
        
        #Phase analysis
        if phases:
            phase_types = {}
            for phase in phases:
                if phase.phase_type not in phase_types:
                    phase_types[phase.phase_type] = []
                phase_types[phase.phase_type].append(phase.duration)
            
            for phase_type, durations in phase_types.items():
                insights['phase_analysis'][phase_type] = {
                    'count': len(durations),
                    'avg_duration': np.mean(durations),
                    'duration_consistency': np.std(durations)
                }
        
        #Cycle analysis
        if len(cycles) > 1:
            quality_trend = [c.cycle_quality for c in cycles]
            if len(quality_trend) > 2:
                #Simple trend analysis
                trend_slope = np.polyfit(range(len(quality_trend)), quality_trend, 1)[0]
                insights['cycle_analysis'] = {
                    'quality_trend': 'improving' if trend_slope > 0.05 else 'declining' if trend_slope < -0.05 else 'stable',
                    'best_cycle': max(cycles, key=lambda c: c.cycle_quality).cycle_id,
                    'quality_range': [min(quality_trend), max(quality_trend)]
                }
        
        #Key moments analysis
        ball_contacts = [e for e in key_events if e['type'] == 'ball_contact']
        direction_changes = [e for e in key_events if e['type'] == 'direction_change']
        
        insights['key_moments'] = {
            'ball_contacts': len(ball_contacts),
            'direction_changes': len(direction_changes),
            'events_per_second': len(key_events) / (key_events[-1]['timestamp'] - key_events[0]['timestamp']) if key_events else 0
        }
        
        #Recommendations
        if insights['movement_summary']:
            if insights['movement_summary']['avg_cycle_quality'] < 0.5:
                insights['recommendations'].append("Focus on smoother, more consistent movement patterns")
            
            if len(ball_contacts) == 0:
                insights['recommendations'].append("Ensure clear ball contact during skill execution")
            
            if insights['movement_summary']['total_cycles'] < 2:
                insights['recommendations'].append("Record longer sequences to capture multiple skill repetitions")
        
        return insights
    
    #Get the motion pahse at a specific timestamp
    def get_phase_at_timestamp(self, segmented_motion: SegmentedMotion, timestamp: float) -> Optional[MotionPhase]:
        """Get the motion phase at a specific timestamp"""
        
        for phase in segmented_motion.overall_phases:
            if phase.start_time <= timestamp <= phase.end_time:
                return phase
        
        return None
    
    #Get the movement cycle at a specific timestamp
    def get_cycle_at_timestamp(self, segmented_motion: SegmentedMotion, timestamp: float) -> Optional[MovementCycle]:
        
        for cycle in segmented_motion.movement_cycles:
            if cycle.start_time <= timestamp <= cycle.end_time:
                return cycle
        
        return None
    
    #Export segmentation data for web intergration
    def export_segmentation_data(self, segmented_motion: SegmentedMotion) -> Dict[str, Any]:
        
        #Convert dataclasses to dictionaries for JSON serialization
        export_data = {
            'metadata': {
                'total_duration': segmented_motion.total_duration,
                'total_frames': segmented_motion.total_frames,
                'cycles_count': len(segmented_motion.movement_cycles),
                'phases_count': len(segmented_motion.overall_phases),
                'key_events_count': len(segmented_motion.key_events)
            },
            'cycles': [],
            'phases': [],
            'key_events': segmented_motion.key_events,
            'loop_recommendations': segmented_motion.loop_recommendations,
            'coaching_insights': segmented_motion.coaching_insights
        }
        
        #Export cycles
        for cycle in segmented_motion.movement_cycles:
            cycle_data = {
                'cycle_id': cycle.cycle_id,
                'start_frame': cycle.start_frame,
                'end_frame': cycle.end_frame,
                'start_time': cycle.start_time,
                'end_time': cycle.end_time,
                'duration': cycle.duration,
                'cycle_quality': cycle.cycle_quality,
                'peak_moments': cycle.peak_moments,
                'loop_points': cycle.loop_points,
                'phases': []
            }
            
            #Add phases within this cycle
            for phase in cycle.phases:
                phase_data = {
                    'name': phase.name,
                    'start_frame': phase.start_frame,
                    'end_frame': phase.end_frame,
                    'start_time': phase.start_time,
                    'end_time': phase.end_time,
                    'duration': phase.duration,
                    'phase_type': phase.phase_type,
                    'movement_intensity': phase.movement_intensity,
                    'primary_body_parts': phase.primary_body_parts,
                    'key_events': phase.key_events
                }
                cycle_data['phases'].append(phase_data)
            
            export_data['cycles'].append(cycle_data)
        
        #Export overall phases
        for phase in segmented_motion.overall_phases:
            phase_data = {
                'name': phase.name,
                'start_frame': phase.start_frame,
                'end_frame': phase.end_frame,
                'start_time': phase.start_time,
                'end_time': phase.end_time,
                'duration': phase.duration,
                'phase_type': phase.phase_type,
                'movement_intensity': phase.movement_intensity,
                'primary_body_parts': phase.primary_body_parts,
                'key_events': phase.key_events
            }
            export_data['phases'].append(phase_data)
        
        return export_data