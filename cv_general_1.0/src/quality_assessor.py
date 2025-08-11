import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import cv2

from src.pose_detector import PoseFrame
from src.object_detector import ObjectFrame
from src.coordinate_3d_generator import Pose3D
from src.data_fusion import FusedSequence

#Quality assesment metrics for process video
@dataclass
class QualityMetrics:
    overall_score: float #0-100 overall quality score
    processing_success: bool #Whether processing was successful
    confidence_level: str #"excellent", "good", "fair", "poor"
    
    #Detailed metrics
    pose_quality: Dict[str, float]
    object_quality: Dict[str, float]
    fusion_quality: Dict[str, float]
    video_quality: Dict[str, float]
    movement_quality: Dict[str, float]
    
    #Issues and recommendations
    warnings: List[str]
    recommendations: List[str]
    processing_time: float

#Video suitabiliuty assessment before processing
@dataclass
class VideoSuitability: 
    is_suitable: bool
    suitability_score: float  # 0-100
    issues: List[str]
    recommendations: List[str]

#Assesses video processing quality and video suitability for pipeline
class QualityAssessor: 
    
    def __init__(self,
                 min_resolution=(720, 480),    # Minimum width x height
                 min_fps=24,                   # Minimum frame rate
                 min_duration=0.5,             # Minimum duration in seconds
                 max_duration=30.0,            # Maximum duration in seconds
                 min_movement_threshold=0.05): # Minimum movement range in meters
        
        self.min_resolution = min_resolution
        self.min_fps = min_fps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_movement_threshold = min_movement_threshold
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 85,
            'good': 70,
            'fair': 50,
            'poor': 0
        }
        
        # Weight factors for overall score
        self.score_weights = {
            'pose_quality': 0.30,      #30% - Human pose detection quality
            'object_quality': 0.20,    #20% - Object detection quality
            'fusion_quality': 0.20,    #20% - Data fusion success
            'video_quality': 0.15,     #15% - Raw video quality
            'movement_quality': 0.15   #15% - Movement analysis
        }
    
    #Assess if video is suitable for processing before running the pipeline
    def assess_video_suitability(self, video_path: str, video_metadata: Dict) -> VideoSuitability:
        
        print(f"Assessing video suitability: {video_path}")
        
        issues = []
        recommendations = []
        score_components = []
        
        #Resolution check
        width, height = video_metadata.get('width', 0), video_metadata.get('height', 0)
        if width >= self.min_resolution[0] and height >= self.min_resolution[1]:
            score_components.append(25)  #Good resolution
        else:
            issues.append(f"Resolution {width}x{height} below minimum {self.min_resolution[0]}x{self.min_resolution[1]}")
            recommendations.append("Use higher resolution camera (720p minimum)")
            score_components.append(0)
        
        #Frame rate check  
        fps = video_metadata.get('fps', 0)
        if fps >= self.min_fps:
            score_components.append(25)  #Good frame rate
        else:
            issues.append(f"Frame rate {fps} below minimum {self.min_fps}")
            recommendations.append("Record at higher frame rate (24fps minimum)")
            score_components.append(0)
        
        #Duration check
        duration = video_metadata.get('duration', 0)
        if self.min_duration <= duration <= self.max_duration:
            score_components.append(25) #Good duration
        elif duration < self.min_duration:
            issues.append(f"Video too short: {duration:.2f}s (minimum {self.min_duration}s)")
            recommendations.append("Record longer videos to capture complete movements")
            score_components.append(5)
        else:
            issues.append(f"Video too long: {duration:.2f}s (maximum {self.max_duration}s)")
            recommendations.append("Trim video to focus on specific skill execution")
            score_components.append(15)
        
        #File format check (basic)
        if video_path.lower().endswith(('.mp4', '.avi', '.mov')):
            score_components.append(25) #Good format
        else:
            issues.append("Unsupported video format")
            recommendations.append("Use MP4, AVI, or MOV format")
            score_components.append(10)
        
        suitability_score = sum(score_components)
        is_suitable = suitability_score >= 60 and len([i for i in issues if 'below minimum' in i]) == 0
        
        return VideoSuitability(
            is_suitable=is_suitable,
            suitability_score=suitability_score,
            issues=issues,
            recommendations=recommendations
        )
    
    #Comprehensive quality assessment of processed video data
    def assess_processing_quality(self, poses_3d: List[Pose3D], object_frames: List[ObjectFrame], fused_sequence: Optional[FusedSequence], video_metadata: Dict, processing_time: float) -> QualityMetrics:
        
        print(f"Assessing processing quality...")
        
        #Individual quality assessments
        pose_quality = self._assess_pose_quality(poses_3d)
        object_quality = self._assess_object_quality(object_frames)
        fusion_quality = self._assess_fusion_quality(fused_sequence)
        video_quality = self._assess_video_technical_quality(video_metadata)
        movement_quality = self._assess_movement_quality(poses_3d)
        
        #Calculate weighted overall score
        overall_score = (
            pose_quality['overall_score'] * self.score_weights['pose_quality'] +
            object_quality['overall_score'] * self.score_weights['object_quality'] +
            fusion_quality['overall_score'] * self.score_weights['fusion_quality'] +
            video_quality['overall_score'] * self.score_weights['video_quality'] +
            movement_quality['overall_score'] * self.score_weights['movement_quality']
        )
        
        #Determine confidence level
        confidence_level = self._determine_confidence_level(overall_score)
        
        #Processing success determination
        processing_success = overall_score >= 50 and pose_quality['overall_score'] >= 40
        
        #Collect warnings and recommendations
        warnings = []
        recommendations = []
        
        for quality_dict in [pose_quality, object_quality, fusion_quality, video_quality, movement_quality]:
            warnings.extend(quality_dict.get('warnings', []))
            recommendations.extend(quality_dict.get('recommendations', []))
        
        return QualityMetrics(
            overall_score=overall_score,
            processing_success=processing_success,
            confidence_level=confidence_level,
            pose_quality=pose_quality,
            object_quality=object_quality,
            fusion_quality=fusion_quality,
            video_quality=video_quality,
            movement_quality=movement_quality,
            warnings=warnings,
            recommendations=recommendations,
            processing_time=processing_time
        )
    
    #Assess human pose detection quality
    def _assess_pose_quality(self, poses_3d: List[Pose3D]) -> Dict[str, Any]: 
        
        if not poses_3d:
            return {
                'overall_score': 0,
                'detection_rate': 0,
                'avg_confidence': 0,
                'key_points_quality': {},
                'warnings': ['No human poses detected'],
                'recommendations': ['Ensure person is clearly visible in frame']
            }
        
        #Calculate average confidence across all keypoints
        all_confidences = []
        keypoint_confidences = {}
        
        #Key sports-relevant keypoints
        key_indices = {
            'wrists': [15, 16], #left, right
            'ankles': [27, 28], #left, right
            'knees': [25, 26], #left, right
            'shoulders': [11, 12] #left, right
        }
        
        for pose in poses_3d:
            for i, keypoint in enumerate(pose.keypoints_3d):
                all_confidences.append(keypoint.confidence)
                
                #Categorize key sports points
                for category, indices in key_indices.items():
                    if i in indices:
                        if category not in keypoint_confidences:
                            keypoint_confidences[category] = []
                        keypoint_confidences[category].append(keypoint.confidence)
        
        avg_confidence = np.mean(all_confidences) if all_confidences else 0
        detection_rate = len(poses_3d) / max(len(poses_3d), 1)  #All frames with poses
        
        #Key points quality
        key_points_quality = {}
        for category, confidences in keypoint_confidences.items():
            key_points_quality[category] = np.mean(confidences) if confidences else 0
        
        #Overall pose score
        pose_score = (avg_confidence * 60) + (detection_rate * 40)
        
        #Warnings and recommendations
        warnings = []
        recommendations = []
        
        if avg_confidence < 0.7:
            warnings.append(f"Low average pose confidence: {avg_confidence:.2f}")
            recommendations.append("Improve lighting and ensure person is clearly visible")
        
        if key_points_quality.get('ankles', 0) < 0.6:
            warnings.append("Poor ankle tracking quality")
            recommendations.append("Ensure feet are visible and not obscured")
        
        return {
            'overall_score': pose_score,
            'detection_rate': detection_rate,
            'avg_confidence': avg_confidence,
            'key_points_quality': key_points_quality,
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    #Assess object detection quality
    def _assess_object_quality(self, object_frames: List[ObjectFrame]) -> Dict[str, Any]: 
        
        if not object_frames:
            return {
                'overall_score': 30,  #Not critical failure if no objects
                'detection_rate': 0,
                'avg_confidence': 0,
                'object_consistency': 0,
                'warnings': ['No objects detected'],
                'recommendations': ['Ensure sports equipment is visible in frame']
            }
        
        #Calculate detection statistics
        frames_with_objects = len([f for f in object_frames if f.detected_objects])
        detection_rate = frames_with_objects / len(object_frames)
        
        #Average confidence
        all_object_confidences = []
        object_types = {}
        
        for frame in object_frames:
            for obj in frame.detected_objects:
                all_object_confidences.append(obj.confidence)
                
                if obj.class_name not in object_types:
                    object_types[obj.class_name] = []
                object_types[obj.class_name].append(obj.confidence)
        
        avg_confidence = np.mean(all_object_confidences) if all_object_confidences else 0
        
        #Object consistency (same objects detected across frames)
        most_common_object = max(object_types.keys(), key=lambda x: len(object_types[x])) if object_types else None
        consistency = len(object_types[most_common_object]) / len(object_frames) if most_common_object else 0
        
        #Overall object score
        object_score = (detection_rate * 40) + (avg_confidence * 40) + (consistency * 20)
        
        #Warnings and recommendations
        warnings = []
        recommendations = []
        
        if detection_rate < 0.5:
            warnings.append(f"Low object detection rate: {detection_rate:.2f}")
            recommendations.append("Ensure sports equipment is clearly visible throughout video")
        
        if avg_confidence < 0.6:
            warnings.append(f"Low object detection confidence: {avg_confidence:.2f}")
            recommendations.append("Improve object visibility and contrast with background")
        
        return {
            'overall_score': object_score,
            'detection_rate': detection_rate,
            'avg_confidence': avg_confidence,
            'object_consistency': consistency,
            'object_types': list(object_types.keys()),
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    #Assess data fusion quality
    def _assess_fusion_quality(self, fused_sequence: Optional[FusedSequence]) -> Dict[str, Any]: 
        
        if not fused_sequence:
            return {
                'overall_score': 0,
                'fusion_success_rate': 0,
                'interaction_count': 0,
                'quality_distribution': {},
                'warnings': ['Data fusion failed'],
                'recommendations': ['Check pose and object detection quality']
            }
        
        frames = fused_sequence.frames
        total_frames = len(frames)
        
        if total_frames == 0:
            return {
                'overall_score': 0,
                'fusion_success_rate': 0,
                'interaction_count': 0,
                'quality_distribution': {},
                'warnings': ['No fused frames generated'],
                'recommendations': ['Verify input data quality']
            }
        
        #Quality distribution
        quality_counts = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        fusion_confidences = []
        
        for frame in frames:
            quality_counts[frame.frame_quality] += 1
            fusion_confidences.append(frame.fusion_confidence)
        
        #Success rate (excellent + good frames)
        fusion_success_rate = (quality_counts['excellent'] + quality_counts['good']) / total_frames
        
        #Interaction analysis
        interaction_count = fused_sequence.interaction_analysis.get('total_interactions', 0)
        
        #Overall fusion score
        avg_fusion_confidence = np.mean(fusion_confidences) if fusion_confidences else 0
        fusion_score = (fusion_success_rate * 60) + (avg_fusion_confidence * 40)
        
        #Warnings and recommendations
        warnings = []
        recommendations = []
        
        if fusion_success_rate < 0.6:
            warnings.append(f"Low fusion success rate: {fusion_success_rate:.2f}")
            recommendations.append("Improve pose and object detection quality")
        
        if interaction_count == 0:
            warnings.append("No human-object interactions detected")
            recommendations.append("Ensure video shows actual skill performance with object contact")
        
        return {
            'overall_score': fusion_score,
            'fusion_success_rate': fusion_success_rate,
            'avg_confidence': avg_fusion_confidence,
            'interaction_count': interaction_count,
            'quality_distribution': quality_counts,
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    #Assess technical video quality
    def _assess_video_technical_quality(self, video_metadata: Dict) -> Dict[str, Any]: 
        
        #Extract metadata
        width = video_metadata.get('width', 0)
        height = video_metadata.get('height', 0)
        fps = video_metadata.get('fps', 0)
        duration = video_metadata.get('duration', 0)
        
        #Resolution score
        total_pixels = width * height
        if total_pixels >= 1920 * 1080: #1080p+
            resolution_score = 100
        elif total_pixels >= 1280 * 720: #720p+
            resolution_score = 75
        elif total_pixels >= 640 * 480: #480p+
            resolution_score = 50
        else:
            resolution_score = 25
        
        #Frame rate score
        if fps >= 60:
            fps_score = 100
        elif fps >= 30:
            fps_score = 75
        elif fps >= 24:
            fps_score = 50
        else:
            fps_score = 25
        
        #Duration score
        if 1.0 <= duration <= 10.0: #Ideal range
            duration_score = 100
        elif 0.5 <= duration <= 20.0: #Acceptable range
            duration_score = 75
        else:
            duration_score = 25
        
        #Overall video quality score
        video_score = (resolution_score * 0.4) + (fps_score * 0.4) + (duration_score * 0.2)
        
        #Warnings and recommendations
        warnings = []
        recommendations = []
        
        if resolution_score < 75:
            warnings.append(f"Low resolution: {width}x{height}")
            recommendations.append("Use 720p or higher resolution")
        
        if fps_score < 50:
            warnings.append(f"Low frame rate: {fps} fps")
            recommendations.append("Record at 30fps or higher for sports analysis")
        
        return {
            'overall_score': video_score,
            'resolution_score': resolution_score,
            'fps_score': fps_score,
            'duration_score': duration_score,
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    #Assess movement characteristics and quality
    def _assess_movement_quality(self, poses_3d: List[Pose3D]) -> Dict[str, Any]: 
        
        if len(poses_3d) < 2:
            return {
                'overall_score': 0,
                'movement_range': 0,
                'movement_smoothness': 0,
                'activity_level': 'static',
                'warnings': ['Insufficient frames for movement analysis'],
                'recommendations': ['Record longer videos with clear movement']
            }
        
        #Calculate movement ranges for key body parts
        key_points = {'right_ankle': 28, 'left_ankle': 27, 'right_wrist': 16, 'left_wrist': 15}
        movement_ranges = {}
        movement_velocities = {}
        
        for name, idx in key_points.items():
            positions = []
            for pose in poses_3d:
                if idx < len(pose.keypoints_3d):
                    kp = pose.keypoints_3d[idx]
                    positions.append([kp.x, kp.y, kp.z])
            
            if positions:
                positions = np.array(positions)
                #Calculate range (max - min for each axis)
                range_3d = np.max(positions, axis=0) - np.min(positions, axis=0)
                movement_ranges[name] = np.linalg.norm(range_3d)
                
                #Calculate average velocity
                if len(positions) > 1:
                    velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                    movement_velocities[name] = np.mean(velocities)
        
        #Overall movement assessment
        max_movement_range = max(movement_ranges.values()) if movement_ranges else 0
        avg_velocity = np.mean(list(movement_velocities.values())) if movement_velocities else 0
        
        #Determine activity level
        if max_movement_range < self.min_movement_threshold:
            activity_level = 'static'
            movement_score = 10
        elif max_movement_range < 0.2:
            activity_level = 'minimal'
            movement_score = 30
        elif max_movement_range < 0.5:
            activity_level = 'moderate'
            movement_score = 60
        else:
            activity_level = 'dynamic'
            movement_score = 90
        
        #Smoothness assessment (lower velocity variance = smoother)
        smoothness_scores = []
        for name, velocities in movement_velocities.items():
            if len(poses_3d) > 3:
                #Calculate velocity variance as smoothness indicator
                all_velocities = []
                positions = []
                for pose in poses_3d:
                    if key_points[name] < len(pose.keypoints_3d):
                        kp = pose.keypoints_3d[key_points[name]]
                        positions.append([kp.x, kp.y, kp.z])
                
                if len(positions) > 2:
                    positions = np.array(positions)
                    velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                    velocity_variance = np.var(velocities)
                    #Lower variance = higher smoothness score
                    smoothness = max(0, 100 - (velocity_variance * 1000))
                    smoothness_scores.append(smoothness)
        
        movement_smoothness = np.mean(smoothness_scores) if smoothness_scores else 50
        
        #Combine scores
        final_movement_score = (movement_score * 0.6) + (movement_smoothness * 0.4)
        
        #Warnings and recommendations
        warnings = []
        recommendations = []
        
        if activity_level == 'static':
            warnings.append("Very little movement detected")
            recommendations.append("Ensure video captures actual skill performance with clear movement")
        
        if movement_smoothness < 30:
            warnings.append("Jerky or inconsistent movement detected")
            recommendations.append("Check for tracking errors or very rapid movements")
        
        return {
            'overall_score': final_movement_score,
            'movement_range': max_movement_range,
            'movement_smoothness': movement_smoothness,
            'activity_level': activity_level,
            'movement_ranges': movement_ranges,
            'avg_velocity': avg_velocity,
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    #Determine confidence level based on overall score
    def _determine_confidence_level(self, overall_score: float) -> str: 
        
        for level, threshold in sorted(self.quality_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if overall_score >= threshold:
                return level
        return 'poor'
    
    #Determione if video quality is sufficient for motion segmentation
    def should_proceed_with_segmentation(self, quality_metrics: QualityMetrics) -> Tuple[bool, str]: 
        
        #Minimum requirements for motion segmentation
        min_requirements = {
            'overall_score': 50,
            'processing_success': True,
            'movement_activity': ['moderate', 'dynamic'],
            'pose_confidence': 0.6
        }
        
        #Check each requirement
        if not quality_metrics.processing_success:
            return False, "Processing failed - cannot proceed with segmentation"
        
        if quality_metrics.overall_score < min_requirements['overall_score']:
            return False, f"Overall quality too low: {quality_metrics.overall_score:.1f}/100"
        
        if quality_metrics.movement_quality['activity_level'] not in min_requirements['movement_activity']:
            return False, f"Insufficient movement for segmentation: {quality_metrics.movement_quality['activity_level']}"
        
        if quality_metrics.pose_quality['avg_confidence'] < min_requirements['pose_confidence']:
            return False, f"Pose tracking quality too low: {quality_metrics.pose_quality['avg_confidence']:.2f}"
        
        return True, f"Quality sufficient for segmentation (Score: {quality_metrics.overall_score:.1f}/100)"
    
    #Generate human readable quality assessment report
    def generate_quality_report(self, quality_metrics: QualityMetrics) -> str: 
        
        report = []
        report.append("VIDEO PROCESSING QUALITY REPORT")
        report.append("=" * 100)
        
        #Overall assessment
        report.append(f"Overall Score: {quality_metrics.overall_score:.1f}/100 ({quality_metrics.confidence_level})")
        report.append(f"    Processing Success: {'Yes' if quality_metrics.processing_success else 'No'}")
        report.append(f"    Processing Time: {quality_metrics.processing_time:.2f}s")
        report.append("")
        
        #Detailed breakdown
        report.append("DETAILED BREAKDOWN:")
        report.append(f"    Pose Quality: {quality_metrics.pose_quality['overall_score']:.1f}/100")
        report.append(f"    Object Quality: {quality_metrics.object_quality['overall_score']:.1f}/100")
        report.append(f"    Fusion Quality: {quality_metrics.fusion_quality['overall_score']:.1f}/100")
        report.append(f"    Video Quality: {quality_metrics.video_quality['overall_score']:.1f}/100")
        report.append(f"    Movement Quality: {quality_metrics.movement_quality['overall_score']:.1f}/100")
        report.append("")
        
        #Warnings
        if quality_metrics.warnings:
            report.append("WARNINGS:")
            for warning in quality_metrics.warnings:
                report.append(f"   • {warning}")
            report.append("")
        
        #Recommendations
        if quality_metrics.recommendations:
            report.append("RECOMMENDATIONS:")
            for rec in quality_metrics.recommendations:
                report.append(f"    - {rec}")
            report.append("")
        
        # Segmentation readiness
        can_segment, reason = self.should_proceed_with_segmentation(quality_metrics)
        report.append(f"Motion Segmentation: {'Ready' if can_segment else 'Not Ready'}")
        report.append(f"    {reason}")
        
        return "\n".join(report)