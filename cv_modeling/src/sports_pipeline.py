#!/usr/bin/env python3
"""
Complete Sports Video Analysis Pipeline
=======================================

Processes a single sports video through the complete pipeline:
1. Video preprocessing
2. Pose detection  
3. Object detection (YOLO)
4. Data smoothing
5. 3D coordinate generation
6. Data fusion
7. Quality assessment
8. Motion segmentation
9. Web integration

Output: Three.js-ready JSON file for web visualization
"""

import sys
import os
import time
import argparse
from pathlib import Path

#Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.video_processor import VideoProcessor
from src.pose_detector import PoseDetector
from src.data_smoother import DataSmoother
from src.coordinate_3d_generator import Coordinate3DGenerator
from src.object_detector import ObjectDetector
from src.data_fusion import DataFusion
from src.quality_assessor import QualityAssessor
from src.motion_segmenter import MotionSegmenter
from src.model_creator_fusion import ModelCreatorFusion
from src.web_integrator import WebIntegrator

#Complete sports video analysis pipeline
class SportsVideoPipeline:
    
    def __init__(self, 
                 sport_type='soccer',
                 reference_height=1.75,
                 output_dir='./output',
                 enable_yolo=True,
                 optimize_for_web=True):
        
        self.sport_type = sport_type
        self.reference_height = reference_height
        self.output_dir = Path(output_dir)
        self.enable_yolo = enable_yolo
        self.optimize_for_web = optimize_for_web
        
        #Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        #Initialize all components
        self.video_processor = None
        self.pose_detector = PoseDetector()
        self.smoother = DataSmoother()
        self.coord_3d = Coordinate3DGenerator(reference_height=reference_height)
        self.object_detector = None
        self.data_fusion = DataFusion(context_smoothing=True)
        self.quality_assessor = QualityAssessor()
        self.motion_segmenter = MotionSegmenter()
        self.fusion_creator = ModelCreatorFusion(optimize_for_web=optimize_for_web)
        self.web_integrator = WebIntegrator(optimize_for_mobile=False)
        
        #Initialize YOLO if enabled
        if self.enable_yolo:
            self.object_detector = ObjectDetector(
                model_size='n', #Fast nano model
                confidence_threshold=0.3,
                sports_objects_only=True,
                light_smoothing=True
            )
    
    #Process a single video through the complete pipeline
    def process_video(self, video_path: str, drill_name: str = None, max_frames: int = None) -> dict: 
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if drill_name is None:
            drill_name = video_path.stem
        
        print("SPORTS VIDEO ANALYSIS PIPELINE")
        print("=" * 100)
        print(f"Video: {video_path.name}")
        print(f"Sport: {self.sport_type}")
        print(f"Drill: {drill_name}")
        print(f"YOLO: {'Enabled' if self.enable_yolo else 'Disabled'}")
        print(f"Web Optimized: {'Yes' if self.optimize_for_web else 'No'}")
        if max_frames:
            print(f"Processing: First {max_frames} frames")
        print()
        
        pipeline_start = time.time()
        results = {}
        
        try:
            print("📥 STEP 1: Video Preprocessing")
            print("-" * 50)
            
            self.video_processor = VideoProcessor(str(video_path))
            if not self.video_processor.load_video():
                raise RuntimeError("Failed to load video")
            
            video_metadata = self.video_processor.metadata
            print(f"Video loaded: {video_metadata['width']}x{video_metadata['height']} @ {video_metadata['fps']} FPS")
            print(f"    Duration: {video_metadata['duration']:.2f}s ({video_metadata['frame_count']} frames)")
            
            #Check video suitability
            suitability = self.quality_assessor.assess_video_suitability(str(video_path), video_metadata)
            if not suitability.is_suitable:
                print("Video quality issues detected:")
                for issue in suitability.issues:
                    print(f"    - {issue}")
                print("Continuing anyway.")
            
            print()
            
            print("🤸 STEP 2: Human Pose Detection")
            print("-" * 50)
            
            pose_frames = []
            fps = video_metadata['fps']
            
            for frame_idx, frame in self.video_processor.get_frames(max_frames=max_frames):
                timestamp = frame_idx / fps
                pose_frame = self.pose_detector.process_frame(frame, frame_idx, timestamp)
                if pose_frame:
                    pose_frames.append(pose_frame)
            
            print(f"Pose detection complete: {len(pose_frames)} frames processed")
            if pose_frames:
                avg_confidence = sum(sum(kp.confidence for kp in frame.keypoints) for frame in pose_frames) / (len(pose_frames) * 33)
                print(f"    Average confidence: {avg_confidence:.3f}")
            print()
            
            object_frames = []
            if self.enable_yolo and self.object_detector:
                print("STEP 3: Object Detection (YOLO)")
                print("-" * 50)
                
                if not self.object_detector.initialize_model():
                    print("Failed to initialize YOLO - continuing without objects")
                    self.enable_yolo = False
                else:
                    for frame_idx, frame in self.video_processor.get_frames(max_frames=max_frames):
                        timestamp = frame_idx / fps
                        object_frame = self.object_detector.process_frame(frame, frame_idx, timestamp)
                        if object_frame:
                            enhanced_frame = self.object_detector.enhance_sports_detection(
                                object_frame, context_info={'sport_type': self.sport_type}
                            )
                            object_frames.append(enhanced_frame)
                    
                    #Apply tracking
                    if len(object_frames) > 2:
                        object_frames = self.object_detector.track_objects_across_frames(object_frames)
                    
                    total_detections = sum(len(f.detected_objects) for f in object_frames)
                    print(f"Object detection complete: {total_detections} objects in {len(object_frames)} frames")
                    
                    if object_frames:
                        stats = self.object_detector.get_detection_statistics(object_frames)
                        print(f"    Object types: {', '.join(stats['object_counts'].keys())}")
            else:
                print("STEP 3: Object Detection - SKIPPED")
                print("-" * 50)
                print("YOLO disabled or not available")
            
            print()
            
            print("STEP 4: Data Smoothing")
            print("-" * 50)
            
            if len(pose_frames) < 3:
                raise RuntimeError("Insufficient pose frames for processing")
            
            smoothed_poses = self.smoother.smooth_pose_sequence(pose_frames)
            print(f"Temporal smoothing complete: {len(smoothed_poses)} frames")
            print()
            
            print("STEP 5: 3D Coordinate Generation")
            print("-" * 50)
            
            poses_3d = self.coord_3d.convert_pose_sequence_to_3d(smoothed_poses)
            print(f"3D conversion complete: {len(poses_3d)} poses")
            print(f"    World scale: {poses_3d[0].world_scale:.3f} meters/unit")
            print()
            
            print("STEP 6: Data Fusion")
            print("-" * 50)
            
            fused_sequence = self.data_fusion.fuse_sequences(
                poses_3d=poses_3d,
                object_frames=object_frames,
                drill_name=drill_name
            )
            print(f"Data fusion complete: {len(fused_sequence.frames)} fused frames")
            print(f"    Interactions detected: {fused_sequence.interaction_analysis.get('total_interactions', 0)}")
            print()
            
            
            print("STEP 7: Quality Assessment")
            print("-" * 50)
            
            processing_time = time.time() - pipeline_start
            quality_metrics = self.quality_assessor.assess_processing_quality(
                poses_3d, object_frames, fused_sequence, video_metadata, processing_time
            )
            
            print(f"Quality assessment complete")
            print(f"    Overall score: {quality_metrics.overall_score:.1f}/100 ({quality_metrics.confidence_level})")
            print(f"    Processing success: {'Yes' if quality_metrics.processing_success else 'No'}")
            
            #Check if we should continue with segmentation
            can_segment, reason = self.quality_assessor.should_proceed_with_segmentation(quality_metrics)
            if not can_segment:
                print(f"    {reason}")
                print(" Skipping motion segmentation")
                segmented_motion = None
            else:
                print(f"Quality sufficient for segmentation")
            print()
            
            segmented_motion = None
            if can_segment:
                print("STEP 8: Motion Segmentation")
                print("-" * 50)
                
                segmented_motion = self.motion_segmenter.segment_motion(fused_sequence, self.sport_type)
                print(f"Motion segmentation complete")
                print(f"    Movement cycles: {len(segmented_motion.movement_cycles)}")
                print(f"    Movement phases: {len(segmented_motion.overall_phases)}")
                print(f"    Key events: {len(segmented_motion.key_events)}")
                print()
                
            print("STEP 9: Fusion Model Creation")
            print("-" * 50)
            
            fused_model = self.fusion_creator.create_fused_3d_model(fused_sequence)
            
            #Export fusion model
            fusion_output = self.output_dir / f"{drill_name}_fusion_model.json"
            fusion_export_info = self.fusion_creator.export_fused_to_json(fused_model, fusion_output)
            
            print(f"Fusion model exported")
            print(f"    File: {fusion_export_info.get('compressed_path', fusion_export_info.get('output_path'))}")
            print(f"    Size: {fusion_export_info.get('compressed_size', fusion_export_info.get('file_size')):,} bytes")
            print()
            
            
            print("🌐 STEP 10: Web Integration")
            print("-" * 50)
            
            web_model = self.web_integrator.create_web_model(fused_model, segmented_motion, self.sport_type)
            
            #Export web model
            web_output = self.output_dir / f"{drill_name}_web_display.json"
            web_export_info = self.web_integrator.export_web_model(web_model, web_output)
            
            #Create Three.js loader config
            loader_config = self.web_integrator.create_three_js_loader_config(web_model)
            loader_output = self.output_dir / f"{drill_name}_loader_config.json"
            with open(loader_output, 'w') as f:
                import json
                json.dump(loader_config, f, indent=2)
            
            print(f"Web integration complete")
            print(f"    Web model: {web_export_info.get('compressed_path', web_export_info.get('output_path'))}")
            print(f"    Loader config: {loader_output}")
            print()
            
            #PIPELINE COMPLETE
            total_time = time.time() - pipeline_start
            
            print("PIPELINE COMPLETE!")
            print("=" * 100)
            print(f"Total processing time: {total_time:.2f} seconds")
            print(f"Overall quality: {quality_metrics.overall_score:.1f}/100")
            print(f"Output directory: {self.output_dir}")
            print()
            print("Files created:")
            print(f"    - {fusion_output.name} - Complete fusion model")
            print(f"    - {web_output.name} - Three.js web display")
            print(f"    - {loader_output.name} - Three.js loader config")
            print()
            
            #Generate quality report
            if quality_metrics.warnings or quality_metrics.recommendations:
                print("Quality Report:")
                if quality_metrics.warnings:
                    for warning in quality_metrics.warnings:
                        print(f"    {warning}")
                if quality_metrics.recommendations:
                    for rec in quality_metrics.recommendations:
                        print(f"    {rec}")
                print()
            
            #Success results
            results = {
                'success': True,
                'processing_time': total_time,
                'quality_score': quality_metrics.overall_score,
                'files_created': {
                    'fusion_model': str(fusion_output),
                    'web_display': str(web_output),
                    'loader_config': str(loader_output)
                },
                'stats': {
                    'frames_processed': len(pose_frames),
                    'objects_detected': sum(len(f.detected_objects) for f in object_frames),
                    'interactions_found': fused_sequence.interaction_analysis.get('total_interactions', 0),
                    'movement_cycles': len(segmented_motion.movement_cycles) if segmented_motion else 0,
                    'file_sizes': {
                        'fusion_model_bytes': fusion_export_info.get('compressed_size', fusion_export_info.get('file_size')),
                        'web_display_bytes': web_export_info.get('compressed_size', web_export_info.get('file_size'))
                    }
                }
            }
            
        except Exception as e:
            print(f"PIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()
            
            results = {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - pipeline_start
            }
        
        finally:
            self._cleanup()
        
        return results
    
    #Clean up resources not picked up by garbage collector
    def _cleanup(self): 
        if self.pose_detector:
            self.pose_detector.cleanup()
        if self.video_processor:
            self.video_processor.cleanup()
        if self.object_detector:
            self.object_detector.cleanup()

#Command Line Interface
def main(): 
    parser = argparse.ArgumentParser(description='Complete Sports Video Analysis Pipeline')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--drill-name', help='Name of the drill/skill (default: video filename)')
    parser.add_argument('--sport', choices=['soccer', 'basketball', 'general'], default='soccer', 
                       help='Sport type for analysis (default: soccer)')
    parser.add_argument('--output-dir', default='./output', help='Output directory (default: ./output)')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process (default: all)')
    parser.add_argument('--reference-height', type=float, default=1.75, 
                       help='Reference person height in meters (default: 1.75)')
    parser.add_argument('--disable-yolo', action='store_true', help='Disable object detection')
    parser.add_argument('--no-web-optimization', action='store_true', help='Disable web optimization')
    
    args = parser.parse_args()
    
    #Create pipeline
    pipeline = SportsVideoPipeline(
        sport_type=args.sport,
        reference_height=args.reference_height,
        output_dir=args.output_dir,
        enable_yolo=not args.disable_yolo,
        optimize_for_web=not args.no_web_optimization
    )
    
    #Process video
    results = pipeline.process_video(
        video_path=args.video_path,
        drill_name=args.drill_name,
        max_frames=args.max_frames
    )
    
    #Exit with appropriate code
    sys.exit(0 if results['success'] else 1)

if __name__ == '__main__':
    main()