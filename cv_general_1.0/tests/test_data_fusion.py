import sys
import os
import json
import time

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.video_processor import VideoProcessor
from src.pose_detector import PoseDetector
from src.data_smoother import DataSmoother
from src.coordinate_3d_generator import Coordinate3DGenerator
from src.object_detector import ObjectDetector
from src.data_fusion import DataFusion
from src.model_creator import ModelCreator

def test_complete_pipeline_fusion():
    """Test complete pipeline: video → pose + objects → fusion → 3D model"""
    
    print("🔗 Testing Complete Data Fusion Pipeline")
    print("=" * 60)
    
    video_path = './data/trimmed_data/elastico_skill_1.mp4'
    
    try:
        # Initialize all components
        video_processor = VideoProcessor(video_path)
        pose_detector = PoseDetector()
        smoother = DataSmoother()
        coord_3d = Coordinate3DGenerator(reference_height=1.75, depth_scale_factor=0.5)
        data_fusion = DataFusion(
            object_depth_estimation='simple',
            interaction_threshold=0.3,
            context_smoothing=True  # CHANGED: New parameter name
        )
        model_creator = ModelCreator(compress_output=True, optimize_for_web=True)
        
        # Initialize object detector if available
        object_detector = None
        
        object_detector = ObjectDetector(
            model_size='n',
            confidence_threshold=0.3,
            sports_objects_only=True,
            light_smoothing=True  # NEW: Enable light YOLO smoothing
        )
        if not object_detector.initialize_model():
            print("⚠️  YOLO initialization failed, continuing with pose-only")
            object_detector = None
        
        
        # Load video
        video_processor.load_video()
        fps = video_processor.metadata['fps']
        print(f"📹 Video loaded: {video_processor.metadata}")
        
        # Process frames
        pose_frames = []
        object_frames = []
        
        print(f"🔄 Processing frames...")
        start_time = time.time()
        
        # Process 30 frames for comprehensive test
        for frame_idx, frame in video_processor.get_frames(max_frames=30):
            timestamp = frame_idx / fps
            
            # Human pose detection
            pose_frame = pose_detector.process_frame(frame, frame_idx, timestamp)
            if pose_frame:
                pose_frames.append(pose_frame)
            
            # Object detection (if available)
            if object_detector:
                object_frame = object_detector.process_frame(frame, frame_idx, timestamp)
                if object_frame:
                    # Apply sports enhancement
                    enhanced_frame = object_detector.enhance_sports_detection(
                        object_frame, 
                        context_info={'sport_type': 'soccer'}
                    )
                    object_frames.append(enhanced_frame)
        
        processing_time = time.time() - start_time
        
        print(f"✅ Frame processing complete in {processing_time:.2f}s")
        print(f"   🤸 Pose frames: {len(pose_frames)}")
        print(f"   🎯 Object frames: {len(object_frames)}")
        
        if len(pose_frames) < 3:
            print("❌ Not enough pose frames for testing")
            return
        
        # Apply smoothing
        print(f"🔄 Applying temporal smoothing...")
        smoothed_poses = smoother.smooth_pose_sequence(pose_frames)
        
        if object_detector and len(object_frames) > 2:
            tracked_objects = object_detector.track_objects_across_frames(object_frames)
        else:
            tracked_objects = object_frames
        
        # Convert to 3D
        print(f"🌐 Converting to 3D coordinates...")
        poses_3d = coord_3d.convert_pose_sequence_to_3d(smoothed_poses)
        
        # FUSION STEP
        print(f"🔗 Performing data fusion...")
        fusion_start = time.time()
        
        fused_sequence = data_fusion.fuse_sequences(
            poses_3d=poses_3d,
            object_frames=tracked_objects,
            drill_name="Soccer Elastico with Objects"
        )
        
        fusion_time = time.time() - fusion_start
        print(f"   ⚡ Fusion completed in {fusion_time:.2f}s")
        
        # Display fusion results
        print(f"\n📊 Fusion Results:")
        metadata = fused_sequence.metadata
        print(f"   Total fused frames: {metadata['total_frames']}")
        print(f"   Duration: {metadata['duration_seconds']:.2f}s")
        
        # Quality distribution
        quality_dist = metadata['quality_distribution']
        print(f"   Frame quality distribution:")
        for quality, count in quality_dist.items():
            print(f"     {quality}: {count} frames")
        
        # Human analysis
        print(f"\n🤸 Human Movement Analysis:")
        human_analysis = fused_sequence.human_analysis
        if 'error' not in human_analysis:
            for body_part, data in human_analysis.items():
                print(f"   {body_part}:")
                print(f"     Movement range: {data['range_x']:.3f}m x {data['range_y']:.3f}m x {data['range_z']:.3f}m")
                print(f"     Avg velocity: {data['avg_velocity']:.3f}m/frame")
        
        # Object analysis
        print(f"\n🎯 Object Movement Analysis:")
        object_analysis = fused_sequence.object_analysis
        if object_analysis:
            for obj_class, data in object_analysis.items():
                print(f"   {obj_class}:")
                print(f"     Detections: {data['total_detections']}")
                print(f"     Avg confidence: {data['avg_confidence']:.3f}")
                print(f"     Movement range: {data['range_x']:.3f}m x {data['range_y']:.3f}m x {data['range_z']:.3f}m")
                print(f"     Avg velocity: {data['avg_velocity']:.3f}m/frame")
        else:
            print("   No objects detected")
        
        # Interaction analysis
        print(f"\n🔗 Interaction Analysis:")
        interaction_analysis = fused_sequence.interaction_analysis
        total_interactions = interaction_analysis.get('total_interactions', 0)
        print(f"   Total interactions: {total_interactions}")
        
        if total_interactions > 0:
            summary = interaction_analysis.get('interaction_summary', {})
            
            if 'interaction_types' in summary:
                print(f"   Interaction types:")
                for int_type, count in summary['interaction_types'].items():
                    print(f"     {int_type}: {count}")
            
            if 'most_active_keypoints' in summary:
                print(f"   Most active keypoints:")
                for keypoint, count in summary['most_active_keypoints'].items():
                    print(f"     {keypoint}: {count} interactions")
        
        # Sample fused frame data
        print(f"\n📋 Sample Fused Frame Data:")
        if fused_sequence.frames:
            sample_frame = fused_sequence.frames[0]
            print(f"   Frame {sample_frame.frame_index} (t={sample_frame.timestamp:.3f}s):")
            print(f"     Quality: {sample_frame.frame_quality}")
            print(f"     Fusion confidence: {sample_frame.fusion_confidence:.3f}")
            print(f"     Human pose: {'✅' if sample_frame.human_pose_3d else '❌'}")
            print(f"     Objects detected: {len(sample_frame.objects_3d)}")
            
            if sample_frame.objects_3d:
                for obj in sample_frame.objects_3d:
                    print(f"       {obj.class_name}: ({obj.center_x:.3f}, {obj.center_y:.3f}, {obj.center_z:.3f})")
        
        # Test enhanced model creation with fusion data
        print(f"\n🎯 Creating Enhanced 3D Model with Fusion Data...")
        
        # For now, use the poses_3d for model creation (future: extend ModelCreator for fusion data)
        model_3d = model_creator.create_3d_model(poses_3d, drill_name="Soccer Elastico with Fusion")
        
        # Export model
        output_path = './output/fused_elastico_model.json'
        export_info = model_creator.export_to_json(model_3d, output_path)
        
        print(f"\n🏁 Complete Pipeline Test Results:")
        print(f"   ✅ Video processing: {len(pose_frames)} pose frames, {len(object_frames)} object frames")
        print(f"   ✅ 3D conversion: {len(poses_3d)} 3D poses")
        print(f"   ✅ Data fusion: {len(fused_sequence.frames)} fused frames")
        print(f"   ✅ Interactions: {total_interactions} human-object interactions")
        print(f"   ✅ Model export: {export_info.get('compressed_size', export_info.get('file_size', 'N/A'))} bytes")
        
        # Cleanup
        pose_detector.cleanup()
        video_processor.cleanup()
        if object_detector:
            object_detector.cleanup()
        
    except Exception as e:
        print(f"❌ Error in complete pipeline fusion test: {e}")
        import traceback
        traceback.print_exc()

def test_fusion_specific_features():
    """Test specific data fusion features"""
    
    print(f"\n🔬 Testing Data Fusion Specific Features")
    print("=" * 50)
    
    try:
        # Test different fusion configurations
        configs = [
            {
                'object_depth_estimation': 'simple',
                'interaction_threshold': 0.2,
                'context_smoothing': True
            },
            {
                'object_depth_estimation': 'advanced',
                'interaction_threshold': 0.4,
                'context_smoothing': False
            }
        ]
        
        for i, config in enumerate(configs):
            print(f"\n🔧 Testing configuration {i+1}: {config}")
            
            fusion = DataFusion(**config)
            
            # Test object size estimates
            print(f"   📏 Object size estimates loaded: {len(fusion.object_size_estimates)} types")
            
            # Show some examples
            examples = ['soccer_ball', 'basketball', 'tennis_ball']
            for obj_type in examples:
                if obj_type in fusion.object_size_estimates:
                    size_info = fusion.object_size_estimates[obj_type]
                    print(f"     {obj_type}: {size_info}")
            
            print(f"   ✅ Configuration {i+1} initialized successfully")
        
        print(f"\n✅ All fusion feature tests passed!")
        
    except Exception as e:
        print(f"❌ Error in fusion feature test: {e}")
        import traceback
        traceback.print_exc()

def test_fusion_edge_cases():
    """Test edge cases in data fusion"""
    
    print(f"\n🧪 Testing Data Fusion Edge Cases")
    print("=" * 40)
    
    try:
        fusion = DataFusion()
        
        # Test 1: Empty sequences
        print(f"🔍 Test 1: Empty sequences")
        try:
            empty_fusion = fusion.fuse_sequences([], [], "Empty Test")
            print(f"   ✅ Empty sequences handled: {len(empty_fusion.frames)} frames")
        except Exception as e:
            print(f"   ⚠️  Empty sequences error: {e}")
        
        # Test 2: Mismatched sequence lengths
        print(f"🔍 Test 2: Mismatched sequence lengths")
        # This would require actual Pose3D and ObjectFrame objects to test properly
        print(f"   ℹ️  Would need real data objects for comprehensive testing")
        
        # Test 3: Invalid object types
        print(f"🔍 Test 3: Object size estimation for unknown objects")
        
        # Create mock object
        class MockObject:
            def __init__(self, class_name):
                self.class_name = class_name
                self.width = 0.1
                self.height = 0.1
        
        unknown_obj = MockObject("unknown_sports_equipment")
        depth = fusion._estimate_simple_depth(unknown_obj, 1.0)
        print(f"   Unknown object depth estimate: {depth:.3f}m")
        
        print(f"\n✅ Edge case tests completed!")
        
    except Exception as e:
        print(f"❌ Error in edge case test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("🚀 Starting Data Fusion Tests")
    print("=" * 70)
    
    # Run comprehensive pipeline test
    test_complete_pipeline_fusion()
    
    # Run feature-specific tests
    test_fusion_specific_features()
    
    # Run edge case tests
    test_fusion_edge_cases()
    
    print(f"\n🏁 All data fusion tests completed!")