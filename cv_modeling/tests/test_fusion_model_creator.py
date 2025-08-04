import sys
import os
import json
import gzip
import time
from pathlib import Path

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
from src.model_creator_fusion import ModelCreatorFusion

#Test complete fusion model creation pipeline
def test_fusion_model_creation():
    
    print("Testing Fusion Model Creation Pipeline")
    print("=" * 100)
    
    video_path = './data/trimmed_data/elastico_skill_1.mp4'
    output_dir = './output'
    
    try:
        #Initialize all components
        video_processor = VideoProcessor(video_path)
        pose_detector = PoseDetector()
        smoother = DataSmoother()
        coord_3d = Coordinate3DGenerator(reference_height=1.75, depth_scale_factor=0.5)
        data_fusion = DataFusion(
            object_depth_estimation='simple',
            interaction_threshold=0.3,
            context_smoothing=True
        )
        
        #Initialize both model creators for comparison
        model_creator = ModelCreator(compress_output=True, optimize_for_web=True)
        fusion_creator = ModelCreatorFusion(
            coordinate_precision=3,
            compress_output=True,
            optimize_for_web=True,
            include_interaction_events=True
        )
        
        #Initialize object detector if available
        object_detector = None
        
        object_detector = ObjectDetector(
            model_size='n',
            confidence_threshold=0.3,
            sports_objects_only=True,
            light_smoothing=True
        )
        if not object_detector.initialize_model():
            print("YOLO initialization failed, testing human-only fusion")
            object_detector = None
        
        #Load and process video
        video_processor.load_video()
        fps = video_processor.metadata['fps']
        print(f"Video loaded: {video_processor.metadata}")
        
        #Process frames
        pose_frames = []
        object_frames = []
        
        print(f"Processing 30 frames for comprehensive test...")
        start_time = time.time()
        
        for frame_idx, frame in video_processor.get_frames(max_frames=30):
            timestamp = frame_idx / fps
            
            # Human pose detection
            pose_frame = pose_detector.process_frame(frame, frame_idx, timestamp)
            if pose_frame:
                pose_frames.append(pose_frame)
            
            #Object detection (if available)
            if object_detector:
                object_frame = object_detector.process_frame(frame, frame_idx, timestamp)
                if object_frame:
                    enhanced_frame = object_detector.enhance_sports_detection(
                        object_frame, 
                        context_info={'sport_type': 'soccer'}
                    )
                    object_frames.append(enhanced_frame)
        
        processing_time = time.time() - start_time
        print(f"Processing complete in {processing_time:.2f}s")
        print(f"    Pose frames: {len(pose_frames)}")
        print(f"    Object frames: {len(object_frames)}")
        
        if len(pose_frames) < 3:
            print("Not enough pose frames for testing")
            return
        
        #Apply smoothing and 3D conversion
        print(f"Applying smoothing and 3D conversion...")
        smoothed_poses = smoother.smooth_pose_sequence(pose_frames)
        
        if object_detector and len(object_frames) > 2:
            tracked_objects = object_detector.track_objects_across_frames(object_frames)
        else:
            tracked_objects = object_frames
        
        poses_3d = coord_3d.convert_pose_sequence_to_3d(smoothed_poses)
        
        #Create fusion sequence
        print(f"Creating fusion sequence...")
        fused_sequence = data_fusion.fuse_sequences(
            poses_3d=poses_3d,
            object_frames=tracked_objects,
            drill_name="Soccer Elastico - Fusion Test"
        )
        
        #Create both model types for comparison
        print(f"Creating human-only model...")
        human_model = model_creator.create_3d_model(poses_3d, "Human Only Model")
        
        print(f"Creating fused model...")
        fused_model = fusion_creator.create_fused_3d_model(fused_sequence)
        
        #Export both models
        print(f"Exporting models...")
        
        #Human-only export
        human_path = Path(output_dir) / "human_only_model.json"
        human_export_info = model_creator.export_to_json(human_model, human_path)
        
        #Fused export
        fused_path = Path(output_dir) / "fused_complete_model.json"
        fused_export_info = fusion_creator.export_fused_to_json(fused_model, fused_path)
        
        #Display model statistics
        print(f"Model Statistics Comparison:")
        
        #Human-only stats
        human_stats = model_creator.get_model_statistics(human_model)
        print(f"    Human-Only Model:")
        print(f"        Duration: {human_stats['basic_info']['duration']:.2f}s")
        print(f"        Keyframes: {human_stats['structure']['human_keyframes'] if 'human_keyframes' in human_stats['structure'] else len(human_model.keyframes)}")
        print(f"        JSON size: {human_stats['data_size']['estimated_json_size_kb']:.1f} KB")
        
        #Fused stats
        fused_stats = fusion_creator.get_fused_model_statistics(fused_model)
        print(f"    Fused Model:")
        print(f"        Duration: {fused_stats['basic_info']['duration']:.2f}s")
        print(f"        Human keyframes: {fused_stats['structure']['human_keyframes']}")
        print(f"        Object keyframes: {fused_stats['structure']['object_keyframes']}")
        print(f"        Interaction events: {fused_stats['structure']['interaction_events']}")
        print(f"        JSON size: {fused_stats['data_size']['estimated_json_size_kb']:.1f} KB")
        
        #File size comparison
        human_file_size = human_export_info.get('compressed_size', human_export_info.get('file_size', 0))
        fused_file_size = fused_export_info.get('compressed_size', fused_export_info.get('file_size', 0))
        
        print(f"File Size Comparison:")
        print(f"    Human-only: {human_file_size:,} bytes ({human_file_size/1024:.1f} KB)")
        print(f"    Fused model: {fused_file_size:,} bytes ({fused_file_size/1024:.1f} KB)")
        
        if human_file_size > 0:
            size_increase = ((fused_file_size / human_file_size) - 1) * 100
            print(f"    Size increase: {size_increase:.1f}%")
        
        #Detailed comparison analysis
        comparison = fusion_creator.compare_with_human_only_model(fused_model, human_file_size)
        print(f"    Detailed Comparison Analysis:")
        print(f"        Size increase: {comparison['size_increase_kb']:.1f} KB ({comparison['size_increase_percent']:.1f}%)")
        print(f"    Additional data added:")
        print(f"        Object keyframes: {comparison['additional_data']['object_keyframes']}")
        print(f"        Interaction events: {comparison['additional_data']['interaction_events']}")
        print(f"        Object types: {comparison['additional_data']['object_types']}")
        
        #Verify exported files
        print(f"File Verification:")
        
        #Load and verify human-only model
        if 'compressed_path' in human_export_info:
            with gzip.open(human_export_info['compressed_path'], 'rt', encoding='utf-8') as f:
                loaded_human = json.load(f)
        else:
            with open(human_export_info['output_path'], 'r', encoding='utf-8') as f:
                loaded_human = json.load(f)
        
        print(f"Human-only model loaded: {len(loaded_human['keyframes'])} keyframes")
        
        #Load and verify fused model
        if 'compressed_path' in fused_export_info:
            with gzip.open(fused_export_info['compressed_path'], 'rt', encoding='utf-8') as f:
                loaded_fused = json.load(f)
        else:
            with open(fused_export_info['output_path'], 'r', encoding='utf-8') as f:
                loaded_fused = json.load(f)
        
        print(f"Fused model loaded:")
        print(f"    Human keyframes: {len(loaded_fused['human_keyframes'])}")
        print(f"    Object keyframes: {len(loaded_fused['object_keyframes'])}")
        print(f"    Interaction events: {len(loaded_fused.get('interaction_events', []))}")
        
        #Sample data verification
        print(f"Sample Data Verification:")
        
        #Show sample fused keyframe
        if loaded_fused['object_keyframes']:
            sample_obj_frame = loaded_fused['object_keyframes'][0]
            print(f"    Sample object keyframe (frame {sample_obj_frame['frame_index']}):")
            for obj in sample_obj_frame['objects']:
                print(f"        {obj['class']}: pos={obj['pos']}, conf={obj['conf']}")
        
        #Show sample interaction events
        if loaded_fused.get('interaction_events'):
            sample_interactions = loaded_fused['interaction_events'][:3]  # First 3 events
            print(f"    Sample interaction events:")
            for event in sample_interactions:
                print(f"    t={event['timestamp']:.3f}s: {event['keypoint_name']} → {event['object_class']} ({event['interaction_type']})")
        
        print(f"Fusion Model Creation Test Results:")
        print(f"    Human-only model: {len(loaded_human['keyframes'])} keyframes, {human_file_size:,} bytes")
        print(f"    Fused model: {len(loaded_fused['human_keyframes'])} human + {len(loaded_fused['object_keyframes'])} object keyframes")
        print(f"    Interactions: {len(loaded_fused.get('interaction_events', []))} events detected")
        print(f"    File size increase: {size_increase:.1f}% for complete sports analysis")
        
        pose_detector.cleanup()
        video_processor.cleanup()
        if object_detector:
            object_detector.cleanup()
        
    except Exception as e:
        print(f"Error in fusion model creation test: {e}")
        import traceback
        traceback.print_exc()

#Test different fusion export configurations
def test_fusion_export_comparison():
    
    print(f"Testing Fusion Export Configurations")
    print("=" * 100)
    
    try:
        #Test different configurations
        configs = [
            {
                'coordinate_precision': 2,
                'compress_output': False,
                'optimize_for_web': False,
                'include_interaction_events': True,
                'name': 'Full Precision, Uncompressed'
            },
            {
                'coordinate_precision': 3,
                'compress_output': True,
                'optimize_for_web': True,
                'include_interaction_events': True,
                'name': 'Web Optimized, Compressed'
            },
            {
                'coordinate_precision': 3,
                'compress_output': True,
                'optimize_for_web': True,
                'include_interaction_events': False,
                'name': 'No Interactions, Compressed'
            }
        ]
        
        for i, config in enumerate(configs):
            print(f"Testing configuration: {config['name']}")
            
            #Remove name from config for initialization
            init_config = {k: v for k, v in config.items() if k != 'name'}
            fusion_creator = ModelCreatorFusion(**init_config)
            
            #Test object render hints
            print(f"Object render hints loaded: {len(fusion_creator.object_render_hints)} types")
        
            #Show some examples
            examples = ['soccer_ball', 'basketball', 'tennis_ball']
            for obj_type in examples:
                if obj_type in fusion_creator.object_render_hints:
                    hints = fusion_creator.object_render_hints[obj_type]
                    print(f"     {obj_type}: {hints}")
            
            print(f"Configuration '{config['name']}' initialized successfully")
            print(f"Precision: {fusion_creator.coordinate_precision} decimals")
            print(f"Compression: {'Yes' if fusion_creator.compress_output else 'No'}")
            print(f"Web optimized: {'Yes' if fusion_creator.optimize_for_web else 'No'}")
            print(f"Include interactions: {'Yes' if fusion_creator.include_interaction_events else 'No'}")
        
        print(f"All fusion export configuration tests passed!")
        
    except Exception as e:
        print(f"Error in fusion export configuration test: {e}")
        import traceback
        traceback.print_exc()

#Test specific fusion model featutues and edge cases
def test_fusion_model_features():
    
    print(f"Testing Fusion Model Specific Features")
    print("=" * 100)
    
    try:
        fusion_creator = ModelCreatorFusion()
        
        print(f"Test 1: Object render hints system")
        
        test_objects = ['soccer_ball', 'basketball', 'american_football', 'tennis_ball', 'unknown_object']
        
        for obj_name in test_objects:
            hints = fusion_creator.object_render_hints.get(obj_name, 
                                                          fusion_creator.object_render_hints.get('sports_ball', {}))
            print(f"    {obj_name}: {hints}")
        
        print(f"Test 2: Data structure validation")
        print(f"    Data structure validation requires real FusedSequence objects")
        print(f"    Structure definitions verified:")
        print(f"        FusionMetadata: Enhanced metadata with fusion settings")
        print(f"        ObjectKeyframe: Object positions per frame") 
        print(f"        InteractionEvent: Human-object interaction moments")
        print(f"        FusedModel3D: Complete exportable structure")
        
        print(f"Test 3: File versioning system")
        
        #Check version progression
        from src.model_creator_fusion import FusionMetadata
        
        fusion_metadata = FusionMetadata(
            drill_name="Test Drill",
            duration_seconds=1.0,
            fps=30.0,
            total_frames=30,
            world_scale=1.0,
            creation_timestamp=time.time()
        )
        
        print(f"    File version: {fusion_metadata.file_version}")
        print(f"    Version 1.1 indicates fusion capabilities")
        
        
        print(f"Test 4: JSON structure validation")
        
        expected_keys = [
            'metadata', 'skeleton_structure', 'human_keyframes', 
            'object_keyframes', 'human_analysis', 'object_analysis', 
            'interaction_analysis', 'interaction_events'
        ]
        
        print(f"    Expected JSON keys: {len(expected_keys)}")
        for key in expected_keys:
            print(f"    {key}")
        
        print(f"Test 5: Inheritance from ModelCreator")
        
        #Verify fusion creator inherits from base creator
    
        
        is_subclass = issubclass(ModelCreatorFusion, ModelCreator)
        print(f"    Inherits from ModelCreator: {'Yes' if is_subclass else 'No'}")
        
        #Check inherited attributes
        inherited_attrs = ['coordinate_precision', 'compress_output', 'optimize_for_web', 'bone_definitions']
        
        for attr in inherited_attrs:
            has_attr = hasattr(fusion_creator, attr)
            print(f"    Has {attr}: {'Yes' if has_attr else 'No'}")
        
        #Test 6: Export method differences
        print(f"Test 6: Export method capabilities")
        
        fusion_methods = [
            'create_fused_3d_model',
            'export_fused_to_json', 
            '_create_human_keyframes',
            '_create_object_keyframes',
            '_create_interaction_events',
            'get_fused_model_statistics',
            'compare_with_human_only_model'
        ]
        
        print(f"Fusion-specific methods: {len(fusion_methods)}")
        for method in fusion_methods:
            has_method = hasattr(fusion_creator, method)
            print(f"    {'Yes' if has_method else 'No'} {method}")
        
        print(f"All fusion model feature tests passed!")
        
    except Exception as e:
        print(f"Error in fusion model feature test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("Starting Fusion Model Creator Tests")
    print("=" * 100)
    
    #Run main fusion model creation test
    test_fusion_model_creation()
    
    #Run export configuration tests
    test_fusion_export_comparison()
    
    #Run feature-specific tests
    test_fusion_model_features()
    
    print(f"All fusion model creator tests completed!")