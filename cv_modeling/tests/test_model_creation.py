import sys
import os
import json
import gzip
from pathlib import Path

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.video_processor import VideoProcessor
from src.pose_detector import PoseDetector
from src.data_smoother import DataSmoother
from src.coordinate_3d_generator import Coordinate3DGenerator
from src.model_creator import ModelCreator

#Test the pipeline from: video -> 3D model -> JSON export file
def test_model_creation():
    
    video_path = './data/trimmed_data/elastico_skill_1.mp4'
    output_dir = './output'
    
    try:
        print("Starting 3D model creation pipeline")
        print("=" * 100)
        
        #Initialize all processors
        video_processor = VideoProcessor(video_path)
        pose_detector = PoseDetector()
        smoother = DataSmoother()
        coord_3d = Coordinate3DGenerator(reference_height=1.75, depth_scale_factor=0.5)
        model_creator = ModelCreator(
            coordinate_precision=3,
            compress_output=True,
            optimize_for_web=True
        )
        
        #Step 1: Process video frames
        pose_frames = []
        video_processor.load_video()
        fps = video_processor.metadata['fps']
        
        print(f"Processing video: {video_processor.metadata}")
        
        #Process first 30 frames for testing
        for frame_idx, frame in video_processor.get_frames(max_frames=30):
            timestamp = frame_idx / fps
            pose_frame = pose_detector.process_frame(frame, frame_idx, timestamp)
            
            if pose_frame:
                pose_frames.append(pose_frame)
        
        print(f"Extracted {len(pose_frames)} pose frames")
        
        if len(pose_frames) >= 3:
            #Step 2: Apply smoothing
            print("Applying temporal smoothing...")
            smoothed_frames = smoother.smooth_pose_sequence(pose_frames)
            print(f"Smoothed {len(smoothed_frames)} frames")
            
            #Step 3: Convert to 3D coordinates
            print("Converting to 3D coordinates...")
            poses_3d = coord_3d.convert_pose_sequence_to_3d(smoothed_frames)
            print(f"Generated {len(poses_3d)} 3D poses")
            
            #Step 4: Create 3D model
            print("Creating 3D model structure...")
            model_3d = model_creator.create_3d_model(poses_3d, drill_name="Soccer Elastico Skill")
            
            #Step 5: Show model statistics
            print("Model Statistics:")
            stats = model_creator.get_model_statistics(model_3d)
            
            print(f"    Drill: {stats['basic_info']['drill_name']}")
            print(f"    Duration: {stats['basic_info']['duration']:.2f} seconds")
            print(f"    FPS: {stats['basic_info']['fps']:.1f}")
            print(f"    World Scale: {stats['basic_info']['world_scale']:.3f} meters/unit")
            print(f"    Keypoints per frame: {stats['structure']['total_keypoints_per_frame']}")
            print(f"    Total bones: {stats['structure']['total_bones']}")
            print(f"    Estimated JSON size: {stats['data_size']['estimated_json_size_kb']:.1f} KB")
            
            #Show bone groups
            print(f"Bone Group Distribution:")
            for group, count in stats['structure']['bone_groups'].items():
                print(f"    {group}: {count} bones")
            
            #Step 6: Export to JSON
            print(f"Exporting model to JSON...")
            output_path = Path(output_dir) / "elastico_skill_model.json"
            export_info = model_creator.export_to_json(model_3d, output_path)
            
            #Step 7: Verify export and show sample data
            print(f"Verification & Sample Data:")
            
            if 'compressed_path' in export_info:
                #Load and verify compressed file
                with gzip.open(export_info['compressed_path'], 'rt', encoding='utf-8') as f:
                    loaded_model = json.load(f)
                print(f"Successfully loaded compressed model")
            else:
                #Load and verify uncompressed file
                with open(export_info['output_path'], 'r', encoding='utf-8') as f:
                    loaded_model = json.load(f)
                print(f"Successfully loaded model")
            
            #Show sample keyframe data
            print(f"Sample Keyframe Data (Frame 0):")
            first_keyframe = loaded_model['keyframes'][0]
            print(f"    Timestamp: {first_keyframe['timestamp']}s")
            print(f"    Frame Index: {first_keyframe['frame_index']}")
            
            # Show sample keypoints (right ankle and right wrist)
            right_ankle = first_keyframe['keypoints'][28]  # Right ankle
            right_wrist = first_keyframe['keypoints'][16]  # Right wrist
            
            print(f"    Right Ankle: x={right_ankle['x']}, y={right_ankle['y']}, z={right_ankle['z']}, confidence={right_ankle['c']}")
            print(f"    Right Wrist: x={right_wrist['x']}, y={right_wrist['y']}, z={right_wrist['z']}, confidence={right_wrist['c']}")
            
            #Show movement analysis sample
            print(f"Movement Analysis Sample:")
            movement = loaded_model['movement_analysis']
            if 'right_ankle' in movement:
                ankle_data = movement['right_ankle']
                print(f"    Right Ankle Movement:")
                print(f"        X-range: {ankle_data['range_x']:.3f}m")
                print(f"        Y-range: {ankle_data['range_y']:.3f}m") 
                print(f"        Z-range: {ankle_data['range_z']:.3f}m")
                print(f"        Total distance: {ankle_data['total_distance']:.3f}m")
                print(f"        Avg velocity: {ankle_data['avg_velocity']:.3f}m/frame")
            
            #Show bounding box
            if 'overall' in movement and 'bounding_box' in movement['overall']:
                bbox = movement['overall']['bounding_box']
                print(f"    Overall Bounding Box:")
                print(f"        X: {bbox['min_x']:.3f} to {bbox['max_x']:.3f}m")
                print(f"        Y: {bbox['min_y']:.3f} to {bbox['max_y']:.3f}m")
                print(f"        Z: {bbox['min_z']:.3f} to {bbox['max_z']:.3f}m")
            
            #Verify skeleton structure
            print(f"Skeleton Structure Verification:")
            skeleton = loaded_model['skeleton_structure']
            print(f"    Total bones: {len(skeleton)}")
            
            #Show sample bones from each group
            bone_groups = {}
            for bone in skeleton:
                group = bone['bone_group']
                if group not in bone_groups:
                    bone_groups[group] = []
                bone_groups[group].append(bone['name'])
            
            for group, bones in bone_groups.items():
                print(f"    {group}: {len(bones)} bones (e.g., '{bones[0]}')")
            
            print(f"    Complete pipeline test successful!")
            print(f"        Video processed: {len(pose_frames)} frames")
            print(f"        3D model created: {len(poses_3d)} poses")
            print(f"        JSON exported: {output_path}")
            print(f"    File verification: passed")
            
            #Show final file info
            if 'compressed_path' in export_info:
                final_path = export_info['compressed_path']
                final_size = export_info['compressed_size']
                print(f"    Final file: {final_path} ({final_size:,} bytes)")
            else:
                final_path = export_info['output_path']
                final_size = export_info['file_size']
                print(f"    Final file: {final_path} ({final_size:,} bytes)")
        
        else:
            print("Not enough frames for processing")
        
        #Cleanup
        pose_detector.cleanup()
        video_processor.cleanup()
        
    except Exception as e:
        print(f"Error during model creation test: {e}")
        import traceback
        traceback.print_exc()

#Test Model Creator specific features and edge cases
def test_model_creator_features():
    
    print(f"Testing ModelCreator specific features...")
    print("=" * 100)
    
    try:
        #Test different configurations
        configs = [
            {"coordinate_precision": 2, "compress_output": False, "optimize_for_web": False},
            {"coordinate_precision": 4, "compress_output": True, "optimize_for_web": True},
        ]
        
        for i, config in enumerate(configs):
            print(f"    Testing configuration {i+1}: {config}")
            
            model_creator = ModelCreator(**config)
            
            #Test bone structure
            bones = model_creator.bone_definitions
            print(f"        Bone definitions: {len(bones)} bones loaded")
            
            #Verify bone groups
            groups = set(bone.bone_group for bone in bones)
            print(f"        Bone groups: {', '.join(sorted(groups))}")
            
            #Test with minimal data (would need real poses_3d for full test)
            print(f"        Configuration {i+1} initialized successfully")
        
        print(f"All ModelCreator feature tests passed!")
        
    except Exception as e:
        print(f"Error in ModelCreator feature test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("Starting Model Creation Tests")
    print("=" * 100)
    
    #Run main pipeline test
    test_model_creation()
    
    #Run feature-specific tests
    test_model_creator_features()
    
    print(f"All tests completed!")