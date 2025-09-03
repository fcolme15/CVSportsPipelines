import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.video_processor import VideoProcessor
from src.pose_detector import PoseDetector
from src.data_smoother import DataSmoother
from src.coordinate_3d_generator import Coordinate3DGenerator

def test_3d_generation():
    video_path = './data/trimmed_data/elastico_skill_1.mp4'

    try: 
        pose_detector = PoseDetector()
        smoother = DataSmoother()
        coord_3d = Coordinate3DGenerator(reference_height=1.75, depth_scale_factor=0.5)

        pose_frames = []

        video_processor = VideoProcessor(video_path)

        video_processor.load_video()
        fps = video_processor.metadata['fps']

        #Process 30 frames for testing
        for frame_idx, frame in video_processor.get_frames(max_frames=30):
            timestamp = frame_idx / fps
            pose_frame = pose_detector.process_frame(frame, frame_idx, timestamp)
            
            if pose_frame: 
                pose_frames.append(pose_frame)

        if len(pose_frames) >= 3:
            #Apply smoothing
            smoothed_frames = smoother.smooth_pose_sequence(pose_frames)

            #Convert to 3D
            poses_3d = coord_3d.convert_pose_sequence_to_3d(smoothed_frames)

            print('Sample 3D coords')
            for i in [0, 10, 20]:
                if i < len(poses_3d):
                    pose = poses_3d[i]

                    right_ankle = pose.keypoints_3d[28] 
                    right_wrist = pose.keypoints_3d[16]

                    print(f"Frame {pose.frame_index} (t={pose.timestamp:.3f}s):")
                    print(f"    Right ankle: ({right_ankle.x:.3f}, {right_ankle.y:.3f}, {right_ankle.z:.3f}) m")
                    print(f"    Right wrist: ({right_wrist.x:.3f}, {right_wrist.y:.3f}, {right_wrist.z:.3f}) m")
                
                
            analysis = coord_3d.analyze_movement_quality(poses_3d)
            print(f"Movement Analysis:")
            print(f"    Duration: {analysis['duration_seconds']:.2f} seconds")
            print(f"    World scale: {analysis['world_scale']:.3f} meters/unit")
            
            if 'ankle_movement_range' in analysis:
                ankle = analysis['ankle_movement_range']
                print(f"    Right ankle movement:")
                print(f"    X-range: {ankle['x_range']:.3f} m (side-to-side)")
                print(f"    Y-range: {ankle['y_range']:.3f} m (up-down)")  
                print(f"    Z-range: {ankle['z_range']:.3f} m (forward-back)")
                print(f"    Total distance: {ankle['total_distance']:.3f} m")
            
            #Show bone connections for visualization
            bones = coord_3d.get_bone_connections()
            print(f"Skeleton structure: {len(bones)} bones defined for 3D rendering")
            
            print(f"3D generation test complete!")
            print(f"    Input frames: {len(pose_frames)}")
            print(f"    Smoothed frames: {len(smoothed_frames)}")
            print(f"    3D poses: {len(poses_3d)}")

        else:
            print('Not enough frames')

        pose_detector.cleanup()

    except Exception as e:
        print('Error during 3D generation: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_3d_generation()