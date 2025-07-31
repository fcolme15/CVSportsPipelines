import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
from src.video_processor import VideoProcessor
from src.pose_detector import PoseDetector
from src.data_smoother import DataSmoother

def test_smoothing():
    video_path = './data/trimmed_data/elastico_skill_1.mp4'

    print('Testing temporal smoothing')

    try:
        pose_detector = PoseDetector()
        smoother = DataSmoother(
            confidence_threshold=0.5,
            outlier_std_threshold=2.0,
            smoothing_window=5
        )

        #Collect pose data from more frames
        pose_frames = []
        
        video_processor = VideoProcessor(video_path)
        video_processor.load_video()
        fps = video_processor.metadata['fps']

        #Process 60 frames 
        for frame_idx, frame in video_processor.get_frames(max_frames=60):
            timestamp = frame_idx / fps
            pose_frame = pose_detector.process_frame(frame, frame_idx, timestamp)

            if pose_frame:
                pose_frames.append(pose_frame)

        if len(pose_frames) >= 3:
            #Apply the smoothing
            smoothed_frames = smoother.smooth_pose_sequence(pose_frames)

            for frame_idx in [0, 10, 20]:
                if frame_idx < len(pose_frames):
                    original = pose_frames[frame_idx]
                    smooth = smoothed_frames[frame_idx]

                    #Comparing right ankle 
                    original_ankle = original.keypoints[28]
                    smooth_ankle = smooth.keypoints[28]

                    print(f"Frame {frame_idx}:")
                    print(f"    Original ankle: ({original_ankle.x:.4f}, {original_ankle.y:.4f})")
                    print(f"    Smoothed ankle: ({smooth_ankle.x:.4f}, {smooth_ankle.y:.4f})")
                    print(f"    Difference: ({abs(original_ankle.x - smooth_ankle.x):.4f}, {abs(original_ankle.y - smooth_ankle.y):.4f})")
            
            print(f'Smoothing test completed!')
            print(f"    Original frames: {len(pose_frames)}")
            print(f"    Smoothed frames: {len(smoothed_frames)}")
        else: 
            print('Not enough frames for smoothing test')

        pose_detector.cleanup()

    except Exception as e:
        print(f'Error during smoothing test: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_smoothing()