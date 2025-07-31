import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.video_processor import VideoProcessor
from src.pose_detector import PoseDetector

def test_pose_detection():

    video_path = './data/trimmed_data/elastico_skill_1.mp4'

    try:
        pose_detector = PoseDetector()

        video_processor = VideoProcessor(video_path)
        video_processor.load_video()
        fps = video_processor.metadata['fps']

        #Process first 30 frames

        frames_processed = 0
        successful_detections = 0

        for frame_idx, frame in video_processor.get_frames(max_frames=30):
            timestamp = frame_idx / fps

            pose_frame = pose_detector.process_frame(frame, frame_idx, timestamp)

            if pose_frame:
                successful_detections += 1
                if frame_idx < 5:
                    print(f'Frame {frame_idx} (t={timestamp:.3f}s)')
                    print(f'    Detection Confidence: {pose_frame.detection_confidence:.3f}')

                    #Show wrist positions
                    left_wrist = pose_frame.keypoints[15]
                    right_wrist = pose_frame.keypoints[16]
                    print(f"    Left wrist: ({left_wrist.x:.3f}, {left_wrist.y:.3f}) confidence: {left_wrist.visibility:.3f}")
                    print(f"    Right wrist: ({right_wrist.x:.3f}, {right_wrist.y:.3f}) confidence: {right_wrist.visibility:.3f}")
                    
                    #Show ankle positions  
                    left_ankle = pose_frame.keypoints[27] 
                    right_ankle = pose_frame.keypoints[28]
                    print(f"    Left ankle: ({left_ankle.x:.3f}, {left_ankle.y:.3f}) confidence: {left_ankle.visibility:.3f}")
                    print(f"    Right ankle: ({right_ankle.x:.3f}, {right_ankle.y:.3f}) confidence: {right_ankle.visibility:.3f}")

                frames_processed += 1

            print(f"Pose detection complete!")
            print(f"    Frames processed: {frames_processed}")
            print(f"    Successful detections: {successful_detections}")
            print(f"    Success rate: {successful_detections/frames_processed*100:.1f}%")            
        
        pose_detector.cleanup()

    except Exception as e:
        print(f'Error during pose detection: {e}')

if __name__ == '__main__':
    test_pose_detection()





