import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.video_processor import VideoProcessor

def test_video_loading():
    video_path = './data/trimmed_data/elastico_skill_1.mp4'

    try:
        processor = VideoProcessor(video_path)
        processor.load_video()

        frames_processed = 0

        for frame_idx, frame in processor.get_frames(max_frames=10):
            print(f'Frame {frame_idx}: shape {frame.shape}')
            frames_processed += 1

        print(f'Processed {frames_processed} frames')

    except FileNotFoundError:
        print('Test video not found')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    test_video_loading()