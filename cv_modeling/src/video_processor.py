import cv2
import numpy as np
from pathlib import Path
import logging

class VideoProcessor:

    def __init__ (self, video_path):
        self. video_path = Path(video_path)
        self.cap = None
        self.metadata = {}

    '''Load video and extract the metadata'''
    def load_video(self):
        if not self.video_path.exists():
            raise FileNotFoundError(f'Video file not found: {self.video_path}')
        
        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise ValueError(f'Could not open video file: {self.video_path}')
        
        self.metadata = {
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': None
        }

        if self.metadata['fps'] > 0:
            self.metadata['duration'] = self.metadata['frame_count'] / self.metadata['fps']
        
        print(f"Loaded: {self.video_path.name}")
        print(f"    Resolution: {self.metadata['width']}x{self.metadata['height']}")
        print(f"    FPS: {self.metadata['fps']:.2f}")
        print(f"    Duration: {self.metadata['duration']:.2f} seconds")
        print(f"    Frame count: {self.metadata['frame_count']}")
        
        return True
    
    def get_frames(self, max_frames=None):
        if not self.cap:
            raise ValueError('Video not loaded')
        
        frame_count = 0

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            if max_frames and frame_count >= max_frames:
                break

            #Convert BGR to RGB (MediaPipe needs RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield frame_count, frame_rgb
            frame_count += 1

    def cleanup(self):
        if self.cap:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

