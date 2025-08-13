'''
Handles efficient chunk-based video processing with memory management.
'''

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Generator, Callable, Tuple, Any, Dict
import logging
from collections import deque
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#A chunk of frames with metadata
@dataclass
class FrameChunk:
    frames: List[np.ndarray]
    start_index: int
    end_index: int  # Exclusive
    timestamps: List[float]
    
    #Num of frames
    @property
    def size(self) -> int:
        return len(self.frames)
    
    #Duration of chunk in seconds
    @property
    def duration(self) -> float:
        if len(self.timestamps) >= 2:
            return self.timestamps[-1] - self.timestamps[0]
        return 0.0
    
    #Get frame using the index
    def get_frame(self, relative_index: int) -> Optional[np.ndarray]:
        if 0 <= relative_index < len(self.frames):
            return self.frames[relative_index]
        return None

#Sliding window for processing with overlap
@dataclass
class ProcessingWindow:
    current_chunk: FrameChunk
    overlap_frames: List[np.ndarray]  # Frames from previous chunk for continuity
    overlap_timestamps: List[float]
    window_size: int
    
    @property
    def total_frames(self) -> int:
        return len(self.overlap_frames) + self.current_chunk.size
    
    def get_all_frames(self) -> List[np.ndarray]:
        return self.overlap_frames + self.current_chunk.frames
    
    def get_all_timestamps(self) -> List[float]:
        return self.overlap_timestamps + self.current_chunk.timestamps


@dataclass
class VideoMetadata:
    path: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration: float
    codec: str
    
    def __str__(self) -> str:
        return (f"Video: {Path(self.path).name}\n"
                f"  Resolution: {self.width}x{self.height}\n"
                f"  FPS: {self.fps:.2f}\n"
                f"  Duration: {self.duration:.2f}s\n"
                f"  Frames: {self.total_frames}")

#Frame processor that handles video in chunks supporting overlap between chunks for temporal continuity
class FrameProcessor:
    def __init__(self,
                 video_path: str,
                 chunk_size: int = 30,
                 overlap: int = 5,
                 max_memory_gb: float = 2.0,
                 skip_frames: int = 0,
                 max_frames: Optional[int] = None):
        
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.chunk_size = chunk_size
        self.overlap = min(overlap, chunk_size // 2) #Overlap can't be more than half chunk
        self.max_memory_gb = max_memory_gb
        self.skip_frames = skip_frames
        self.max_frames = max_frames
        
        #Video capture
        self.cap = None
        self.metadata = None
        
        #Processing state
        self.current_frame_index = 0
        self.frames_processed = 0
        self.chunks_processed = 0
        
        #Memory monitoring
        self.estimated_frame_size_mb = 0
        self.estimated_chunk_memory_mb = 0
        
        #Overlap buffer for temporal continuity
        self.overlap_buffer = deque(maxlen=self.overlap)
        self.overlap_timestamps_buffer = deque(maxlen=self.overlap)
        
        #Initialize video
        self._init_video()
    
    #Initialize video capture and extract metadata
    def _init_video(self):
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        #Extract metadata
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        self.metadata = VideoMetadata(
            path=str(self.video_path),
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            duration=total_frames / fps if fps > 0 else 0,
            codec=codec
        )
        
        #Estimate memory usage
        self.estimated_frame_size_mb = (width * height * 3) / (1024 * 1024)
        self.estimated_chunk_memory_mb = self.estimated_frame_size_mb * self.chunk_size
        
        #Check if chunk size is reasonable
        if self.estimated_chunk_memory_mb > self.max_memory_gb * 1024:
            suggested_chunk_size = int((self.max_memory_gb * 1024) / self.estimated_frame_size_mb)
            logger.warning(f"Chunk size {self.chunk_size} may use {self.estimated_chunk_memory_mb:.1f}MB. "
                          f"Consider reducing to {suggested_chunk_size}")
        
        logger.info(f"Initialized video processor:\n{self.metadata}")
        logger.info(f"Chunk size: {self.chunk_size}, Overlap: {self.overlap}")
        logger.info(f"Estimated memory per chunk: {self.estimated_chunk_memory_mb:.1f}MB")

    def get_next_chunk(self) -> Optional[FrameChunk]:
        if not self.cap or not self.cap.isOpened():
            return None
        
        #Check if we've reached max frames
        if self.max_frames and self.frames_processed >= self.max_frames:
            return None
        
        frames = []
        timestamps = []
        start_index = self.current_frame_index
        
        frames_to_read = self.chunk_size
        if self.max_frames:
            frames_to_read = min(frames_to_read, self.max_frames - self.frames_processed)
        
        #Read frames for this chunk
        frames_read = 0
        while frames_read < frames_to_read:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            #Handle frame skipping
            if self.skip_frames > 0 and self.current_frame_index % (self.skip_frames + 1) != 0:
                self.current_frame_index += 1
                continue
            
            #Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frames.append(frame_rgb)
            timestamps.append(self.current_frame_index / self.metadata.fps)
            
            #Update overlap buffer (keep last N frames for next chunk)
            if frames_read >= self.chunk_size - self.overlap:
                self.overlap_buffer.append(frame_rgb)
                self.overlap_timestamps_buffer.append(self.current_frame_index / self.metadata.fps)
            
            self.current_frame_index += 1
            frames_read += 1
        
        if not frames:
            return None
        
        chunk = FrameChunk(
            frames=frames,
            start_index=start_index,
            end_index=self.current_frame_index,
            timestamps=timestamps
        )
        
        self.frames_processed += len(frames)
        self.chunks_processed += 1
        
        logger.info(f"Chunk {self.chunks_processed}: Frames {start_index}-{self.current_frame_index-1} "
                   f"({len(frames)} frames, {chunk.duration:.2f}s)")
        
        return chunk
    
    def get_next_window(self) -> Optional[ProcessingWindow]:
        chunk = self.get_next_chunk()
        
        if chunk is None:
            return None
        
        #Create processing window with overlap
        window = ProcessingWindow(
            current_chunk=chunk,
            overlap_frames=list(self.overlap_buffer),
            overlap_timestamps=list(self.overlap_timestamps_buffer),
            window_size=self.chunk_size + self.overlap
        )
        
        return window
    
    def process_in_chunks(self, 
                         processing_func: Callable[[FrameChunk], Any],
                         with_overlap: bool = False #True, Processing window has overlap
                         ) -> Generator[Any, None, None]:
        logger.info(f"Starting chunk processing of {self.video_path.name}")
        start_time = time.time()
        
        try:
            while True:
                if with_overlap:
                    window = self.get_next_window()
                    if window is None:
                        break
                    result = processing_func(window)
                else:
                    chunk = self.get_next_chunk()
                    if chunk is None:
                        break
                    result = processing_func(chunk)
                
                yield result
                
                #Log progress
                if self.chunks_processed % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_processed = self.frames_processed / elapsed if elapsed > 0 else 0
                    eta = (self.metadata.total_frames - self.frames_processed) / fps_processed if fps_processed > 0 else 0
                    
                    logger.info(f"Progress: {self.frames_processed}/{self.metadata.total_frames} frames "
                               f"({self.frames_processed/self.metadata.total_frames*100:.1f}%), "
                               f"Speed: {fps_processed:.1f} fps, ETA: {eta:.1f}s")
        
        finally:
            elapsed = time.time() - start_time
            logger.info(f"Chunk processing complete: {self.frames_processed} frames in {elapsed:.1f}s "
                       f"({self.frames_processed/elapsed:.1f} fps)")
    
    def seek_to_frame(self, frame_index: int):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            self.current_frame_index = frame_index
            self.overlap_buffer.clear()
            self.overlap_timestamps_buffer.clear()
    
    def reset(self):
        self.seek_to_frame(0)
        self.frames_processed = 0
        self.chunks_processed = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'video_path': str(self.video_path),
            'frames_processed': self.frames_processed,
            'chunks_processed': self.chunks_processed,
            'total_frames': self.metadata.total_frames,
            'progress_percent': (self.frames_processed / self.metadata.total_frames * 100 
                               if self.metadata.total_frames > 0 else 0),
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'estimated_memory_mb': self.estimated_chunk_memory_mb
        }
    
    def cleanup(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info(f"FrameProcessor cleaned up. Processed {self.frames_processed} frames in {self.chunks_processed} chunks")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    #TODO - Implement parallel processing
    # def process_parallel_chunks(self,
    #                            processing_func: Callable,
    #                            num_workers: int = 2) -> Generator[Any, None, None]:
        
    #     with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #         #Submit first batch of chunks
    #         futures = []
    #         for _ in range(num_workers):
    #             chunk = self.get_next_chunk()
    #             if chunk:
    #                 future = executor.submit(processing_func, chunk)
    #                 futures.append((future, chunk.start_index))
            
    #         #Process results and submit new chunks
    #         while futures:
    #             #Wait for any future to complete
    #             done, pending = as_completed(futures), []
                
    #             for future in done:
    #                 #Get result
    #                 result = future.result()
    #                 yield result
                    
    #                 #Submit next chunk
    #                 chunk = self.get_next_chunk()
    #                 if chunk:
    #                     new_future = executor.submit(processing_func, chunk)
    #                     pending.append((new_future, chunk.start_index))
                
    #             futures = pending