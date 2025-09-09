'''
Neural Depth Estimation using MiDaS
Provides pixel-level depth maps for accurate 3D reconstruction
Longer description at the bottom.
'''

import torch
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''Single depth map with metadata'''
@dataclass
class DepthMap:
    depth_data: np.ndarray #Normalized depth values (0-1)
    raw_depth: np.ndarray #Raw MiDaS output
    frame_index: int
    timestamp: float
    input_shape: Tuple[int, int] #Original image dimensions
    confidence: float #Overall depth quality metric

    '''Sample depth at normalized coordinates'''
    def sample_depth(self, x: float, y: float) -> float:
        
        #Convert normalized coords to pixel coords
        px = int(x * self.input_shape[1])
        py = int(y * self.input_shape[0])
        
        #Bounds checking
        px = np.clip(px, 0, self.input_shape[1] - 1)
        py = np.clip(py, 0, self.input_shape[0] - 1)
        
        return float(self.depth_data[py, px])
    
    '''Get depth values for a bounding box region'''
    def get_depth_region(self, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        
        x1, y1, x2, y2 = bbox
        
        #Convert to pixel coordinates
        px1 = int(x1 * self.input_shape[1])
        py1 = int(y1 * self.input_shape[0])
        px2 = int(x2 * self.input_shape[1])
        py2 = int(y2 * self.input_shape[0])
        
        return self.depth_data[py1:py2, px1:px2]

'''
MiDaS-based depth estimation for monocular video
Converts RGB frames to depth maps for 3D reconstruction
'''
class DepthEstimator:

    MODELS = {
        'small': 'MiDaS_small', #Fastest, lowest quality (good for M1 Mac)
        'hybrid': 'DPT_Hybrid', #Balanced (recommended start)
        'large': 'DPT_Large', #Best quality but memory intensive
    }
    
    def __init__(self, 
                 model_type: str = 'small', #Default with small
                 device: str = 'auto', #Computational device: auto, cpu, cuda, mps
                 optimize_memory: bool = True):
        
        self.model_type = model_type
        self.optimize_memory = optimize_memory
        
        #Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps') #M1 Mac
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f'DepthEstimator using device: {self.device}')
        
        #Load model
        self.model = None
        self.transform = None
        self._load_model()
        
        #Statistics
        self.frames_processed = 0
        self.total_processing_time = 0.0
        
    def _load_model(self):
        
        try:
            #Load MiDaS from torch hub
            model_name = self.MODELS[self.model_type]
            logger.info(f'Loading MiDaS model: {model_name}')
            
            self.model = torch.hub.load('intel-isl/MiDaS', model_name) #Load the model from github
            self.model.to(self.device) #CPU/GPU/M1
            self.model.eval() #Change to prediction not training mode
            
            #Load transforms
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms') #Get the needed transforms
            
            #Pick the correct trasnform
            if self.model_type == 'small':
                self.transform = midas_transforms.small_transform
            elif self.model_type in ['hybrid', 'large']:
                self.transform = midas_transforms.dpt_transform
            
            logger.info(f'MiDaS {model_name} loaded successfully')
            
            #Memory optimization for M1
            if self.optimize_memory and self.device.type == 'mps':
                torch.mps.set_per_process_memory_fraction(0.0) #Don't pre-allocate gpu memory, only as needed
                logger.info('Memory optimization enabled for M1 Mac')
                
        except Exception as e:
            logger.error(f'Failed to load MiDaS model: {e}')
            raise

    #Estimate depth map for single frame
    def estimate_depth(self, 
                      frame: np.ndarray,
                      frame_index: int,
                      timestamp: float) -> Optional[DepthMap]:
        
        start_time = time.time()
        
        try:
            #Store original shape
            original_shape = frame.shape[:2]
            
            #Prepare frame for MiDaS by shrinking the px size
            input_batch = self.transform(frame).to(self.device)
            
            #Run inference with no gradient as its not in training
            with torch.no_grad():
                prediction = self.model(input_batch)
                
                #Resize to original resolution using bicubic interpolation
                prediction = torch.nn.functional.interpolate( 
                    prediction.unsqueeze(1), #Add dimension description
                    size=original_shape,
                    mode='bicubic',
                    align_corners=False 
                ).squeeze() #Remove dimension description
            
            #Convert from gpu to cpu and then from torch tensor to numpy
            raw_depth = prediction.cpu().numpy()
            
            #Normalize depth to 0-1 range
            depth_min = raw_depth.min()
            depth_max = raw_depth.max()
            
            if depth_max - depth_min > 0:
                normalized_depth = (raw_depth - depth_min) / (depth_max - depth_min)
            else:
                normalized_depth = np.zeros_like(raw_depth)
            
            #Calculate confidence (based on depth variance)
            confidence = self._calculate_depth_confidence(normalized_depth)
            
            #Update statistics
            self.frames_processed += 1
            self.total_processing_time += time.time() - start_time
            
            #Clear cache if memory optimization enabled
            if self.optimize_memory and self.device.type == 'mps':
                torch.mps.empty_cache()
            
            return DepthMap(
                depth_data=normalized_depth,
                raw_depth=raw_depth,
                frame_index=frame_index,
                timestamp=timestamp,
                input_shape=original_shape,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f'Depth estimation failed for frame {frame_index}: {e}')
            return None
        
    # def estimate_batch(self, 
    #                   frames: List[np.ndarray],
    #                   start_index: int = 0,
    #                   fps: float = 30.0) -> List[Optional[DepthMap]]:
        
    #     depth_maps = []
        
    #     for i, frame in enumerate(frames):
    #         frame_index = start_index + i
    #         timestamp = frame_index / fps
            
    #         depth_map = self.estimate_depth(frame, frame_index, timestamp)
    #         depth_maps.append(depth_map)
            
    #         #Log progress
    #         if (i + 1) % 10 == 0:
    #             avg_time = self.total_processing_time / self.frames_processed
    #             logger.info(f'Processed {i+1}/{len(frames)} frames ({avg_time:.3f}s/frame)')
        
    #     return depth_maps
    
    def _calculate_depth_confidence(self, depth_map: np.ndarray) -> float:
        
        #Compute the standard deviation along the specified axis.
        depth_std = np.std(depth_map)
        
        #Check for edge coherence (gradient magnitude)
        grad_x = np.abs(np.diff(depth_map, axis=1))
        grad_y = np.abs(np.diff(depth_map, axis=0))
        edge_strength = np.mean(grad_x) + np.mean(grad_y)
        
        #Combine metrics
        if depth_std < 0.05: #Too flat
            confidence = 0.3
        elif depth_std > 0.4: #Good variation
            confidence = 0.9
        else:
            confidence = 0.5 + depth_std
        
        #Adjust for edge quality
        if edge_strength > 0.1:
            confidence = min(1.0, confidence * 1.1)
        
        return float(confidence)
    
    def get_statistics(self) -> dict:
        
        avg_time = self.total_processing_time / self.frames_processed if self.frames_processed > 0 else 0
        
        return {
            'frames_processed': self.frames_processed,
            'total_time': self.total_processing_time,
            'avg_time_per_frame': avg_time,
            'fps': 1.0 / avg_time if avg_time > 0 else 0,
            'model_type': self.model_type,
            'device': str(self.device)
        }
    
    def cleanup(self):

        if self.model:
            del self.model
            self.model = None
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()
        
        logger.info(f'DepthEstimator cleaned up. Processed {self.frames_processed} frames')


'''
Longer description: The depth estimator transforms 2D video frames into 3D depth estimations. 
During initialization, it downloads a pre-trained MiDaS neural network from GitHub (then cached locally after 
first use) and selects the appropriate preprocessing transforms for the chosen model variant. 
When processing each frame, the system first shrinks the input image to the model's 
expected resolution (around 384×384), feeds it through the neural network to generate a low-resolution depth 
prediction, then upscales this prediction back to the original frame size using bicubic interpolation. 
The raw neural network output gets normalized to a consistent 0-1 range where 0 represents 
the farthest points and 1 represents the closest points in the scene. It calculates a confidence 
score based on depth variation and edge sharpness, conveting everything into a DepthMap object. 
'''