import numpy as np 
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from typing import List, Optional
from src.pose_detector import PoseFrame, PoseKeyPoint

class DataSmoother:
    def __init__ (self, confidence_threshold=0.5, outlier_std_threshold=2.0, smoothing_window=5):
        self.confidence_threshold = confidence_threshold
        self.outlier_std_threshold = outlier_std_threshold
        self.smoothing_window = smoothing_window
    
    #Apply temporal smoothing to the entire pose sequence
    def smooth_pose_sequence(self, pose_frames: List[PoseFrame]) ->  List[PoseFrame]:
        
        if len(pose_frames) < 3:
            print(f'Not enough frames for smoothing with {pose_frames}, returning original data')
            return pose_frames
        
        #Convert the pose_frames into numpy arrays 
        keypoint_data = self._extract_keypoint_arrays(pose_frames)

        #Apply smoothing to each keypoint
        smoothed_data = {}
        #MediaPipe has 33 keypoints
        for keypoint_idx in range(33): 
            #keypoint_name = f'keypoint_{keypoint_idx}' 
            
            if keypoint_idx in keypoint_data:
                #Smooth coords separately
                smoothed_coords = self._smooth_keypoint_trajectory(keypoint_data[keypoint_idx], keypoint_idx)
                smoothed_data[keypoint_idx] = smoothed_coords

        #Reconstruct the pose frames using the smoothed data
        smoothed_frames = self._reconstruct_pose_frames(pose_frames, smoothed_data)

        return smoothed_frames
    
    #Convert keypoint coordinates into numpy arrays
    def _extract_keypoint_arrays(self, pose_frames:List[PoseFrame]) -> dict:
        keypoint_data = {}

        for frame in pose_frames:
            for kp_idx, keypoint in enumerate(frame.keypoints):
                if kp_idx not in keypoint_data:
                    keypoint_data[kp_idx] = {
                        'x': [],
                        'y': [],
                        'z': [], 
                        'visibility': [],
                        'presence': []
                    }

                keypoint_data[kp_idx]['x'].append(keypoint.x)
                keypoint_data[kp_idx]['y'].append(keypoint.y)
                keypoint_data[kp_idx]['z'].append(keypoint.z)
                keypoint_data[kp_idx]['visibility'].append(keypoint.visibility)
                keypoint_data[kp_idx]['presence'].append(keypoint.presence)

        #Convert lists to numpy arrays
        for kp_idx in keypoint_data:
            for coord in keypoint_data[kp_idx]:
                keypoint_data[kp_idx][coord] = np.array(keypoint_data[kp_idx][coord])
            
        return keypoint_data
    
    def _smooth_keypoint_trajectory(self, keypoint_coords:dict, keypoint_idx: int ) -> dict:
        
        smoothed = {}

        for coord in ['x', 'y', 'z']:
            original = keypoint_coords[coord]
            confidence = keypoint_coords['visibility']

            #1.) Handle low confidence points
            cleaned = self._handle_low_confidence_points(original, confidence)

            #2.) Remove the outliers
            cleaned = self._remove_outliers(cleaned)

            #3.) Apply temporal smoothing filter
            if len(cleaned) >= self.smoothing_window:
                #Apply Savitzky Golay filter(sports movements)
                try:
                    smoothed_coord = signal.savgol_filter(
                        cleaned, 
                        window_length=min(self.smoothing_window, len(cleaned)),
                        polyorder=2,
                        mode='nearest'
                    )
                except:
                    #Simple moving average
                    smoothed_coord = self._moving_average(cleaned, self.smoothing_window)
            else:
                smoothed_coord = cleaned
            
            smoothed[coord] = smoothed_coord

        smoothed['visibility'] = keypoint_coords['visibility']
        smoothed['presence'] = keypoint_coords['presence']

        return smoothed
    
    def _handle_low_confidence_points(self, coordinates: np.ndarray, confidence: np.ndarray) -> np.ndarray:

        #Find high confidence points
        valid_mask = confidence >= self.confidence_threshold

        if np.sum(valid_mask) < 2:
            #Too few valid points. Return original
            return coordinates
        
        #Iterpolate missing points
        valid_indices = np.where(valid_mask)[0]
        valid_coords = coordinates[valid_mask]

        #Create interpolation function
        if len(valid_indices) >= 2:
            interpulation_function = interp1d(valid_indices, valid_coords, kind='linear', bounds_error=False, fill_value='extrapolate')
 
            #Apply interpolation on all points
            all_indices = np.arange(len(coordinates))
            interpolated = interpulation_function(all_indices)

            return interpolated 
        
        return coordinates        

    #Remove statistical outliers from trajectory
    def _remove_outliers(self, coordinates: np.ndarray) -> np.ndarray:

        if len(coordinates) <= 5:
            return coordinates
        
        #Calculate rolling statistics
        mean = np.mean(coordinates)
        std = np.std(coordinates)

        #Identify the outliers
        outliers_mask = np.abs(coordinates - mean) > (self.outlier_std_threshold * std)

        #Replace the outliers with interpolated values
        if np.sum(outliers_mask) > 0:
            clean_coords = coordinates.copy()

            for i in range(len(coordinates)):
                if outliers_mask[i]:
                    #Simple interpolation between neighbor
                    if i > 0 and i < len(coordinates) - 1:
                        clean_coords[i] = (coordinates[i-1] + coordinates[i+1]) / 2
                    elif i == 0:
                        clean_coords[i] = coordinates[1]
                    else:
                        clean_coords[i] = coordinates[-2]

            return clean_coords
        
        return coordinates
    
    #Moving average smoothing
    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:

        if len(data) < window:
            return data
        
        #Using pandas for the rolling average
        df = pd.Series(data)
        smoothed = df.rolling(window=window, center=True, min_periods=1).mean()
        return smoothed.values
    
    #Reconstruct PoseFrame objects with smoothed keypoint data
    def _reconstruct_pose_frames(self, original_frames: List[PoseFrame], smoothed_data: dict) -> List[PoseFrame]:

        smoothed_frames = []

        for frame_idx, original_frame in enumerate(original_frames):
            #New keypoints with smoothed coordinates
            smoothed_keypoints = []

            for kp_idx in range(len(original_frame.keypoints)):
                if kp_idx in smoothed_data:
                    #Use the smoothed coordinates
                    smoothed_kp = PoseKeyPoint(
                        x=float(smoothed_data[kp_idx]['x'][frame_idx]),
                        y=float(smoothed_data[kp_idx]['y'][frame_idx]),
                        z=float(smoothed_data[kp_idx]['z'][frame_idx]),
                        visibility=float(smoothed_data[kp_idx]['visibility'][frame_idx]),
                        presence=float(smoothed_data[kp_idx]['presence'][frame_idx])
                    )
                else:
                    #Use original
                    smoothed_kp = original_frame.keypoints[kp_idx]
                
                smoothed_keypoints.append(smoothed_kp)

            #Create a new frame with smoothed data
            smoothed_frame = PoseFrame(
                frame_index=original_frame.frame_index,
                timestamp=original_frame.timestamp,
                keypoints=smoothed_keypoints,
                detection_confidence=original_frame.detection_confidence
            )

            smoothed_frames.append(smoothed_frame)
        
        return smoothed_frames
            