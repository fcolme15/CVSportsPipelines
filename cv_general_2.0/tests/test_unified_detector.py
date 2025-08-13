'''
Test Script for Unified Detector Component
Manual Input: python test_unified_detector.py --video sample.mp4
'''
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import time
import numpy as np
import cv2
from pathlib import Path
import argparse

#Import component to test
from src.unified_detector import UnifiedDetector, Unified2DFrame

#Test that components can be enabled/disabled properly
def test_component_composition():
    print("TEST: Component Composition")
    print("-" * 100)
    
    #Test with YOLO disabled
    detector_no_yolo = UnifiedDetector(enable_yolo=False)
    assert detector_no_yolo.pose_detector is not None, "Pose detector missing"
    assert detector_no_yolo.object_detector is None, "YOLO should be disabled"
    detector_no_yolo.cleanup()
    print("Detector without YOLO works")
    
    #Test with YOLO enabled
    detector_with_yolo = UnifiedDetector(enable_yolo=True)
    assert detector_with_yolo.pose_detector is not None, "Pose detector missing"
    assert detector_with_yolo.object_detector is not None, "YOLO should be enabled"
    print("Detector with YOLO works")
    
    detector_with_yolo.cleanup()
    return True

#Test sport context passing & application
def test_context_handling(video_path):
    print("TEST: Context Handling")
    print("-" * 100)
    
    detector = UnifiedDetector(enable_yolo=True)
    
    #Get first frame from video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError("Could not read video frame")
    
    contexts = [
        {'sport_type': 'soccer'},
        {'sport_type': 'basketball'},
        {'sport_type': 'tennis'},
        None #No context
    ]
    
    for context in contexts:
        unified_frame = detector.process_frame(frame, 0, 0.0, context)
        assert unified_frame is not None, f"Failed with context {context}"
        context_name = context['sport_type'] if context else 'none'
        print(f"Context '{context_name}' processed")
    
    detector.cleanup()
    return True

def test_video_processing(video_path, max_frames=100):
    print("TEST: Video Processing")
    print("-" * 100)
    
    detector = UnifiedDetector(enable_yolo=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_times = []
    poses_detected = 0
    total_objects = 0
    frame_count = 0
    
    print(f"Processing up to {max_frames} frames.")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        #Process frame
        start = time.time()
        unified_frame = detector.process_frame(
            frame, frame_count, frame_count/fps,
            context={'sport_type': 'soccer'}
        )
        frame_times.append(time.time() - start)
        
        #Verify we get valid output
        assert unified_frame is not None, "Process frame returned None"
        assert hasattr(unified_frame, 'pose_2d'), "Missing pose_2d attribute"
        assert hasattr(unified_frame, 'objects_2d'), "Missing objects_2d attribute"
        assert unified_frame.frame_index == frame_count, "Frame index mismatch"
        
        if unified_frame.pose_2d:
            poses_detected += 1
        total_objects += len(unified_frame.objects_2d)
        
        frame_count += 1
        
        if frame_count % 25 == 0:
            print(f"    Processed {frame_count} frames...")
    
    cap.release()
    
    #Calculate metrics
    avg_time = np.mean(frame_times) * 1000
    pose_rate = poses_detected / frame_count
    avg_objects = total_objects / frame_count
    processing_fps = 1000 / avg_time
    
    print(f"Results for {frame_count} frames:")
    print(f"    Pose detection rate: {pose_rate:.1%}")
    print(f"    Avg objects/frame: {avg_objects:.2f}")
    print(f"    Avg time: {avg_time:.2f}ms")
    print(f"    Processing speed: {processing_fps:.1f} fps")
    
    #Actual pass/fail conditions
    assert pose_rate >= 0.5, f"Pose detection rate too low: {pose_rate:.1%}"
    assert processing_fps >= 10, f"Processing too slow: {processing_fps:.1f} fps"
    assert avg_time < 200, f"Frame processing too slow: {avg_time:.2f}ms"
    
    print("All performance criteria met")
    
    detector.cleanup()
    return True


def test_batch_processing(video_path):
    print("TEST: Batch Processing")
    print("-" * 100)
    
    detector = UnifiedDetector(enable_yolo=False) #Faster without YOLO
    
    #Get frames from video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    batch_size = 10
    frames = []
    for _ in range(batch_size):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    
    if len(frames) < batch_size:
        print(f"Warning: Only got {len(frames)} frames")
        batch_size = len(frames)
    
    #Process batch
    start = time.time()
    results = detector.process_batch(frames, start_index=0, fps=fps)
    batch_time = time.time() - start
    
    #Verify results
    assert len(results) == batch_size, f"Expected {batch_size} results, got {len(results)}"
    
    for i, result in enumerate(results):
        assert result.frame_index == i, f"Frame index mismatch at {i}"
        assert result.timestamp == i/fps, f"Timestamp mismatch at {i}"
    
    print(f"Processed {batch_size} frames in {batch_time:.2f}s")
    print(f"Avg per frame: {batch_time/batch_size*1000:.2f}ms")
    
    detector.cleanup()
    return True

def test_unified_detector():
    manualInput = True
    
    if manualInput:
        parser = argparse.ArgumentParser(description='Test Unified Detector')
        parser.add_argument('--video', required=True, help='Path to test video')
        args = parser.parse_args()
        videoPath = args.video
    else:
        videoPath = 'Path' 
    
    #Check video exists
    if not Path(videoPath).exists():
        print(f"Error: Video not found at {videoPath}")
        print("A video is required for testing")
        sys.exit(1)
    
    print("="*100)
    print("UNIFIED DETECTOR - TEST SUITE")
    print("="*100)
    
    tests_passed = 0
    tests_failed = 0
    
    #Run tests
    tests = [
        ("Component Composition", lambda: test_component_composition()),
        ("Context Handling", lambda: test_context_handling(videoPath)),
        ("Video Processing", lambda: test_video_processing(videoPath)),
        ("Batch Processing", lambda: test_batch_processing(videoPath))
    ]
    
    for test_name, test_func in tests:
        try:
            test_func()
            tests_passed += 1
        except AssertionError as e:
            print(f"✗ {test_name} failed: {e}")
            tests_failed += 1
        except Exception as e:
            print(f"✗ {test_name} error: {e}")
            tests_failed += 1
    
    #Summary
    print("="*100)
    print("TEST SUMMARY")
    print("="*100)
    total = tests_passed + tests_failed
    pass_rate = (tests_passed / total * 100) if total > 0 else 0
    
    print(f"Tests: {tests_passed}/{total} passed ({pass_rate:.0f}%)")
    
    if pass_rate == 100:
        print("All tests passed! UnifiedDetector working correctly")
    elif pass_rate >= 75:
        print("Warning: Some tests failed")
    else:
        print("Error: Multiple test failures")
    
    sys.exit(0 if tests_failed == 0 else 1)

if __name__ == "__main__":
    test_unified_detector()
    