import sys
import os
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.video_processor import VideoProcessor
from src.object_detector import ObjectDetector, DetectedObject, ObjectFrame

#Test basic YOLO object detection functionality
def test_object_detection_basic():
    
    print("Testing YOLO Object Detection - Basic Functionality")
    print("=" * 100)
    
    
    video_path = './data/trimmed_data/elastico_skill_1.mp4'
    
    try:
        #Initialize components
        video_processor = VideoProcessor(video_path)
        object_detector = ObjectDetector(
            model_size='n', #Nano for speed
            confidence_threshold=0.3,
            sports_objects_only=True
        )
        
        #Load video
        video_processor.load_video()
        print(f"Video loaded: {video_processor.metadata}")
        
        #Initialize YOLO model
        print("Initializing YOLO model...")
        if not object_detector.initialize_model():
            print("Failed to initialize YOLO model")
            return
        
        #Double-check model initialization
        if object_detector.model is None:
            print("Model is None after initialization")
            return
        
        #Process frames
        object_frames = []
        fps = video_processor.metadata['fps']
        
        print(f"Processing frames for object detection...")
        start_time = time.time()
        
        #Process first 15 frames for testing
        for frame_idx, frame in video_processor.get_frames(max_frames=15):
            timestamp = frame_idx / fps
            
            object_frame = object_detector.process_frame(frame, frame_idx, timestamp)
            if object_frame:
                object_frames.append(object_frame)
        
        processing_time = time.time() - start_time
        
        print(f"Processed {len(object_frames)} frames in {processing_time:.2f} seconds")
        print(f"    Speed: {len(object_frames)/processing_time:.1f} FPS")
        
        #Show detection results
        print(f"Detection Results:")
        
        total_detections = 0
        frames_with_detections = 0
        
        for i, frame in enumerate(object_frames[:5]):  #Show first 5 frames
            if frame.detected_objects:
                frames_with_detections += 1
                total_detections += len(frame.detected_objects)
                
                print(f"Frame {frame.frame_index} (t={frame.timestamp:.3f}s):")
                print(f"    Detection confidence: {frame.detection_confidence:.3f}")
                
                for obj in frame.detected_objects:
                    print(f"    {obj.class_name}: {obj.confidence:.3f} confidence")
                    print(f"        Center: ({obj.center_x:.3f}, {obj.center_y:.3f})")
                    print(f"        Size: {obj.width:.3f} x {obj.height:.3f}")
            else:
                print(f"    Frame {frame.frame_index}: No objects detected")
        
        # Get overall statistics
        stats = object_detector.get_detection_statistics(object_frames)
        
        print(f"Overall Statistics:")
        print(f"    Total frames processed: {stats['total_frames']}")
        print(f"    Frames with detections: {stats['frames_with_detections']}")
        print(f"    Detection rate: {stats['frames_with_detections']/stats['total_frames']*100:.1f}%")
        
        if stats['object_counts']:
            print(f"Object Counts:")
            for obj_class, count in stats['object_counts'].items():
                avg_conf = stats['average_confidence'][obj_class]
                detection_rate = stats['detection_rate'][obj_class]
                print(f"    {obj_class}: {count} detections")
                print(f"        Avg confidence: {avg_conf:.3f}")
                print(f"        Detection rate: {detection_rate*100:.1f}%")
        
        print(f"Basic object detection test completed!")

        object_detector.cleanup()
        video_processor.cleanup()
        
    except Exception as e:
        print(f"Error in basic object detection test: {e}")
        import traceback
        traceback.print_exc()

#Test object tracking across frames
def test_object_tracking():
    
    print(f"Testing Object Tracking Across Frames")
    print("=" * 100)
    
    
    video_path = './data/trimmed_data/elastico_skill_1.mp4'
    
    try:
        #Initialize components
        video_processor = VideoProcessor(video_path)
        object_detector = ObjectDetector(
            model_size='n',
            confidence_threshold=0.25,
            sports_objects_only=True
        )
        
        video_processor.load_video()
        if not object_detector.initialize_model():
            print("Failed to initialize YOLO model")
            return
        if object_detector.model is None:
            print("Model is None after initialization")
            return
        
        #Process more frames for tracking test
        object_frames = []
        fps = video_processor.metadata['fps']
        
        print(f"Processing 20 frames for tracking test...")
        
        for frame_idx, frame in video_processor.get_frames(max_frames=20):
            timestamp = frame_idx / fps
            object_frame = object_detector.process_frame(frame, frame_idx, timestamp)
            
            if object_frame:
                #Apply sports enhancement
                enhanced_frame = object_detector.enhance_sports_detection(
                    object_frame, 
                    context_info={'sport_type': 'soccer'}
                )
                object_frames.append(enhanced_frame)
        
        print(f"Processed {len(object_frames)} frames")
        
        #Apply tracking
        print(f"Applying object tracking...")
        tracked_frames = object_detector.track_objects_across_frames(object_frames)
        
        #Compare before and after tracking
        print(f"Tracking Results:")
        
        before_stats = object_detector.get_detection_statistics(object_frames)
        after_stats = object_detector.get_detection_statistics(tracked_frames)
        
        print(f"    Before tracking:")
        for obj_class, avg_conf in before_stats['average_confidence'].items():
            print(f"        {obj_class}: {avg_conf:.3f} avg confidence")
        
        print(f"    After tracking:")
        for obj_class, avg_conf in after_stats['average_confidence'].items():
            print(f"        {obj_class}: {avg_conf:.3f} avg confidence")
        
        #Show tracking consistency
        if tracked_frames:
            print(f"Tracking Consistency Check:")
            prev_objects = {}
            consistent_tracks = 0
            total_comparisons = 0
            
            for frame in tracked_frames:
                current_objects = {}
                
                for obj in frame.detected_objects:
                    current_objects[obj.class_name] = (obj.center_x, obj.center_y)
                
                #Compare with previous frame
                for obj_class, (x, y) in current_objects.items():
                    if obj_class in prev_objects:
                        prev_x, prev_y = prev_objects[obj_class]
                        distance = ((x - prev_x)**2 + (y - prev_y)**2)**0.5
                        
                        if distance < 0.1:  #Reasonable movement threshold
                            consistent_tracks += 1
                        
                        total_comparisons += 1
                
                prev_objects = current_objects
            
            if total_comparisons > 0:
                consistency_rate = consistent_tracks / total_comparisons * 100
                print(f"    Tracking consistency: {consistency_rate:.1f}%")
                print(f"    ({consistent_tracks}/{total_comparisons} consistent tracks)")
        
        print(f"Object tracking test completed!")
        
        object_detector.cleanup()
        video_processor.cleanup()
        
    except Exception as e:
        print(f"Error in object tracking test: {e}")
        import traceback
        traceback.print_exc()


#Test sports specific detection enhancement
def test_sports_enhancement():
    
    print(f"Testing Sports-Specific Enhancement")
    print("=" * 100)
    
    
    try:
        #Test different sport contexts
        contexts = [
            {'sport_type': 'soccer'},
            {'sport_type': 'basketball'},
            {'sport_type': 'american_football'},
            {'sport_type': 'tennis'}
        ]
        
        object_detector = ObjectDetector(sports_objects_only=True)
        if not object_detector.initialize_model():
            print("Failed to initialize model for enhancement test")
            return
        
        print(f"Testing enhancement logic with different sport contexts:")
        
        #Create mock detection for testing (import classes directly)
        mock_ball = object_detector.__class__.__module__
        
        mock_ball = DetectedObject(
            class_name='sports_ball',
            confidence=0.85,
            bbox=(0.4, 0.3, 0.6, 0.5),
            center_x=0.5,
            center_y=0.4,
            width=0.2,
            height=0.2
        )
        
        mock_frame = ObjectFrame(
            frame_index=0,
            timestamp=0.0,
            detected_objects=[mock_ball],
            detection_confidence=0.85
        )
        
        for context in contexts:
            enhanced_frame = object_detector.enhance_sports_detection(mock_frame, context)
            
            sport_type = context['sport_type']
            enhanced_name = enhanced_frame.detected_objects[0].class_name
            
            print(f"    {sport_type} context: 'sports_ball' → '{enhanced_name}'")
        
        print(f"Sports enhancement test completed!")
        
        object_detector.cleanup()
        
    except Exception as e:
        print(f"Error in sports enhancement test: {e}")
        import traceback
        traceback.print_exc()

#Test performace with different YOLO Model sizes
def test_performance_comparison():
    
    print(f"Testing Performance with Different Model Sizes")
    print("=" * 100)
    
    
    
    video_path = './data/trimmed_data/elastico_skill_1.mp4'
    model_sizes = ['n', 's']  #Test nano and small models
    
    try:
        video_processor = VideoProcessor(video_path)
        video_processor.load_video()
        
        #Get test frames 
        test_frames = []
        fps = video_processor.metadata['fps']
        
        for frame_idx, frame in video_processor.get_frames(max_frames=10):
            timestamp = frame_idx / fps
            test_frames.append((frame, frame_idx, timestamp))
        
        print(f"Testing with {len(test_frames)} frames")
        
        results = {}
        
        for model_size in model_sizes:
            print(f"Testing YOLO{model_size.upper()} model.")
            
            detector = ObjectDetector(
                model_size=model_size,
                confidence_threshold=0.3,
                sports_objects_only=True
            )
            
            if not detector.initialize_model():
                print(f"Failed to initialize YOLO{model_size.upper()}")
                continue
            
            if detector.model is None:
                print(f"Model is None for YOLO{model_size.upper()}")
                continue
            
            #Time the processing
            start_time = time.time()
            detections = 0
            
            for frame, frame_idx, timestamp in test_frames:
                object_frame = detector.process_frame(frame, frame_idx, timestamp)
                if object_frame and object_frame.detected_objects:
                    detections += len(object_frame.detected_objects)
            
            processing_time = time.time() - start_time
            fps_performance = len(test_frames) / processing_time
            
            results[model_size] = {
                'processing_time': processing_time,
                'fps_performance': fps_performance,
                'total_detections': detections
            }
            
            print(f"Processing time: {processing_time:.2f}s")
            print(f"Performance: {fps_performance:.1f} FPS")
            print(f"Total detections: {detections}")
            
            detector.cleanup()
        
        #Compare results
        if len(results) > 1:
            print(f"Performance Comparison:")
            fastest = max(results.items(), key=lambda x: x[1]['fps_performance'])
            print(f"Fastest: YOLO{fastest[0].upper()} at {fastest[1]['fps_performance']:.1f} FPS")
            
            for model, data in results.items():
                print(f"YOLO{model.upper()}: {data['fps_performance']:.1f} FPS, {data['total_detections']} detections")
        
        print(f"Performance comparison completed!")
        
        video_processor.cleanup()
        
    except Exception as e:
        print(f"Error in performance test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("Starting Object Detection Tests")
    print("=" * 100)
    
    test_object_detection_basic()
    test_object_tracking()
    test_sports_enhancement()
    test_performance_comparison()
    
    print(f"All object detection tests completed!")