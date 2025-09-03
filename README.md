# This is the repository that will hold versions of Computer vision Pipelines for sports use cases.
The goal is to make sports videos into 3d models such that they can be replayed in a fashion such as motion capture. 
This is basically what motion capture is, but implemented solely using computer vision models.
Other similar products include DeepMotion.com and Plask.ai

V1.0 implementations use MediaPipe for player/pose detection and YOLO for sports equipment detection. 

V2.0 built upon V1.0 but combined 2d for mediapipe and player pose into 3d. The earlier made a worse implementation of trying to combine 2d object detection with 3d pose data.

V3.0 complete restructure from the depth estimation in a way sophisticated guessing with MIDAS proper depth estimation. This gives a proper depth perception for the 3d conversion.
