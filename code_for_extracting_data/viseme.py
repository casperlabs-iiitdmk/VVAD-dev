import cv2
import numpy as np
import os
# Import the extractor from the code we wrote earlier
from viseme_logic import VisemeFeatureExtractor 

extractor = VisemeFeatureExtractor()

def process_video_to_math(video_path, save_path):
    # 1. Open the raw video
    cap = cv2.VideoCapture(video_path)
    
    math_vectors = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        # Convert frame to grayscale for our 32-shade logic
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. EXTRACT THE JUICE! (This does the 32-shades, Fibonacci, etc.)
        # Assuming frontal face for this specific frame
        vector_401 = extractor.extract(gray, pose_type='A') 
        
        math_vectors.append(vector_401)

    cap.release()
    
    # 3. SAVE THE JUICE (Do NOT use JSON. Use NumPy)
    # math_vectors is now a list of arrays. Convert to one giant array.
    final_array = np.array(math_vectors) 
    
    # Save it to your hard drive as a .npy file
    np.save(save_path, final_array)
    print(f"Saved {save_path} - Shape: {final_array.shape}")

# Example: Run this on all your LRS3 videos
process_video_to_math("LRS3_raw_videos/video_001.mp4", "processed_data/video_001_features.npy")