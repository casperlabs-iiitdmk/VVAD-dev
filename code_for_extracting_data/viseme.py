import imageio.v3 as iio
import numpy as np
import os
# Import the extractor from the code we wrote earlier
from viseme_logic import VisemeFeatureExtractor 

extractor = VisemeFeatureExtractor()

def process_video_to_math(video_path, save_path):
    # Quick safety check: does the video actually exist?
    if not os.path.exists(video_path):
        print(f"❌ ERROR: Could not find the video at {video_path}")
        return

    math_vectors = []
    
    try:
        # 1. Open the raw video using imageio (bypasses OpenCV .mpg codec issues)
        for frame in iio.imiter(video_path):
            
            # Convert frame to grayscale for our 32-shade logic
            # (imageio reads in RGB, so we do standard math to make it grayscale)
            gray = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            
            # 2. EXTRACT THE JUICE! (This does the 32-shades, Fibonacci, etc.)
            # Assuming frontal face for this specific frame
            vector_401 = extractor.extract(gray, pose_type='A') 
            
            math_vectors.append(vector_401)
            
    except Exception as e:
        print(f"❌ ERROR processing video: {e}")
        return

    # 3. SAVE THE JUICE (Do NOT use JSON. Use NumPy)
    final_array = np.array(math_vectors) 
    
    # If the array is still empty, something is severely wrong with the video file
    if len(final_array) == 0:
        print("❌ ERROR: 0 frames extracted. The video file might be corrupted.")
        return

    # Automatically create the 'processed_data' folder if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save it to your hard drive as a .npy file
    np.save(save_path, final_array)
    print(f"✅ Saved {save_path} - Shape: {final_array.shape}")

# Example: Run this on your GRID .mpg video
process_video_to_math("LRS3_raw_videos", "processed_data")