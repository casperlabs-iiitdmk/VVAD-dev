import numpy as np

# Load the ancient script back into a Python array
data = np.load("processed_data/video_001_features.npy")

print(f"Total Frames Processed: {data.shape[0]}")
print(f"Features per Frame: {data.shape[1]}")
print("-" * 30)

# Print the 401 numbers from the very FIRST frame (Frame 0)
print("Here is the math for Frame 1:")
print(np.round(data[0], 4)) # rounding to 4 decimal places so it's readable