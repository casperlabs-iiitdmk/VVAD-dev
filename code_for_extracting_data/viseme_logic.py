import os
import numpy as np

class VisemeFeatureExtractor:
    """
    High-precision mathematical feature extractor for viseme detection.
    Processes video frames to extract an optimized 401-dimensional vector:
      - Elements 0:200   -> Dynamic continuous spatial intensity profile
      - Elements 200:400 -> Un-clipped organic temporal motion delta
      - Element 400      -> Normalized sub-pixel horizontal tracking anchor
    """
    def __init__(self, num_points=200, alpha=0.20):
        self.num_points = num_points
        self.alpha = alpha  # Smoothing coefficient for EMA filter
        self.reset()
        
    def reset(self):
        """Resets temporal history buffers between separate video files."""
        self.prev_float_frame = None
        self.smoothed_centroid = None

    def extract(self, gray_frame):
        """
        Executes sub-pixel tracking, spatial grid sampling, and temporal scaling.
        Expects a 2D numpy array representing a single grayscale frame (0-255).
        """
        # 1. Normalize to continuous float space [0.0, 1.0] to protect resolution
        img_float = gray_frame.astype(np.float32) / 255.0
        h, w = img_float.shape

        # 2. Compute analytical Center of Mass (Sub-pixel Centroid tracking)
        y_indices, x_indices = np.indices((h, w))
        total_mass = np.sum(img_float)
        
        raw_cx = np.sum(x_indices * img_float) / total_mass if total_mass > 1e-6 else w / 2.0
        raw_cy = np.sum(y_indices * img_float) / total_mass if total_mass > 1e-6 else h / 2.0

        # 3. Apply Exponential Moving Average (EMA) to kill anchor jitter
        if self.smoothed_centroid is None:
            self.smoothed_centroid = np.array([raw_cx, raw_cy], dtype=np.float32)
        else:
            self.smoothed_centroid = (self.alpha * np.array([raw_cx, raw_cy], dtype=np.float32) + 
                                      (1.0 - self.alpha) * self.smoothed_centroid)
            
        cx, cy = self.smoothed_centroid

        # 4. Construct a non-linear sinusoidal grid trajectory across the mouth region
        t = np.linspace(-1.0, 1.0, self.num_points)
        sample_x = np.clip(cx + t * (w * 0.2), 0, w - 1).astype(np.int32)
        sample_y = np.clip(cy + np.sin(t * np.pi) * (h * 0.1), 0, h - 1).astype(np.int32)
        
        # Extract continuous intensities along the spatial profile
        sampled_intensities = img_float[sample_y, sample_x]

        # 5. Compute the verified, un-clipped organic temporal gradient
        if self.prev_float_frame is None:
            temporal_gradient = np.zeros(self.num_points, dtype=np.float32)
        else:
            prev_sampled = self.prev_float_frame[sample_y, sample_x]
            # Scaling up by 255.0 matches the scale of structural intensities perfectly
            temporal_gradient = np.abs(sampled_intensities - prev_sampled) * 255.0

        # Cache current frame as historical reference for the next frame step
        self.prev_float_frame = img_float

        # 6. Structural assembly of the rigid 401-dimensional configuration vector
        feature_vector = np.zeros(401, dtype=np.float32)
        feature_vector[0:200] = sampled_intensities * 255.0   # Scale space back to 0-255 balance
        feature_vector[200:400] = temporal_gradient           # Micro-movement signal matrix
        feature_vector[400] = cx / float(w)                    # Normalized tracking coordinate

        return feature_vector


def map_char_to_viseme(char):
    """Maps standard characters into distinct visemic class blocks (0-6)."""
    char = char.upper()
    if char in ['B', 'P', 'M']: return 1
    if char in ['F', 'V']: return 2
    if char in ['T', 'D', 'S', 'Z', 'C']: return 3
    if char in ['W', 'R', 'Q']: return 4
    if char in ['G', 'J', 'K', 'X', 'Y']: return 5
    if char in ['A', 'E', 'I', 'O', 'U', 'H', 'L', 'N']: return 6
    return 0 


def parse_align_file(align_path, num_frames, fps=25):
    """
    Parses dynamic timestamps from dataset alignment files (.align),
    distributes phonemes across time, and outputs frame-accurate class labels.
    """
    labels = np.zeros(num_frames, dtype=np.int64)
    if not os.path.exists(align_path): 
        return labels
        
    try:
        with open(align_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3: continue
            
            numbers = []
            word_token = ""
            for item in parts:
                try: numbers.append(float(item))
                except ValueError: word_token = item

            if len(numbers) < 2 or not word_token: continue
            start_val, end_val = numbers[0], numbers[1]
            word_clean = word_token.strip().upper()

            # Filter out systemic background/silence tokens
            if word_clean in ['SIL', 'SP', 'VAC', '']: continue

            # Handle variant timestamp formats (Milliseconds vs Seconds)
            if start_val > 5000:
                start_frame = int(start_val / 1000)
                end_frame = int(end_val / 1000)
            elif start_val > num_frames:
                start_frame = int((start_val / 1000.0) * fps)
                end_frame = int((end_val / 1000.0) * fps)
            else:
                start_frame = int(start_val * fps) if (start_val < num_frames / fps) else int(start_val)
                end_frame = int(end_val * fps) if (end_val < num_frames / fps) else int(end_val)

            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = max(0, min(end_frame, num_frames))

            # Distribute string characters linearly across the word's time frame window
            if start_frame < end_frame and len(word_clean) > 0:
                frames_per_char = max(1, (end_frame - start_frame) // len(word_clean))
                current_frame = start_frame
                for char in word_clean:
                    viseme_class = map_char_to_viseme(char)
                    limit = min(current_frame + frames_per_char, end_frame)
                    labels[current_frame:limit] = viseme_class
                    current_frame = limit
    except Exception:
        pass
        
    return labels