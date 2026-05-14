import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# 1. THE EXTRACTOR (Pixels -> Math)
class VisemeFeatureExtractor:
    def __init__(self, num_points=200):
        self.num_points = num_points
        # Pre-generate the Fibonacci spiral points
        self.grid = self._generate_fibonacci_grid()

    def _generate_fibonacci_grid(self):
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))
        for i in range(self.num_points):
            r = math.sqrt(i / self.num_points)
            theta = i * phi
            points.append((r * math.cos(theta), r * math.sin(theta)))
        return np.array(points)

    def preprocess_frame(self, frame):
        # Applies 32-shade quantization (5-bit) via bit-shifting
        return (frame >> 3) << 3

    def extract(self, frame_raw, pose_type='A'):
        """
        Converts a single frame into your 401-dimension vector.
        """
        # A. Apply 32-shade filter
        clean_frame = self.preprocess_frame(frame_raw)
        
        # B. Mocking the logic for 200 points (to keep code clean)
        # In your final version, use the Barycentric centroid logic here
        magnitudes = np.random.uniform(0, 15, size=self.num_points) # Movement
        shade_means = np.random.uniform(0, 31, size=self.num_points) # Texture
        
        # C. Pose Bit (Case A=0.0, Case B=1.0)
        pose_bit = 0.0 if pose_type == 'A' else 1.0
        
        # Combine everything: 200 + 200 + 1 = 401
        return np.concatenate([magnitudes, shade_means, [pose_bit]])


# 2. THE BRAIN (Math -> Prediction)
class RobustVisemeNet(nn.Module):
    def __init__(self, num_points=200, hidden_dim=128):
        super(RobustVisemeNet, self).__init__()
        # Input: 200 (Mags) + 200 (Shades) + 1 (Pose) = 401
        self.input_dim = (num_points * 2) + 1 
        self.projection = nn.Linear(self.input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        # x shape: [Batch, Time, 401]
        x = F.relu(self.projection(x))
        gru_out, _ = self.gru(x)
        last_frame = gru_out[:, -1, :]
        return F.softmax(self.classifier(last_frame), dim=1)