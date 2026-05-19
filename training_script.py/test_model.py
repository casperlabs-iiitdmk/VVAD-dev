import os
import torch
import torch.nn as nn
import numpy as np

# ==========================================================
# 1. RECREATED MODEL ARCHITECTURE (Fully Inline)
# ==========================================================
class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  
        return self.relu(out)

class VisemeHybridPro(nn.Module):
    def __init__(self):
        super(VisemeHybridPro, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=401, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ResidualBlock1D(channels=256, kernel_size=3),
            nn.Dropout(0.4),  # Matches structural dropout from updated training configuration
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.memory = nn.GRU(input_size=128, hidden_size=64, num_layers=2, 
                             batch_first=True, bidirectional=True, dropout=0.4)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x, _ = self.memory(x)
        out = self.classifier(x) 
        return out.transpose(1, 2)


# ==========================================================
# 2. EVALUATION LOGIC
# ==========================================================
def evaluate_single_file(feature_file_path, weight_path="best_model.pth", max_len=75):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Initializing Hybrid Model Core on {device}...")
    
    # Instantiate the inline architecture
    model = VisemeHybridPro().to(device)
    if not os.path.exists(weight_path):
        print(f"❌ Error: Weights file '{weight_path}' missing! Please run the training script first.")
        return
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    print("💾 Model weights loaded successfully!")

    label_file_path = feature_file_path.replace("_features.npy", "_labels.npy")
    if not os.path.exists(label_file_path):
        print(f"❌ Error: Could not find paired label file at '{label_file_path}'")
        return

    # Load sequence arrays
    X_raw = np.load(feature_file_path).astype(np.float32)
    y_raw = np.load(label_file_path).astype(np.int64)
    original_length = len(X_raw)
    
    print(f"📁 Source Target: {os.path.basename(feature_file_path)}")
    print(f"📊 Array Profile: Shape = {X_raw.shape} | Expected Labels = {y_raw.shape}")

    # CRITICAL FIX: Explicitly match training scale normalization bounds [0.0, 1.0]
    if original_length > 0:
        X_raw[:, :400] = X_raw[:, :400] / 255.0  

    # Reconstruct exact tensor sequence dimensions
    X_padded = np.zeros((max_len, 401), dtype=np.float32)
    y_padded = np.full((max_len,), -100, dtype=np.int64)
    
    actual_len = min(original_length, max_len)
    X_padded[:actual_len, :] = X_raw[:actual_len, :]
    y_padded[:actual_len] = y_raw[:actual_len]

    # Reshape into evaluation batch format (1, Sequence_Length, Features)
    tensor_in = torch.from_numpy(X_padded).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor_in)  
        predictions = torch.argmax(output, dim=1).squeeze(0).cpu().numpy() 

    # Strip padding out to analyze true frame accuracy
    valid_predictions = predictions[:actual_len]
    valid_targets = y_padded[:actual_len]

    # Calculate real accuracy metrics
    correct_mask = valid_predictions == valid_targets
    total_valid_frames = len(valid_targets)
    correct_count = np.sum(correct_mask)
    accuracy = (correct_count / total_valid_frames) * 100 if total_valid_frames > 0 else 0.0

    # Operational distribution check (Instantly catches class collapse bugs)
    unique_pred, counts_pred = np.unique(valid_predictions, return_counts=True)
    distribution_report = dict(zip(unique_pred, counts_pred))

    print("\n📊 --- Target Evaluation Metrics ---")
    print(f"File Reference: {os.path.basename(feature_file_path)}")
    print(f"Frames Evaluated: {total_valid_frames}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Frame-Level Accuracy: {accuracy:.2f}%")
    print(f"Prediction Value Spread (Class -> Frame Count): {distribution_report}")
    print("-" * 50)

    # Print out raw slice to manually verify variance
    display_limit = min(25, total_valid_frames)
    print(f"First {display_limit} Frames Verification:")
    print(f"  Ground Truth: {list(valid_targets[:display_limit])}")
    print(f"  Predictions:  {list(valid_predictions[:display_limit])}")


if __name__ == "__main__":
    # Point this directly to your target extracted npy file
    TARGET_TEST_FILE = "./processed_data/bbal6n_features.npy"
    
    evaluate_single_file(
        feature_file_path=TARGET_TEST_FILE, 
        weight_path="best_model.pth", 
        max_len=75
    )