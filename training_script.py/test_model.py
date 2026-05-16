import os
import glob
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm 

# ==========================================
# 1. THE UPGRADED BRAIN (Must match Training exactly!)
# ==========================================
class VisemeBrainPro(nn.Module):
    def __init__(self):
        super(VisemeBrainPro, self).__init__()
        
        # UPGRADE 1: Wider (256), Deeper (4 layers), and Bidirectional
        self.memory_layer = nn.GRU(
            input_size=401, 
            hidden_size=256, 
            num_layers=4, 
            batch_first=True, 
            dropout=0.5,
            bidirectional=True # Gives context from both past and future frames
        )
        
        # UPGRADE 2: Multi-Layer Perceptron (MLP) Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.LayerNorm(256),       
            nn.ReLU(),
            nn.Dropout(0.5),         
            
            nn.Linear(in_features=256, out_features=128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(in_features=128, out_features=7)
        )

    def forward(self, x):
        brain_thoughts, _ = self.memory_layer(x)
        final_guess = self.classifier(brain_thoughts)
        return final_guess

# ==========================================
# 2. BULK EVALUATION LOGIC
# ==========================================
def evaluate_folder(data_folder, model_path, max_len=75):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧐 Evaluating all data in {data_folder} using {device}...")

    # Load the NEW Bidirectional Pro Brain
    student = VisemeBrainPro().to(device)
    student.load_state_dict(torch.load(model_path, map_location=device))
    student.eval()

    feature_files = sorted(glob.glob(os.path.join(data_folder, "*_features.npy")))
    
    total_frames = 0
    total_correct = 0
    video_count = len(feature_files)

    print(f"📂 Found {video_count} videos to test. Starting bulk process...")

    with torch.no_grad():
        for feat_path in tqdm(feature_files):
            X_raw = np.load(feat_path).astype(np.float32)
            label_path = feat_path.replace("_features.npy", "_labels.npy")
            y_true_raw = np.load(label_path).astype(np.int64)

            actual_len = min(len(X_raw), max_len)
            X_padded = np.zeros((max_len, 401), dtype=np.float32)
            X_padded[:actual_len, :] = X_raw[:actual_len, :]
            
            # Normalize to match training
            for i in range(X_padded.shape[1]):
                col = X_padded[:, i]
                mean = np.mean(col)
                std = np.std(col) + 1e-6
                X_padded[:, i] = (col - mean) / std

            X_tensor = torch.from_numpy(X_padded).unsqueeze(0).to(device)
            output = student(X_tensor)
            guesses = torch.argmax(output, dim=2).squeeze(0).cpu().numpy()

            guesses_clipped = guesses[:actual_len]
            y_true_clipped = y_true_raw[:actual_len]

            total_correct += np.sum(guesses_clipped == y_true_clipped)
            total_frames += actual_len

    final_accuracy = (total_correct / total_frames) * 100
    print("\n" + "="*40)
    print("📊 FINAL BULK REPORT (PRO MODEL)")
    print("="*40)
    print(f"Total Videos Scanned: {video_count}")
    print(f"Total Frames Checked: {total_frames}")
    print(f"Total Correct Guesses: {total_correct}")
    print(f"OVERALL ACCURACY:     {final_accuracy:.2f}%")
    print("="*40)

if __name__ == "__main__":
    DATA_PATH = r"C:\Users\sudheendraa A G\OneDrive\Desktop\my learnings\python.py\VVAD\test_data"
    MODEL_FILE = "best_pro_model.pth"
    evaluate_folder(DATA_PATH, MODEL_FILE)