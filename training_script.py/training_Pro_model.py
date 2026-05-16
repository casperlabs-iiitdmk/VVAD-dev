import os
import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

# ==========================================
# 1. THE DATA LOADER
# ==========================================
class VisemeDataset(Dataset):
    def __init__(self, data_folder, max_len=75):
        self.feature_files = sorted(glob.glob(os.path.join(data_folder, "*_features.npy")))
        self.max_len = max_len
        
    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, index):
        feat_path = self.feature_files[index]
        X_raw = np.load(feat_path).astype(np.float32)
        
        label_path = feat_path.replace("_features.npy", "_labels.npy")
        y_raw = np.load(label_path).astype(np.int64)

        X_padded = np.zeros((self.max_len, 401), dtype=np.float32)
        y_padded = np.zeros((self.max_len,), dtype=np.int64)
        
        actual_len = min(len(X_raw), self.max_len)
        X_padded[:actual_len, :] = X_raw[:actual_len, :]
        y_padded[:actual_len] = y_raw[:actual_len]

        for i in range(X_padded.shape[1]):
            col = X_padded[:, i]
            mean = np.mean(col)
            std = np.std(col) + 1e-6
            X_padded[:, i] = (col - mean) / std
        
        return torch.from_numpy(X_padded), torch.from_numpy(y_padded)

# ==========================================
# 2. THE UPGRADED BRAIN (Deeper & Bidirectional)
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
        # Bidirectional GRU doubles the hidden size (256 * 2 = 512)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.LayerNorm(256),       # Stabilizes deep network training
            nn.ReLU(),
            nn.Dropout(0.5),         # Prevents overfitting
            
            nn.Linear(in_features=256, out_features=128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(in_features=128, out_features=7) # Final 7 viseme classes
        )

    def forward(self, x):
        # Pass through the recurrent layers
        brain_thoughts, _ = self.memory_layer(x)
        
        # Pass the sequence through the deeper classifier
        final_guess = self.classifier(brain_thoughts)
        return final_guess

# ==========================================
# 3. ANTI-OVERFIT TRAINING LOOP
# ==========================================
def train_pro_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_path = r"C:\Users\sudheendraa A G\OneDrive\Desktop\my learnings\python.py\VVAD\processed_data"
    full_dataset = VisemeDataset(data_path) 
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"🚀 Starting 150-Cycle BOOTCAMP Training on {device}...")
    print(f"📈 Dataset Split: {train_size} training videos | {val_size} validation videos")

    # Use the new, upgraded model
    student = VisemeBrainPro().to(device)
    teacher_grader = nn.CrossEntropyLoss()
    
    # Slightly lowered weight decay since we added more dropout/layernorm
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) 
    
    best_val_error = float('inf')
    patience = 15 
    epochs_no_improve = 0 
    
    for epoch in range(150): 
        # ==========================
        # PHASE A: TRAINING
        # ==========================
        student.train()
        train_error = 0
        
        for video_math, correct_answers in train_loader:
            video_math, correct_answers = video_math.to(device), correct_answers.to(device)
            
            # 🛡️ DATA AUGMENTATION: Inject noise so the model can't memorize pixels
            noise = torch.randn_like(video_math) * 0.05 # 5% visual static
            video_math_noisy = video_math + noise
            
            optimizer.zero_grad()
            guesses = student(video_math_noisy)
            loss = teacher_grader(guesses.view(-1, 7), correct_answers.view(-1))
            
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_error += loss.item()
            
        # ==========================
        # PHASE B: VALIDATION
        # ==========================
        student.eval() 
        val_error = 0
        
        with torch.no_grad(): 
            for video_math, correct_answers in val_loader:
                video_math, correct_answers = video_math.to(device), correct_answers.to(device)
                
                # NO NOISE IN VALIDATION - The exam must be clean!
                guesses = student(video_math)
                loss = teacher_grader(guesses.view(-1, 7), correct_answers.view(-1))
                val_error += loss.item()
        
        avg_train = train_error / len(train_loader)
        avg_val = val_error / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Cycle {epoch + 1:03d}/150 | Train Error: {avg_train:.4f} | Val Error: {avg_val:.4f} | LR: {current_lr:.6f}")
        
        scheduler.step()

        if avg_val < best_val_error:
            best_val_error = avg_val
            epochs_no_improve = 0 
            torch.save(student.state_dict(), "best_pro_model.pth")
            print("   🌟 New best validation score! Model saved.")
        else:
            epochs_no_improve += 1
            print(f"   ⚠️ No improvement for {epochs_no_improve}/{patience} cycles.")
            
        if epochs_no_improve >= patience:
            print(f"\n🛑 EARLY STOPPING TRIGGERED! The student hasn't improved on unseen data in {patience} cycles.")
            break

    print("\n🎓 Training Complete!")
    print(f"💾 The absolute best version of the brain is saved as 'best_pro_model.pth' with a Validation Error of {best_val_error:.4f}")

if __name__ == "__main__":
    train_pro_model()