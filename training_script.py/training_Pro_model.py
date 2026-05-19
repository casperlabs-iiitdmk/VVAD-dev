import os
import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

# ==========================================================
# 1. DATA LOADER (With Uniform Max Scaling Preservation)
# ==========================================================
class VisemeDataset(Dataset):
    def __init__(self, data_folder, max_len=75):
        raw_feature_files = sorted(glob.glob(os.path.join(data_folder, "*_features.npy")))
        self.max_len = max_len
        self.feature_files = []
        
        for feat_path in raw_feature_files:
            label_path = feat_path.replace("_features.npy", "_labels.npy")
            if os.path.exists(label_path):
                self.feature_files.append(feat_path)
        print(f"📦 Dataset initialized. Valid paired sequences: {len(self.feature_files)}")
        
    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, index):
        feat_path = self.feature_files[index]
        X_raw = np.load(feat_path).astype(np.float32)
        y_raw = np.load(feat_path.replace("_features.npy", "_labels.npy")).astype(np.int64)

        if len(X_raw) > 0:
            X_raw[:, :400] = X_raw[:, :400] / 255.0  # Crucial Scale Normalization

        X_padded = np.zeros((self.max_len, 401), dtype=np.float32)
        y_padded = np.full((self.max_len,), -100, dtype=np.int64)
        
        actual_len = min(len(X_raw), self.max_len)
        X_padded[:actual_len, :] = X_raw[:actual_len, :]
        y_padded[:actual_len] = y_raw[:actual_len]
        
        return torch.from_numpy(X_padded), torch.from_numpy(y_padded)


# ==========================================================
# 2. HYBRID MODEL ARCHITECTURE
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
            nn.Dropout(0.4),  # Increased to prevent early memorization
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
# 3. TRAINING LOOP WITH LIVE ACCURACY METRICS
# ==========================================================
def train_hybrid_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_dataset = VisemeDataset("./balanced_data") 
    if len(full_dataset) == 0: return

    # Calculate Smoothed Class Weights (Log-Scale to prevent collapse)
    all_labels = []
    for feat_path in full_dataset.feature_files:
        all_labels.extend(np.load(feat_path.replace("_features.npy", "_labels.npy")).flatten())
    classes, counts = np.unique(all_labels, return_counts=True)
    
    computed_weights = np.ones(7, dtype=np.float32)
    for c, count in zip(classes, counts):
        if c < 7:
            computed_weights[c] = 1.0 / (np.log1p(count) + 1e-5) # Smooth Log dampening
    computed_weights = computed_weights / np.sum(computed_weights) * 7.0
    weights_tensor = torch.tensor(computed_weights, dtype=torch.float32).to(device)

    # Cross-Entropy with Label Smoothing breaks local minima cheating loops
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=-100, label_smoothing=0.1)

    train_size = int(0.8 * len(full_dataset))
    val_dataset, train_dataset = random_split(full_dataset, [len(full_dataset)-train_size, train_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    student = VisemeHybridPro().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.0005, weight_decay=1e-3) # Lowered LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4) 

    best_val_acc = 0.0
    patience, no_improvement = 15, 0

    for epoch in range(100):
        student.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for video_math, targets in train_loader:
            video_math, targets = video_math.to(device), targets.to(device)
            video_math = video_math + torch.randn_like(video_math) * 0.01 # Subtle Noise
            
            optimizer.zero_grad()
            guesses = student(video_math)
            loss = criterion(guesses, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate live operational accuracy
            preds = torch.argmax(guesses, dim=1)
            mask = targets != -100
            train_correct += (preds[mask] == targets[mask]).sum().item()
            train_total += mask.sum().item()

        # Validation Pass
        student.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for video_math, targets in val_loader:
                video_math, targets = video_math.to(device), targets.to(device)
                guesses = student(video_math)
                loss = criterion(guesses, targets)
                val_loss += loss.item()
                
                preds = torch.argmax(guesses, dim=1)
                mask = targets != -100
                val_correct += (preds[mask] == targets[mask]).sum().item()
                val_total += mask.sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = (train_correct / train_total) * 100
        val_acc = (val_correct / val_total) * 100
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(val_acc) # Scale learning rate based on real accuracy, not loss!

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improvement = 0
            torch.save(student.state_dict(), "best_model.pth")
            print(f"   🌟 Real Validation Accuracy Improved to {val_acc:.2f}%! Weights saved.")
        else:
            no_improvement += 1
            
        if no_improvement >= patience:
            print("\n🛑 Stop trigger reached. Class collapse successfully averted.")
            break

if __name__ == "__main__":
    train_hybrid_model()