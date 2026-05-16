import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque

# ==========================================
# 1. THE ML MODEL (VisemeBrainPro)
# ==========================================
class VisemeBrainPro(nn.Module):
    def __init__(self):
        super(VisemeBrainPro, self).__init__()
        self.memory_layer = nn.GRU(
            input_size=401, 
            hidden_size=256, 
            num_layers=4, 
            batch_first=True, 
            dropout=0.5,
            bidirectional=True 
        )
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
        return self.classifier(brain_thoughts)

# ==========================================
# 2. THE REAL-TIME ENGINE
# ==========================================
def run_real_detector(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = VisemeBrainPro().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # OpenCV built-in face detector (No mediapipe required)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    viseme_labels = {0:"Silence", 1:"P/B/M", 2:"F/V", 3:"Th", 4:"L/N", 5:"O/U", 6:"A/E/I"}
    frame_buffer = deque(maxlen=75) 
    cap = cv2.VideoCapture(0)

    print("🎬 Real-Time Detection Started. Feed the AI real pixels...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        display_label = "Waiting for Face..."

        for (x, y, w, h) in faces:
            # 1. Define the Mouth ROI (approx. bottom 1/3 of the face)
            mouth_y = y + int(h * 0.65)
            mouth_h = int(h * 0.35)
            mouth_roi = gray[mouth_y:mouth_y+mouth_h, x:x+w]

            if mouth_roi.size > 0:
                # 2. Extract 401 Features from ACTUAL pixels
                # We resize the mouth to 20x20 (400 pixels) + 1 pose bit = 401
                resized_mouth = cv2.resize(mouth_roi, (20, 20)).flatten()
                pose_bit = np.array([0.0]) # Default Pose A
                
                raw_features = np.concatenate([resized_mouth, pose_bit]).astype(np.float32)

                # 3. Normalization (CRITICAL: Makes live data look like training data)
                mean, std = np.mean(raw_features), np.std(raw_features) + 1e-6
                norm_features = (raw_features - mean) / std
                
                frame_buffer.append(norm_features)
                display_label = f"Processing... {len(frame_buffer)}/75"

            # Draw a box only around the mouth area for feedback
            cv2.rectangle(frame, (x, mouth_y), (x+w, mouth_y+mouth_h), (0, 255, 255), 1)
            break 

        # 4. Run the Neural Network
        if len(frame_buffer) == 75:
            input_tensor = torch.FloatTensor(np.array(frame_buffer)).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                # Middle-frame prediction for Bidirectional context
                prediction = torch.argmax(output[0, 37, :]).item()
                display_label = viseme_labels.get(prediction, "Calculating...")

        # HUD Output
        cv2.putText(frame, f"VISEME: {display_label}", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow('Pro Viseme Detector (No-Solutions Version)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_real_detector("best_pro_model.pth")