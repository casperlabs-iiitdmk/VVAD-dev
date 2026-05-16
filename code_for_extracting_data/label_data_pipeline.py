import os
import glob
import numpy as np
import imageio.v3 as iio
from g2p_en import G2p
from viseme_logic import VisemeFeatureExtractor 

extractor = VisemeFeatureExtractor()
g2p = G2p()

VISEME_MAP = {
    'F': 1, 'V': 1,
    'SH': 2, 'ZH': 2, 'CH': 2, 'JH': 2,
    'T': 3, 'D': 3, 'S': 3, 'Z': 3, 'TH': 3, 'DH': 3, 'N': 3, 'L': 3, 'R': 3,
    'W': 4, 'UW': 4, 'OW': 4, 'OY': 4, 'AO': 4, 'UH': 4,
    'P': 5, 'B': 5, 'M': 5,
    'AA': 6, 'AE': 6, 'AH': 6, 'AY': 6, 'EY': 6, 'EH': 6, 'IH': 6, 'IY': 6, 
    'AW': 6, 'ER': 6, 'Y': 6, 'K': 6, 'G': 6, 'HH': 6, 'NG': 6
}

def process_separated_folders(video_folder, align_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Grab every .mpg file from the specific video folder
    video_files = glob.glob(os.path.join(video_folder, "*.mpg"))
    print(f"📂 Found {len(video_files)} videos. Starting batch process...\n")
    
    for video_path in video_files:
        base_name = os.path.basename(video_path).replace(".mpg", "")
        
        # Look for the matching .align file in the specific align folder
        align_path = os.path.join(align_folder, f"{base_name}.align")
        
        feature_save_path = os.path.join(output_folder, f"{base_name}_features.npy")
        label_save_path = os.path.join(output_folder, f"{base_name}_labels.npy")
        
        if os.path.exists(feature_save_path) and os.path.exists(label_save_path):
            print(f"⏭️ Skipping {base_name} (Already processed)")
            continue
            
        if not os.path.exists(align_path):
            print(f"⚠️ WARNING: No .align file found for {base_name} in {align_folder}. Skipping.")
            continue
            
        print(f"⏳ Processing: {base_name}...")
        
        # --- 1. EXTRACT THE JUICE ---
        math_vectors = []
        try:
            for frame in iio.imiter(video_path):
                gray = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                vector_401 = extractor.extract(gray, pose_type='A') 
                math_vectors.append(vector_401)
        except Exception as e:
            print(f"❌ Error reading video {base_name}: {e}")
            continue
            
        features_array = np.array(math_vectors)
        total_frames = len(features_array)
        
        if total_frames == 0:
            print(f"❌ Error: 0 frames extracted for {base_name}.")
            continue
            
        # --- 2. TRANSLATE THE LABELS ---
        labels = np.zeros(total_frames, dtype=int)
        
        with open(align_path, 'r') as file:
            lines = file.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 3: continue
                
            start_frame = int(parts[0]) // 1000
            end_frame = int(parts[1]) // 1000
            word = parts[2]
            
            if word in ['sil', 'sp']: continue
                
            phonemes = g2p(word)
            clean_phonemes = [p.rstrip('012') for p in phonemes if p.isalpha()]
            if not clean_phonemes: continue
                
            word_frame_count = end_frame - start_frame
            frames_per_phoneme = max(1, word_frame_count // len(clean_phonemes))
            
            current_frame = start_frame
            
            for phoneme in clean_phonemes:
                viseme_group = VISEME_MAP.get(phoneme, 6)
                for i in range(frames_per_phoneme):
                    if current_frame < total_frames:
                        labels[current_frame] = viseme_group
                    current_frame += 1
                    
            while current_frame < end_frame and current_frame < total_frames:
                labels[current_frame] = viseme_group
                current_frame += 1

        # --- 3. SAVE BOTH ---
        np.save(feature_save_path, features_array)
        np.save(label_save_path, labels)
        
        print(f"✅ Success: {base_name} | Mapped {total_frames} frames.")

# ==========================================
# 🚀 RUN THE PIPELINE
# ==========================================
# UPDATE THESE THREE PATHS TO MATCH YOUR FOLDERS EXACTLY
# Make sure to include the exact names of the folders inside your 'data' folder
VIDEO_FOLDER = "LRS3_raw_data/data/s1" 
ALIGN_FOLDER = "LRS3_raw_data/alignments/s1"
OUTPUT_SAVE_FOLDER = "processed_data"

process_separated_folders(VIDEO_FOLDER, ALIGN_FOLDER, OUTPUT_SAVE_FOLDER)