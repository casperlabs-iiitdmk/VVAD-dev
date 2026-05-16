import numpy as np
import os
from g2p_en import G2p

# Initialize the word-to-sound translator
g2p = G2p()

# 1. THE DICTIONARY: Map Phonemes to Your 6 Viseme Groups
# Group 0 is reserved for Silence / Background
VISEME_MAP = {
    # Group 1: Labiodental (Lip-to-teeth)
    'F': 1, 'V': 1,
    
    # Group 2: Postalveolar (Lips slightly puckered, teeth together)
    'SH': 2, 'ZH': 2, 'CH': 2, 'JH': 2,
    
    # Group 3: Alveolar (Tongue to roof of mouth, teeth close)
    'T': 3, 'D': 3, 'S': 3, 'Z': 3, 'TH': 3, 'DH': 3, 'N': 3, 'L': 3, 'R': 3,
    
    # Group 4: Rounded (Lips shaped like an 'O')
    'W': 4, 'UW': 4, 'OW': 4, 'OY': 4, 'AO': 4, 'UH': 4,
    
    # Group 5: Bilabial (Lips tightly pressed together)
    'P': 5, 'B': 5, 'M': 5,
    
    # Group 6: Open Mouth / Vowels / Velar
    'AA': 6, 'AE': 6, 'AH': 6, 'AY': 6, 'EY': 6, 'EH': 6, 'IH': 6, 'IY': 6, 
    'AW': 6, 'ER': 6, 'Y': 6, 'K': 6, 'G': 6, 'HH': 6, 'NG': 6
}

def translate_align_to_labels(align_path, total_frames, save_path):
    print(f"--- TRANSLATING: {align_path} ---")
    
    # Create an empty array of zeros (Group 0 = Silence) for all 75 frames
    labels = np.zeros(total_frames, dtype=int)
    
    with open(align_path, 'r') as file:
        lines = file.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 3:
            continue
            
        # GRID alignments use 1000 units per frame (e.g., 25000 = Frame 25)
        start_frame = int(parts[0]) // 1000
        end_frame = int(parts[1]) // 1000
        word = parts[2]
        
        # Skip silence tokens
        if word in ['sil', 'sp']:
            continue
            
        # 2. Get the sounds for the word (e.g., "BIN" -> ['B', 'IH', 'N'])
        phonemes = g2p(word)
        
        # Clean up phonemes (g2p sometimes adds numbers for stress, like 'IH1')
        clean_phonemes = [p.rstrip('012') for p in phonemes if p.isalpha()]
        
        if not clean_phonemes:
            continue
            
        # 3. Distribute frames evenly across the phonemes
        word_frame_count = end_frame - start_frame
        frames_per_phoneme = max(1, word_frame_count // len(clean_phonemes))
        
        current_frame = start_frame
        
        for phoneme in clean_phonemes:
            # Look up the Viseme Group (default to 6 if unknown)
            viseme_group = VISEME_MAP.get(phoneme, 6)
            
            # Assign this group to the allocated frames
            for i in range(frames_per_phoneme):
                if current_frame < total_frames:
                    labels[current_frame] = viseme_group
                current_frame += 1
                
        # Fill any remainder frames for this word with the last phoneme's group
        while current_frame < end_frame and current_frame < total_frames:
            labels[current_frame] = viseme_group
            current_frame += 1

    # 4. Save the final label array
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, labels)
    
    print(f"✅ Saved Labels to {save_path} - Shape: {labels.shape}")
    
    # Print a preview so you can see it working!
    print("Preview of first 25 frames:")
    print(labels[:25])

# Example: Run this on the align file that matches your completed video
# NOTE: Make sure the total_frames exactly matches what your Juicer output (75)
translate_align_to_labels(
    align_path="bbaf2n.align", 
    total_frames=75, 
    save_path="processed_data/video_001_labels.npy"
)