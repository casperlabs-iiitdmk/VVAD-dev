import os
import glob
import numpy as np
import imageio.v3 as iio
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import explicit dependencies from our logic module
from viseme_logic import VisemeFeatureExtractor, parse_align_file

def process_single_video(video_path, align_data_dir, output_dir):
    """Worker core executing sequential feature extraction on a video tracking pass."""
    try:
        base_name = os.path.basename(video_path).replace(".mpg", "")
        
        if align_data_dir:
            align_path = os.path.join(align_data_dir, f"{base_name}.align")
        else:
            align_path = video_path.replace(".mpg", ".align")
            
        if not os.path.exists(align_path):
            alt_path = align_path.replace(".align", ".ALIGN")
            if os.path.exists(alt_path): align_path = alt_path
        
        if not os.path.exists(align_path):
            return base_name, False, f"Missing alignment target: {align_path}"
            
        extractor = VisemeFeatureExtractor(num_points=200, alpha=0.20)
        
        math_vectors = []
        for frame in iio.imiter(video_path):
            # Compute analytical continuous luminance channel
            gray = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            vector_401 = extractor.extract(gray) 
            math_vectors.append(vector_401)
            
        features_array = np.array(math_vectors, dtype=np.float32)
        labels_array = parse_align_file(align_path, len(features_array))
        
        # Output paths for target arrays
        out_features = os.path.join(output_dir, f"{base_name}_features.npy")
        out_labels = os.path.join(output_dir, f"{base_name}_labels.npy")
        
        np.save(out_features, features_array)
        np.save(out_labels, labels_array)
        
        return base_name, True, f"Successfully extracted ({len(features_array)} frames processed)"
        
    except Exception as e:
        return os.path.basename(video_path), False, f"Process Exception: {str(e)}"


def process_dataset_parallel(raw_video_dir, align_data_dir, output_dir="processed_data", num_workers=32):
    """Executes high-throughput multi-core video extraction."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    video_files = sorted(glob.glob(os.path.join(raw_video_dir, "*.mpg")))
    total_videos = len(video_files)
    
    if total_videos == 0:
        print(f"❌ Execution Halting: No source video files (.mpg) identified within '{raw_video_dir}'")
        return

    print(f"🚀 Initializing high-throughput pipeline. Processing {total_videos} videos across {num_workers} multi-core workers...")
    
    processed_count = 0
    failed_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_video, vid, align_data_dir, output_dir): vid for vid in video_files}
        
        for idx, future in enumerate(as_completed(futures), 1):
            base_name, success, message = future.result()
            status_icon = "✅" if success else "❌"
            if success: processed_count += 1
            else: failed_count += 1
                
            progress_pct = (idx / total_videos) * 100
            print(f"[{idx}/{total_videos} - {progress_pct:.1f}%] {status_icon} {base_name}: {message}")
            
    print(f"\n🎉 Generation Pipeline Finished. Cleanly Extracted: {processed_count} files. Errors/Skips: {failed_count}")


if __name__ == "__main__":
    # Point these parameters directly to your local LRS3 file trees
    VIDEOS_DIR = "LRS3_raw_data/data/s1"
    ALIGNMENTS_DIR = "LRS3_raw_data/alignments/s1" 
    OUTPUT_TARGET_DIR = "processed_data"
    
    process_dataset_parallel(
        raw_video_dir=VIDEOS_DIR, 
        align_data_dir=ALIGNMENTS_DIR, 
        output_dir=OUTPUT_TARGET_DIR, 
        num_workers=32
    )