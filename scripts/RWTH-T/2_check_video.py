import os
import pathlib
from imageio_ffmpeg import get_ffmpeg_exe
from tqdm import tqdm

# Configure paths (same as in 1_make_video.py)
root = '/deepo_data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T'
frames_dir = os.path.join(root, 'features/fullFrame-210x260px')
videos_dir = os.path.join(root, 'videos')

def check_video(video_path):
    """Check if video exists and can be opened with ffmpeg."""
    if not os.path.exists(video_path):
        return False, "Video file missing"
    
    check_cmd = f'{get_ffmpeg_exe()} -v error -i {video_path} -f null - 2>&1'
    if os.system(check_cmd) != 0:
        return False, "Video file corrupted"
    
    return True, "OK"

# Process each split
splits = ['dev', 'train', 'test']
failed_videos = []

for split in splits:
    print(f"\nChecking {split} split...")
    
    # Get source directories (frames)
    split_frames_path = os.path.join(frames_dir, split)
    if not os.path.exists(split_frames_path):
        print(f"Split frames path not found: {split_frames_path}")
        continue
    
    # Get expected sequences from frames directory
    expected_sequences = [d for d in os.listdir(split_frames_path) 
                         if os.path.isdir(os.path.join(split_frames_path, d))]
    
    # Check corresponding videos
    split_videos_path = os.path.join(videos_dir, split)
    for sequence in tqdm(expected_sequences, desc=f"Checking {split} videos"):
        video_path = os.path.join(split_videos_path, f"{sequence}.mp4")
        is_valid, status = check_video(video_path)
        
        if not is_valid:
            failed_videos.append((split, sequence, status))

# Report results
print("\nCheck completed!")
print(f"Total failed videos: {len(failed_videos)}")
if failed_videos:
    print("\nFailed videos:")
    for split, sequence, status in failed_videos:
        print(f"- {split}/{sequence}: {status}")
