import os
import pathlib
import threading
import math
from imageio_ffmpeg import get_ffmpeg_exe
from tqdm import tqdm

# Configure paths
root = '/deepo_data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T'
frames_dir = os.path.join(root, 'features/fullFrame-210x260px')
output_dir = os.path.join(root, 'videos')
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

# Process each split (dev, train, test)
splits = ['dev', 'train', 'test']
num_threads = 16

def make_video(sequence_info):
    split_path, sequence_dir = sequence_info
    input_path = os.path.join(split_path, sequence_dir, 'images%04d.png')
    
    split_name = os.path.basename(split_path)
    output_subdir = os.path.join(output_dir, split_name)
    pathlib.Path(output_subdir).mkdir(exist_ok=True)
    
    output_path = os.path.join(output_subdir, f'{sequence_dir}.mp4')
    
    # Check if video exists and can be opened
    if os.path.exists(output_path):
        # Try to open the video with ffmpeg
        check_cmd = f'{get_ffmpeg_exe()} -v error -i {output_path} -f null - 2>&1'
        if os.system(check_cmd) == 0:
            return None
        # If check fails, video is corrupted and should be regenerated
    
    cmd = f'{get_ffmpeg_exe()} -framerate 24 -i {input_path} -c:v libx264 -crf 0 -pix_fmt yuv420p {output_path} -y -loglevel error'
    os.system(cmd)
    return output_path

def process_sequences(thread_idx, sequences, split_path):
    num_seqs_per_thread = math.ceil(len(sequences) / num_threads)
    start_idx = thread_idx * num_seqs_per_thread
    end_idx = min((thread_idx + 1) * num_seqs_per_thread, len(sequences))
    
    processor = tqdm(sequences[start_idx:end_idx], 
                    desc=f'Thread {thread_idx}',
                    position=thread_idx)
    
    for sequence_dir in processor:
        make_video((split_path, sequence_dir))
        processor.set_description(f'Thread {thread_idx}: {sequence_dir}')

# Process each split
for split in splits:
    split_path = os.path.join(frames_dir, split)
    if not os.path.exists(split_path):
        print(f"Split path not found: {split_path}")
        continue
    
    # Get all sequence directories in this split
    sequences = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    print(f"Processing {split} split: {len(sequences)} sequences")
    
    # Create and start threads
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(
            target=process_sequences,
            args=(i, sequences, split_path),
            name=f'thread {i}'
        )
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
