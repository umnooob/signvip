import math
import os
import pathlib
import threading

import pandas
from imageio_ffmpeg import get_ffmpeg_exe
from tqdm import tqdm

data_type = 'val'

root = '/home/test/dzx/How2Sign'
csv_file = os.path.join(root, f'how2sign_realigned_{data_type}.csv')
raw_dir = os.path.join(root, f'{data_type}_raw_videos')
tgt_dir = os.path.join(root, f'{data_type}_clip_videos')
pathlib.Path(tgt_dir).mkdir(parents=True, exist_ok=True)

obj = pandas.read_csv(csv_file, sep='\t')
obj = obj.sample(frac=1).reset_index(drop=True)
print(obj)
num_threads = 16
num_paths_per_thread = math.ceil(len(obj) / num_threads)


def clip(thread_idx):
    processor = tqdm(range(
        thread_idx * num_paths_per_thread,
        min((thread_idx + 1) * num_paths_per_thread, len(obj)),
    ))
    for i in processor:
        raw_name = obj.loc[i, "VIDEO_NAME"]
        tgt_name = obj.loc[i, "SENTENCE_NAME"]
        st = obj.loc[i, "START_REALIGNED"]
        et = obj.loc[i, "END_REALIGNED"]
        raw_vid_path = os.path.join(raw_dir, raw_name + '.mp4')
        tgt_vid_path = os.path.join(tgt_dir, tgt_name + '.mp4')
        
        # Check if target video exists and is valid
        if os.path.exists(tgt_vid_path):
            # Check if file size is reasonable (e.g., > 1KB)
            if os.path.getsize(tgt_vid_path) > 1024:
                continue
        
        # If we reach here, either the file doesn't exist or is invalid
        if not os.path.exists(raw_vid_path):
            print(f"Warning: Raw video not found: {raw_vid_path}")
            continue
            
        print(f"Re-clipping video: {tgt_name}")
        os.system(f'{get_ffmpeg_exe()} -i {raw_vid_path} -ss {st} -to {et} {tgt_vid_path} -y -loglevel error')
        processor.set_description(f'{thread_idx}, {tgt_name}')


for i in range(num_threads):
    thread = threading.Thread(target=clip, name=f'thread {i}', args=(i,))
    thread.start()
