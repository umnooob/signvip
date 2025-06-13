import math
import os
import pathlib
import threading

import pandas
from imageio_ffmpeg import get_ffmpeg_exe
from tqdm import tqdm

data_type = 'train'

root = '/home/test/dzx/How2Sign'
csv_file = os.path.join(root, f'how2sign_realigned_{data_type}.csv')
raw_dir = os.path.join(root, f'{data_type}_clip_videos')
tgt_dir = os.path.join(root, f'{data_type}_processed_videos')
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
        tgt_name = obj.loc[i, "SENTENCE_NAME"]
        raw_vid_path = os.path.join(raw_dir, tgt_name + '.mp4')
        tgt_vid_path = os.path.join(tgt_dir, tgt_name + '.mp4')

        os.system(
            f'{get_ffmpeg_exe()} -i {raw_vid_path} '
            f'-vf "crop=720:720:280:0,scale=512:512" '
            f'{tgt_vid_path} -y -loglevel error'
        )
        processor.set_description(f'{thread_idx}, {tgt_name}')


for i in range(num_threads):
    thread = threading.Thread(target=clip, name=f'thread {i}', args=(i,))
    thread.start()
