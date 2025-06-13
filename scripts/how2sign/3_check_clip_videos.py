import os
import sys
import subprocess

import cv2
import pandas
from tqdm import tqdm
from imageio_ffmpeg import get_ffmpeg_exe

data_type = 'train'

root = '/home/test/dzx/How2Sign'
csv_file = os.path.join(root, f'how2sign_realigned_{data_type}.csv')
clip_dir = os.path.join(root, f'{data_type}_clip_videos')
raw_dir = os.path.join(root, f'{data_type}_raw_videos')

obj = pandas.read_csv(csv_file, sep='\t')
processor = tqdm(range(len(obj)))
missing_videos = []
corrupted_videos = []

for i in processor:
    clip_name = obj.loc[i, "SENTENCE_NAME"]
    clip_path = os.path.join(clip_dir, clip_name + '.mp4')
    
    # Check if video exists
    if not os.path.exists(clip_path):
        print(f'{clip_path} not found')
        missing_videos.append(i)
        continue
        
    # Check if video file is valid
    video_capture = cv2.VideoCapture(clip_path)
    if not video_capture.isOpened():
        print(f'{clip_path} cannot be opened')
        corrupted_videos.append(i)
        video_capture.release()
        continue
        
    num_video_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Check if frames can be read
    is_corrupted = False
    for j in range(num_video_frames):
        ret, frame = video_capture.read()
        if not ret or frame is None:
            print(f'{clip_path} is corrupted at frame {j+1}/{num_video_frames}')
            corrupted_videos.append(i)
            is_corrupted = True
            break
        processor.set_description(f'{data_type}, {clip_path}, {j + 1}/{num_video_frames}')
    video_capture.release()
    
    if is_corrupted:
        continue

# Process missing or corrupted videos
problem_indices = list(set(missing_videos + corrupted_videos))
if problem_indices:
    print(f"Found {len(problem_indices)} videos that need to be clipped")
    for i in problem_indices:
        raw_name = obj.loc[i, "VIDEO_NAME"]
        tgt_name = obj.loc[i, "SENTENCE_NAME"]
        st = obj.loc[i, "START_REALIGNED"]
        et = obj.loc[i, "END_REALIGNED"]
        raw_vid_path = os.path.join(raw_dir, raw_name + '.mp4')
        tgt_vid_path = os.path.join(clip_dir, tgt_name + '.mp4')
        
        if not os.path.exists(raw_vid_path):
            print(f"Warning: Raw video not found: {raw_vid_path}")
            continue
            
        print(f"Clipping video: {tgt_name}")
        cmd = f'{get_ffmpeg_exe()} -i {raw_vid_path} -ss {st} -to {et} {tgt_vid_path} -y -loglevel error'
        subprocess.call(cmd, shell=True)
    
    print(f"Finished clipping {len(problem_indices)} videos")
else:
    print("All videos are valid!")
