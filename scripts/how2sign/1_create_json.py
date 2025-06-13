import json
import os

import pandas
from tqdm import tqdm

data_type = 'train'

root = '/home/test/dzx/How2Sign'
csv_file = os.path.join(root, f'how2sign_realigned_{data_type}.csv')
raw_dir = os.path.join(root, f'{data_type}_raw_videos')
# tgt_dir = os.path.join(root, f'{data_type}_processed_videos')
output_json = os.path.join(root, f'{data_type}_videos.json')

obj = pandas.read_csv(csv_file, sep='\t')
processor = tqdm(range(len(obj)))
json_obj = []
for i in processor:
    vid_name = obj.loc[i, "VIDEO_NAME"]
    tgt_name = obj.loc[i, "SENTENCE_NAME"]
    text = obj.loc[i, "SENTENCE"]
    if data_type == 'no':
        if not (os.path.exists(os.path.join(raw_dir, 'train_raw_videos-s1', vid_name + '.mp4')) or
                os.path.exists(os.path.join(raw_dir, 'train_raw_videos-s2', vid_name + '.mp4')) or
                os.path.exists(os.path.join(raw_dir, 'train_raw_videos-s3', vid_name + '.mp4'))):
            input(f'{vid_name} not found')
    else:
        vid_path = os.path.join(raw_dir, vid_name + '.mp4')
        if not os.path.exists(vid_path):
            input(f'{vid_name} not found')
    json_obj.append({
        "video": os.path.join(raw_dir, tgt_name + '.mp4'),
        "text": text,
    })
    processor.set_description(f'{tgt_name}')
json.dump(json_obj, open(output_json, 'w'))
