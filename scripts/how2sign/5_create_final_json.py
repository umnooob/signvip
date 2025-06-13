import json
import os

import pandas
from tqdm import tqdm

data_type = 'train'

root = '/home/test/dzx/How2Sign'
csv_file = os.path.join(root, f'how2sign_realigned_{data_type}.csv')
data_dir = os.path.join(root, f'{data_type}_processed_videos')
output_json = os.path.join(root, f'{data_type}_processed_videos.json')

obj = pandas.read_csv(csv_file, sep='\t')
processor = tqdm(range(len(obj)))
json_obj = []
not_found_files = []
for i in processor:
    vid_name = obj.loc[i, "SENTENCE_NAME"]
    text = obj.loc[i, "SENTENCE"]

    vid_path = os.path.join(data_dir, vid_name + '.mp4')
    if not os.path.exists(vid_path):
        not_found_files.append(vid_path)
    json_obj.append({
        "video": os.path.join(f'{data_type}_processed_videos', vid_name + '.mp4'),
        "text": text,
    })
    processor.set_description(f'{data_type}, {vid_name}')
json.dump(json_obj, open(output_json, 'w'))
for file in not_found_files:
    print(file)
print(f'num_not_found: {len(not_found_files)}')
