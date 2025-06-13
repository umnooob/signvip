# process annotation
# /deepo_data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv

import json
import os

import pandas as pd

splits = ["dev", "train", "test"]

for split in splits:
    path = f"/deepo_data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.{split}.corpus.csv"
    df = pd.read_csv(path, sep="|")

    # Create list of dictionaries with video and text pairs
    processed_data = []
    for _, row in df.iterrows():
        processed_data.append(
            {
                "video": f"{split}_processed_videos/{row['name']}.mp4",  # Assuming 'name' is the video filename column
                "text": row["translation"],  # Assuming 'translation' is the text column
                "orth": row["orth"],
            }
        )

    # Write to JSON file
    output_path = f"{split}_processed_videos.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
