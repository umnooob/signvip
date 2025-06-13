# CUDA_VISIBLE_DEVICES=0 python dwpose_process.py --input_file /deepo_data/signvipworkspace/datasets/How2Sign/test_processed_videos.json --output_dir /deepo_data/signvipworkspace/datasets/How2SignSK/test_processed_videos --input_dir /deepo_data/signvipworkspace/datasets/How2Sign/test_processed_videos
import argparse
import json
import os
import pickle
import random

import cv2
import torch
from dwpose.preprocess import DWposeDetector, get_video_pose_unscale_keypoints
from tqdm import tqdm

# argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos with DWPose")
    parser.add_argument("--input_dir", type=str, default="", help="input directory")
    parser.add_argument("--input_file", type=str, default="", help="input file")
    parser.add_argument("--output_dir", type=str, default="", help="output directory")
    args = parser.parse_args()

    # traverse all videos in the folder and save the processed video to another folder
    dwprocessor = DWposeDetector(
        model_det="models/DWPose/yolox_l.onnx",
        model_pose="models/DWPose/dw-ll_ucoco_384.onnx",
        device="cuda:0",
    )
    folder_path = args.input_dir
    output_folder_path = args.output_dir
    os.makedirs(output_folder_path, exist_ok=True)
    # shuffle the video files
    if args.input_file:
        with open(args.input_file, "r") as f:
            json_data = json.load(f)
        video_files = [os.path.basename(item["video"]) for item in json_data]
        print(video_files[0])
        print(f"Processing {len(video_files)} videos")
        existing_videos = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
        video_files = [f for f in video_files if f in existing_videos]
        print(f"Found {len(video_files)} videos")
    else:
        video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]

    # remove the video files that have already been processed
    # video_files = [
    #     f
    #     for f in video_files
    #     if not os.path.exists(
    #         os.path.join(output_folder_path, f.replace(".mp4", ".pkl"))
    #     )
    # ]
    random.shuffle(video_files)

    for video_file in tqdm(video_files):
        # check if the video is already processed
        output_path = os.path.join(
            output_folder_path, video_file.replace(".mp4", ".pkl")
        )
        if os.path.exists(output_path):
            try:
                with open(output_path, "rb") as f:
                    pickle.load(f)
                continue
            except Exception as e:
                print(f"Error loading {output_path}: {e}")
                os.remove(output_path)

        video_path = os.path.join(folder_path, video_file)
        video_keypoints, fps = get_video_pose_unscale_keypoints(
            video_path, dwprocessor=dwprocessor
        )
        output_path = os.path.join(
            output_folder_path, video_file.replace(".mp4", ".pkl")
        )
        with open(output_path, "wb") as f:
            pickle.dump(video_keypoints, f)
