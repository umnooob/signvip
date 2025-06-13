import glob
import os
import sys

import cv2

sys.path.append("/openpose/build/python/openpose")
import argparse
import json
import random

import numpy as np
import pyopenpose as op
from tqdm import tqdm

sys.path.pop(0)


class Openpose_Human:
    def __init__(self):
        super(Openpose_Human, self).__init__()

        self.params = dict()
        self.params["model_folder"] = "/openpose/models"
        self.params["num_gpu"] = 2
        self.params["num_gpu_start"] = 0
        self.params["display"] = 0
        self.params["face"] = True
        self.params["hand"] = True
        self.params["disable_blending"] = True

        # Starting OpenPose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(self.params)
        self.opWrapper.start()

    def gen_image_kpts(self, img):
        datum = op.Datum()
        datum.cvInputData = img
        # Process
        self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        return (
            datum.cvOutputData,
            datum.poseKeypoints,
            datum.faceKeypoints,
            datum.handKeypoints,
        )

    def gen_image_kpts_pre(self, img):
        datum = op.Datum()
        datum.cvInputData = img
        # Process
        self.opWrapper.waitAndEmplace(op.VectorDatum([datum]))

    def gen_image_kpts_after(self):
        datums = op.VectorDatum()
        self.opWrapper.waitAndPop(datums)
        return datums[0].cvOutputData

    def gen_video_kpts(self, video, max_num_person=1):
        cap = cv2.VideoCapture(video)
        assert cap.isOpened(), "Cannot capture source"

        # Get video properties
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        processed_frames = []
        processed_keypoints = []
        for frame_num in tqdm(range(video_length)):
            ret, frame = cap.read()

            if not ret:
                print(f"Failed to read frame {frame_num}")
                break

            try:
                cvOutputData, poseKeypoints, faceKeypoints, handKeypoints = (
                    self.gen_image_kpts(frame)
                )
                processed_frames.append(cvOutputData)
                keypoints_data = {
                    "frame": frame_num,
                    "body_keypoints": (
                        poseKeypoints.tolist() if poseKeypoints is not None else []
                    ),
                    "face_keypoints": (
                        faceKeypoints.tolist() if faceKeypoints is not None else []
                    ),
                    "left_hand_keypoints": (
                        handKeypoints[0].tolist()
                        if handKeypoints is not None and len(handKeypoints) > 0
                        else []
                    ),
                    "right_hand_keypoints": (
                        handKeypoints[1].tolist()
                        if handKeypoints is not None and len(handKeypoints) > 1
                        else []
                    ),
                }
                processed_keypoints.append(keypoints_data)
            except Exception as e:
                print(f"Error processing frame {frame_num}: {str(e)}")
                break

        cap.release()
        return processed_frames, processed_keypoints, fps

    def save_video(self, frames, output_path, fps=30):
        if not frames:
            print("No frames to save.")
            return

        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()
        print(f"Video saved to {output_path}")


if __name__ == "__main__":
    openpose = Openpose_Human()

    # traverse all videos in the folder and save the processed video to another folder
    folder_path = (
        "/deepo_data/signvipworkspace/datasets/How2Sign/train_processed_videos"
    )
    output_folder_path = (
        "/deepo_data/signvipworkspace/datasets/How2SignSK/train_processed_videos"
    )
    output_keypoints_folder_path = (
        "/deepo_data/signvipworkspace/datasets/How2SignSK/train_processed_keypoints"
    )
    os.makedirs(output_folder_path, exist_ok=True)
    os.makedirs(output_keypoints_folder_path, exist_ok=True)
    # shuffle the video files
    video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
    # remove the video files that have already been processed
    video_files = [
        f
        for f in video_files
        if not os.path.exists(os.path.join(output_folder_path, f))
    ]
    random.shuffle(video_files)

    for video_file in tqdm(video_files):
        # check if the video is already processed
        output_path = os.path.join(output_folder_path, video_file)
        if os.path.exists(output_path):
            continue

        video_path = os.path.join(folder_path, video_file)
        video, keypoints, fps = openpose.gen_video_kpts(video_path)
        output_path = os.path.join(output_folder_path, video_file)
        openpose.save_video(video, output_path, fps=fps)

        with open(
            os.path.join(
                output_keypoints_folder_path, f"{video_file.split('.')[0]}.json"
            ),
            "w",
        ) as f:
            json.dump(keypoints, f)
