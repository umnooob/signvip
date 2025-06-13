import json
import os
import pickle
import random
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_video
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Normalize,
    PILToTensor,
    RandomGrayscale,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from scripts.hamer.renderer import Renderer, render_hands
from scripts.sk.dwpose.util import draw_pose, get_hand_area

"""
This dataset is used to generate the sign pose tokens with text.
"""


class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


class SignCondDataset(Dataset):

    def __init__(
        self,
        output_dir: str = None,
        frame_size: Tuple[int, int] = (512, 512),
        frame_scale: Tuple[float, float] = (1.0, 1.0),
        sample_rate: int = 1,
        roots: Union[List[str], str] = None,
        sk_roots: Union[List[str], str] = None,
        hamer_roots: Union[List[str], str] = None,
        meta_paths: Union[List[str], str] = None,
    ):
        super().__init__()

        self.output_dir = output_dir
        self.frame_size = frame_size
        self.sample_rate = sample_rate

        roots = [roots] if isinstance(roots, str) else roots
        sk_roots = [sk_roots] if isinstance(sk_roots, str) else sk_roots
        hamer_roots = [hamer_roots] if isinstance(hamer_roots, str) else hamer_roots
        meta_paths = [meta_paths] if isinstance(meta_paths, str) else meta_paths

        self.videos_info = []
        for root, meta_path in zip(roots, meta_paths):
            obj = json.load(open(meta_path, "r"))
            obj = [{"root": root, **o} for o in obj]
            self.videos_info.extend(obj)

        self.sk_videos_info = []
        for sk_root, meta_path in zip(sk_roots, meta_paths):
            obj = json.load(open(meta_path, "r"))
            obj = [{"root": sk_root, **o} for o in obj]
            self.sk_videos_info.extend(obj)

        self.hamer_videos_info = []
        for hamer_root, meta_path in zip(hamer_roots, meta_paths):
            obj = json.load(open(meta_path, "r"))
            obj = [{"root": hamer_root, **o} for o in obj]
            self.hamer_videos_info.extend(obj)

        self.img_transform = Compose(
            [
                # Resize(frame_size),
                PILToTensor(),
                ZeroOneNormalize(),
                RandomResizedCrop(
                    frame_size,
                    scale=frame_scale,
                    ratio=(
                        frame_size[1] / frame_size[0],
                        frame_size[1] / frame_size[0],
                    ),
                    antialias=True,
                ),
                # RandomGrayscale(p=0.1),
                # ToTensor(),
                Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    @staticmethod
    def read_frame(vid_capture, frame_idx):
        vid_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, frame = vid_capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        return frame

    def process_frames(
        self,
        target_sk_images,
        target_hamer_images,
        rand_state,
    ):
        processed_sk_images = []
        for image in target_sk_images:
            target_sk_image = self.augmentation(image, self.img_transform, rand_state)
            processed_sk_images.append(target_sk_image)
        processed_hamer_images = []
        for image in target_hamer_images:
            target_hamer_image = self.augmentation(
                image, self.img_transform, rand_state
            )
            processed_hamer_images.append(target_hamer_image)
        return (
            (
                torch.stack(processed_sk_images, dim=0).permute(1, 0, 2, 3)
                if processed_sk_images
                else []
            ),
            (
                torch.stack(processed_hamer_images, dim=0).permute(1, 0, 2, 3)
                if processed_hamer_images
                else []
            ),
        )

    @staticmethod
    def augmentation(img, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(img)

    def __getitem__(self, index):
        flag = True
        while flag:
            vid_info = dict(self.videos_info[index])
            vid_path = os.path.join(vid_info["root"], vid_info["video"])
            vid_capture = cv2.VideoCapture(vid_path)
            vid_len = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            out_path = os.path.join(
                self.output_dir, os.path.basename(vid_path).replace(".mp4", ".pkl")
            )
            if os.path.exists(out_path) or vid_len > 2400:
                return {"path": vid_path}

            if self.sk_videos_info:
                sk_vid_info = dict(self.sk_videos_info[index])
                sk_vid_path = os.path.join(sk_vid_info["root"], sk_vid_info["video"])
                sk_vid_path = sk_vid_path.replace(".mp4", ".pkl")
                try:
                    with open(sk_vid_path, "rb") as f:
                        sk_vid_capture = pickle.load(f)
                except:
                    print(f"Error: Failed to load {sk_vid_path}")
                    return {"path": ""}

                sk_vid_len = len(sk_vid_capture)

            if self.hamer_videos_info:
                hamer_vid_info = dict(self.hamer_videos_info[index])
                hamer_vid_path = os.path.join(
                    hamer_vid_info["root"], hamer_vid_info["video"]
                )
                hamer_vid_path = hamer_vid_path.replace(
                    "processed_videos", "processed_videos2"
                )
                try:
                    hamer_vid_capture = cv2.VideoCapture(hamer_vid_path)
                    hamer_vid_len = int(hamer_vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    if hamer_vid_len != sk_vid_len or not hamer_vid_capture.isOpened():
                        print(
                            f"The hamer video_len of {hamer_vid_path} is {hamer_vid_len}, which is not equal to {sk_vid_len} of {sk_vid_path}. "
                            f"NOW SKIP IT!"
                        )
                        index = (index + 1) % len(self.videos_info)
                        vid_capture.release()
                        hamer_vid_capture.release()
                        return {"path": ""}
                except:
                    print(f"Error: Failed to load {hamer_vid_path}")
                    index = (index + 1) % len(self.videos_info)
                    vid_capture.release()
                    hamer_vid_capture.release()
                    return {"path": ""}

            if self.sk_videos_info and self.hamer_videos_info:
                if vid_len != sk_vid_len or not vid_len == hamer_vid_len:
                    print(
                        f"The video_len of {vid_path} is {vid_len}, which is not equal to {sk_vid_path} of {sk_vid_len} or {hamer_vid_path} of {hamer_vid_len}. "
                        f"NOW SKIP IT!"
                    )
                    index = (index + 1) % len(self.videos_info)
                    vid_capture.release()
                    hamer_vid_capture.release()
                    return {"path": ""}

                batch_ids = list(range(sk_vid_len))
                batch_ids = batch_ids[:: self.sample_rate]

                tgt_sk_frames = []
                tgt_hamer_frames = []

                for frame_idx in batch_ids:
                    if self.sk_videos_info:
                        tgt_sk_frame = sk_vid_capture[frame_idx]

                        tgt_sk_frame = draw_pose(
                            tgt_sk_frame, 260, 210
                        )  # 512,512 for How2Sign
                        # numpy to image
                        tgt_sk_frame = Image.fromarray(tgt_sk_frame)
                        tgt_sk_frames.append(tgt_sk_frame)

                    if self.hamer_videos_info:
                        tgt_hamer_frame = self.read_frame(hamer_vid_capture, frame_idx)
                        tgt_hamer_frames.append(tgt_hamer_frame)

                transform_rand_state = torch.get_rng_state()

                tgt_sk_frames, tgt_hamer_frames = self.process_frames(
                    tgt_sk_frames,
                    tgt_hamer_frames,
                    transform_rand_state,
                )

                sample = dict(
                    tgt_sk_frames=tgt_sk_frames,
                    tgt_hamer_frames=tgt_hamer_frames,
                    path=vid_path,
                )
                return sample

    def __len__(self):
        return len(self.videos_info)
