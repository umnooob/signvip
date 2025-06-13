import json
import os
import pickle
import random
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Normalize,
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


class SignTextDataset(Dataset):

    def __init__(
        self,
        frame_size: Tuple[int, int] = (512, 512),
        frame_scale: Tuple[float, float] = (1.0, 1.0),
        sample_rate: int = 1,
        tokenizer=None,
        max_pose_len: int = 600,
        padding_value: int = 0,
        roots: Union[List[str], str] = None,
        sk_roots: Union[List[str], str] = None,
        hamer_roots: Union[List[str], str] = None,
        pose_roots: Union[List[str], str] = None,
        meta_paths: Union[List[str], str] = None,
    ):
        super().__init__()

        self.frame_size = frame_size
        self.tokenizer = tokenizer
        self.max_pose_len = max_pose_len
        self.padding_value = padding_value
        self.sample_rate = sample_rate

        roots = [roots] if isinstance(roots, str) else roots
        sk_roots = [sk_roots] if isinstance(sk_roots, str) else sk_roots
        hamer_roots = [hamer_roots] if isinstance(hamer_roots, str) else hamer_roots
        pose_roots = [pose_roots] if isinstance(pose_roots, str) else pose_roots
        meta_paths = [meta_paths] if isinstance(meta_paths, str) else meta_paths

        self.videos_info = []
        for root, meta_path in zip(roots, meta_paths):
            obj = json.load(open(meta_path, "r"))
            obj = [{"root": root, **o} for o in obj]
            self.videos_info.extend(obj)
        self.sk_videos_info = []
        if sk_roots:
            for sk_root, meta_path in zip(sk_roots, meta_paths):
                obj = json.load(open(meta_path, "r"))
                obj = [{"root": sk_root, **o} for o in obj]
                self.sk_videos_info.extend(obj)

        self.hamer_videos_info = []
        if hamer_roots:
            for hamer_root, meta_path in zip(hamer_roots, meta_paths):
                obj = json.load(open(meta_path, "r"))
                obj = [{"root": hamer_root, **o} for o in obj]
                self.hamer_videos_info.extend(obj)

        if self.hamer_videos_info:
            data = np.load("/deepo_data/hamer/faces.npz")
            faces = data["arr_0"]
            self.renderer = Renderer(5000, 256, faces=faces)

        self.pose_videos_info = []
        if pose_roots:
            for pose_root, meta_path in zip(pose_roots, meta_paths):
                obj = json.load(open(meta_path, "r"))
                obj = [{"root": pose_root, **o} for o in obj]
                self.pose_videos_info.extend(obj)

        self.img_transform = Compose(
            [
                # Resize(frame_size),
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
                ToTensor(),
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
            text = vid_info["text"]
            text_inputs = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_ids = text_inputs.input_ids
            text_attn_mask = text_inputs.attention_mask

            untruncated_ids = self.tokenizer(
                text, padding="longest", return_tensors="pt"
            ).input_ids
            if untruncated_ids.shape[-1] >= text_ids.shape[-1] and not torch.equal(
                text_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                print(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}."
                )

            text_ids = text_ids[0, ...]
            text_attn_mask = text_attn_mask[0, ...]

            if self.sk_videos_info:
                sk_vid_info = dict(self.sk_videos_info[index])
                sk_vid_path = os.path.join(sk_vid_info["root"], sk_vid_info["video"])
                sk_vid_path = sk_vid_path.replace(".mp4", ".pkl")
                try:
                    with open(sk_vid_path, "rb") as f:
                        sk_vid_capture = pickle.load(f)
                except:
                    print(f"Error: Failed to load {sk_vid_path}")
                    index = (index + 1) % len(self.videos_info)
                    continue

                sk_vid_len = len(sk_vid_capture)

            if self.hamer_videos_info:
                hamer_vid_info = dict(self.hamer_videos_info[index])
                hamer_vid_path = os.path.join(
                    hamer_vid_info["root"], hamer_vid_info["video"]
                )
                hamer_vid_path = hamer_vid_path.replace(
                    "train_processed_videos", "train_processed_videos2"
                )
                try:
                    hamer_vid_capture = cv2.VideoCapture(hamer_vid_path)
                    hamer_vid_len = int(hamer_vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    if not hamer_vid_capture.isOpened():
                        print(f"Error: Failed to load {hamer_vid_path}")
                        index = (index + 1) % len(self.videos_info)
                        vid_capture.release()
                        hamer_vid_capture.release()
                        continue
                except:
                    print(f"Error: Failed to load {hamer_vid_path}")
                    index = (index + 1) % len(self.videos_info)
                    continue
            if self.sk_videos_info and self.hamer_videos_info:
                if vid_len != sk_vid_len or not vid_len == hamer_vid_len:
                    print(
                        f"The video_len of {vid_path} is {vid_len}, which is not equal to {sk_vid_path} of {sk_vid_len} or {hamer_vid_path} of {hamer_vid_len}. "
                        f"NOW SKIP IT!"
                    )
                    index = (index + 1) % len(self.videos_info)
                    vid_capture.release()
                    hamer_vid_capture.release()
                    continue

                batch_ids = list(range(sk_vid_len))
                batch_ids = batch_ids[:: self.sample_rate]

                tgt_sk_frames = []
                tgt_hamer_frames = []

                for frame_idx in batch_ids:
                    if self.sk_videos_info:
                        tgt_sk_frame = sk_vid_capture[frame_idx]

                        tgt_sk_frame = draw_pose(
                            tgt_sk_frame, self.frame_size[0], self.frame_size[1]
                        )
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
                vid_capture.release()
                hamer_vid_capture.release()
                sample = dict(
                    path=os.path.basename(vid_path),
                    tgt_sk_frames=tgt_sk_frames,
                    tgt_hamer_frames=tgt_hamer_frames,
                    text_input_ids=text_ids,
                    text_attn_mask=text_attn_mask,
                )
                return sample
            elif self.pose_videos_info:
                pose_vid_info = dict(self.pose_videos_info[index])
                pose_vid_path = os.path.join(
                    pose_vid_info["root"], pose_vid_info["video"]
                )
                pose_vid_path = pose_vid_path.replace(".mp4", ".pkl")
                try:
                    with open(pose_vid_path, "rb") as f:
                        pose_latents = pickle.load(f)
                    # flatten the pose_latents
                    pose_latents = rearrange(pose_latents, "f d h w -> f (h w d)")
                    # pose_latents = pose_latents.reshape(pose_latents.shape[0], -1)
                    pose_latents = pose_latents[:: self.sample_rate]
                    pose_latents = pose_latents[
                        : self.max_pose_len
                    ]  # [Frame, pose_dim]
                    pose_latents = torch.tensor(pose_latents, dtype=torch.float32)
                    pose_len = len(pose_latents)
                    # add counter at last pose_dim
                    counter = torch.linspace(0, 1, pose_len)
                    counter = counter.unsqueeze(1)  # [pose_len, 1]
                    pose_latents = torch.cat(
                        [pose_latents, counter], dim=1
                    )  # [pose_len, pose_dim+1]
                    pose_latents = torch.nn.functional.pad(
                        pose_latents,
                        (0, 0, 0, self.max_pose_len - len(pose_latents)),
                        value=self.padding_value,
                    )

                except Exception as e:
                    print(f"Error: Failed to load {pose_vid_path} {e}")
                    index = (index + 1) % len(self.videos_info)
                    continue
                sample = dict(
                    path=os.path.basename(vid_path),
                    pose_latents=pose_latents,
                    pose_len=pose_len,
                    text_input_ids=text_ids,
                    text_attn_mask=text_attn_mask,
                )
                return sample
            else:
                sample = dict(
                    path=os.path.basename(vid_path),
                    text_input_ids=text_ids,
                    text_attn_mask=text_attn_mask,
                )
                return sample

    def __len__(self):
        return len(self.videos_info)
