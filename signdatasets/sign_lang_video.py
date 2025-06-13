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
from torchvision.transforms import (
    ColorJitter,
    Compose,
    InterpolationMode,
    Normalize,
    PILToTensor,
    RandomGrayscale,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import CLIPImageProcessor, CLIPTokenizer

from scripts.hamer.renderer import Renderer, render_hands
from scripts.sk.dwpose.util import draw_pose, get_hand_area
from utils import CannyDetector


class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)


class SignLangVideoDataset(Dataset):

    def __init__(
        self,
        frame_size: Tuple[int, int] = (512, 512),
        frame_scale: Tuple[float, float] = (1.0, 1.0),
        frame_ratio: Tuple[float, float] = (0.9, 1.0),
        roots: Union[List[str], str] = None,
        sk_roots: Union[List[str], str] = None,
        hamer_roots: Union[List[str], str] = None,
        meta_paths: Union[List[str], str] = None,
        sample_rate: int = 1,
        num_frames: int = 12,
        ref_margin: int = 30,
        uncond_ratio: float = 0.1,
        mask_ratio: float = 1,
        mask_thershold: float = 0.85,
        skip_ratio: float = 0,
        use_sample_random: bool = False,
        random_sample: bool = True,
        sk_mask_ratio: float = 0,
        hamer_mask_ratio: float = 0,
        both_mask_ratio: float = 0,
    ):
        super().__init__()

        self.frame_size = frame_size
        if use_sample_random:
            self.sample_rate = sample_rate if random.random() > 0.5 else 1
        else:
            self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.ref_margin = ref_margin
        self.uncond_ratio = uncond_ratio
        self.mask_ratio = mask_ratio
        self.skip_ratio = skip_ratio
        self.mask_thershold = mask_thershold
        self.random_sample = random_sample
        self.sk_mask_ratio = sk_mask_ratio
        self.hamer_mask_ratio = hamer_mask_ratio
        self.both_mask_ratio = both_mask_ratio

        roots = [roots] if isinstance(roots, str) else roots
        sk_roots = [sk_roots] if isinstance(sk_roots, str) else sk_roots
        hamer_roots = [hamer_roots] if isinstance(hamer_roots, str) else hamer_roots
        meta_paths = [meta_paths] if isinstance(meta_paths, str) else meta_paths
        assert len(roots) == len(
            meta_paths
        ), f"The number of roots is not equal to that of meta_paths! {len(roots)} != {len(meta_paths)}"

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
        # if self.hamer_videos_info:
        #     data = np.load("/deepo_data/hamer/faces.npz")
        #     faces = data["arr_0"]
        #     self.renderer = Renderer(5000, 256, faces=faces)

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

        self.clip_image_processor = CLIPImageProcessor()
        # self.canny_detector = CannyDetector()

    @staticmethod
    def read_frame(vid_capture, frame_idx):
        vid_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, frame = vid_capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        return frame

    def process_frames(
        self,
        reference_image,
        target_images,
        target_sk_images,
        target_hamer_images,
        rand_state,
    ):
        images = [reference_image] + target_images
        processed_images = []
        for image in images:
            target_image = self.augmentation(image, self.img_transform, rand_state)
            processed_images.append(target_image)
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
            processed_images[0],
            torch.stack(processed_images[1:], dim=0).permute(1, 0, 2, 3),
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
            vid_height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid_width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            text = vid_info["text"]

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
                    vid_capture.release()
                    continue

                sk_vid_len = len(sk_vid_capture)

                if sk_vid_len != vid_len:
                    print(
                        f"The sk video_len of {sk_vid_path} is {sk_vid_len}, which is not equal to {vid_len} of {vid_path}. "
                        f"NOW SKIP IT!"
                    )
                    index = (index + 1) % len(self.videos_info)
                    vid_capture.release()
                    continue

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
                    if hamer_vid_len != vid_len or not hamer_vid_capture.isOpened():
                        print(
                            f"The hamer video_len of {hamer_vid_path} is {hamer_vid_len}, which is not equal to {vid_len} of {vid_path}. "
                            f"NOW SKIP IT!"
                        )
                        index = (index + 1) % len(self.videos_info)
                        vid_capture.release()
                        hamer_vid_capture.release()
                        continue
                except:
                    print(f"Error: Failed to load {hamer_vid_path}")
                    index = (index + 1) % len(self.videos_info)
                    vid_capture.release()
                    hamer_vid_capture.release()
                    continue

            if vid_len < self.num_frames:
                print(
                    f"The video_len of {vid_path} is {vid_len}, which is less than num_frames {self.num_frames}. "
                    f"NOW SKIP IT!"
                )
                index = (index + 1) % len(self.videos_info)
                vid_capture.release()
                continue

            clip_vid_len = min(vid_len, (self.num_frames - 1) * self.sample_rate + 1)
            if self.random_sample:
                start_idx = random.randint(0, vid_len - clip_vid_len)
            else:
                start_idx = 0
            batch_ids = np.linspace(
                start_idx, start_idx + clip_vid_len - 1, self.num_frames, dtype=int
            ).tolist()
            tgt_frame_ids = torch.tensor(batch_ids)

            tgt_frames = []
            for frame_idx in batch_ids:
                tgt_frame = self.read_frame(vid_capture, frame_idx)
                tgt_frames.append(tgt_frame)

            tgt_sk_frames = []
            tgt_hand_masks = []
            tgt_hamer_frames = []

            for frame_idx in batch_ids:
                if self.sk_videos_info:
                    tgt_sk_frame = sk_vid_capture[frame_idx]

                    hand_areas, avg_hand_scores, _ = get_hand_area(
                        tgt_sk_frame["hands"], tgt_sk_frame["hands_score"]
                    )

                    # Create mask
                    mask = np.ones(
                        (self.frame_size[0], self.frame_size[1]), dtype=np.float32
                    )
                    avg_hand_score = np.mean(avg_hand_scores)
                    for area, score in zip(hand_areas, avg_hand_scores):
                        x1, y1, x2, y2 = area
                        x1, y1, x2, y2 = (
                            int(x1 * self.frame_size[0]),
                            int(y1 * self.frame_size[1]),
                            int(x2 * self.frame_size[0]),
                            int(y2 * self.frame_size[1]),
                        )
                        if score < self.mask_thershold:
                            mask[y1:y2, x1:x2] = 0.0

                    tgt_hand_masks.append(torch.from_numpy(mask))
                    tgt_sk_frame = draw_pose(tgt_sk_frame, vid_height, vid_width)
                    # numpy to image
                    tgt_sk_frame = Image.fromarray(tgt_sk_frame)
                    tgt_sk_frames.append(tgt_sk_frame)

                if self.hamer_videos_info:
                    tgt_hamer_frame = self.read_frame(hamer_vid_capture, frame_idx)
                    tgt_hamer_frames.append(tgt_hamer_frame)

            if self.sk_videos_info and self.hamer_videos_info and self.num_frames == 1:
                if (
                    avg_hand_score < self.mask_thershold
                    and random.random() < self.skip_ratio
                ):
                    print(
                        f"Skip this sample because the hand score is {avg_hand_score}!"
                    )
                    index = (index + 1) % len(self.videos_info)
                    vid_capture.release()
                    continue

            if self.sk_videos_info:
                tgt_hand_masks = torch.stack(tgt_hand_masks)

            left_max_ref_idx = min(batch_ids) - self.ref_margin - 1
            right_min_ref_idx = max(batch_ids) + self.ref_margin + 1
            if left_max_ref_idx < 0 and right_min_ref_idx > vid_len:
                print(
                    f"There is no space to select a reference image in {vid_path}, "
                    f"because the maximum left reference index is {left_max_ref_idx}, "
                    f"the minimum right reference index is {right_min_ref_idx}. "
                    f"Both of them are not satisfied the condition: "
                    f"(1) the maximum left reference index is bigger than 0; "
                    f"(2) the minimum right reference index is smaller than video_len (it is {vid_len} here). "
                    f"NOW SKIP IT!"
                )
                index = (index + 1) % len(self.videos_info)
                vid_capture.release()
                continue

            ref_idx_range = list(range(vid_len))
            remove_ids = np.arange(
                left_max_ref_idx + 1, right_min_ref_idx - 1, dtype=int
            ).tolist()

            for remove_idx in remove_ids:
                if remove_idx not in ref_idx_range:
                    continue
                ref_idx_range.remove(remove_idx)
            if ref_idx_range:
                ref_idx = random.choice(ref_idx_range)
            else:
                ref_idx = ref_idx_range[0]

            ref_frame = self.read_frame(vid_capture, ref_idx)

            clip_image = self.clip_image_processor(
                images=ref_frame, return_tensors="pt"
            ).pixel_values[0, ...]
            transform_rand_state = torch.get_rng_state()
            ref_frame, tgt_frames, tgt_sk_frames, tgt_hamer_frames = (
                self.process_frames(
                    ref_frame,
                    tgt_frames,
                    tgt_sk_frames,
                    tgt_hamer_frames,
                    transform_rand_state,
                )
            )

            vid_capture.release()

            if random.random() < self.mask_ratio:
                # mask the sk and hamer frames blurry hands
                mask = tgt_hand_masks.unsqueeze(0)
                tgt_sk_frames = tgt_sk_frames * mask + (1 - mask) * -1
                tgt_hamer_frames = tgt_hamer_frames * mask + (1 - mask) * -1

            # Add per-frame random masking
            for frame_idx in range(tgt_sk_frames.shape[1]):  # Iterate through frames
                rand = random.random()
                if rand < self.both_mask_ratio:
                    # Mask both sk and hamer frame to black
                    tgt_sk_frames[:, frame_idx] = (
                        torch.ones_like(tgt_sk_frames[:, frame_idx]) * -1
                    )
                    tgt_hamer_frames[:, frame_idx] = (
                        torch.ones_like(tgt_hamer_frames[:, frame_idx]) * -1
                    )
                elif rand < (self.both_mask_ratio + self.sk_mask_ratio):
                    # Mask only sk frame
                    tgt_sk_frames[:, frame_idx] = (
                        torch.ones_like(tgt_sk_frames[:, frame_idx]) * -1
                    )
                elif rand < (
                    self.both_mask_ratio + self.sk_mask_ratio + self.hamer_mask_ratio
                ):
                    # Mask only hamer frame
                    tgt_hamer_frames[:, frame_idx] = (
                        torch.ones_like(tgt_hamer_frames[:, frame_idx]) * -1
                    )

            sample = dict(
                clip_image=clip_image,
                ref_frame=ref_frame,
                tgt_frames=tgt_frames,
                tgt_sk_frames=tgt_sk_frames,
                tgt_hamer_frames=tgt_hamer_frames,
                tgt_hand_masks=tgt_hand_masks,
                tgt_frame_ids=tgt_frame_ids,
                vid_len=vid_len,
            )
            return sample

    def __len__(self):
        return len(self.videos_info)
