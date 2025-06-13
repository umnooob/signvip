import json
import os
import pickle
import random
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import CLIPTokenizer


class SignCondDataset(Dataset):

    def __init__(
        self,
        pose_roots: Union[List[str], str] = None,
        meta_paths: Union[List[str], str] = None,
        tokenizer: CLIPTokenizer = None,
        seq_len: int = 128,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        pose_roots = [pose_roots] if isinstance(pose_roots, str) else pose_roots

        self.pose_videos_info = []
        for pose_root, meta_path in zip(pose_roots, meta_paths):
            obj = json.load(open(meta_path, "r"))
            obj = [{"root": pose_root, **o} for o in obj]
            self.pose_videos_info.extend(obj)

    def __getitem__(self, index):
        pose_video_info = dict(self.pose_videos_info[index])
        pose_video_path = os.path.join(
            pose_video_info["root"], pose_video_info["video"]
        )

        with open(pose_video_path, "rb") as f:
            pose_video = pickle.load(f)
        pose_video = pose_video.astype(np.float32)
        pose_video = torch.from_numpy(pose_video).reshape(1, -1, self.seq_len)

        text = pose_video_info["text"]
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_ids = text_inputs.input_ids[0, ...]
        text_attn_mask = text_inputs.attention_mask[0, ...]

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
        sample = {
            "pose": pose_video,
            "text_input_ids": text_ids,
            "text_attn_mask": text_attn_mask,
        }
        return sample

    def __len__(self):
        return len(self.pose_videos_info)
