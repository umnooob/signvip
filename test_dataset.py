import argparse
import datetime
import logging
import math
import os
import pathlib
import random
import shutil
import time
import warnings

import accelerate
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.transforms as transforms
from accelerate.logging import get_logger
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    MotionAdapter,
    UNet2DConditionModel,
)
from diffusers.configuration_utils import FrozenDict
from diffusers.optimization import get_scheduler
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.utils import make_grid
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from signdatasets import SignLangVideoDataset

# def canny_edge_detection(image):

#     canny_detector = CannyDetector()
#     edges = canny_detector(image)

#     # Convert back to PIL Image
#     return edges


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/deepo_data/signvip/configs/stage1/stage_1_multicond_RWTH.yaml",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    args = parser.parse_args()

    # Load the configuration file
    cfg = OmegaConf.load(args.config)
    cfg.exp_name = pathlib.Path(args.config).stem

    # Update cfg with values from args
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    args_dict = vars(args)
    merged_dict = {**cfg_dict, **args_dict}

    # Convert merged_dict back to OmegaConf object
    merged_cfg = OmegaConf.create(merged_dict)

    return merged_cfg


cfg = parse_config()
fig_size = list((i // 30) * 3 for i in cfg.dataset.frame_size)
print(fig_size)
tokenizer = CLIPTokenizer.from_pretrained(cfg.modules.tokenizer)
dataset = SignLangVideoDataset(
    frame_size=cfg.dataset.frame_size,
    frame_scale=cfg.dataset.frame_scale,
    frame_ratio=cfg.dataset.frame_ratio,
    roots=cfg.dataset.roots,
    sk_roots=cfg.dataset.sk_roots,
    hamer_roots=cfg.dataset.hamer_roots,
    meta_paths=cfg.dataset.meta_paths,
    sample_rate=cfg.dataset.sample_rate,
    num_frames=8,
    ref_margin=cfg.dataset.ref_margin,
    uncond_ratio=cfg.dataset.uncond_ratio,
    mask_ratio=cfg.dataset.mask_ratio,
    mask_thershold=cfg.dataset.mask_thershold,
    skip_ratio=cfg.dataset.skip_ratio,
    sk_mask_ratio=cfg.dataset.sk_mask_ratio,
    hamer_mask_ratio=cfg.dataset.hamer_mask_ratio,
    both_mask_ratio=cfg.dataset.both_mask_ratio,
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 获取一个样本
sample = next(iter(dataloader))

# 提取 ref_frame 和 tgt_frames
ref_frame = sample["ref_frame"].squeeze().cpu().numpy()
tgt_frames = sample["tgt_frames"].squeeze().cpu().numpy()
tgt_sk_frames = sample["tgt_sk_frames"].squeeze().cpu().numpy()
tgt_hamer_frames = sample["tgt_hamer_frames"].squeeze().cpu().numpy()
# tgt_canny_frames = sample["tgt_canny_frames"].squeeze().cpu().numpy()
# tgt_gray_frames = sample["tgt_gray_frames"].squeeze().cpu().numpy()
# tgt_hand_masks = sample["tgt_hand_masks"].squeeze().cpu().numpy()

ori_tgt_frames = tgt_frames.copy()  # detect on hands
print(f"tgt_sk_frames: {tgt_sk_frames.shape}")  # tgt_sk_frames: (3, 8, 216, 264)
print(f"tgt_frames: {tgt_frames.shape}")  # tgt_frames: (3, 8, 216, 264)


if ori_tgt_frames.shape[0] == 3:  # 如果形状是 (3, 8, 512, 512)
    ori_tgt_frames = ori_tgt_frames.transpose(1, 0, 2, 3)
# 转换 ref_frame 的形状从 (3, 512, 512) 到 (512, 512, 3)
ref_frame = ref_frame.transpose(1, 2, 0)


# 转换 tgt_frames 的形状
if tgt_frames.shape[0] == 3:  # 如果形状是 (3, 8, 512, 512)
    tgt_frames = tgt_frames.transpose(1, 2, 3, 0)
elif tgt_frames.shape[0] == 8:  # 如果形状是 (8, 3, 512, 512)
    tgt_frames = tgt_frames.transpose(0, 2, 3, 1)

# 转换 tgt_sk_frames 的形状
if tgt_sk_frames.shape[0] == 3:  # 如果形状是 (3, 8, 512, 512)
    tgt_sk_frames = tgt_sk_frames.transpose(1, 2, 3, 0)
elif tgt_sk_frames.shape[0] == 8:  # 如果形状是 (8, 3, 512, 512)
    tgt_sk_frames = tgt_sk_frames.transpose(0, 2, 3, 1)

# 转换 tgt_hamer_frames 的形状
if tgt_hamer_frames.shape[0] == 3:  # 如果形状是 (3, 8, 512, 512)
    tgt_hamer_frames = tgt_hamer_frames.transpose(1, 2, 3, 0)
elif tgt_hamer_frames.shape[0] == 8:  # 如果形状是 (8, 3, 512, 512)
    tgt_hamer_frames = tgt_hamer_frames.transpose(0, 2, 3, 1)

# tgt_gray_frames = tgt_gray_frames[:,None,...]
# # 转换 tgt_gray_frames 的形状
# if tgt_gray_frames.shape[0] == 1:  # 如果形状是 (1, 8, 512, 512)
#     tgt_gray_frames = tgt_gray_frames.transpose(0, 2, 3, 1)
# elif tgt_gray_frames.shape[0] == 8:  # 如果形状是 (8, 1, 512, 512)
#     tgt_gray_frames = tgt_gray_frames.transpose(0, 2, 3, 1)


tgt_sk_frame = (tgt_sk_frames[0] + 1) / 2
tgt_frame = (tgt_frames[0] + 1) / 2
# save images
from PIL import Image

Image.fromarray((tgt_sk_frame * 255).astype(np.uint8)).save("tgt_sk_frame.png")
Image.fromarray((tgt_frame * 255).astype(np.uint8)).save("tgt_frame.png")


# 创建一个图形来显示图像
fig, axes = plt.subplots(3, 3, figsize=fig_size)
fig.suptitle("Reference Frame and Target Frames", fontsize=16)

# 显示 ref_frame
axes[0, 0].imshow((ref_frame + 1) / 2)  # 将 [-1, 1] 范围转换为 [0, 1]
axes[0, 0].set_title("Reference Frame")
axes[0, 0].axis("off")

# 显示 tgt_frames
for i in range(8):
    row = (i + 1) // 3
    col = (i + 1) % 3
    tgt_frame = tgt_frames[i]
    axes[row, col].imshow((tgt_frame + 1) / 2)  # 将 [-1, 1] 范围转换为 [0, 1]
    axes[row, col].set_title(f"Target Frame {i+1}")
    axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("debug.png")


# # 创建一个图形来显示图像
# fig, axes = plt.subplots(3, 3, figsize=fig_size)
# fig.suptitle("Reference Frame and Target Frames", fontsize=16)

# # 显示 ref_frame
# axes[0, 0].imshow((ref_frame + 1) / 2)  # 将 [-1, 1] 范围转换为 [0, 1]
# axes[0, 0].set_title("Reference Frame")
# axes[0, 0].axis("off")

# # 显示 tgt_frames
# for i in range(8):
#     row = (i + 1) // 3
#     col = (i + 1) % 3
#     tgt_frame = (ori_tgt_frames[i] + 1) / 2
#     tgt_frame = tgt_frame.squeeze().numpy()
#     axes[row, col].imshow(tgt_frame)
#     axes[row, col].set_title(f"Target Frame {i+1}")
#     axes[row, col].axis("off")

# plt.tight_layout()
# plt.savefig("debug_canny.png")

fig, axes = plt.subplots(3, 3, figsize=fig_size)
fig.suptitle("Target SK Frames", fontsize=16)

# 显示 ref_frame
axes[0, 0].imshow((ref_frame + 1) / 2)  # 将 [-1, 1] 范围转换为 [0, 1]
axes[0, 0].set_title("Reference Frame")
axes[0, 0].axis("off")

# 显示 tgt_sk_frames
for i in range(8):
    row = (i + 1) // 3
    col = (i + 1) % 3
    tgt_sk_frame = (tgt_sk_frames[i] + 1) / 2
    axes[row, col].imshow(tgt_sk_frame)  # 将 [-1, 1] 范围转换为 [0, 1]
    axes[row, col].set_title(f"Target SK Frame {i+1}")
    axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("debug_sk.png")


fig, axes = plt.subplots(3, 3, figsize=fig_size)
fig.suptitle("Target HAMER Frames", fontsize=16)
# 显示 ref_frame
axes[0, 0].imshow((ref_frame + 1) / 2)  # 将 [-1, 1] 范围转换为 [0, 1]
axes[0, 0].set_title("Reference Frame")
axes[0, 0].axis("off")

# 显示 tgt_hamer_frames
for i in range(8):
    row = (i + 1) // 3
    col = (i + 1) % 3
    tgt_hamer_frame = (tgt_hamer_frames[i] + 1) / 2
    axes[row, col].imshow(tgt_hamer_frame)
    axes[row, col].set_title(f"Target HAMER Frame {i+1}")
    axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("debug_hamer.png")


fig, axes = plt.subplots(3, 3, figsize=fig_size)
fig.suptitle("Target HAMER canny Frames", fontsize=16)
# 显示 ref_frame
axes[0, 0].imshow((ref_frame + 1) / 2)  # 将 [-1, 1] 范围转换为 [0, 1]
axes[0, 0].set_title("Reference Frame")
axes[0, 0].axis("off")

# # 显示 tgt_canny_frames
# for i in range(8):
#     row = (i + 1) // 3
#     col = (i + 1) % 3
#     tgt_canny_frame = (tgt_canny_frames[i] + 1) / 2
#     axes[row, col].imshow(tgt_canny_frame)
#     axes[row, col].set_title(f"Target HAMER canny Frame {i+1}")
#     axes[row, col].axis("off")

# plt.tight_layout()
# plt.savefig("debug_hamer_canny.png")


fig, axes = plt.subplots(3, 3, figsize=fig_size)
fig.suptitle("Target HAMER gray Frames", fontsize=16)
# 显示 ref_frame
axes[0, 0].imshow((ref_frame + 1) / 2)  # 将 [-1, 1] 范围转换为 [0, 1]
axes[0, 0].set_title("Reference Frame")
axes[0, 0].axis("off")

# # 显示 tgt_gray_frames
# for i in range(8):
#     row = (i + 1) // 3
#     col = (i + 1) % 3
#     tgt_gray_frame = tgt_gray_frames[i]
#     print(tgt_gray_frame)
#     axes[row, col].imshow(tgt_gray_frame)
#     axes[row, col].set_title(f"Target HAMER gray Frame {i+1}")
#     axes[row, col].axis("off")

# plt.tight_layout()
# plt.savefig("debug_hamer_gray.png")


fig, axes = plt.subplots(3, 3, figsize=fig_size)
fig.suptitle("Reference Frame and Target Frames", fontsize=16)

# 显示 ref_frame
axes[0, 0].imshow((ref_frame + 1) / 2)  # 将 [-1, 1] 范围转换为 [0, 1]
axes[0, 0].set_title("Reference Frame")
axes[0, 0].axis("off")


gray_transform = transforms.Grayscale(num_output_channels=1)
# 显示 tgt_frames
for i in range(8):
    row = (i + 1) // 3
    col = (i + 1) % 3
    tgt_frame = ori_tgt_frames[i]
    tgt_frame = gray_transform(torch.from_numpy(tgt_frame))
    tgt_frame = tgt_frame.squeeze().numpy()
    axes[row, col].imshow(tgt_frame)
    axes[row, col].set_title(f"Target Frame {i+1}")
    axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("debug_gray.png")
