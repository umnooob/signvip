import math
import os
import pathlib
import random
from typing import List, Optional

import cv2
import numpy as np
import pyrender
import torch
import torch.nn as nn
import torch.nn.functional as func
import tqdm
import trimesh
from einops import rearrange, repeat
from imageio_ffmpeg import get_ffmpeg_exe

# from scipy.signal import gaussian

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def get_num_params(params):
    trainable_params = list(filter(lambda p: p.requires_grad, params))
    num_params = sum(p.numel() for p in params) / 1e6
    num_trainable_params = sum(p.numel() for p in trainable_params) / 1e6
    return num_params, num_trainable_params


def get_sinusoidal_frames_pe(frame_ids, vid_len, dim, max_sin_pe_len):
    position = torch.arange(max_sin_pe_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe = torch.zeros(1, max_sin_pe_len, dim)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)  # [1, num_seq, dim]

    rel_pos = frame_ids / vid_len
    pos = torch.tensor(
        rel_pos * max_sin_pe_len, dtype=torch.long
    )  # [batch_size, num_frames]

    batch_size, _ = pos.shape
    _, _, dim = pe.shape

    pe = repeat(pe, "b l d -> (n b) l d", n=batch_size).to(
        pos.device
    )  # [batch_size, len, dim]
    pos = repeat(pos, "b l -> b l d", d=dim)  # [batch_size, num_seq, dim]
    frames_pe = torch.gather(pe, dim=1, index=pos)
    return frames_pe

def median_filter_3d(video_tensor, kernel_size, device):
    _, video_length, height, width = video_tensor.shape

    pad_size = kernel_size // 2
    video_tensor = func.pad(
        video_tensor,
        (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size),
        mode="reflect",
    )

    filtered_video_tensor = []
    for i in tqdm.tqdm(range(video_length), desc="Median Filtering"):
        video_segment = video_tensor[:, i : i + kernel_size, ...].to(device)
        video_segment = video_segment.unfold(dimension=2, size=kernel_size, step=1)
        video_segment = video_segment.unfold(dimension=3, size=kernel_size, step=1)
        video_segment = video_segment.permute(0, 2, 3, 1, 4, 5).reshape(
            3, height, width, -1
        )
        filtered_video_frame = torch.median(video_segment, dim=-1)[0]
        filtered_video_tensor.append(filtered_video_frame.cpu())
    filtered_video_tensor = torch.stack(filtered_video_tensor, dim=1)
    return filtered_video_tensor


def save_video(video_tensor, output_path, device, fps=30.0):
    # Create the full directory path
    output_dir = pathlib.Path(output_path).parent
    output_dir.mkdir(exist_ok=True, parents=True)

    video_tensor = video_tensor[0, ...]
    _, num_frames, height, width = video_tensor.shape

    # 方案1：移除中值滤波
    # video_tensor = median_filter_3d(video_tensor, kernel_size=3, device=device)

    # 方案2：仅在空间维度进行中值滤波，不在时间维度上过滤
    # video_tensor = spatial_median_filter(video_tensor, kernel_size=3, device=device)

    video_tensor = video_tensor.cpu()
    video_tensor = video_tensor.permute(1, 2, 3, 0)
    video_frames = (video_tensor * 255).numpy().astype(np.uint8)

    output_name = pathlib.Path(output_path).stem
    temp_output_path = str(output_dir / f"{output_name}-temp.mp4")
    video_writer = cv2.VideoWriter(
        temp_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for i in tqdm.tqdm(range(num_frames), "Writing frames into file"):
        frame_image = video_frames[i, ...]
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_image)
    video_writer.release()

    cmd = (
        f'{get_ffmpeg_exe()} -i "{temp_output_path}" '
        f'-map 0:v -c:v h264 -shortest -y "{output_path}" -loglevel quiet'
    )
    os.system(cmd)
    os.remove(temp_output_path)


def decode_latents(vae, latents):
    video_length = latents.shape[2]
    latents = 1 / vae.config.scaling_factor * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    video = []
    for frame_idx in tqdm.tqdm(
        range(latents.shape[0]), desc="Decoding latents into frames"
    ):
        image = vae.decode(latents[frame_idx : frame_idx + 1].to(vae.device)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().float()
        video.append(image)
    video = torch.cat(video)
    video = rearrange(video, "(b f) c h w -> b f c h w", f=video_length)

    return video


def frames_to_image(frames):
    b, f, c, h, w = frames.shape
    # Rearrange to (frames, height, width, channels)
    frames = rearrange(frames, "b f c h w -> (b f) h w c")

    # Calculate the dimensions of the final big image
    rows = int(torch.sqrt(torch.tensor(f)).item())
    cols = (f + rows - 1) // rows

    # Pad frames to make sure they can fit into a grid
    pad_size = rows * cols - f
    if pad_size > 0:
        pad = torch.zeros((pad_size, h, w, c))
        frames = torch.cat((frames, pad), dim=0)

    # Arrange frames into grid
    frames = frames.reshape(rows, cols, h, w, c)
    big_image = rearrange(frames, "r c h w k -> (r h) (c w) k")

    return big_image


def encode_prompt(
    tokenizer,
    text_encoder,
    prompt,
    device,
    num_videos_per_prompt,
    do_classifier_free_guidance,
    negative_prompt,
):
    batch_size = len(prompt) if isinstance(prompt, list) else 1

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(
        prompt, padding="longest", return_tensors="pt"
    ).input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        text_input_ids, untruncated_ids
    ):
        removed_text = tokenizer.batch_decode(
            untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
        )

    if (
        hasattr(text_encoder.config, "use_attention_mask")
        and text_encoder.config.use_attention_mask
    ):
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    text_embeddings = text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask,
    )
    text_embeddings = text_embeddings[0]

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    bs_embed, seq_len, _ = text_embeddings.shape
    text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
    text_embeddings = text_embeddings.view(
        bs_embed * num_videos_per_prompt, seq_len, -1
    )

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance:
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        max_length = text_input_ids.shape[-1]
        uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        if (
            hasattr(text_encoder.config, "use_attention_mask")
            and text_encoder.config.use_attention_mask
        ):
            attention_mask = uncond_input.attention_mask.to(device)
        else:
            attention_mask = None

        uncond_embeddings = text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=attention_mask,
        )
        uncond_embeddings = uncond_embeddings[0]

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
        uncond_embeddings = uncond_embeddings.view(
            batch_size * num_videos_per_prompt, seq_len, -1
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    return text_embeddings


def get_state_dict(filter_size=5, std=1.0, map_func=lambda x: x):
    generated_filters = (
        gaussian(filter_size, std=std).reshape([1, filter_size]).astype(np.float32)
    )

    gaussian_filter_horizontal = generated_filters[None, None, ...]

    gaussian_filter_vertical = generated_filters.T[None, None, ...]

    sobel_filter_horizontal = np.array(
        [[[[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]]], dtype="float32"
    )

    sobel_filter_vertical = np.array(
        [[[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]]], dtype="float32"
    )

    directional_filter = np.array(
        [
            [[[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
        ],
        dtype=np.float32,
    )

    connect_filter = np.array(
        [[[[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]], dtype=np.float32
    )

    return {
        "gaussian_filter_horizontal.weight": map_func(gaussian_filter_horizontal),
        "gaussian_filter_vertical.weight": map_func(gaussian_filter_vertical),
        "sobel_filter_horizontal.weight": map_func(sobel_filter_horizontal),
        "sobel_filter_vertical.weight": map_func(sobel_filter_vertical),
        "directional_filter.weight": map_func(directional_filter),
        "connect_filter.weight": map_func(connect_filter),
    }


class CannyDetector(nn.Module):
    def __init__(self, filter_size=5, std=1.0, device="cpu"):
        super(CannyDetector, self).__init__()
        # 配置运行设备
        self.device = device

        # 高斯滤波器
        self.gaussian_filter_horizontal = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, filter_size),
            padding=(0, filter_size // 2),
            bias=False,
        )
        self.gaussian_filter_vertical = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(filter_size, 1),
            padding=(filter_size // 2, 0),
            bias=False,
        )

        # Sobel 滤波器
        self.sobel_filter_horizontal = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False
        )
        self.sobel_filter_vertical = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False
        )

        # 定向滤波器
        self.directional_filter = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False
        )

        # 连通滤波器
        self.connect_filter = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False
        )

        # 初始化参数
        params = get_state_dict(
            filter_size=filter_size,
            std=std,
            map_func=lambda x: torch.from_numpy(x).to(self.device),
        )
        self.load_state_dict(params)

    @torch.no_grad()
    def forward(self, img, threshold1=1.5, threshold2=5):
        # 拆分图像通道
        img_r = img[:, 0:1]  # red channel
        img_g = img[:, 1:2]  # green channel
        img_b = img[:, 2:3]  # blue channel

        # Step1: 应用高斯滤波进行模糊降噪
        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        # Step2: 用 Sobel 算子求图像的强度梯度
        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # Step2: 确定边缘梯度和方向
        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = torch.atan2(
            grad_y_r + grad_y_g + grad_y_b, grad_x_r + grad_x_g + grad_x_b
        ) * (180.0 / math.pi)
        grad_orientation += 180.0
        grad_orientation = torch.round(grad_orientation / 45.0) * 45.0

        # Step3: 非最大抑制，边缘细化
        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        channel_select_filtered_positive = torch.gather(
            all_filtered, 1, inidices_positive.long()
        )
        channel_select_filtered_negative = torch.gather(
            all_filtered, 1, inidices_negative.long()
        )

        channel_select_filtered = torch.stack(
            [channel_select_filtered_positive, channel_select_filtered_negative]
        )

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0

        thin_edges = grad_mag.clone()
        thin_edges[is_max == 0] = 0.0

        # Step4: 双阈值
        low_threshold = min(threshold1, threshold2)
        high_threshold = max(threshold1, threshold2)
        thresholded = thin_edges.clone()
        lower = thin_edges < low_threshold
        thresholded[lower] = 0.0
        higher = thin_edges > high_threshold
        thresholded[higher] = 1.0
        connect_map = self.connect_filter(higher.float())
        middle = torch.logical_and(
            thin_edges >= low_threshold, thin_edges <= high_threshold
        )
        thresholded[middle] = 0.0
        connect_map[torch.logical_not(middle)] = 0
        thresholded[connect_map > 0] = 1.0
        thresholded[..., 0, :] = 0.0
        thresholded[..., -1, :] = 0.0
        thresholded[..., :, 0] = 0.0
        thresholded[..., :, -1] = 0.0
        thresholded = (thresholded > 0.0).float()

        return thresholded


# 新增函数：仅空间维度的中值滤波
def spatial_median_filter(video_tensor, kernel_size, device):
    _, num_frames, height, width = video_tensor.shape
    pad_size = kernel_size // 2

    filtered_frames = []
    for i in range(num_frames):
        frame = video_tensor[:, i : i + 1, :, :]  # [3, 1, H, W]
        padded = func.pad(
            frame, (pad_size, pad_size, pad_size, pad_size), mode="reflect"
        )
        frame = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        frame = frame.permute(0, 1, 2, 3, 4, 5).reshape(3, 1, height, width, -1)
        filtered = torch.median(frame, dim=-1)[0]
        filtered_frames.append(filtered)

    return torch.cat(filtered_frames, dim=1)
