import argparse
import logging
import math
import os
import pathlib
import pickle
import random
import time
import warnings
from collections import OrderedDict

import accelerate
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.transforms as transforms
from accelerate.logging import get_logger
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    MotionAdapter,
    UNet2DConditionModel,
    UNetMotionModel,
)
from einops import rearrange

# from fastdtw import fastdtw
from omegaconf import OmegaConf
from PIL import Image
from scipy.spatial.distance import cdist
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from models.appearance_encoder import AppearanceEncoderModel
from models.condition_encoder import VQConditionEncoder
from models.unet import UNet3DConditionModel
from pipelines.pipeline_multicond import SignViPPipeline
from utils import save_video, seed_everything

warnings.filterwarnings("ignore")
logger = get_logger(__name__, log_level="INFO")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/deepo_data/signvip/workspace/vq_multicond_RWTH_back/20250108-2127/config.yaml",
    )

    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_inference_steps", type=int, default=20)

    parser.add_argument(
        "--reference_image_path",
        type=str,
        default=None,
    )
    parser.add_argument("--output_path", type=str, default="./output/1.mp4")
    parser.add_argument("--input_path", type=str, default="./input/1.pkl")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--pose_size", type=int, default=64)
    parser.add_argument(
        "--video_base_path",
        type=str,
        default="/deepo_data/signvipworkspace/datasets/RWTH-T/dev_processed_videos",
    )
    args = parser.parse_args()
    cfg_file = parser.parse_args().config

    cfg = OmegaConf.load(cfg_file)
    cfg.exp_name = pathlib.Path(cfg_file).stem

    return args, cfg


def load_modules(cfg, device, weight_dtype):
    modules_cfg = cfg.modules
    vae = AutoencoderKL.from_pretrained(modules_cfg.vae).to(device, weight_dtype)
    logger.info(f"Loaded VAE from {modules_cfg.vae}.")

    unet = UNet3DConditionModel.from_pretrained_2d(
        modules_cfg.unet_2d,
        unet_additional_kwargs=OmegaConf.to_container(
            modules_cfg.unet_additional_kwargs
        ),
    )

    logger.info(f"Loaded U-Net from {modules_cfg.unet_2d} (unet_2d)")
    appearance_encoder = AppearanceEncoderModel.from_pretrained(
        modules_cfg.apperance_encoder
    ).to(device, weight_dtype)
    logger.info(f"Loaded appearance encoder from {modules_cfg.apperance_encoder}.")

    condition_encoder = VQConditionEncoder(
        conditioning_channels=3,
        image_finetune=False,
        num_conds=2,
        motion_module_type=modules_cfg.unet_additional_kwargs.motion_module_type,
        motion_module_kwargs=modules_cfg.unet_additional_kwargs.motion_module_kwargs,
        **modules_cfg.condition_encoder_kwargs,
    )
    if modules_cfg.condition_encoder:
        state_dict = torch.load(modules_cfg.condition_encoder, map_location="cpu")
        motion_module_state_dict = torch.load(
            modules_cfg.condition_encoder_motion, map_location="cpu"
        )
        state_dict.update(motion_module_state_dict)
        logger.info(f"Loaded condition encoder from {modules_cfg.condition_encoder}.")
        if modules_cfg.get("vq_model", None):
            vq_state_dict = torch.load(modules_cfg.vq_model, map_location="cpu")
            vq_state_dict = {
                k[7:] if k.startswith("module.") else k: v
                for k, v in vq_state_dict.items()
            }
            state_dict.update(vq_state_dict)
        missing, unexpected = condition_encoder.load_state_dict(
            state_dict, strict=False
        )
        assert len(unexpected) == 0
        logger.info(f"missing: {missing}")
    assert len(unexpected) == 0
    assert len(missing) == 0

    condition_encoder.to(device, weight_dtype)
    condition_encoder = condition_encoder.to(device, weight_dtype)

    if modules_cfg.unet:
        unet.load_state_dict(torch.load(modules_cfg.unet))
        unet.to(device, weight_dtype)
        logger.info(f"Loaded full UNET from {modules_cfg.unet}.")

    # Load empty_text_emb
    empty_text_emb = torch.load(cfg.modules.empty_text_emb).to(device, weight_dtype)

    scheduler = DDIMScheduler.from_pretrained(
        modules_cfg.scheduler,
    )
    return (
        vae,
        unet,
        empty_text_emb,
        scheduler,
        appearance_encoder,
        condition_encoder,
    )


def infer_one_video(cfg, args, pipeline, condition_encoder, ref_frame, tgt_frames):
    tgt_frames = condition_encoder.preprocess(tgt_frames)
    video_tensor = pipeline(
        condition_encoder=condition_encoder,
        sk_images=None,
        hamer_images=None,
        ref_image=ref_frame,
        pose_latent=tgt_frames,
        width=cfg.dataset.frame_size[1],
        height=cfg.dataset.frame_size[0],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        context_batch_size=1,
        context_frames=16,
    )
    return video_tensor


def main():
    args, cfg = parse_config()
    start_time = time.time()
    accelerator = accelerate.Accelerator()
    device = accelerator.device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # if cfg.seed is not None:
    #     seed_everything(cfg.seed)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    elif cfg.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training!"
        )
    weight_dtype = torch.float16

    vae, unet, empty_text_emb, scheduler, appearance_encoder, condition_encoder = (
        load_modules(cfg, device, weight_dtype)
    )
    pipeline = SignViPPipeline(
        vae=vae,
        denoising_unet=unet,
        scheduler=scheduler,
        empty_text_emb=empty_text_emb,
        appearance_encoder=appearance_encoder,
    ).to(dtype=weight_dtype, device=device)

    if args.input_dir is not None:
        input_paths = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".pkl")
        ]
    else:
        input_paths = [args.input_path]

    random.shuffle(input_paths)
    for input_path in tqdm(input_paths):
        output_path = input_path.replace(".pkl", ".mp4")
        if os.path.exists(output_path):
            continue
        with open(input_path, "rb") as f:
            sample = pickle.load(f)

        eos = cfg.modules.condition_encoder_kwargs.vq_kwargs.n_e
        if isinstance(sample, dict):
            pred_pose_latent = torch.from_numpy(sample["pred"]).to(device, weight_dtype)
            eos_idx = (pred_pose_latent == eos).nonzero()
            if len(eos_idx) == 0:
                continue
            # calculate nearest eos_idx
            eos_idx = (eos_idx[0] // args.pose_size) * args.pose_size

            pred_pose_latent = pred_pose_latent[:eos_idx]
            pred_pose_latent = pred_pose_latent.reshape(-1, args.pose_size).to(
                torch.long
            )
        else:
            pred_pose_latent = (
                torch.from_numpy(sample)
                .to(device)
                .reshape(-1, args.pose_size)
                .to(torch.long)
            )

        # read ref_frame from video first frame
        origin_path = os.path.join(
            args.video_base_path,
            os.path.basename(input_path).replace(".pkl", ".mp4"),
        )
        cap = cv2.VideoCapture(origin_path)
        ref_frames = []

        while True:
            ret, img_cv2 = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            ref_frame = Image.fromarray(frame_rgb).resize(
                (cfg.dataset.frame_size[1], cfg.dataset.frame_size[0])
            )
            # ref_frame = transforms.ToTensor()(ref_frame)
            ref_frames.append(ref_frame)
            break

        cap.release()

        if args.reference_image_path is None:
            first_frame = ref_frames[0]
        else:
            first_frame = Image.open(args.reference_image_path).convert("RGB")

        pred_video = infer_one_video(
            cfg, args, pipeline, condition_encoder, first_frame, pred_pose_latent
        )

        # concat ref_video and tgt_video in width

        save_video(
            pred_video,
            output_path,
            device=device,
            fps=24,
        )


if __name__ == "__main__":
    main()
