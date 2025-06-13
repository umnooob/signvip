import argparse
import logging
import os
import pathlib

import accelerate
import cv2  # Add OpenCV import
import numpy as np
import torch
import torchvision.transforms as transforms
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.appearance_encoder import AppearanceEncoderModel
from models.condition_encoder import ConditionEncoder
from models.unet import UNet3DConditionModel
from pipelines.pipeline_static import SignViPStaticPipeline
from signdatasets import SignCondDataset
from utils import seed_everything

logger = get_logger(__name__, log_level="INFO")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_frames",
        help="Directory to save output frames",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt to use for generation",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=5,
        help="Maximum number of videos to process",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of frames to process at once to avoid OOM errors",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    # If prompt is provided through command line, use it instead
    if args.prompt:
        cfg.validation_data.prompt = args.prompt

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

    condition_encoder = ConditionEncoder(
        conditioning_channels=3,
        image_finetune=True,
        num_conds=2,
    )
    if modules_cfg.condition_encoder:
        condition_encoder.load_state_dict(torch.load(modules_cfg.condition_encoder))
        logger.info(f"Loaded condition encoder from {modules_cfg.condition_encoder}.")
    condition_encoder.to(device, weight_dtype)

    if modules_cfg.unet:
        unet.load_state_dict(torch.load(modules_cfg.unet))
        unet.to(device, weight_dtype)
        logger.info(f"Loaded full UNET from {modules_cfg.unet}.")

    scheduler = DDIMScheduler.from_pretrained(
        cfg.modules.scheduler,
    )

    empty_text_emb = torch.load(modules_cfg.empty_text_emb).to(device, weight_dtype)

    return (
        vae,
        unet,
        scheduler,
        appearance_encoder,
        condition_encoder,
        empty_text_emb,
    )


def save_frames(video_tensor, output_dir, video_name, start_idx=0):
    """Save individual frames from video tensor as separate image files."""
    # Create output directory for this video
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # Extract frames from tensor (assuming shape [1, C, F, H, W])
    video_tensor = video_tensor[0]  # Remove batch dimension
    _, num_frames, height, width = video_tensor.shape

    # Convert to numpy arrays and save as images
    to_pil = transforms.ToPILImage()

    for frame_idx in tqdm(
        range(num_frames), desc=f"Saving frames {start_idx}-{start_idx+num_frames-1}"
    ):
        # Extract frame and convert to PIL image
        frame = video_tensor[:, frame_idx, :, :]
        frame_pil = to_pil(frame)

        # Save image with global frame index
        frame_path = os.path.join(
            video_output_dir, f"frame_{start_idx + frame_idx:04d}.png"
        )
        frame_pil.save(frame_path)

    logger.info(f"Saved {num_frames} frames to {video_output_dir}")
    return video_output_dir


def process_video_in_batches(
    pipeline,
    first_frame,
    tgt_sk_frames,
    tgt_hamer_frames,
    cfg,
    args,
    video_name,
    output_dir,
):
    """Process video in smaller batches to avoid CUDA OOM errors."""
    b, c, total_frames, h, w = tgt_sk_frames.shape
    processed_frames = 0
    batch_size = args.batch_size

    # Process video in batches
    while processed_frames < total_frames:
        # Calculate end index for current batch (handle last smaller batch)
        end_idx = min(processed_frames + batch_size, total_frames)
        curr_batch_size = end_idx - processed_frames

        logger.info(
            f"Processing frames {processed_frames} to {end_idx-1} (batch size: {curr_batch_size})"
        )

        # Extract current batch of skeleton and HAMER frames
        batch_sk_frames = tgt_sk_frames[:, :, processed_frames:end_idx, :, :]
        batch_hamer_frames = tgt_hamer_frames[:, :, processed_frames:end_idx, :, :]

        # Generate frames for current batch
        generated_video = pipeline(
            prompt=cfg.validation_data.prompt,
            reference_image=first_frame,
            sk_image=batch_sk_frames,
            hamer_image=batch_hamer_frames,
            width=cfg.dataset.frame_size[1],
            height=cfg.dataset.frame_size[0],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            batch_size=curr_batch_size,
        )

        # Save frames from current batch
        save_frames(generated_video, output_dir, video_name, start_idx=processed_frames)

        # Update processed frames count
        processed_frames = end_idx

        # Clear CUDA cache to avoid memory fragmentation
        torch.cuda.empty_cache()

    return processed_frames


def main():
    args, cfg = parse_config()
    accelerator = accelerate.Accelerator()
    device = accelerator.device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    # Use float16 for faster inference
    weight_dtype = torch.float16

    # Load models
    (
        vae,
        unet,
        scheduler,
        appearance_encoder,
        condition_encoder,
        empty_text_emb,
    ) = load_modules(cfg, device, weight_dtype)

    # Create pipeline
    pipeline = SignViPStaticPipeline(
        vae=vae,
        denoising_unet=unet,
        scheduler=scheduler,
        appearance_encoder=appearance_encoder,
        condition_encoder=condition_encoder,
        empty_text_emb=empty_text_emb,
    ).to(dtype=weight_dtype, device=device)

    # Create dataset
    dataset = SignCondDataset(
        output_dir=args.output_dir,
        frame_size=cfg.dataset.frame_size,
        frame_scale=cfg.dataset.frame_scale,
        roots=cfg.dataset.roots,
        sk_roots=cfg.dataset.sk_roots,
        hamer_roots=cfg.dataset.hamer_roots,
        meta_paths=cfg.dataset.meta_paths,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate frames
    with torch.no_grad():
        for i, batch in enumerate(
            tqdm(dataloader, total=len(dataloader), desc="Processing videos")
        ):
            if i >= args.max_videos:
                break

            try:
                # Get path and extract video name
                path = batch["path"][0]
                if path == "":
                    continue
                video_name = os.path.splitext(os.path.basename(path))[0]

                # Read reference frame from video using OpenCV
                ref_frames = []
                cap = cv2.VideoCapture(path)
                while True:
                    ret, img_cv2 = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
                    ref_frame = Image.fromarray(frame_rgb).resize(
                        (cfg.dataset.frame_size[1], cfg.dataset.frame_size[0])
                    )
                    ref_frames.append(ref_frame)
                    break  # Only need the first frame as reference

                cap.release()

                # Check if we got a reference frame
                if not ref_frames:
                    logger.error(f"Could not read reference frame from {path}")
                    continue

                first_frame = ref_frames[0]

                # Get skeleton and HAMER frames
                tgt_sk_frames = batch["tgt_sk_frames"].to(device, torch.float32)
                tgt_hamer_frames = batch["tgt_hamer_frames"].to(device, torch.float32)

                # Get dimensions
                b, c, f, h, w = tgt_sk_frames.shape

                logger.info(f"Video {video_name} has {f} frames")

                # Process video in batches to avoid OOM
                total_processed = process_video_in_batches(
                    pipeline,
                    first_frame,
                    tgt_sk_frames,
                    tgt_hamer_frames,
                    cfg,
                    args,
                    video_name,
                    args.output_dir,
                )

                logger.info(
                    f"Generated {total_processed} frames for video: {video_name}"
                )

            except Exception as e:
                logger.error(f"Error processing video {path}: {str(e)}")
                import traceback

                logger.error(traceback.format_exc())
                continue

            # Clear CUDA cache between videos
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
