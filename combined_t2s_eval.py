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
from models.appearance_encoder import AppearanceEncoderModel
from models.condition_encoder import VQConditionEncoder
from models.multihead_t2vqpgpt import Text2VQPoseGPT
from models.unet import UNet3DConditionModel
from omegaconf import OmegaConf
from PIL import Image
from pipelines.pipeline_multicond import SignViPPipeline
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from signdatasets import VQSignTextDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from utils import save_video, seed_everything

warnings.filterwarnings("ignore")
logger = get_logger(__name__, log_level="INFO")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--t2s_config",
        type=str,
        help="Config file for text-to-pose model (GPT)",
        default="/deepo_data/signvip_v2/configs/gpt/eval_multihead_t2vqpgpt_RWTH.yaml",
    )
    parser.add_argument(
        "--vq_config",
        type=str,
        help="Config file for pose-to-video model (VQ)",
        default="/deepo_data/signvip_v2/workspace/vq_multicond_RWTH_compress/20250518-1840/config_test.yaml",
    )
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument(
        "--reference_image_path",
        type=str,
        default=None,
        help="Path to reference image for appearance",
    )
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--text_input", type=str, default=None, help="Text input for direct inference"
    )
    parser.add_argument(
        "--video_base_path",
        type=str,
        default="/deepo_data/signvipworkspace/datasets/RWTH-T/test_processed_videos",
    )
    args = parser.parse_args()

    t2s_cfg = OmegaConf.load(args.t2s_config)
    vq_cfg = OmegaConf.load(args.vq_config)
    vq_cfg.exp_name = pathlib.Path(args.vq_config).stem

    return args, t2s_cfg, vq_cfg


def load_t2s_model(cfg, device, weight_dtype):
    # Load text encoder
    text_model = SentenceTransformer(
        cfg.modules.text_model,
        cache_folder="/deepo_data/signvipworkspace/models",
        local_files_only=True,
    )
    text_model = text_model.to(device, weight_dtype)
    text_model.eval()
    tokenizer = text_model.tokenizer

    # Load T2VQPGPT model
    t2pgpt = Text2VQPoseGPT(
        num_vq=cfg.modules.codebook_size + 2,
        embed_dim=cfg.modules.embed_dim,
        clip_dim=cfg.modules.clip_dim,
        block_size=cfg.modules.block_size,
        num_layers=cfg.modules.num_layers,
        n_head=cfg.modules.n_head,
        drop_out_rate=cfg.modules.drop_out_rate,
        fc_rate=cfg.modules.fc_rate,
        pose_size=cfg.eval.pose_size,
        head_layers=cfg.modules.head_layers,
    )

    if cfg.modules.ckpt:
        state_dict = torch.load(cfg.modules.ckpt)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {
                k[7:]: v for k, v in state_dict.items() if k.startswith("module.")
            }
        t2pgpt.load_state_dict(state_dict)
        logger.info(f"Loaded T2VQPGPT checkpoint from {cfg.modules.ckpt}")

    t2pgpt = t2pgpt.to(device, weight_dtype)
    t2pgpt.eval()

    return tokenizer, text_model, t2pgpt


def load_vq_modules(cfg, device, weight_dtype):
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
        logger.info(f"Missing keys: {missing}")
        logger.info(f"Unexpected keys: {unexpected}")

    condition_encoder = condition_encoder.to(device, weight_dtype)

    if modules_cfg.unet:
        unet.load_state_dict(torch.load(modules_cfg.unet))
        unet.to(device, weight_dtype)
        logger.info(f"Loaded full UNET from {modules_cfg.unet}.")

    scheduler = DDIMScheduler.from_pretrained(
        modules_cfg.scheduler,
    )

    # Load empty_text_emb
    empty_text_emb = torch.load(cfg.modules.empty_text_emb).to(device, weight_dtype)
    return (
        vae,
        unet,
        empty_text_emb,
        scheduler,
        appearance_encoder,
        condition_encoder,
    )


def create_dataset_dataloader(t2s_cfg, tokenizer):
    dataloaders = []

    for metapath in t2s_cfg.dataset.meta_paths:
        # Create dataset and dataloader
        dataset = VQSignTextDataset(
            frame_size=t2s_cfg.dataset.frame_size,
            frame_scale=t2s_cfg.dataset.frame_scale,
            sample_rate=t2s_cfg.dataset.sample_rate,
            tokenizer=tokenizer,
            max_pose_len=t2s_cfg.eval.max_pose_len,
            pose_size=t2s_cfg.eval.pose_size,
            codebook_size=t2s_cfg.modules.codebook_size,
            roots=t2s_cfg.dataset.roots,
            pose_roots=t2s_cfg.dataset.pose_roots,
            meta_paths=[metapath],
        )

        dataloader = DataLoader(
            dataset,
            batch_size=t2s_cfg.eval.batch_size,
            shuffle=False,
            num_workers=t2s_cfg.eval.num_workers,
        )
        dataloaders.append(dataloader)

    return dataloaders


def generate_pose_from_text(
    text,
    tokenizer,
    text_model,
    t2s_model,
    max_pose_len,
    pose_size,
    device,
    weight_dtype,
):
    # Process text input
    text_inputs = tokenizer(
        text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    text_attn_mask = text_inputs.attention_mask.to(device)

    # Generate text embeddings
    text_embeds = text_model(
        {"input_ids": text_input_ids, "attention_mask": text_attn_mask}
    )
    text_embeds = text_embeds["sentence_embedding"].to(weight_dtype)

    # Generate pose tokens
    with torch.no_grad():
        pred_idx = t2s_model.sample(text_embeds, max_pose_len + 1)

    return pred_idx


def infer_video_from_pose(
    vq_cfg, args, pipeline, condition_encoder, ref_frame, pose_latents
):
    pose_latents = condition_encoder.preprocess(pose_latents)
    video_tensor = pipeline(
        condition_encoder=condition_encoder,
        sk_images=None,
        hamer_images=None,
        ref_image=ref_frame,
        pose_latent=pose_latents,
        width=vq_cfg.dataset.frame_size[1],
        height=vq_cfg.dataset.frame_size[0],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        context_batch_size=1,
        context_frames=24,
    )
    return video_tensor


def get_reference_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, img_cv2 = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        ref_frame = Image.fromarray(frame_rgb)
        cap.release()
        return ref_frame
    cap.release()
    return None


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args, t2s_cfg, vq_cfg = parse_config()
    start_time = time.time()

    # Initialize accelerator
    accelerator = accelerate.Accelerator(
        mixed_precision=(
            t2s_cfg.weight_dtype if t2s_cfg.weight_dtype != "fp32" else "no"
        ),
    )
    device = accelerator.device

    # Set weight dtype
    if t2s_cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif t2s_cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    elif t2s_cfg.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float16  # Default to fp16

    # Set random seed if specified
    if vq_cfg.seed is not None:
        seed_everything(vq_cfg.seed)

    # Load models
    t2s_tokenizer, text_model, t2s_model = load_t2s_model(t2s_cfg, device, weight_dtype)
    (
        vae,
        unet,
        empty_text_emb,
        scheduler,
        appearance_encoder,
        condition_encoder,
    ) = load_vq_modules(vq_cfg, device, weight_dtype)

    print(
        "T2S Model (Text2VQPoseGPT) 参数量：{:.2f} M".format(
            count_parameters(t2s_model) / 1_000_000
        )
    )
    print("VAE 参数量：{:.2f} M".format(count_parameters(vae) / 1_000_000))
    print("UNet 参数量：{:.2f} M".format(count_parameters(unet) / 1_000_000))
    print(
        "AppearanceEncoder 参数量：{:.2f} M".format(
            count_parameters(appearance_encoder) / 1_000_000
        )
    )
    print(
        "ConditionEncoder 参数量：{:.2f} M".format(
            count_parameters(condition_encoder) / 1_000_000
        )
    )
    print(
        "ConditionEncoder 无用参数量：{:.2f} M".format(
            (
                count_parameters(condition_encoder.backbone)
                + count_parameters(condition_encoder.gate_module)
                + count_parameters(condition_encoder.vq.downsample_encoder)
            )
            / 1_000_000
        )
    )
    total_params = (
        count_parameters(t2s_model)
        + count_parameters(vae)
        + count_parameters(unet)
        + count_parameters(appearance_encoder)
        + count_parameters(condition_encoder)
        - count_parameters(condition_encoder.backbone)
        - count_parameters(condition_encoder.gate_module)
        - count_parameters(condition_encoder.vq.downsample_encoder)
    )
    print("推理总参数量：{:.2f} M".format(total_params / 1_000_000))
    # Create pipeline for video generation
    pipeline = SignViPPipeline(
        vae=vae,
        denoising_unet=unet,
        scheduler=scheduler,
        empty_text_emb=empty_text_emb,
        appearance_encoder=appearance_encoder,
    ).to(dtype=weight_dtype, device=device)

    # Make output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Direct text input mode
    if args.text_input:
        logger.info(f"Generating sign video for text: {args.text_input}")

        # Step 1: Generate pose tokens from text
        pred_pose_latent = generate_pose_from_text(
            args.text_input,
            t2s_tokenizer,
            text_model,
            t2s_model,
            t2s_cfg.eval.max_pose_len,
            t2s_cfg.eval.pose_size,
            device,
            weight_dtype,
        )

        # Find EOS token (codebook_size)
        eos_idx = t2s_cfg.modules.codebook_size
        eos_positions = (pred_pose_latent[0] == eos_idx).nonzero()
        pose_size = t2s_cfg.eval.pose_size

        if len(eos_positions) > 0:
            # Calculate nearest eos_idx in multiples of pose_size
            truncate_pos = (eos_positions[0].item() // pose_size) * pose_size
            pred_pose_latent = (
                pred_pose_latent[0, :truncate_pos].reshape(-1, pose_size).to(torch.long)
            )
        else:
            pred_pose_latent = pred_pose_latent[0].reshape(-1, pose_size).to(torch.long)

        # Step 2: Get reference image or use provided one
        if args.reference_image_path:
            ref_frame = (
                Image.open(args.reference_image_path)
                .convert("RGB")
                .resize((vq_cfg.dataset.frame_size[1], vq_cfg.dataset.frame_size[0]))
            )
        else:
            # Use first frame from a sample video if available
            sample_videos = os.listdir(args.video_base_path)
            if sample_videos:
                sample_video_path = os.path.join(args.video_base_path, sample_videos[0])
                ref_frame = get_reference_frame(sample_video_path)
                if ref_frame is None:
                    raise ValueError(
                        "Could not get reference frame from sample video. Please provide a reference image."
                    )
                ref_frame = ref_frame.resize(
                    (vq_cfg.dataset.frame_size[1], vq_cfg.dataset.frame_size[0])
                )
            else:
                raise ValueError(
                    "No reference image provided and no sample videos found."
                )

        # Step 3: Generate video from pose tokens
        output_video = infer_video_from_pose(
            vq_cfg, args, pipeline, condition_encoder, ref_frame, pred_pose_latent
        )

        # Save output video
        text_slug = args.text_input.replace(" ", "_")[
            :50
        ]  # Create a filename from text
        output_path = os.path.join(args.output_dir, f"{text_slug}.mp4")
        save_video(output_video, output_path, device=device, fps=24)
        logger.info(f"Video saved to {output_path}")

    # Dataset evaluation mode
    else:
        # Load datasets for batch processing
        dataloaders = create_dataset_dataloader(t2s_cfg, t2s_tokenizer)
        t2s_model, *dataloaders = accelerator.prepare(t2s_model, *dataloaders)

        # Process each dataset
        for dataloader_idx, dataloader in enumerate(dataloaders):
            output_dir = (
                t2s_cfg.eval.output_dirs[dataloader_idx]
                if dataloader_idx < len(t2s_cfg.eval.output_dirs)
                else os.path.join(args.output_dir, f"dataset_{dataloader_idx}")
            )
            os.makedirs(output_dir, exist_ok=True)

            for step, batch in tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                desc=f"Processing dataset {dataloader_idx}",
            ):
                pose_latents = (
                    batch["pose_latents"].to(device)
                    if "pose_latents" in batch
                    else None
                )
                pose_len = batch["pose_len"].to(device) if "pose_len" in batch else None
                text_input_ids = batch["text_input_ids"].to(device)
                text_attn_mask = batch["text_attn_mask"].to(device)
                paths = batch["path"]

                # Step 1: Generate text embeddings
                text_embeds = text_model(
                    {"input_ids": text_input_ids, "attention_mask": text_attn_mask}
                )
                text_embeds = text_embeds["sentence_embedding"].to(weight_dtype)

                # Step 2: Generate pose tokens from text
                with torch.no_grad():
                    pred_idx = t2s_model.sample(
                        text_embeds, t2s_cfg.eval.max_pose_len + 1
                    )

                # Gather results from all processes
                pred_idx, text_input_ids, text_attn_mask = (
                    accelerator.gather_for_metrics(
                        (pred_idx, text_input_ids, text_attn_mask)
                    )
                )
                if pose_latents is not None and pose_len is not None:
                    pose_latents, pose_len = accelerator.gather_for_metrics(
                        (pose_latents, pose_len)
                    )
                paths = accelerator.gather_for_metrics(paths, use_gather_object=True)

                # Process each item in the batch
                for i in range(len(paths)):
                    sample_path = paths[i]
                    pred = pred_idx[i]

                    # Find EOS token
                    eos_idx = t2s_cfg.modules.codebook_size
                    eos_positions = (pred == eos_idx).nonzero()
                    pose_size = t2s_cfg.eval.pose_size

                    if len(eos_positions) > 0:
                        # Calculate nearest eos_idx in multiples of pose_size
                        truncate_pos = (
                            eos_positions[0].item() // pose_size
                        ) * pose_size
                        pred = pred[:truncate_pos].reshape(-1, pose_size).to(torch.long)
                    else:
                        pred = pred.reshape(-1, pose_size).to(torch.long)

                    # Save intermediate pose tokens
                    token_output_path = os.path.join(
                        output_dir, sample_path.replace(".mp4", ".pkl")
                    )
                    os.makedirs(os.path.dirname(token_output_path), exist_ok=True)

                    if accelerator.is_main_process:
                        with open(token_output_path, "wb") as f:
                            if pose_latents is not None and pose_len is not None:
                                pickle.dump(
                                    {
                                        "gt": pose_latents[i].cpu().numpy(),
                                        "pred": pred.cpu().numpy(),
                                        "len": (
                                            pose_len[i].cpu().numpy()
                                            if i < len(pose_len)
                                            else None
                                        ),
                                    },
                                    f,
                                )
                            else:
                                pickle.dump(
                                    {
                                        "pred": pred.cpu().numpy(),
                                    },
                                    f,
                                )

                    # Step 3: Get reference frame from original video
                    origin_path = os.path.join(args.video_base_path, sample_path)
                    if os.path.exists(origin_path):
                        ref_frame = get_reference_frame(origin_path)
                        if ref_frame is None:
                            if args.reference_image_path:
                                ref_frame = Image.open(
                                    args.reference_image_path
                                ).convert("RGB")
                            else:
                                logger.warning(
                                    f"Could not get reference frame for {origin_path}, skipping video generation"
                                )
                                continue
                    elif args.reference_image_path:

                        ref_frame = Image.open(args.reference_image_path).convert("RGB")
                    else:
                        logger.warning(
                            f"Could not get reference frame for {origin_path}, skipping video generation"
                        )
                        continue

                    ref_frame = ref_frame.resize(
                        (vq_cfg.dataset.frame_size[1], vq_cfg.dataset.frame_size[0])
                    )

                    # Step 4: Generate video from pose tokens
                    video_output_path = os.path.join(output_dir, sample_path)
                    os.makedirs(os.path.dirname(video_output_path), exist_ok=True)

                    if (
                        not os.path.exists(video_output_path)
                        and accelerator.is_main_process
                    ):
                        pred_video = infer_video_from_pose(
                            vq_cfg, args, pipeline, condition_encoder, ref_frame, pred
                        )
                        save_video(pred_video, video_output_path, device=device, fps=24)
                        logger.info(f"Video saved to {video_output_path}")

    logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
