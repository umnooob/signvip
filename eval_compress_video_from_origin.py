import argparse
import logging
import os
import pathlib
import random
import time
import traceback
import warnings
from collections import OrderedDict

import accelerate
import cv2
import numpy as np
import torch
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler
from einops import rearrange
from models.appearance_encoder import AppearanceEncoderModel
from models.condition_encoder import VQConditionEncoder
from models.unet import UNet3DConditionModel

# from fastdtw import fastdtw
from omegaconf import OmegaConf
from PIL import Image
from pipelines.pipeline_multicond import SignViPPipeline
from signdatasets import SignCondDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import save_video, seed_everything

warnings.filterwarnings("ignore")
logger = get_logger(__name__, log_level="INFO")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/deepo_data/signvip/workspace/vq_multicond_RWTH_compress/20250105-0235-FSQ1000/config.yaml",
    )

    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/deepo_data/signvipworkspace/eval_videos/compress_FSQ1000",
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
        **modules_cfg.get("condition_encoder_kwargs", {}),
    )
    if modules_cfg.condition_encoder:
        state_dict = torch.load(modules_cfg.condition_encoder, map_location="cpu")
        motion_module_state_dict = torch.load(
            modules_cfg.condition_encoder_motion, map_location="cpu"
        )
        state_dict.update(motion_module_state_dict)
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

    # assert len(missing) == 0
    logger.info(f"Loaded condition encoder from {modules_cfg.condition_encoder}.")
    condition_encoder.to(device, weight_dtype)

    if modules_cfg.mm:
        motion_module_state_dict = torch.load(modules_cfg.mm, map_location="cpu")
        # motion_module_state_dict = torch.load(motion_module, map_location="cpu")
        motion_module_state_dict = (
            motion_module_state_dict["state_dict"]
            if "state_dict" in motion_module_state_dict
            else motion_module_state_dict
        )
        try:
            # extra steps for self-trained models
            state_dict = OrderedDict()
            for key in motion_module_state_dict.keys():
                if key.startswith("module."):
                    _key = key.split("module.")[-1]
                    state_dict[_key] = motion_module_state_dict[key]
                else:
                    state_dict[key] = motion_module_state_dict[key]
            motion_module_state_dict = state_dict
            del state_dict
            logger.info(f"motion_module_state_dict: {motion_module_state_dict.keys()}")
            missing, unexpected = unet.load_state_dict(
                motion_module_state_dict, strict=False
            )
            assert len(unexpected) == 0
            logger.info(f"mm missing: {missing}")
        except:
            _tmp_ = OrderedDict()
            for key in motion_module_state_dict.keys():
                if "motion_modules" in key:
                    if key.startswith("unet."):
                        _key = key.split("unet.")[-1]
                        _tmp_[_key] = motion_module_state_dict[key]
                    else:
                        _tmp_[key] = motion_module_state_dict[key]
            missing, unexpected = unet.load_state_dict(_tmp_, strict=False)
            assert len(unexpected) == 0
            del _tmp_
        del motion_module_state_dict

    if modules_cfg.unet:
        unet.load_state_dict(torch.load(modules_cfg.unet))
        unet.to(device, weight_dtype)
        logger.info(f"Loaded full UNET from {modules_cfg.unet}.")

    empty_text_emb = torch.load(modules_cfg.empty_text_emb).to(device, weight_dtype)

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


def infer_one_video(cfg, args, pipeline, ref_frame, tgt_sk_frames, tgt_hamer_frames):
    b, c, f, h, w = tgt_sk_frames.shape
    # tgt_hamer_frames = torch.ones_like(tgt_hamer_frames) * -1

    video_tensor = pipeline(
        ref_image=ref_frame,
        sk_images=tgt_sk_frames,
        hamer_images=tgt_hamer_frames,
        width=cfg.dataset.frame_size[1],
        height=cfg.dataset.frame_size[0],
        video_length=f,
        num_inference_steps=cfg.validation_data.num_inference_steps,
        guidance_scale=args.guidance_scale,
        context_frames=24,
    )
    return video_tensor


def main():
    args, cfg = parse_config()
    accelerator = accelerate.Accelerator()
    device = accelerator.device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # if cfg.seed is not None:
    #     seed_everything(cfg.seed)

    weight_dtype = torch.float16

    (
        vae,
        unet,
        empty_text_emb,
        scheduler,
        appearance_encoder,
        condition_encoder,
    ) = load_modules(cfg, device, weight_dtype)

    pipeline = SignViPPipeline(
        vae=vae,
        denoising_unet=unet,
        scheduler=scheduler,
        empty_text_emb=empty_text_emb,
        appearance_encoder=appearance_encoder,
        condition_encoder=condition_encoder,
    ).to(dtype=weight_dtype, device=device)

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
    os.makedirs(args.output_dir, exist_ok=True)
    sub_batch_size = 256
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            try:
                path = batch["path"][0]
                if path == "":
                    continue
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
                    # ref_frame = transforms.ToTensor()(ref_frame)
                    ref_frames.append(ref_frame)
                    break

                cap.release()
                first_frame = ref_frames[0]

                out_path = os.path.join(args.output_dir, os.path.basename(path))
                if os.path.exists(out_path):
                    continue
                tgt_sk_frames = batch["tgt_sk_frames"].to(device, torch.float32)
                tgt_hamer_frames = batch["tgt_hamer_frames"].to(device, torch.float32)

                # 获取当前帧数并计算需要pad的数量
                b, c, f, h, w = tgt_sk_frames.shape

                pred_video = infer_one_video(
                    cfg,
                    args,
                    pipeline,
                    first_frame,
                    tgt_sk_frames,
                    tgt_hamer_frames,
                )
                save_video(
                    pred_video,
                    out_path,
                    device=device,
                    fps=24,
                )

            except Exception as e:
                logger.error(f"Error processing batch with path {path}: {str(e)}")
                logger.debug(traceback.format_exc())
                continue


if __name__ == "__main__":
    main()
