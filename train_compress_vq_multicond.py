import argparse
import datetime
import logging
import math
import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
import pathlib
import random
import shutil
import time
import warnings
from collections import OrderedDict

import accelerate
import matplotlib.pyplot as plt
import numpy as np
import torch

# torch.backends.cudnn.enabled = False
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
from diffusers.optimization import get_scheduler
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from models.appearance_encoder import AppearanceEncoderModel
from models.condition_encoder import VQConditionEncoder
from models.unet import UNet3DConditionModel
from pipelines.pipeline_multicond import SignViPPipeline
from signdatasets import SignLangVideoDataset
from utils import get_num_params, save_video, seed_everything

warnings.filterwarnings("ignore")
logger = get_logger(__name__, log_level="INFO")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Model(nn.Module):

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet3DConditionModel,
        appearance_encoder: UNet2DConditionModel,
        origin_condition_encoder,
    ):
        super().__init__()

        self.vae = vae
        self.unet = unet
        self.appearance_encoder = appearance_encoder
        self.origin_condition_encoder = origin_condition_encoder


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/stage_1.yaml")
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
    unet = unet.to(device, weight_dtype)
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

    modules_cfg.condition_encoder_kwargs.use_vq = False
    original_condition_encoder = VQConditionEncoder(
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
        original_condition_encoder.load_state_dict(state_dict)
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

    condition_encoder = condition_encoder.to(device, torch.float32)
    original_condition_encoder = original_condition_encoder.to(device, torch.float32)

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
            missing, unexpected = unet.load_state_dict(
                motion_module_state_dict, strict=False
            )
            assert len(unexpected) == 0
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

    # Load empty_text_emb
    empty_text_emb = torch.load(cfg.modules.empty_text_emb).to(device, weight_dtype)

    scheduler_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    noise_scheduler = DDPMScheduler.from_pretrained(
        cfg.modules.scheduler,
    )

    return (
        vae,
        unet,
        appearance_encoder,
        condition_encoder,
        original_condition_encoder,
        empty_text_emb,
        noise_scheduler,
    )


def count_params(model: Model):
    logger.info("***** Parameters Counting *****")

    # VAE
    params = list(model.vae.parameters())
    num_params, num_trainable_params = get_num_params(params)
    logger.info(f"  VAE: {num_params:.3f} M (trainable: {num_trainable_params:.3f}M)")

    # unet
    params = list(model.unet.parameters())
    num_params, num_trainable_params = get_num_params(params)
    logger.info(
        f"  U-Net (unet_2d + mm): {num_params:.3f} M (trainable: {num_trainable_params:.3f}M)"
    )

    # appearance_encoder
    params = list(model.appearance_encoder.parameters())
    num_params, num_trainable_params = get_num_params(params)
    logger.info(
        f"  Appearance Encoder: {num_params:.3f} M (trainable: {num_trainable_params:.3f}M)"
    )


def compute_snr(noisy_scheduler, timesteps):
    alphas_cumprod = noisy_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    snr = (alpha / sigma) ** 2
    return snr


def save_model(condition_encoder, cfg, model_path, is_main_process, weight_dtype):

    if is_main_process:
        pathlib.Path(os.path.join(model_path, "condition_encoder")).mkdir(
            parents=True, exist_ok=True
        )
        state_dict = condition_encoder.state_dict()
        # Then extract only motion module parameters
        vq_state_dict = {
            name: param for name, param in state_dict.items() if "vq" in name
        }
        torch.save(
            vq_state_dict,
            os.path.join(model_path, "condition_encoder/model.bin"),
        )
        logger.info(
            f"Saved condition encoder to {os.path.join(model_path, 'condition_encoder')}."
        )
        logger.info(f"vq_state_dict: {vq_state_dict.keys()}")


@torch.no_grad()
def log_valid(
    cfg,
    accelerator,
    valid_data,
    valid_dataloader,
    model,
    condition_encoder,
    empty_text_emb,
    device,
    weight_dtype,
    global_step,
    out_path,
):
    weight_dtype = torch.float16
    model.eval()
    condition_encoder.eval()
    condition_encoder.to(device, weight_dtype)

    scheduler = DDIMScheduler.from_pretrained(
        cfg.modules.scheduler,
    )

    pipeline = SignViPPipeline(
        vae=model.vae,
        denoising_unet=model.unet,
        scheduler=scheduler,
        appearance_encoder=model.appearance_encoder,
        condition_encoder=condition_encoder,
        empty_text_emb=empty_text_emb,
    ).to(dtype=weight_dtype, device=device)

    ref_frame = valid_data["ref_frame"].unsqueeze(0).to(device, weight_dtype)
    tgt_frames = (valid_data["tgt_frames"].unsqueeze(0) / 2 + 0.5).cpu().clamp(0, 1)
    tgt_sk_frames = valid_data["tgt_sk_frames"].unsqueeze(0).to(device, weight_dtype)
    tgt_hamer_frames = (
        valid_data["tgt_hamer_frames"].unsqueeze(0).to(device, weight_dtype)
    )

    video_tensor = pipeline(
        ref_image=ref_frame,
        sk_images=tgt_sk_frames,
        hamer_images=tgt_hamer_frames,
        width=cfg.dataset.frame_size[1],
        height=cfg.dataset.frame_size[0],
        video_length=36,
        num_inference_steps=cfg.validation_data.num_inference_steps,
        guidance_scale=cfg.validation_data.guidance_scale)

    video_tensor = torch.cat([tgt_frames, video_tensor], dim=4)
    # save as video
    if accelerator.is_main_process:
        save_video(video_tensor, out_path + f"/{global_step}.mp4", device, fps=24)

    del video_tensor
    torch.cuda.empty_cache()

    valid_loss = []
    for step, batch in tqdm(
        enumerate(valid_dataloader), total=len(valid_dataloader), desc="Validating"
    ):
        tgt_sk_frames = batch["tgt_sk_frames"].to(
            device, weight_dtype
        )  # [bs, c, f, h, w]
        tgt_hamer_frames = batch["tgt_hamer_frames"].to(
            device, weight_dtype
        )  # [bs, c, f, h, w]
        tgt_sk_frames = rearrange(tgt_sk_frames, "b c f h w -> (b f) c h w")
        tgt_hamer_frames = rearrange(tgt_hamer_frames, "b c f h w -> (b f) c h w")

        # Replace with this - compute loss per-process first, then aggregate only the loss values
        with torch.cuda.amp.autocast(enabled=True):
            sk_cond, perplexity, vq_loss = condition_encoder.encode(
                tgt_sk_frames, tgt_hamer_frames, return_indices=False
            )

            origin_cond = model.origin_condition_encoder.encode(
                tgt_sk_frames, tgt_hamer_frames
            )

            # Calculate loss locally before gathering
            distill_loss = func.mse_loss(sk_cond.float(), origin_cond.float())

            # Free memory as soon as possible
            del sk_cond, origin_cond
            torch.cuda.empty_cache()

            # Only gather scalar loss values instead of full tensors
            distill_loss = accelerator.gather(
                distill_loss.repeat(tgt_sk_frames.size(0))
            ).mean()
        valid_loss.append(distill_loss.item())
        if step>3:
            break

    valid_loss = torch.tensor(valid_loss).mean()
    logger.info(f"Valid loss: {valid_loss:.4f}")

    condition_encoder.to(device, torch.float32)
    return valid_loss


def main():
    cfg = parse_config()
    experiment_index = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    workspace_dir = f"./workspace/{cfg.exp_name}/{experiment_index}"

    if cfg.resume_from_checkpoint:
        workspace_dir = cfg.resume_from_checkpoint

    pathlib.Path(workspace_dir).mkdir(parents=True, exist_ok=True)

    config = accelerate.utils.ProjectConfiguration(
        project_dir=".", logging_dir=workspace_dir + "/log"
    )
    # save config under workspace_dir
    with open(os.path.join(workspace_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    accelerator = accelerate.Accelerator(
        log_with=cfg.report_to,
        project_config=config,
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
    )
    accelerator.init_trackers("signvip")
    device = accelerator.device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if cfg.seed is not None:
        seed_everything(cfg.seed)

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

    (
        vae,
        unet,
        appearance_encoder,
        condition_encoder,
        origin_condition_encoder,
        empty_text_emb,
        noisy_scheduler,
    ) = load_modules(cfg, device, torch.float16)

    vae.requires_grad_(cfg.grad.vae)
    unet.requires_grad_(cfg.grad.unet_2d)
    appearance_encoder.requires_grad_(cfg.grad.appearance_encoder)
    condition_encoder.requires_grad_(cfg.grad.condition_encoder)
    origin_condition_encoder.requires_grad_(False)
    origin_condition_encoder.eval()

    for name, param in condition_encoder.named_parameters():
        for trainable_module_name in cfg.trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    model = Model(
        vae,
        unet,
        appearance_encoder,
        origin_condition_encoder,
    ).to(device, torch.float16)

    condition_encoder = condition_encoder.to(device, torch.float32)

    if cfg.solver.optimizer.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam by running `pip install bitsandbytes`."
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(
        filter(lambda p: p.requires_grad, condition_encoder.parameters())
    )
    optimizer = optimizer_cls(
        trainable_params,
        lr=cfg.solver.optimizer.learning_rate,
        betas=(cfg.solver.optimizer.adam_beta1, cfg.solver.optimizer.adam_beta2),
        weight_decay=cfg.solver.optimizer.adam_weight_decay,
        eps=cfg.solver.optimizer.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_scheduler.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps
        * accelerator.num_processes,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps
        * accelerator.num_processes,
        num_cycles=cfg.solver.lr_scheduler.num_cycles,
    )

    dataset = SignLangVideoDataset(
        frame_size=cfg.dataset.frame_size,
        frame_scale=cfg.dataset.frame_scale,
        frame_ratio=cfg.dataset.frame_ratio,
        roots=cfg.dataset.roots,
        sk_roots=cfg.dataset.sk_roots,
        hamer_roots=cfg.dataset.hamer_roots,
        meta_paths=cfg.dataset.meta_paths,
        sample_rate=cfg.dataset.sample_rate,
        num_frames=cfg.dataset.num_frames,
        ref_margin=cfg.dataset.ref_margin,
        uncond_ratio=cfg.dataset.uncond_ratio,
        mask_ratio=cfg.dataset.mask_ratio,
        mask_thershold=cfg.dataset.mask_thershold,
        skip_ratio=cfg.dataset.skip_ratio,
        sk_mask_ratio=cfg.dataset.sk_mask_ratio,
        hamer_mask_ratio=cfg.dataset.hamer_mask_ratio,
        both_mask_ratio=cfg.dataset.both_mask_ratio,
    )
    valid_dataset = SignLangVideoDataset(
        frame_size=cfg.dataset.frame_size,
        frame_scale=cfg.dataset.frame_scale,
        frame_ratio=cfg.dataset.frame_ratio,
        roots=cfg.dataset.roots[0],
        sk_roots=cfg.dataset.sk_roots[0],
        hamer_roots=cfg.dataset.hamer_roots[0],
        meta_paths=cfg.validation_data.meta_paths,
        sample_rate=cfg.dataset.sample_rate,
        num_frames=36,
        ref_margin=cfg.dataset.ref_margin,
        uncond_ratio=cfg.dataset.uncond_ratio,
        mask_ratio=cfg.dataset.mask_ratio,
        mask_thershold=cfg.dataset.mask_thershold,
        skip_ratio=cfg.dataset.skip_ratio,
        sk_mask_ratio=cfg.dataset.sk_mask_ratio,
        hamer_mask_ratio=cfg.dataset.hamer_mask_ratio,
        both_mask_ratio=cfg.dataset.both_mask_ratio,
    )
    valid_data = valid_dataset[0]
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=0,
    )
    count_params(model)

    (condition_encoder, optimizer, dataloader, valid_dataloader, lr_scheduler) = (
        accelerator.prepare(
            condition_encoder, optimizer, dataloader, valid_dataloader, lr_scheduler
        )
    )

    weight_dtype = torch.float16

    num_steps_per_epoch = math.ceil(
        len(dataloader) / cfg.solver.gradient_accumulation_steps
    )
    num_epochs = math.ceil(cfg.solver.max_train_steps / num_steps_per_epoch)

    total_batch_size = (
        cfg.dataloader.batch_size
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.dataloader.batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")

    global_step = 0
    start_epoch = 0

    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = workspace_dir

        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path), load_module_strict=False)
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        start_epoch = global_step // num_steps_per_epoch

    accelerator.wait_for_everyone()

    valid_loss = log_valid(
        cfg,
        accelerator,
        valid_data,
        valid_dataloader,
        model,
        condition_encoder,
        empty_text_emb,
        device,
        weight_dtype,
        global_step,
        workspace_dir + "/log",
    )
    accelerator.log({"valid_loss": valid_loss}, step=global_step)

    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description(f"Exp: {cfg.exp_name}")

    train_loss = 0.0
    train_vq_loss = 0.0
    train_distill_loss = 0.0
    train_perplexity = 0.0
    best_train_distill_loss = valid_loss
    for epoch in range(start_epoch, num_epochs):
        t_data_start = time.time()
        for step, batch in enumerate(dataloader):
            t_data = time.time() - t_data_start
            condition_encoder.train()
            condition_encoder.to(weight_dtype)
            with accelerator.accumulate(model):
                ref_frame = batch["ref_frame"].to(device, weight_dtype)  # [bs, c, h, w]
                tgt_frames = batch["tgt_frames"].to(
                    device, weight_dtype
                )  # [bs, c, f, h, w]
                tgt_sk_frames = batch["tgt_sk_frames"].to(
                    device, weight_dtype
                )  # [bs, c, f, h, w]
                tgt_hamer_frames = batch["tgt_hamer_frames"].to(
                    device, weight_dtype
                )  # [bs, c, f, h, w]

                tgt_sk_frames = rearrange(tgt_sk_frames, "b c f h w -> (b f) c h w")
                tgt_hamer_frames = rearrange(
                    tgt_hamer_frames, "b c f h w -> (b f) c h w"
                )
                sk_cond, perplexity, vq_loss = condition_encoder.encode(
                    tgt_sk_frames, tgt_hamer_frames, return_indices=False
                )
                origin_cond = model.origin_condition_encoder.encode(
                    tgt_sk_frames, tgt_hamer_frames
                )

                distill_loss = func.mse_loss(sk_cond.float(), origin_cond.float())

                total_loss = cfg.distill_loss_weight * distill_loss + vq_loss

                avg_loss = accelerator.gather(
                    total_loss.repeat(cfg.dataloader.batch_size)
                ).mean()
                avg_vq_loss = accelerator.gather(
                    vq_loss.repeat(cfg.dataloader.batch_size)
                ).mean()
                avg_distill_loss = accelerator.gather(
                    distill_loss.repeat(cfg.dataloader.batch_size)
                ).mean()
                avg_perplexity = accelerator.gather(
                    perplexity.repeat(cfg.dataloader.batch_size)
                ).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps
                train_vq_loss += (
                    avg_vq_loss.item() / cfg.solver.gradient_accumulation_steps
                )
                train_distill_loss += (
                    avg_distill_loss.item() / cfg.solver.gradient_accumulation_steps
                )
                train_perplexity += (
                    avg_perplexity.item() / cfg.solver.gradient_accumulation_steps
                )
                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params, cfg.solver.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Log metrics
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "train_vq_loss": train_vq_loss,
                        "train_distill_loss": train_distill_loss,
                        "train_perplexity": train_perplexity,
                    },
                    step=global_step,
                )

                train_loss = 0.0
                train_vq_loss = 0.0
                train_distill_loss = 0.0
                train_perplexity = 0.0
                if global_step % cfg.valid_steps == 0 and global_step > 0:
                    valid_loss = log_valid(
                        cfg,
                        accelerator,
                        valid_data,
                        valid_dataloader,
                        model,
                        condition_encoder,
                        empty_text_emb,
                        device,
                        weight_dtype,
                        global_step,
                        workspace_dir + "/log",
                    )
                    accelerator.log({"valid_loss": valid_loss}, step=global_step)
                    if (
                        accelerator.is_main_process
                        and best_train_distill_loss > valid_loss
                    ):
                        best_train_distill_loss = valid_loss
                        model_path = os.path.join(workspace_dir, f"best")
                        save_model(
                            condition_encoder,
                            cfg,
                            model_path,
                            accelerator.is_main_process,
                            weight_dtype,
                        )

            logs = {
                "step_loss": total_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "td": f"{t_data:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break

            if global_step % cfg.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    # Save DiT checkpoint:
                    if cfg.max_ckpt is not None:
                        checkpoints = os.listdir(workspace_dir)
                        checkpoints = [
                            d for d in checkpoints if d.startswith("checkpoint")
                        ]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1])
                        )

                        # before we save the new checkpoint, we need to have at _most_ `max_ckpt - 1` checkpoints
                        if len(checkpoints) >= cfg.max_ckpt:
                            num_to_remove = len(checkpoints) - cfg.max_ckpt + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}"
                            )

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(
                                    workspace_dir, removing_checkpoint
                                )
                                shutil.rmtree(removing_checkpoint)

                    if cfg.max_model is not None:
                        models = os.listdir(workspace_dir)
                        models = [d for d in models if d.startswith("model")]
                        models = sorted(models, key=lambda x: int(x.split("-")[1]))

                        # before we save the new model, we need to have at most `max_model - 1` models
                        if len(models) >= cfg.max_model:
                            num_to_remove = len(models) - cfg.max_model + 1
                            removing_models = models[0:num_to_remove]

                            logger.info(
                                f"{len(models)} models already exist, removing {len(removing_models)} models"
                            )
                            logger.info(
                                f"removing models: {', '.join(removing_models)}"
                            )

                            for removing_model in removing_models:
                                removing_model_path = os.path.join(
                                    workspace_dir, removing_model
                                )
                                shutil.rmtree(removing_model_path)

                ckpt_path = os.path.join(workspace_dir, f"checkpoint-{global_step}")
                model_path = os.path.join(workspace_dir, f"model-{global_step}")

                accelerator.save_state(ckpt_path, exclude_frozen_parameters=True)
                save_model(
                    condition_encoder,
                    cfg,
                    model_path,
                    accelerator.is_main_process,
                    weight_dtype,
                )

    ckpt_path = os.path.join(workspace_dir, f"checkpoint-{global_step}")
    model_path = os.path.join(workspace_dir, f"model-{global_step}")

    accelerator.save_state(ckpt_path, exclude_frozen_parameters=True)
    save_model(
        condition_encoder, cfg, model_path, accelerator.is_main_process, weight_dtype
    )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
