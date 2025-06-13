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
from collections import OrderedDict

import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics.calculate_fvd import calculate_fvd as calc_fvd
from models.appearance_encoder import AppearanceEncoderModel
from models.condition_encoder import ConditionEncoder, VQConditionEncoder
from models.mutual_self_attention import ReferenceAttentionControl
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
        condition_encoder: ConditionEncoder,
        batch_size: int,
    ):
        super().__init__()

        self.vae = vae
        self.unet = unet
        self.appearance_encoder = appearance_encoder
        self.reference_control_writer = ReferenceAttentionControl(
            appearance_encoder,
            do_classifier_free_guidance=False,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        self.reference_control_reader = ReferenceAttentionControl(
            self.unet,
            do_classifier_free_guidance=False,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        self.condition_encoder = condition_encoder


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/stage_2.yaml")
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
    )
    if modules_cfg.condition_encoder:
        state_dict = torch.load(modules_cfg.condition_encoder)
        if modules_cfg.get("condition_encoder_motion", None):
            motion_module_state_dict = torch.load(
                modules_cfg.condition_encoder_motion, map_location="cpu"
            )
            state_dict.update(motion_module_state_dict)
        missing, unexpected = condition_encoder.load_state_dict(
            state_dict, strict=False
        )
        assert len(unexpected) == 0

    logger.info(f"Loaded condition encoder from {modules_cfg.condition_encoder}.")
    condition_encoder.to(device, weight_dtype)

    if modules_cfg.mm:
        motion_module_state_dict = torch.load(modules_cfg.mm, map_location="cpu")
        motion_module_state_dict = (
            motion_module_state_dict["state_dict"]
            if "state_dict" in motion_module_state_dict
            else motion_module_state_dict
        )
        try:
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
            logger.info(unexpected)
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
        **scheduler_kwargs,
    )

    return (
        vae,
        unet,
        appearance_encoder,
        condition_encoder,
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

    # condition_encoder
    params = list(model.condition_encoder.parameters())
    num_params, num_trainable_params = get_num_params(params)
    logger.info(
        f"  Condition Encoder: {num_params:.3f} M (trainable: {num_trainable_params:.3f}M)"
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


def save_model(model, cfg, model_path, is_main_process, weight_dtype):
    if is_main_process:
        pathlib.Path(os.path.join(model_path, "condition_encoder")).mkdir(
            parents=True, exist_ok=True
        )
        # Extract and save motion modules from condition encoder
        if hasattr(model.condition_encoder, "motion_modules"):
            # First get the full state dict
            condition_encoder_state_dict = model.condition_encoder.state_dict()

            # Then extract only motion module parameters
            motion_state_dict = {
                name: param
                for name, param in condition_encoder_state_dict.items()
                if "motion_module" in name
            }

            if motion_state_dict:
                torch.save(
                    motion_state_dict,
                    os.path.join(model_path, "condition_encoder/model.bin"),
                )
                logger.info(f"save motion {list(motion_state_dict.keys())}")
                logger.info(
                    f"Saved condition encoder motion modules to {os.path.join(model_path, 'condition_encoder')}."
                )

        pathlib.Path(os.path.join(model_path, "unet")).mkdir(
            parents=True, exist_ok=True
        )
        # First get the full state dict
        unet_state_dict = model.unet.state_dict()

        # Then extract only motion module parameters
        motion_state_dict = {
            name: param
            for name, param in unet_state_dict.items()
            if "motion_module" in name
        }

        if motion_state_dict:
            torch.save(motion_state_dict, os.path.join(model_path, "unet/model.bin"))
            logger.info(f"save motion {list(motion_state_dict.keys())}")
            logger.info(f"Saved unet to {os.path.join(model_path, 'unet')}.")


@torch.no_grad()
def log_valid(
    cfg,
    accelerator,
    valid_loader,
    model,
    empty_text_emb,
    device,
    weight_dtype,
    global_step,
    out_path,
):
    model.eval()
    scheduler_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    scheduler = DDIMScheduler.from_pretrained(
        cfg.modules.scheduler,
        **scheduler_kwargs,
    )

    pipeline = SignViPPipeline(
        vae=model.vae,
        denoising_unet=model.unet,
        scheduler=scheduler,
        appearance_encoder=model.appearance_encoder,
        condition_encoder=model.condition_encoder,
        empty_text_emb=empty_text_emb,
    ).to(device=device)

    # 用于存储所有生成和真实视频
    all_generated = []
    all_real = []

    # 遍历验证集
    for batch in tqdm(valid_loader, total=len(valid_loader), desc="Validating"):
        ref_frame = batch["ref_frame"].to(device)
        tgt_frames = batch["tgt_frames"].to(device)  # 保持原始范围 [-1, 1]
        tgt_frames = (tgt_frames / 2 + 0.5).clamp(0, 1)
        tgt_sk_frames = batch["tgt_sk_frames"].to(device)
        tgt_hamer_frames = batch["tgt_hamer_frames"].to(device)

        # 生成视频
        video_tensor = pipeline(
            ref_image=ref_frame,
            sk_images=tgt_sk_frames,
            hamer_images=tgt_hamer_frames,
            width=cfg.dataset.frame_size[1],
            height=cfg.dataset.frame_size[0],
            video_length=32,
            num_inference_steps=cfg.validation_data.num_inference_steps,
            guidance_scale=cfg.validation_data.guidance_scale,
            context_frames=cfg.dataset.num_frames,
            
        )
        video_tensor = video_tensor.contiguous().to(device)
        tgt_frames = tgt_frames.contiguous().to(device)
        video_tensor, tgt_frames = accelerator.gather_for_metrics(
            (video_tensor, tgt_frames)
        )
        # 收集生成和真实视频
        all_generated.append(video_tensor)
        all_real.append(tgt_frames)

        # 可以只处理部分批次以节省时间
        if len(all_generated) >= cfg.validation_data.get("max_eval_batches", 10):
            break

    # 合并所有批次并归一化到 [0,1] 范围
    all_generated = torch.cat(all_generated, dim=0)  # [N, C, T, H, W]
    all_real = torch.cat(all_real, dim=0)  # [N, C, T, H, W]

    # transpose for FVD calculation
    fvd_all_generated = all_generated.transpose(1, 2)
    fvd_all_real = all_real.transpose(1, 2)
    fvd_score = 0
    if accelerator.is_main_process:
        # 计算 FVD
        fvd_result = calc_fvd(
            fvd_all_real,  # 真实视频 [N, T, C, H, W]
            fvd_all_generated,  # 生成视频 [N, T, C, H, W]
            device=device,
            method="styleganv",  # 使用 styleganv 方法
            only_final=True,  # 只计算最终的 FVD 分数
        )

        fvd_score = np.mean(fvd_result["value"])  # 获取 FVD 分数

        # 记录 FVD 分数
        accelerator.log(
            {
                "validation/fvd": fvd_score,
            },
            step=global_step,
        )

        # 保存示例视频(使用第一个批次)
        # 转换到 [0, 1] 范围用于保存
        real_video = all_real[0:1]
        generated_video = all_generated[0:1]
        example_video = torch.cat([real_video, generated_video], dim=4)
        save_video(example_video, out_path + f"/{global_step}.mp4", device, fps=24)

    return fvd_score  # 返回 FVD 分数用于模型选择


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
    elif cfg.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training!"
        )

    (
        vae,
        unet,
        appearance_encoder,
        condition_encoder,
        empty_text_emb,
        noisy_scheduler,
    ) = load_modules(cfg, device, weight_dtype)

    vae.requires_grad_(cfg.grad.vae)
    unet.requires_grad_(cfg.grad.unet_2d)
    appearance_encoder.requires_grad_(cfg.grad.appearance_encoder)
    condition_encoder.requires_grad_(cfg.grad.condition_encoder)

    model = Model(
        vae,
        unet,
        appearance_encoder,
        condition_encoder,
        cfg.dataloader.batch_size,
    ).to(device, weight_dtype)

    for name, param in model.named_parameters():
        for trainable_module_name in cfg.trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    if cfg.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()
        model.appearance_encoder.enable_gradient_checkpointing()

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

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    # trainable_params keys
    trainable_params_keys = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    logger.info(f"Trainable params keys: {trainable_params_keys}")
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
        num_frames=32,
        ref_margin=cfg.dataset.ref_margin,
        uncond_ratio=cfg.dataset.uncond_ratio,
        mask_ratio=cfg.dataset.mask_ratio,
        mask_thershold=cfg.dataset.mask_thershold,
        skip_ratio=cfg.dataset.skip_ratio,
        sk_mask_ratio=cfg.dataset.sk_mask_ratio,
        hamer_mask_ratio=cfg.dataset.hamer_mask_ratio,
        both_mask_ratio=cfg.dataset.both_mask_ratio,
        random_sample=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    count_params(model)

    (model, optimizer, dataloader, valid_loader, lr_scheduler) = accelerator.prepare(
        model, optimizer, dataloader, valid_loader, lr_scheduler
    )

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
    best_fvd = float("inf")

    fvd_score = log_valid(
        cfg,
        accelerator,
        valid_loader,
        model,
        empty_text_emb,
        device,
        weight_dtype,
        global_step,
        workspace_dir + "/log",
    )

    # 保存最佳模型
    if accelerator.is_main_process and fvd_score < best_fvd:
        best_fvd = fvd_score
        best_model_path = os.path.join(workspace_dir, f"best")
        save_model(
            model, cfg, best_model_path, accelerator.is_main_process, weight_dtype
        )
        logger.info(f"Saved best model with FVD score: {best_fvd}")

    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description(f"Exp: {cfg.exp_name}")

    for epoch in range(start_epoch, num_epochs):
        train_loss = 0.0
        train_diff_loss = 0.0
        train_vq_loss = 0.0
        train_perplexity = 0.0

        t_data_start = time.time()
        for step, batch in enumerate(dataloader):
            t_data = time.time() - t_data_start
            model.train()
            with accelerator.accumulate(model):
                ref_frame = batch["ref_frame"].to(device, dtype=weight_dtype)
                tgt_frames = batch["tgt_frames"].to(device, dtype=weight_dtype)
                tgt_sk_frames = batch["tgt_sk_frames"].to(device, dtype=weight_dtype)
                tgt_hamer_frames = batch["tgt_hamer_frames"].to(
                    device, dtype=weight_dtype
                )

                # Apply pose augmentation for each sample and frame in batch
                b, c, f, h, w = tgt_sk_frames.shape
                for i in range(b):
                    for j in range(f):
                        if random.random() < cfg.pose_aug_rate:
                            # Randomly select another sample from batch
                            other_sample = random.randint(0, b - 1)
                            # Randomly select a frame from that sample
                            other_frame = random.randint(0, f - 1)

                            # Replace current frame conditions with randomly selected frame
                            tgt_sk_frames[i, :, j] = tgt_sk_frames[
                                other_sample, :, other_frame
                            ].clone()
                            tgt_hamer_frames[i, :, j] = tgt_hamer_frames[
                                other_sample, :, other_frame
                            ].clone()

                with torch.no_grad():
                    frames = torch.cat(
                        [ref_frame.unsqueeze(2), tgt_frames], dim=2
                    )  # [bs, c, f+1, h, w]
                    _, _, num_frames, _, _ = frames.shape

                    frames = rearrange(frames, "b c f h w -> (b f) c h w")
                    latents = model.vae.encode(frames).latent_dist.sample()
                    latents = rearrange(
                        latents, "(b f) c h w -> b c f h w", f=num_frames
                    )
                    latents = latents * 0.18215
                    tgt_latents = latents[:, :, 1:, ...]  # [bs, c, f, h, w]
                    ref_latents = latents[:, :, 0, ...]  # [bs, c, h, w]

                    num_visible_frames = (
                        0
                        if cfg.stage_init
                        else random.randint(0, cfg.max_visible_frames)
                    )
                    visible_tgt_latents = tgt_latents[:, :, :num_visible_frames, ...]
                    non_visible_tgt_latents = tgt_latents[
                        :, :, num_visible_frames:, ...
                    ]

                    # Use empty_text_emb for text_embeds
                    bs = non_visible_tgt_latents.shape[0]
                    text_embeds = empty_text_emb.repeat(bs, 1, 1)

                bs, nc, _, _, _ = non_visible_tgt_latents.shape
                noise = torch.randn_like(non_visible_tgt_latents)
                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn(
                        (bs, nc, 1, 1, 1), device=latents.device
                    )
                noise = (
                    torch.cat(
                        [
                            torch.zeros_like(visible_tgt_latents),
                            noise,
                        ],
                        dim=2,
                    )
                    if not cfg.stage_init
                    else noise
                )

                timesteps = torch.randint(
                    0, noisy_scheduler.num_train_timesteps, (bs,), device=latents.device
                ).long()
                noisy_latents = noisy_scheduler.add_noise(tgt_latents, noise, timesteps)
                tgt_sk_frames = rearrange(tgt_sk_frames, "b c f h w -> (b f) c h w")
                tgt_hamer_frames = rearrange(
                    tgt_hamer_frames, "b c f h w -> (b f) c h w"
                )

                if model.condition_encoder.use_vq:
                    sk_features, perplexity, vq_loss = model.condition_encoder(
                        tgt_sk_frames,
                        tgt_hamer_frames,
                        video_length=num_frames - 1,
                        return_vq=True,
                    )
                    vq_loss = vq_loss * cfg.vq_loss_weight

                else:
                    sk_features = model.condition_encoder(
                        tgt_sk_frames, tgt_hamer_frames, video_length=num_frames - 1
                    )
                    vq_loss = torch.tensor(0.0, device=device, dtype=weight_dtype)
                    perplexity = torch.tensor(0.0, device=device, dtype=weight_dtype)
                sk_features = [
                    rearrange(sk_feature, "(b f) c h w -> b c f h w", f=num_frames - 1)
                    for sk_feature in sk_features
                ]

                if noisy_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noisy_scheduler.prediction_type == "v_prediction":
                    target = noisy_scheduler.get_velocity(tgt_latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noisy_scheduler.prediction_type}"
                    )
                model.appearance_encoder(ref_latents, timesteps, text_embeds)
                model.reference_control_reader.update(
                    model.reference_control_writer, dtype=weight_dtype
                )
                pred = model.unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=text_embeds,
                    sk_features=sk_features,
                ).sample

                loss = func.mse_loss(pred.float(), target.float(), reduction="none")

                if cfg.snr_gamma != 0:
                    snr = compute_snr(noisy_scheduler, timesteps)
                    if noisy_scheduler.prediction_type == "v_prediction":
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [
                                snr,
                                cfg.snr_gamma * torch.ones_like(timesteps),
                            ],
                            dim=1,
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                loss = loss.mean()

                total_loss = loss + vq_loss if model.condition_encoder.use_vq else loss

                avg_loss = accelerator.gather(
                    total_loss.repeat(cfg.dataloader.batch_size)
                ).mean()
                if model.condition_encoder.use_vq:
                    avg_vq_loss = accelerator.gather(
                        vq_loss.repeat(cfg.dataloader.batch_size)
                    ).mean()
                    avg_perplexity = accelerator.gather(
                        perplexity.repeat(cfg.dataloader.batch_size)
                    ).mean()
                    train_vq_loss += (
                        avg_vq_loss.item() / cfg.solver.gradient_accumulation_steps
                    )
                    train_perplexity += (
                        avg_perplexity.item() / cfg.solver.gradient_accumulation_steps
                    )

                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params, cfg.solver.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                model.reference_control_reader.clear()
                model.reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                log_dict = {"train_loss": train_loss}
                if model.condition_encoder.use_vq:
                    log_dict.update(
                        {
                            "train_vq_loss": train_vq_loss,
                            "train_perplexity": train_perplexity,
                        }
                    )
                accelerator.log(log_dict, step=global_step)

                train_loss = 0.0
                train_vq_loss = 0.0
                train_perplexity = 0.0

                if global_step % cfg.valid_steps == 0 and global_step > 0:
                    fvd_score = log_valid(
                        cfg,
                        accelerator,
                        valid_loader,
                        model,
                        empty_text_emb,
                        device,
                        weight_dtype,
                        global_step,
                        workspace_dir + "/log",
                    )

                    # 保存最佳模型
                    if accelerator.is_main_process and fvd_score < best_fvd:
                        best_fvd = fvd_score
                        best_model_path = os.path.join(workspace_dir, f"best")
                        save_model(
                            model,
                            cfg,
                            best_model_path,
                            accelerator.is_main_process,
                            weight_dtype,
                        )
                        logger.info(f"Saved best model with FVD score: {best_fvd}")

            logs = {
                "step_loss": loss.detach().item(),
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

    ckpt_path = os.path.join(workspace_dir, f"checkpoint-{global_step}")
    model_path = os.path.join(workspace_dir, f"model-{global_step}")

    accelerator.save_state(ckpt_path, exclude_frozen_parameters=True)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
