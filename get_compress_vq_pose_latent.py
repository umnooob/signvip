import argparse
import logging
import os
import pathlib
import pickle
import traceback
import warnings

import accelerate
import numpy as np
import torch
from accelerate.logging import get_logger
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.condition_encoder import VQConditionEncoder
from signdatasets import SignCondDataset

warnings.filterwarnings("ignore")
logger = get_logger(__name__, log_level="DEBUG")


def load_modules(cfg, device, weight_dtype):
    modules_cfg = cfg.modules

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

    condition_encoder = condition_encoder.to(device, torch.float32)

    return (condition_encoder,)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="pose_latent")
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


def main():
    try:
        cfg = parse_config()

        accelerator = accelerate.Accelerator()
        device = accelerator.local_process_index

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

        # if cfg.seed is not None:
        #     seed_everything(cfg.seed)

        if cfg.weight_dtype == "fp16":
            weight_dtype = torch.float16
        elif cfg.weight_dtype == "fp32":
            weight_dtype = torch.float32
        else:
            raise ValueError(
                f"Do not support weight dtype: {cfg.weight_dtype} during training!"
            )
        condition_encoder = load_modules(cfg, device, weight_dtype)[0]
        num_compress = condition_encoder.num_compress

        dataset = SignCondDataset(
            output_dir=cfg.output_dir,
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
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        os.makedirs(cfg.output_dir, exist_ok=True)

        sub_batch_size = 256
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                try:
                    path = batch["path"][0]
                    if path == "":
                        continue

                    out_path = os.path.join(
                        cfg.output_dir, os.path.basename(path).replace(".mp4", ".pkl")
                    )
                    if os.path.exists(out_path):
                        continue
                    tgt_sk_frames = batch["tgt_sk_frames"].to(device, torch.float32)
                    tgt_hamer_frames = batch["tgt_hamer_frames"].to(
                        device, torch.float32
                    )

                    # 获取当前帧数并计算需要pad的数量
                    b, c, f, h, w = tgt_sk_frames.shape
                    pad_frames = (
                        num_compress - f % num_compress
                    ) % num_compress  # 计算需要补充多少帧才能达到2的倍数

                    if pad_frames > 0:
                        # 获取最后一帧并重复所需次数
                        last_sk_frame = tgt_sk_frames[:, :, -1:, :, :]
                        last_hamer_frame = tgt_hamer_frames[:, :, -1:, :, :]

                        # 在时间维度上拼接
                        tgt_sk_frames = torch.cat(
                            [
                                tgt_sk_frames,
                                last_sk_frame.repeat(1, 1, pad_frames, 1, 1),
                            ],
                            dim=2,
                        )
                        tgt_hamer_frames = torch.cat(
                            [
                                tgt_hamer_frames,
                                last_hamer_frame.repeat(1, 1, pad_frames, 1, 1),
                            ],
                            dim=2,
                        )

                    # 重新排列维度
                    tgt_sk_frames = rearrange(tgt_sk_frames, "b c f h w -> (b f) c h w")
                    tgt_hamer_frames = rearrange(
                        tgt_hamer_frames, "b c f h w -> (b f) c h w"
                    )

                    if tgt_sk_frames.shape[0] > 2400:
                        continue
                    pose_latents = []
                    for i in range(0, tgt_sk_frames.shape[0], sub_batch_size):
                        end_idx = min(i + sub_batch_size, tgt_sk_frames.shape[0])
                        tgt_sk_frames_sub = tgt_sk_frames[i:end_idx]
                        tgt_hamer_frames_sub = tgt_hamer_frames[i:end_idx]
                        pose_latent = condition_encoder.encode(
                            tgt_sk_frames_sub,
                            tgt_hamer_frames_sub,
                        )
                        print(pose_latent.shape)
                        pose_latents.append(pose_latent.cpu().numpy())

                        # clear cache
                        torch.cuda.empty_cache()
                    pose_latents = np.concatenate(pose_latents, axis=0)
                    assert pose_latents.shape[0] == (f + pad_frames) // num_compress
                    with open(out_path, "wb") as f:
                        pickle.dump(pose_latents, f)
                    del pose_latents

                except Exception as e:
                    logger.error(f"Error processing batch with path {path}: {str(e)}")
                    logger.debug(traceback.format_exc())
                    continue

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
