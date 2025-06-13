import argparse
import logging
import os
import pickle
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.multihead_t2vqpgpt import Text2VQPoseGPT
from signdatasets import VQSignTextDataset

logger = get_logger(__name__, log_level="INFO")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/deepo_data/signvip/configs/test/multihead_t2vqpgpt.yaml",
    )
    parser.add_argument("--report_to", type=str, default="tensorboard")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    args_dict = vars(args)
    merged_dict = {**cfg_dict, **args_dict}
    return OmegaConf.create(merged_dict)


def load_model(cfg, device, weight_dtype):
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
        logger.info(f"Loaded checkpoint from {cfg.modules.ckpt}")

    t2pgpt = t2pgpt.to(device, weight_dtype)
    t2pgpt.eval()

    return tokenizer, text_model, t2pgpt


@torch.no_grad()
def evaluate(
    cfg, model, text_encoder, dataloader, accelerator, device, weight_dtype, output_dir
):
    valid_acc = []

    for step, batch in tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Evaluating"
    ):
        pose_latents = batch["pose_latents"].to(device)
        pose_len = batch["pose_len"].to(device)
        text_input_ids = batch["text_input_ids"].to(device)
        text_attn_mask = batch["text_attn_mask"].to(device)
        paths = batch["path"]

        text_embeds = text_encoder(
            {"input_ids": text_input_ids, "attention_mask": text_attn_mask}
        )
        text_embeds = text_embeds["sentence_embedding"].to(weight_dtype)

        pred_idx = model.sample(text_embeds, cfg.eval.max_pose_len + 1)

        # Gather tensor results from all processes
        pred_idx, pose_latents, pose_len = accelerator.gather_for_metrics(
            (pred_idx, pose_latents, pose_len)
        )

        # Gather paths separately using gather_for_metrics with use_gather_object=True
        paths = accelerator.gather_for_metrics(paths, use_gather_object=True)

        # Calculate accuracy
        for i in range(pose_latents.shape[0]):
            pred = pred_idx[i, : pose_len[i]]
            gt = pose_latents[i, : pose_len[i]]
            # print(f"pred: {pred.shape}, gt: {gt.shape}")
            valid_acc.append((pred == gt).sum().item() / pose_len[i].item())

        # Save first batch predictions
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            for i in range(pose_latents.shape[0]):
                save_path = os.path.join(output_dir, paths[i].replace(".mp4", ".pkl"))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    pickle.dump(
                        {
                            "gt": pose_latents[i].cpu().numpy(),
                            "pred": pred_idx[i].cpu().numpy(),
                            "len": pose_len[i].cpu().numpy(),
                        },
                        f,
                    )
    # Calculate final accuracy
    accuracy = sum(valid_acc) / len(valid_acc)
    if accelerator.is_main_process:
        logger.info(f"Evaluation accuracy: {accuracy:.4f}")

    return accuracy


def main():
    cfg = parse_config()

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=cfg.weight_dtype if cfg.weight_dtype != "fp32" else "no",
    )
    device = accelerator.device

    # Set weight dtype
    weight_dtype = torch.float32 if cfg.weight_dtype == "fp32" else torch.float16

    # Load models
    tokenizer, text_encoder, model = load_model(cfg, device, weight_dtype)
    dataloaders = []

    for metapath in cfg.dataset.meta_paths:
        # Create dataset and dataloader
        dataset = VQSignTextDataset(
            frame_size=cfg.dataset.frame_size,
            frame_scale=cfg.dataset.frame_scale,
            sample_rate=cfg.dataset.sample_rate,
            tokenizer=tokenizer,
            max_pose_len=cfg.eval.max_pose_len,
            pose_size=cfg.eval.pose_size,
            codebook_size=cfg.modules.codebook_size,
            roots=cfg.dataset.roots,
            pose_roots=cfg.dataset.pose_roots,
            meta_paths=[metapath],
        )

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.eval.batch_size,
            shuffle=False,
            num_workers=cfg.eval.num_workers,
        )
        dataloaders.append(dataloader)
    model, *dataloaders = accelerator.prepare(model, *dataloaders)

    # Evaluate
    for dataloader, output_dir in zip(dataloaders, cfg.eval.output_dirs):
        evaluate(
            cfg,
            model,
            text_encoder,
            dataloader,
            accelerator,
            device,
            weight_dtype,
            output_dir,
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
