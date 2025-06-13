# CUDA_VISIBLE_DEVICES=0 python infer_videos_dw.py --video_folder /deepo_data/signvipworkspace/datasets/RWTH-T/train_processed_videos --pose_folder /deepo_data/signvipworkspace/datasets/RWTH-TSK/train_processed_videos --out_folder /deepo_data/signvipworkspace/datasets/RWTH-TSmplerx/train_processed_videos
import argparse
import os
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from hamer.configs import CACHE_DIR_HAMER
from hamer.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD, ViTDetDataset
from hamer.models import DEFAULT_CHECKPOINT, HAMER, download_models, load_hamer
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full
from tqdm import tqdm

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

import json
import os
from typing import Dict, Optional

from dwpose.preprocess import DWposeDetector, get_video_pose_unscale_keypoints
from vitpose_model import ViTPoseModel

os.environ["PYOPENGL_PLATFORM"] = "osmesa"

# dwprocessor = DWposeDetector(
#     model_det="models/DWPose/yolox_l.onnx",
#     model_pose="models/DWPose/dw-ll_ucoco_384.onnx",
#     device="cuda:0",
# )


def process_video(video_path, pose_path, model_cfg, model, args, device):

    with open(pose_path, "rb") as f:
        video_keypoints = pickle.load(f)

    cap = cv2.VideoCapture(video_path)

    results = {}

    valid_bboxes = []
    valid_is_right = []
    valid_idxes = []
    valid_img_cv2 = []

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if len(video_keypoints) != total_frames:
        print(f"Mismatch in number of frames: {len(video_keypoints)} != {total_frames}")
        with open("./log.txt", "a") as f:
            f.write(
                f"{video_path} {pose_path}: Mismatch in number of frames: {len(video_keypoints)} != {total_frames}\n"
            )
        return {}

    while True:
        ret, img_cv2 = cap.read()
        if not ret:
            break

        # Detect human keypoints for each person
        vitposes_out = video_keypoints[frame_idx]

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections

        left_hand_keyp, right_hand_keyp = vitposes_out["hands"][:2]
        left_hand_score, right_hand_score = vitposes_out["hands_score"][:2]
        left_hand_keyp = np.concatenate(
            [left_hand_keyp, left_hand_score[:, None]], axis=1
        )  # (21, 2) + (21,) = (21, 3)
        right_hand_keyp = np.concatenate(
            [right_hand_keyp, right_hand_score[:, None]], axis=1
        )  # (21, 2) + (21,) = (21, 3)

        # Rejecting not confident detections
        keyp = left_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [
                keyp[valid, 0].min() * img_cv2.shape[1],
                keyp[valid, 1].min() * img_cv2.shape[0],
                keyp[valid, 0].max() * img_cv2.shape[1],
                keyp[valid, 1].max() * img_cv2.shape[0],
            ]

            bboxes.append(bbox)
            is_right.append(0)
        keyp = right_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [
                keyp[valid, 0].min() * img_cv2.shape[1],
                keyp[valid, 1].min() * img_cv2.shape[0],
                keyp[valid, 0].max() * img_cv2.shape[1],
                keyp[valid, 1].max() * img_cv2.shape[0],
            ]
            bboxes.append(bbox)
            is_right.append(1)

        if len(bboxes) == 0:
            print(f"missing{frame_idx}")
        else:
            valid_bboxes.append(bboxes)
            valid_is_right.append(is_right)
            valid_img_cv2.extend([img_cv2] * len(bboxes))
            valid_idxes.extend([frame_idx] * len(bboxes))

        frame_idx += 1
    flat_bboxes = [s for ss in valid_bboxes for s in ss]
    flat_is_right = [s for ss in valid_is_right for s in ss]
    if len(flat_bboxes) == 0 or len(flat_is_right) == 0:
        return {}
    boxes = np.stack(flat_bboxes)
    right = np.stack(flat_is_right)

    # Run reconstruction on all detected hands
    dataset = ViTDetDataset(
        model_cfg,
        valid_img_cv2,
        boxes,
        right,
        rescale_factor=args.rescale_factor,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    results = {}
    all_verts = []
    all_cam_t = []
    all_right = []
    for batch in tqdm(dataloader):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        multiplier = 2 * batch["right"] - 1
        pred_cam = out["pred_cam"]
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        multiplier = 2 * batch["right"] - 1

        scaled_focal_length = (
            model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        )
        pred_cam_t_full = (
            cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            )
            .detach()
            .cpu()
            .numpy()
        )

        # Render the result
        for n in range(box_size.shape[0]):
            # Add all verts and cams to list
            verts = out["pred_vertices"][n].detach().cpu().numpy()
            is_right = batch["right"][n].cpu().numpy()
            verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
            cam_t = pred_cam_t_full[n]
            all_verts.append(verts)
            all_cam_t.append(cam_t)
            all_right.append(is_right)

    for image_id, verts, cam_t, is_right in zip(
        valid_idxes, all_verts, all_cam_t, all_right
    ):
        if image_id not in results:
            results[image_id] = []
        results[image_id].append((verts, cam_t, is_right))

    cap.release()
    return results


def main():
    parser = argparse.ArgumentParser(description="HaMeR demo code")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="out_demo",
        help="Output folder to save rendered results",
    )
    parser.add_argument(
        "--side_view",
        dest="side_view",
        action="store_true",
        default=False,
        help="If set, render side view also",
    )
    parser.add_argument(
        "--full_frame",
        dest="full_frame",
        action="store_true",
        default=True,
        help="If set, render all people together also",
    )
    parser.add_argument(
        "--save_mesh",
        dest="save_mesh",
        action="store_true",
        default=False,
        help="If set, save meshes to disk also",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for inference/fitting"
    )
    parser.add_argument(
        "--rescale_factor", type=float, default=2.0, help="Factor for padding the bbox"
    )
    parser.add_argument(
        "--body_detector",
        type=str,
        default="vitdet",
        choices=["vitdet", "regnety"],
        help="Using regnety improves runtime and reduces memory",
    )
    parser.add_argument(
        "--file_type",
        nargs="+",
        default=["*.jpg", "*.png"],
        help="List of file extensions to consider",
    )
    parser.add_argument(
        "--video_folder", type=str, required=True, help="Folder with input videos"
    )
    parser.add_argument(
        "--pose_folder", type=str, required=True, help="Folder with input poses"
    )
    args = parser.parse_args()

    # Download and load checkpoints
    # download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all video files
    video_files = [
        f.replace(".pkl", ".mp4")
        for f in os.listdir(args.pose_folder)
        if f.endswith(".pkl")
    ]

    random.shuffle(video_files)

    for video_file in tqdm(video_files):
        video_path = os.path.join(args.video_folder, video_file)

        output_path = os.path.join(args.out_folder, video_file.replace(".mp4", ".pkl"))

        pose_path = os.path.join(args.pose_folder, video_file.replace(".mp4", ".pkl"))

        if os.path.exists(output_path):
            try:
                with open(output_path, "rb") as f:
                    video_results = pickle.load(f)
                continue
            except Exception as e:
                print(f"Error loading {output_path}: {e}")
                os.remove(output_path)

        video_results = process_video(
            video_path, pose_path, model_cfg, model, args, device
        )

        with open(output_path, "wb") as f:
            pickle.dump(video_results, f)


if __name__ == "__main__":
    main()
