import argparse
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from hamer.utils.renderer import Renderer

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

import json
import os
from typing import Dict, Optional

os.environ["PYOPENGL_PLATFORM"] = "osmesa"


def main():
    parser = argparse.ArgumentParser(description="HaMeR demo code")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="out_demo",
        help="Output folder to save rendered results",
    )
    args = parser.parse_args()

    with open(args.path, "rb") as f:
        results = pickle.load(f)
    scaled_focal_length = 10000.0
    # model, model_cfg = load_hamer("./_DATA/hamer_ckpts/checkpoints/hamer.ckpt")
    # print(model_cfg.EXTRA.FOCAL_LENGTH, model_cfg.MODEL.IMAGE_SIZE)
    # faces = model.mano.faces
    # np.savez("./faces.npz", faces)
    data = np.load("/deepo_data/hamer/faces.npz")
    faces = data["arr_0"]
    renderer = Renderer(5000, 256, faces=faces)
    for idx, result in results.items():
        all_verts = []
        all_cam_t = []
        all_right = []

        for verts, cam_t, is_right in result:
            all_verts.append(verts)
            all_cam_t.append(cam_t)
            all_right.append(is_right)

        # Render front view
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(0, 0, 0),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(
                all_verts,
                cam_t=all_cam_t,
                render_res=[260, 210],
                is_right=all_right,
                **misc_args,
            )

            input_img_overlay = cam_view[:, :, :3]

            cv2.imwrite(
                os.path.join(args.out_folder, f"{idx}_all.jpg"),
                255 * input_img_overlay[:, :, ::-1],
            )


if __name__ == "__main__":
    main()
