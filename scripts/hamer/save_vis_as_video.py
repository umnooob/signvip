import argparse
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from hamer.configs import CACHE_DIR_HAMER
from hamer.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD, ViTDetDataset
from hamer.models import DEFAULT_CHECKPOINT, HAMER, download_models, load_hamer
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

import json
import os
from typing import Dict, Optional

from vitpose_model import ViTPoseModel

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
    data = np.load("/deepo_data/hamer/faces.npz")
    faces = data["arr_0"]
    renderer = Renderer(5000, 256, faces=faces)
    scaled_focal_length = 10000.0
    # traverse all files in args.path
    for root, dirs, files in os.walk(args.path):
        for file in files:
            if file.endswith(".pkl"):
                with open(os.path.join(root, file), "rb") as f:
                    results = pickle.load(f)
            
            # render results as video to out_folder
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
                        render_res=[512, 512],
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
                render_res=[512, 512],
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
