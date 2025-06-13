import argparse
import multiprocessing
import os
import pickle
import random
from multiprocessing import Manager
from pathlib import Path

os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import cv2
import numpy as np
import torch
from renderer import Renderer
from tqdm import tqdm

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

import json
import os
from typing import Dict, Optional


def process_video(video_path, pose_path, renderer, output_path, queue):
    try:
        # Quick file existence check without opening video
        if os.path.exists(output_path):
            try:
                # Get frame count using ffprobe/ffmpeg
                cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 '{video_path}'"
                total_frames = int(os.popen(cmd).read().strip())

                cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 '{output_path}'"
                generated_total_frames = int(os.popen(cmd).read().strip())

                if generated_total_frames == total_frames:
                    queue.put(1)
                    return

            except (ValueError, Exception) as e:
                print(
                    f"Error checking frame count for {output_path}, will regenerate: {str(e)}"
                )

            # If we get here, either the frame counts didn't match or there was an error
            if os.path.exists(output_path):
                os.remove(output_path)
        # Load keypoints once
        with open(pose_path, "rb") as f:
            video_keypoints = pickle.load(f)

        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        original_res = 512
        new_res = 260
        scaled_focal_length = 10000.0 * (new_res / original_res)

        out_video = cv2.VideoWriter(output_path, fourcc, fps, (210, 260))
        for frame_idx in range(total_frames):
            if frame_idx in video_keypoints.keys():
                result = video_keypoints[frame_idx]
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
                        render_res=[210, 260],
                        is_right=all_right,
                        **misc_args,
                    )

                    input_img_overlay = 255 * cam_view[:, :, :3][:, :, ::-1]
                else:
                    # all black
                    input_img_overlay = np.zeros((260, 210, 3))
            else:
                input_img_overlay = np.zeros((260, 210, 3))
            input_img_overlay = input_img_overlay.astype(np.uint8)
            out_video.write(input_img_overlay)

        out_video.release()
        cap.release()

    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
    finally:
        # Always send a signal, even if there's an error
        queue.put(1)


def main():
    parser = argparse.ArgumentParser(description="HaMeR demo code")
    parser.add_argument(
        "--out_folder",
        type=str,
        default="out_demo",
        help="Output folder to save rendered results",
    )
    parser.add_argument(
        "--pose_folder", type=str, required=True, help="Folder with input poses"
    )
    parser.add_argument(
        "--video_folder", type=str, required=True, help="Folder with input videos"
    )
    args = parser.parse_args()

    scaled_focal_length = 10000.0
    data = np.load("/deepo_data/hamer/faces.npz")
    faces = data["arr_0"]
    renderer = Renderer(5000, 256, faces=faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # # Get all video files
    video_files = [f.replace(".pkl", ".mp4") for f in os.listdir(args.pose_folder) if f.endswith(".pkl")]

    # random.shuffle(video_files)
    # with open("files2.txt", "r") as f:
    #     video_files = f.read().splitlines()

    print(len(video_files))

    # Create a multiprocessing manager
    manager = Manager()
    queue = manager.Queue()

    # Use multiprocessing to process videos in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Prepare arguments for each video
        args = [
            (
                os.path.join(args.video_folder, video_file),
                os.path.join(
                    args.pose_folder,
                    video_file.replace(".mp4", ".pkl"),
                ),
                renderer,
                os.path.join(args.out_folder, video_file),
                queue,
            )
            for video_file in video_files
        ]

        # Create a tqdm progress bar
        with tqdm(total=len(video_files)) as pbar:
            # Process videos in parallel using starmap_async
            result = pool.starmap_async(process_video, args)

            # Update the progress bar based on the queue
            completed = 0
            while completed < len(video_files):
                queue.get()  # Wait for a signal from a worker
                completed += 1
                pbar.update(1)  # Update the progress bar

            # Ensure all processes have finished
            result.wait()

    print(f"Processed {len(video_files)} videos.")

if __name__ == "__main__":
    main()
