import json
import pickle

import cv2
from dwpose.util import draw_bodypose, draw_normalized_pose, draw_pose, get_hand_area
from PIL import Image
from tqdm import tqdm

# video_path = "/deepo_data/signvipworkspace/datasets/How2Sign/train_processed_videos/_20g7MG8K1U_5-8-rgb_front.mp4"
ref_path = "/deepo_data/signvipworkspace/datasets/RWTH-TSK/dev_processed_videos/01April_2010_Thursday_heute-6698.pkl"
# traverse all files in the directory
# import os
# vid_len = []
# for file in tqdm(os.listdir(ref_path)):
#     with open(os.path.join(ref_path, file), "rb") as f:
#         video_keypoints = pickle.load(f)
#     vid_len.append(len(video_keypoints))
# # print the max length of videos
# print(max(vid_len))
# # save ditribution of video lengths
# import matplotlib.pyplot as plt
# plt.hist(vid_len, bins=range(1, max(vid_len) + 1))
# plt.savefig("vid_len_distribution.png")

# print(video_keypoints[0]["faces_score"].shape)

# openpose_path = "/deepo_data/signvipworkspace/datasets/How2SignSK/train_processed_keypoints/_0-JkwZ9o4Q_7-5-rgb_front.json"
# with open(openpose_path, "r") as f:
#     openpose_keypoints = json.load(f)
# print(len(openpose_keypoints[0]["body_keypoints"]))

with open(ref_path, "rb") as f:
    ref_keypoints = pickle.load(f)

image = ref_keypoints[0]
image = draw_pose(image, 260, 210)
Image.fromarray(image).save("test_pose.png")
# img = draw_normalized_pose(video_keypoints[10],ref_keypoints[0], 512, 512)
# cv2.imwrite("test_normalized.png", img)
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter("hand_vis.mp4", fourcc, 24, (512, 512))

# cap = cv2.VideoCapture(video_path)
# video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# for i in range(video_length):

#     ret, frame = cap.read()
#     if ret:
#         sk_image = video_keypoints[i]

#         hand_areas, avg_hand_scores, min_hand_scores = get_hand_area(
#             sk_image["hands"], sk_image["hands_score"]
#         )
#         # Draw hand areas on the image
#         for area in hand_areas:
#             x1, y1, x2, y2 = area
#             # Convert normalized coordinates to pixel coordinates
#             x1, y1 = int(x1 * 512), int(y1 * 512)
#             x2, y2 = int(x2 * 512), int(y2 * 512)
#             cv2.rectangle(
#                 frame, (x1, y1), (x2, y2), (0, 255, 0), 2
#             )  # Green rectangle with thickness 2

#         # Label the hand areas with their index
#         for idx, area in enumerate(hand_areas):
#             x1, y1, x2, y2 = area
#             # Convert normalized coordinates to pixel coordinates
#             x1, y1 = int(x1 * 512), int(y1 * 512)
#             cv2.putText(frame, f"Hand {idx+1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

#         # Write avg_hand_scores and min_hand_scores to frame
#         for idx, (avg_score, min_score) in enumerate(zip(avg_hand_scores, min_hand_scores)):
#             text = f"A: {avg_score:.2f}, M: {min_score:.2f}"
#             cv2.putText(frame, text, (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#         out.write(frame)

# cap.release()
# out.release()
