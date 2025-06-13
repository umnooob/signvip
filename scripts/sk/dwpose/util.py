import math

import cv2
import matplotlib
import numpy as np

eps = 0.01


def alpha_blend_color(color, alpha):
    """blend color according to point conf"""
    return [int(c * alpha) for c in color]


# def draw_bodypose(canvas, candidate, subset, score):
#     H, W, C = canvas.shape
#     candidate = np.array(candidate)
#     subset = np.array(subset)

#     stickwidth = 4

#     limbSeq = [
#         [2, 3],
#         [2, 6],
#         [3, 4],
#         [4, 5],
#         [6, 7],
#         [7, 8],
#         [2, 9],
#         [9, 10],
#         [10, 11],
#         [2, 12],
#         [12, 13],
#         [13, 14],
#         [2, 1],
#         [1, 15],
#         [15, 17],
#         [1, 16],
#         [16, 18],
#         [3, 17],
#         [6, 18],
#     ]

#     colors = [
#         [255, 0, 0],
#         [255, 85, 0],
#         [255, 170, 0],
#         [255, 255, 0],
#         [170, 255, 0],
#         [85, 255, 0],
#         [0, 255, 0],
#         [0, 255, 85],
#         [0, 255, 170],
#         [0, 255, 255],
#         [0, 170, 255],
#         [0, 85, 255],
#         [0, 0, 255],
#         [85, 0, 255],
#         [170, 0, 255],
#         [255, 0, 255],
#         [255, 0, 170],
#         [255, 0, 85],
#     ]

#     for i in range(17):
#         for n in range(len(subset)):
#             index = subset[n][np.array(limbSeq[i]) - 1]
#             conf = score[n][np.array(limbSeq[i]) - 1]
#             if conf[0] < 0.3 or conf[1] < 0.3:
#                 continue
#             Y = candidate[index.astype(int), 0] * float(W)
#             X = candidate[index.astype(int), 1] * float(H)
#             mX = np.mean(X)
#             mY = np.mean(Y)
#             length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
#             angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
#             polygon = cv2.ellipse2Poly(
#                 (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
#             )
#             cv2.fillConvexPoly(
#                 canvas, polygon, alpha_blend_color(colors[i], conf[0] * conf[1])
#             )

#     canvas = (canvas * 0.6).astype(np.uint8)

#     for i in range(18):
#         for n in range(len(subset)):
#             index = int(subset[n][i])
#             if index == -1:
#                 continue
#             x, y = candidate[index][0:2]
#             conf = score[n][i]
#             x = int(x * W)
#             y = int(y * H)
#             cv2.circle(
#                 canvas,
#                 (int(x), int(y)),
#                 4,
#                 alpha_blend_color(colors[i], conf),
#                 thickness=-1,
#             )

#     return canvas


# def draw_handpose(canvas, all_hand_peaks, all_hand_scores):
#     H, W, C = canvas.shape

#     edges = [
#         [0, 1],
#         [1, 2],
#         [2, 3],
#         [3, 4],
#         [0, 5],
#         [5, 6],
#         [6, 7],
#         [7, 8],
#         [0, 9],
#         [9, 10],
#         [10, 11],
#         [11, 12],
#         [0, 13],
#         [13, 14],
#         [14, 15],
#         [15, 16],
#         [0, 17],
#         [17, 18],
#         [18, 19],
#         [19, 20],
#     ]

#     for peaks, scores in zip(all_hand_peaks, all_hand_scores):

#         for ie, e in enumerate(edges):
#             x1, y1 = peaks[e[0]]
#             x2, y2 = peaks[e[1]]
#             x1 = int(x1 * W)
#             y1 = int(y1 * H)
#             x2 = int(x2 * W)
#             y2 = int(y2 * H)
#             score = int(scores[e[0]] * scores[e[1]] * 255)
#             if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
#                 cv2.line(
#                     canvas,
#                     (x1, y1),
#                     (x2, y2),
#                     matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
#                     * score,
#                     thickness=2,
#                 )

#         for i, keyponit in enumerate(peaks):
#             x, y = keyponit
#             x = int(x * W)
#             y = int(y * H)
#             score = int(scores[i] * 255)
#             if x > eps and y > eps:
#                 cv2.circle(canvas, (x, y), 4, (0, 0, score), thickness=-1)
#     return canvas


def get_hand_area(all_hand_peaks, all_hand_scores):
    # get two squares include hands
    hand_areas = []
    avg_hand_scores = [np.mean(hand_scores) for hand_scores in all_hand_scores]
    min_hand_scores = [np.min(hand_scores) for hand_scores in all_hand_scores]

    for hand_peaks, hand_scores in zip(all_hand_peaks, all_hand_scores):
        if len(hand_peaks) == 0:
            continue

        # Convert to numpy array for easier calculations
        hand_peaks = np.array(hand_peaks)

        # Find min and max coordinates
        min_x, min_y = np.min(hand_peaks, axis=0)
        max_x, max_y = np.max(hand_peaks, axis=0)

        # Calculate center and side length
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        side_length = max(max_x - min_x, max_y - min_y) * 1.2  # Add 20% margin

        # Calculate square coordinates
        x1 = max(0, center_x - side_length / 2)
        y1 = max(0, center_y - side_length / 2)
        x2 = min(1, center_x + side_length / 2)
        y2 = min(1, center_y + side_length / 2)

        hand_areas.append([x1, y1, x2, y2])

    return hand_areas, avg_hand_scores, min_hand_scores


# def draw_facepose(canvas, all_lmks, all_scores):
#     H, W, C = canvas.shape
#     for lmks, scores in zip(all_lmks, all_scores):
#         for lmk, score in zip(lmks, scores):
#             x, y = lmk
#             x = int(x * W)
#             y = int(y * H)
#             conf = int(score * 255)
#             if x > eps and y > eps:
#                 cv2.circle(canvas, (x, y), 3, (conf, conf, conf), thickness=-1)
#     return canvas


def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks):
    H, W, C = canvas.shape

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
                    * 255,
                    thickness=2,
                )

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, all_lmks):
    H, W, C = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas





# def draw_pose(pose, H, W, ref_w=512):
#     """vis dwpose outputs

#     Args:
#         pose (List): DWposeDetector outputs in dwpose_detector.py
#         H (int): height
#         W (int): width
#         ref_w (int, optional) Defaults to 2160.

#     Returns:
#         np.ndarray: image pixel value in RGB mode
#     """
#     bodies = pose["bodies"]
#     faces = pose["faces"]
#     hands = pose["hands"]
#     candidate = bodies["candidate"]
#     subset = bodies["subset"]

#     sz = min(H, W)
#     sr = (ref_w / sz) if sz != ref_w else 1

#     ########################################## create zero canvas ##################################################
#     canvas = np.zeros(shape=(int(H * sr), int(W * sr), 3), dtype=np.uint8)

#     ########################################### draw body pose #####################################################
#     canvas = draw_bodypose(canvas, candidate, subset, score=bodies["score"])

#     ########################################### draw hand pose #####################################################
#     canvas = draw_handpose(canvas, hands, pose["hands_score"])

#     ########################################### draw face pose #####################################################
#     canvas = draw_facepose(canvas, faces, pose["faces_score"])

#     return cv2.resize(canvas, (W, H))


def draw_pose(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = draw_bodypose(canvas, candidate, subset)

    canvas = draw_handpose(canvas, hands)

    canvas = draw_facepose(canvas, faces)

    return canvas


def draw_normalized_pose(pose, ref_pose, H, W):
    """Visualize normalized DWpose outputs

    Args:
        pose (dict): DWposeDetector outputs
        ref_pose (dict): DWposeDetector outputs
        H (int): height
        W (int): width

    Returns:
        np.ndarray: image pixel value in RGB mode
    """
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [
        i
        for i in ref_keypoint_id
        if len(ref_pose["bodies"]["subset"]) > 0
        and ref_pose["bodies"]["subset"][0][i] >= 0.0
    ]
    ref_body = ref_pose["bodies"]["candidate"][ref_keypoint_id]

    detected_body = pose["bodies"]["candidate"][ref_keypoint_id]
    # compute linear-rescale params
    ay, by = np.polyfit(
        detected_body[:, 1].flatten(),
        ref_body[:, 1],
        1,
    )
    ax = 1
    bx = np.mean(ref_body[:, 0] - detected_body[:, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    # pose rescale
    pose["bodies"]["candidate"] = pose["bodies"]["candidate"] * a + b
    pose["faces"] = pose["faces"] * a + b
    pose["hands"] = pose["hands"] * a + b
    im = draw_pose(pose, H, W)

    return im
