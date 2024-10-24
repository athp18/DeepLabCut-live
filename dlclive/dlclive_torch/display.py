import cv2
import numpy as np
from typing import Dict
from utils import generate_colors


def draw_poses(
    frame: np.ndarray, poses: np.ndarray, cfg: Dict, threshold: float = 0.5
) -> np.ndarray:
    """Draw pose predictions on the frame."""
    bodyparts = cfg.get("bodyparts", [])
    skeleton = cfg.get("skeleton", [])

    # Generate colors
    colors = generate_colors(len(bodyparts))

    # Draw keypoints
    frame = draw_keypoints(frame, poses, colors, threshold)

    # Draw skeleton
    if skeleton:
        # Check if skeleton uses names or indices
        if isinstance(skeleton[0][0], str):
            # Map names to indices
            bodypart_to_idx = {name: idx for idx, name in enumerate(bodyparts)}
            skeleton = [
                [bodypart_to_idx[part1], bodypart_to_idx[part2]]
                for part1, part2 in skeleton
                if part1 in bodypart_to_idx and part2 in bodypart_to_idx
            ]
        frame = draw_skeleton(frame, poses, skeleton, colors, threshold)

    return frame


def draw_keypoints(
    frame: np.ndarray, poses: np.ndarray, colors: np.ndarray, threshold: float
) -> np.ndarray:
    """Draw keypoint markers"""
    for i, (x, y, conf) in enumerate(poses):
        if conf > threshold:
            cv2.circle(frame, (int(x), int(y)), 3, colors[i].tolist(), -1)
    return frame


def draw_skeleton(
    frame: np.ndarray,
    poses: np.ndarray,
    skeleton: list,
    colors: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Draw skeleton connections"""
    for i, (bp1_idx, bp2_idx) in enumerate(skeleton):
        if poses[bp1_idx, 2] > threshold and poses[bp2_idx, 2] > threshold:
            pt1 = tuple(poses[bp1_idx, :2].astype(int))
            pt2 = tuple(poses[bp2_idx, :2].astype(int))
            cv2.line(frame, pt1, pt2, colors[i].tolist(), 1)
    return frame
