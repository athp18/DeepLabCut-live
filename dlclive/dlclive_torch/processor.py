import cv2
import torch
import numpy as np
from typing import Optional, Tuple, Dict
import albumentations as A


def process_frame(
    frame: np.ndarray,
    transform: A.Compose,
    cropping: Optional[list[int]] = None,
    dynamic_crop: Optional[Tuple[np.ndarray, float, float]] = None,
    resize: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Process frame for inference"""
    track_info = {
        "original_size": frame.shape[:2],
        "crop_offset": [0, 0],
        "scale_factor": 1.0,
    }

    # Apply static crop
    if cropping is not None:
        frame = frame[cropping[2] : cropping[3], cropping[0] : cropping[1]]
        track_info["crop_offset"] = [cropping[0], cropping[2]]

    # Apply dynamic crop
    if dynamic_crop is not None:
        frame, track_info = _apply_dynamic_crop(frame, *dynamic_crop, track_info)

    # Apply resize
    if resize is not None and resize != 1.0:
        frame, track_info = _apply_resize(frame, resize, track_info)

    # Apply training transforms
    frame_tensor = _apply_transforms(frame, transform)

    return frame_tensor.unsqueeze(0), track_info


def _apply_dynamic_crop(
    frame: np.ndarray,
    prev_pose: np.ndarray,
    threshold: float,
    margin: float,
    track_info: Dict,
) -> Tuple[np.ndarray, Dict]:
    """Apply dynamic cropping based on previous pose"""
    detected = prev_pose[:, 2] > threshold
    if np.any(detected):
        x = prev_pose[detected, 0]
        y = prev_pose[detected, 1]

        x1 = int(max(0, np.min(x) - margin))
        x2 = int(min(frame.shape[1], np.max(x) + margin))
        y1 = int(max(0, np.min(y) - margin))
        y2 = int(min(frame.shape[0], np.max(y) + margin))

        frame = frame[y1:y2, x1:x2]
        track_info["crop_offset"] = [x1, y1]

    return frame, track_info


def _apply_resize(
    frame: np.ndarray, resize: float, track_info: Dict
) -> Tuple[np.ndarray, Dict]:
    """Apply resize transform"""
    h, w = frame.shape[:2]
    new_h = int(h * resize)
    new_w = int(w * resize)
    frame = cv2.resize(frame, (new_w, new_h))
    track_info["scale_factor"] = 1.0 / resize
    return frame, track_info


def _apply_transforms(frame: np.ndarray, transform: A.Compose) -> torch.Tensor:
    """Apply training transforms"""
    transformed = transform(image=frame)
    frame_tensor = transformed["image"]
    if not isinstance(frame_tensor, torch.Tensor):
        frame_tensor = torch.from_numpy(frame_tensor)
    return frame_tensor
