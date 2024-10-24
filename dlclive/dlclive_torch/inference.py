import torch
import numpy as np
from typing import Dict, Optional


def predict_pose(
    model: torch.nn.Module, frame: torch.Tensor, cfg: Dict, device: torch.device
) -> np.ndarray:
    """Run model inference"""
    with torch.no_grad():
        frame = frame.to(device)
        outputs = model(frame)

        # Extract coordinates
        if cfg["location_refinement"]:  # Check config for locref
            scmap, locref = outputs
            poses = extract_poses(scmap, locref, cfg["stride"], cfg["locref_stdev"])
        else:
            scmap = outputs
            poses = extract_poses(scmap, None, cfg["stride"], None)  # No locref
            poses = poses[:, [1, 0, 2]]  # Reorder to x,y,conf

    return poses


def extract_poses(
    scmap: torch.Tensor,
    locref: Optional[torch.Tensor],
    stride: int,
    locref_stdev: Optional[float],
) -> np.ndarray:
    """Extract poses from scoremap and location refinement heatmaps."""

    if locref is not None:
        scmap = scmap.cpu()
        locref = locref.cpu()

        batch_size, num_keypoints, h, w = scmap.shape
        locref = (
            locref.reshape(batch_size, num_keypoints, 2, h, w)
            .transpose(2, 3)
            .reshape(batch_size, num_keypoints, h, w, 2)
        )

        max_vals, max_indices = scmap.view(batch_size, num_keypoints, -1).max(
            dim=-1
        )  # Get max values and their indices
        x = max_indices % w
        y = max_indices // w

        locref_x = (
            locref[
                np.arange(batch_size)[:, None], np.arange(num_keypoints)[None], y, x, 0
            ]
            * locref_stdev
        )
        locref_y = (
            locref[
                np.arange(batch_size)[:, None], np.arange(num_keypoints)[None], y, x, 1
            ]
            * locref_stdev
        )

        x = (
            x.view(batch_size, num_keypoints, 1) * stride
            + stride / 2
            + locref_x.view(batch_size, num_keypoints, 1)
        )
        y = (
            y.view(batch_size, num_keypoints, 1) * stride
            + stride / 2
            + locref_y.view(batch_size, num_keypoints, 1)
        )
        likelihood = max_vals.view(batch_size, num_keypoints, 1)

        pred = (
            torch.cat((x, y, likelihood), dim=2).squeeze(0).numpy()
        )  # Assuming batch_size=1 for live

    else:  # No locref
        scmap = scmap.cpu().numpy()
        h, w = scmap.shape[2:]
        poses = []

        for p in scmap[0]:  # Single batch assumed
            y, x = np.unravel_index(np.argmax(p), (h, w))
            likelihood = p.max()
            poses.append([x * stride, y * stride, likelihood])

        pred = np.array(poses)

    return pred


def map_to_original_coords(poses: np.ndarray, track_info: Dict) -> np.ndarray:
    """Map predictions to original coordinates"""
    poses = poses.copy()

    # Apply scale
    scale = track_info["scale_factor"]
    poses[:, :2] *= scale

    # Add offsets
    x_off, y_off = track_info["crop_offset"]
    poses[:, 0] += x_off
    poses[:, 1] += y_off

    return poses
