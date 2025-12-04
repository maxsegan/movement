#!/usr/bin/env python3
"""
Overlay short-horizon future skeletons on top of the source video frames.

For every frame t in the input video, this script draws the skeletons for
frames t+1, t+2, and t+3 with progressively lower opacity (1.0, 0.66, 0.33).
Pelvis translation is fixed to the pelvis pixel of the *current* frame to
avoid ambiguities from missing global orientation/translation in the 3D trace.

Inputs:
- A video file.
- A 3D pose .npz produced by the data prep pipeline (expects `pose3d` in H36M order).
The frame counts should match; if not, the shorter one defines the usable range.

The script re-detects 2D keypoints with ViTPose (GPU) to estimate per-frame
scale/rotation, but keeps pelvis anchoring fixed to the current frame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, VitPoseForPoseEstimation

# Allow running as a standalone script (`python movement/inference/overlay_future_skeletons.py`)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data_prep.geometry import fit_similarity_2d  # noqa: E402
from data_prep.keypoints import coco_h36m  # noqa: E402
from data_prep.vitpose import infer_sequence  # noqa: E402
from data_prep.constants import H36M_I, H36M_J  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Overlay future skeletons onto a video for quick visualisation.")
    ap.add_argument("--video", required=True, help="Path to the source video file.")
    ap.add_argument("--pose", required=True, help="Path to .npz containing pose3d in H36M order.")
    ap.add_argument("--output", default=None, help="Output mp4 path (default: alongside video with _future_overlay).")
    ap.add_argument("--limit", type=int, default=None, help="Optional frame cap for quick previews.")
    ap.add_argument("--device", default="cuda:0", help="Torch device for ViTPose (e.g., cuda:0).")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for ViTPose.")
    return ap.parse_args()


def read_video_frames(video_path: Path, limit: int | None) -> Tuple[List[np.ndarray], int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(frame_bgr[:, :, ::-1])  # to RGB
        count += 1
        if limit is not None and count >= limit:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to read frames from {video_path}")
    h, w = frames[0].shape[:2]
    return frames, w, h, fps


def full_frame_boxes(num: int, width: int, height: int) -> np.ndarray:
    return np.tile(np.array([0, 0, width - 1, height - 1], dtype=np.float32), (num, 1))


def run_vitpose(frames: List[np.ndarray], device: str, batch_size: int) -> np.ndarray:
    vitpose_id = "usyd-community/vitpose-plus-large"
    processor = AutoImageProcessor.from_pretrained(vitpose_id, trust_remote_code=True)
    model = VitPoseForPoseEstimation.from_pretrained(
        vitpose_id, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    idxs = np.arange(len(frames))
    boxes = full_frame_boxes(len(frames), frames[0].shape[1], frames[0].shape[0])
    keypoints2d, _scores = infer_sequence(
        raw_video_path="",
        image_processor=processor,
        model=model,
        device=torch.device(device),
        idxs=idxs,
        boxes_xyxy=boxes,
        frames=np.array(frames),
        batch_size=batch_size,
    )
    keypoints_coco = keypoints2d[0]  # (T,17,2)
    keypoints_h36m, _valid = coco_h36m(keypoints_coco)
    return keypoints_h36m


def procrustes_xy(src_xy: np.ndarray, dst_xy: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return scale and rotation aligning src->dst (both (17,2) root-centered)."""
    if np.isnan(dst_xy).any() or np.isnan(src_xy).any():
        return 1.0, np.eye(2, dtype=np.float32)
    s, R, _t = fit_similarity_2d(src_xy, dst_xy)
    return float(s), R


def draw_pose(
    base_image: np.ndarray,
    joints_xy: np.ndarray,
    alpha: float,
    color: Tuple[int, int, int] = (0, 200, 255),
    radius: int = 3,
    thickness: int = 2,
):
    overlay = base_image.copy()
    for i, j in zip(H36M_I, H36M_J):
        p1 = tuple(np.round(joints_xy[i]).astype(int))
        p2 = tuple(np.round(joints_xy[j]).astype(int))
        cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
    for p in joints_xy:
        cv2.circle(overlay, tuple(np.round(p).astype(int)), radius, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, base_image, 1 - alpha, 0, dst=base_image)


def main():
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    pose_path = Path(args.pose).expanduser().resolve()
    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else video_path.with_name(video_path.stem + "_future_overlay.mp4")
    )

    frames, width, height, fps = read_video_frames(video_path, args.limit)
    pose_npz = np.load(pose_path, allow_pickle=True)
    if "pose3d" not in pose_npz:
        raise RuntimeError(f"{pose_path} missing pose3d")
    pose3d = np.array(pose_npz["pose3d"], dtype=np.float32)
    T = min(len(frames), pose3d.shape[0])
    frames = frames[:T]
    pose3d = pose3d[:T]

    device = torch.device(args.device)
    keypoints_h36m = run_vitpose(frames, device, args.batch_size)
    keypoints_h36m = keypoints_h36m[:T]

    # Precompute root-centered 3D xy for efficiency
    pose3d_xy_root = pose3d[:, :, :2] - pose3d[:, :1, :2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    alphas = [1.0, 0.66, 0.33]
    for t in range(T):
        frame_bgr = frames[t][:, :, ::-1].copy()  # back to BGR for cv2
        anchor = keypoints_h36m[t, 0] if t < keypoints_h36m.shape[0] else np.array([width / 2, height / 2])
        for step, alpha in enumerate(alphas, start=1):
            f = t + step
            if f >= T:
                continue
            src = pose3d_xy_root[f]
            dst_full = keypoints_h36m[f] if f < keypoints_h36m.shape[0] else None
            dst = dst_full - dst_full[0] if dst_full is not None else None
            if dst is None or np.isnan(dst).any():
                scale, R = 1.0, np.eye(2, dtype=np.float32)
            else:
                scale, R = procrustes_xy(src, dst)
            posed = (src @ R) * scale + anchor
            draw_pose(frame_bgr, posed, alpha=alpha)
        writer.write(frame_bgr)

    writer.release()
    print(f"Overlay saved to {out_path}")


if __name__ == "__main__":
    main()
