#!/usr/bin/env python3
"""
Generate a pipeline figure for the paper showing:
  Panel 1: Raw video frame with bounding box (tracking)
  Panel 2: 2D pose estimation overlay (ViTPose)
  Panel 3: 3D pose lifting (MotionAGFormer)

Uses pre-computed data from NPZ files — no model inference needed.
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_prep.constants import H36M_I, H36M_J, H36M_LR_EDGE_MASK

# H36M joint names for reference
JOINT_NAMES = [
    "Hip", "R_Hip", "R_Knee", "R_Ankle",
    "L_Hip", "L_Knee", "L_Ankle",
    "Spine", "Thorax", "Neck", "Head",
    "L_Shoulder", "L_Elbow", "L_Wrist",
    "R_Shoulder", "R_Elbow", "R_Wrist",
]


def extract_video_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Extract a single RGB frame from a video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return frame_bgr[:, :, ::-1]  # BGR -> RGB


def draw_bbox(frame: np.ndarray, bbox: np.ndarray, color=(30, 144, 255), thickness=3) -> np.ndarray:
    """Draw bounding box on frame. bbox = [x1, y1, x2, y2]."""
    img = frame.copy()
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return img


def draw_2d_skeleton(frame: np.ndarray, keypoints: np.ndarray, scores: np.ndarray,
                     score_thresh: float = 0.2, joint_radius: int = 4, bone_width: int = 3) -> np.ndarray:
    """Draw 2D skeleton overlay on frame."""
    img = frame.copy()
    kp = keypoints.astype(int)

    # Draw bones
    for i, (a, b) in enumerate(zip(H36M_I, H36M_J)):
        if scores[a] > score_thresh and scores[b] > score_thresh:
            color = (0, 200, 0) if H36M_LR_EDGE_MASK[i] == 1 else (200, 50, 50)
            cv2.line(img, tuple(kp[a]), tuple(kp[b]), color, bone_width, cv2.LINE_AA)

    # Draw joints
    for j in range(len(kp)):
        if scores[j] > score_thresh:
            cv2.circle(img, tuple(kp[j]), joint_radius, (255, 220, 50), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(kp[j]), joint_radius, (0, 0, 0), 1, cv2.LINE_AA)

    return img


def draw_3d_pose(ax, pose3d: np.ndarray, elev: float = 15, azim: float = -50):
    """Render 3D skeleton on a matplotlib 3D axis."""
    pts = pose3d - pose3d[0:1]  # Center at hip
    # Raw data: X=left-right, Y=up-down (negative=up), Z=depth
    # Remap so Z is vertical (up) for matplotlib 3D plot
    pts = pts[:, [0, 2, 1]]    # X, Z, Y -> plot X, plot Y (depth), plot Z (vertical)
    pts[:, 2] = -pts[:, 2]     # Negate so head (negative Y) points up

    # Draw bones with thicker lines
    for i, (a, b) in enumerate(zip(H36M_I, H36M_J)):
        color = '#2ecc71' if H36M_LR_EDGE_MASK[i] == 1 else '#e74c3c'
        ax.plot([pts[a, 0], pts[b, 0]],
                [pts[a, 1], pts[b, 1]],
                [pts[a, 2], pts[b, 2]],
                color=color, linewidth=4.5, solid_capstyle='round')

    # Draw joints larger
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c='#f39c12', s=70, edgecolors='black', linewidths=0.8, zorder=5)

    # Equal aspect ratio with tighter bounds
    max_range = np.abs(pts).max() * 1.1
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)

    # Clean, minimal axes
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(length=0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#e0e0e0')
    ax.yaxis.pane.set_edgecolor('#e0e0e0')
    ax.zaxis.pane.set_edgecolor('#e0e0e0')
    ax.xaxis.line.set_color('#e0e0e0')
    ax.yaxis.line.set_color('#e0e0e0')
    ax.zaxis.line.set_color('#e0e0e0')
    ax.grid(True, alpha=0.15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True, help="Path to processed NPZ file")
    parser.add_argument("--frame", type=int, default=None, help="Frame index within clip (default: middle)")
    parser.add_argument("--output", type=str, default="paper_draft_docs/figures/pipeline_figure.png")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    npz_path = Path(args.npz)
    data = np.load(npz_path, allow_pickle=True)

    pose3d = data["pose3d"].astype(np.float32)
    keypoints2d = data["keypoints2d"].astype(np.float32)
    scores2d = data["scores2d"].astype(np.float32)
    bboxes = data["bboxes"].astype(np.float32)
    indices = data["indices"].astype(np.int32)
    meta = data["meta"]
    fps, nframes, vid_w, vid_h = meta

    # Pick frame (default: middle of clip)
    if args.frame is not None:
        fidx = args.frame
    else:
        fidx = len(indices) // 2

    frame_idx_in_video = int(indices[fidx])
    bbox = bboxes[fidx]
    kp2d = keypoints2d[fidx]
    sc2d = scores2d[fidx]
    p3d = pose3d[fidx]

    # Find the video file
    stem = npz_path.stem
    action_class = npz_path.parent.name
    vid_candidates = [
        Path(f"data/kinetics-dataset/k700-2020/train/{action_class}/{stem}.mp4"),
        Path(f"data/kinetics-dataset/k700-2020/val/{action_class}/{stem}.mp4"),
    ]
    video_path = None
    for v in vid_candidates:
        if v.exists():
            video_path = str(v)
            break
    if video_path is None:
        raise FileNotFoundError(f"Video not found for {action_class}/{stem}")

    print(f"Clip: {action_class}/{stem}")
    print(f"Frame {fidx} (video frame {frame_idx_in_video}), {int(vid_w)}x{int(vid_h)} @ {fps:.1f}fps")
    print(f"Mean 2D score: {sc2d.mean():.2f}")

    # Extract video frame
    rgb_frame = extract_video_frame(video_path, frame_idx_in_video)

    # Crop region around the bounding box with padding
    h_frame, w_frame = rgb_frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad = max(bw, bh) * 0.4  # 40% padding around bbox
    cx1 = max(0, int(x1 - pad))
    cy1 = max(0, int(y1 - pad * 0.3))  # Less top padding
    cx2 = min(w_frame, int(x2 + pad))
    cy2 = min(h_frame, int(y2 + pad * 0.6))  # More bottom padding for feet
    crop = rgb_frame[cy1:cy2, cx1:cx2]

    # Adjust keypoints and bbox for crop
    kp2d_crop = kp2d.copy()
    kp2d_crop[:, 0] -= cx1
    kp2d_crop[:, 1] -= cy1
    bbox_crop = bbox.copy()
    bbox_crop[0] -= cx1
    bbox_crop[1] -= cy1
    bbox_crop[2] -= cx1
    bbox_crop[3] -= cy1

    # Generate the three panels
    panel1 = draw_bbox(crop, bbox_crop)
    panel2 = draw_2d_skeleton(crop, kp2d_crop, sc2d)

    # Create figure: 3 panels side by side
    fig = plt.figure(figsize=(15, 5.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.1], wspace=0.03)

    # Panel 1: Tracking (bbox)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(panel1)
    ax1.set_title("(a) Person Tracking", fontsize=14, fontweight='bold', pad=8)
    ax1.axis('off')

    # Panel 2: 2D Pose
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(panel2)
    ax2.set_title("(b) 2D Pose Estimation", fontsize=14, fontweight='bold', pad=8)
    ax2.axis('off')

    # Panel 3: 3D Pose
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    draw_3d_pose(ax3, p3d)
    ax3.set_title("(c) 3D Pose Lifting", fontsize=14, fontweight='bold', pad=0)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved to {out_path}")

    # Also save as PDF for LaTeX
    pdf_path = out_path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved to {pdf_path}")

    plt.close()


if __name__ == "__main__":
    main()
