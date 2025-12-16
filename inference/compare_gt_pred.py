#!/usr/bin/env python3
"""
Compare ground truth vs model-predicted poses.

Renders both GT and predicted skeletons overlaid on the actual video frames
for each timestep in the action horizon.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import yaml

# Allow running as standalone
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_prep.constants import H36M_I, H36M_J
from training.kinetics_dataset import (
    KineticsPoseDataset, _parse_description,
    joint_angles_to_pose3d, pose3d_to_joint_angles, JOINT_ANGLES_DIM
)
from training.vla_model import VLAModel, VLAConfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare GT vs predicted poses")
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    ap.add_argument("--config", required=True, help="Path to training config YAML")
    ap.add_argument("--output", default=str(Path(__file__).parent / "gt_vs_pred.mp4"), help="Output video path")
    ap.add_argument("--action-class", type=str, default=None, help="Filter by action class (e.g., 'push up')")
    ap.add_argument("--sample-idx", type=int, default=0, help="Sample index within filtered set")
    ap.add_argument("--device", default="cuda:0", help="Device to run inference on")
    ap.add_argument("--gt-color", default="0,200,255", help="BGR color for GT skeleton (orange)")
    ap.add_argument("--pred-color", default="255,100,100", help="BGR color for predicted skeleton (blue)")
    ap.add_argument("--full-video", action="store_true", help="Render all frames in video (not just action horizon)")
    ap.add_argument("--native-res", action="store_true", help="Use native video resolution instead of 224x224")
    return ap.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, config: dict, device: str) -> VLAModel:
    """Load model from checkpoint."""
    mc = config['model_config']
    vla_config = VLAConfig(
        qwen_model_name=mc.get('qwen_model_name', 'Qwen/Qwen3-VL-4B-Instruct'),
        qwen_hidden_size=mc.get('qwen_hidden_size', 2560),
        use_intermediate_hidden=mc.get('use_intermediate_hidden', True),
        hidden_layer_index=mc.get('hidden_layer_index', 18),
        use_early_exit=mc.get('use_early_exit', True),
        use_deepstack_features=mc.get('use_deepstack_features', True),
        use_flash_attention=mc.get('use_flash_attention', False),
        projection_dim=mc.get('projection_dim', 512),
        action_dim=mc.get('action_dim', 51),
        diffusion_hidden_dim=mc.get('diffusion_hidden_dim', 512),
        num_diffusion_layers=mc.get('num_diffusion_layers', 4),
        num_diffusion_heads=mc.get('num_diffusion_heads', 8),
        num_future_tokens=mc.get('num_future_tokens', 4),
        action_horizon=mc.get('action_horizon', 16),
        num_frames=mc.get('num_frames', 4),
        use_lora=mc.get('use_lora', True),
        lora_rank=mc.get('lora_rank', 128),
        lora_alpha=mc.get('lora_alpha', 128),
        lora_dropout=mc.get('lora_dropout', 0.05),
        freeze_vision_encoder=mc.get('freeze_vision_encoder', True),
        freeze_qwen_layers=mc.get('freeze_qwen_layers', 0),
        use_thinking_mode=mc.get('use_thinking_mode', False),
        diffusion_steps=mc.get('diffusion_steps', 8),
    )

    print("Loading model...")
    model = VLAModel(vla_config)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle DDP state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model


def procrustes_align_2d(
    source: np.ndarray,
    target: np.ndarray,
    allow_scale: bool = True,
) -> np.ndarray:
    """
    Align source points to target points using Procrustes analysis.

    This finds the optimal rotation, scale, and translation to minimize
    the distance between source and target point sets.

    Args:
        source: Source points [N, 2]
        target: Target points [N, 2]
        allow_scale: Whether to allow uniform scaling

    Returns:
        Aligned source points [N, 2]
    """
    # Filter out invalid points (NaN)
    valid_mask = ~(np.isnan(source).any(axis=1) | np.isnan(target).any(axis=1))
    if valid_mask.sum() < 3:
        return source  # Not enough points for alignment

    src_valid = source[valid_mask]
    tgt_valid = target[valid_mask]

    # Center both point sets
    src_center = src_valid.mean(axis=0)
    tgt_center = tgt_valid.mean(axis=0)

    src_centered = src_valid - src_center
    tgt_centered = tgt_valid - tgt_center

    # Compute optimal rotation using SVD
    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute scale if allowed
    if allow_scale:
        src_var = (src_centered ** 2).sum()
        if src_var > 1e-8:
            # S is 1D array of singular values, not a matrix
            scale = S.sum() / src_var
        else:
            scale = 1.0
    else:
        scale = 1.0

    # Apply transformation to all source points
    result = source.copy()
    result = (result - src_center) @ R.T * scale + tgt_center

    return result


def draw_skeleton(
    frame: np.ndarray,
    joints_xy: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.8,
    radius: int = 5,
    thickness: int = 3,
) -> np.ndarray:
    """Draw skeleton with specified color and alpha blending."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    def is_valid(p):
        if np.any(np.isnan(p)):
            return False
        if p[0] < -w or p[0] > 2*w or p[1] < -h or p[1] > 2*h:
            return False
        return True

    def to_point(p):
        return (int(np.clip(np.round(p[0]), 0, w-1)),
                int(np.clip(np.round(p[1]), 0, h-1)))

    # Draw bones
    for i, j in zip(H36M_I, H36M_J):
        if i < len(joints_xy) and j < len(joints_xy):
            if is_valid(joints_xy[i]) and is_valid(joints_xy[j]):
                p1 = to_point(joints_xy[i])
                p2 = to_point(joints_xy[j])
                cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)

    # Draw joints
    for idx, p in enumerate(joints_xy):
        if is_valid(p):
            pt = to_point(p)
            cv2.circle(overlay, pt, radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, pt, radius - 1, color, -1, cv2.LINE_AA)

    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return result


def project_3d_to_2d_with_bbox(
    pose3d: np.ndarray,
    bbox: np.ndarray,
    resize_factor: float = 1.0,
) -> np.ndarray:
    """Project normalized 3D pose to 2D using bounding box."""
    if np.any(np.isnan(bbox)):
        return np.full((pose3d.shape[0], 2), np.nan)

    x1, y1, x2, y2 = bbox * resize_factor
    bbox_cx = (x1 + x2) / 2
    bbox_cy = (y1 + y2) / 2
    bbox_w = max(x2 - x1, 10)
    bbox_h = max(y2 - y1, 10)

    scale = min(bbox_w, bbox_h) * 0.8

    joints_2d = np.zeros((pose3d.shape[0], 2))
    joints_2d[:, 0] = pose3d[:, 0] * scale + bbox_cx
    # Pose Y is typically up-positive, screen Y is down-positive
    # But normalized poses may already be flipped - try without negation
    joints_2d[:, 1] = pose3d[:, 1] * scale + bbox_cy

    return joints_2d


def main():
    args = parse_args()
    config = load_config(args.config)

    gt_color = tuple(map(int, args.gt_color.split(',')))
    pred_color = tuple(map(int, args.pred_color.split(',')))

    device = args.device
    model = load_model(args.checkpoint, config, device)
    action_horizon = config['model_config'].get('action_horizon', 16)
    num_frames = config['model_config'].get('num_frames', 4)
    action_dim = config['model_config'].get('action_dim', 51)
    use_joint_angles = config['dataset'].get('use_joint_angles', False)

    # Detect joint angle mode from action_dim
    if action_dim == JOINT_ANGLES_DIM:
        use_joint_angles = True
        print(f"Using joint angle mode (action_dim={action_dim})")
    else:
        print(f"Using 3D position mode (action_dim={action_dim})")

    dc = config['dataset']
    print("Loading validation dataset...")
    # Always load dataset with use_joint_angles=False for GT visualization
    # We need raw 3D poses for rendering
    val_dataset = KineticsPoseDataset(
        pose_dir=dc['pose_dir'],
        desc_dir=dc['desc_dir'],
        video_dir=dc['video_dir'],
        split='val',
        val_split=dc.get('val_split', 0.02),
        action_horizon=action_horizon,
        num_frames=num_frames,
        sample_stride=dc.get('sample_stride', 3),
        resize=dc.get('image_size', 224),
        normalize_pose=dc.get('normalize_pose', True),
        use_joint_angles=False,  # Always False - we want raw 3D poses for GT
        seed=dc.get('seed', 42),
    )

    print(f"Validation dataset: {len(val_dataset)} samples")

    # Find sample by action class if specified
    if args.action_class:
        print(f"Searching for action class: {args.action_class}")
        matching_indices = []
        for i, (clip_idx, _) in enumerate(val_dataset.samples):
            clip = val_dataset.clips[clip_idx]
            if args.action_class.lower() in clip.action_class.lower():
                matching_indices.append(i)
                if len(matching_indices) > args.sample_idx:
                    break

        if not matching_indices:
            print(f"No samples found for action class: {args.action_class}")
            return

        sample_idx = matching_indices[min(args.sample_idx, len(matching_indices) - 1)]
        print(f"Found {len(matching_indices)} matching samples, using index {sample_idx}")
    else:
        sample_idx = args.sample_idx

    # Get sample info
    clip_idx, start_frame = val_dataset.samples[sample_idx]
    clip = val_dataset.clips[clip_idx]

    print(f"\nProcessing: {clip.action_class}/{clip.clip_id}")
    print(f"  Start frame: {start_frame}")

    # Load the sample
    sample = val_dataset[sample_idx]
    images = sample['images']
    instruction = sample['instruction']
    gt_actions = sample['actions']
    robot_state = sample['robot_state']

    print(f"  Instruction: {instruction[:80]}...")

    # Load raw pose data for bboxes and frame indices
    data = np.load(clip.pose_path, allow_pickle=True)
    bboxes = data["bboxes"].astype(np.float32)
    pose_indices = data["indices"].astype(np.int32)

    # Get bboxes for action window
    end_frame = start_frame + action_horizon
    action_bboxes = bboxes[start_frame:end_frame]
    action_video_indices = pose_indices[start_frame:end_frame]

    # Pad if needed
    if len(action_bboxes) < action_horizon:
        pad_len = action_horizon - len(action_bboxes)
        action_bboxes = np.concatenate([action_bboxes, np.repeat(action_bboxes[-1:], pad_len, axis=0)])
        action_video_indices = np.concatenate([action_video_indices, np.repeat(action_video_indices[-1:], pad_len)])

    # Get video info
    cap = cv2.VideoCapture(str(clip.video_path))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Resolution settings
    if args.native_res:
        out_w, out_h = orig_w, orig_h
        resize_factor = 1.0
    else:
        display_size = val_dataset.resize or 224
        out_w = out_h = display_size
        resize_factor = display_size / max(orig_w, orig_h)

    print(f"  Video: {orig_w}x{orig_h} @ {video_fps}fps")
    print(f"  Output: {out_w}x{out_h}, resize_factor: {resize_factor:.3f}")

    # Run inference
    img_arrays = [np.array(img) for img in images]

    # Convert robot_state to joint angles if model expects them
    if use_joint_angles:
        # robot_state from dataset is 3D pose (51,), convert to joint angles (22,)
        robot_state_3d = robot_state.reshape(17, 3)
        robot_state = pose3d_to_joint_angles(robot_state_3d)

    with torch.no_grad():
        pred_actions = model.get_action(
            images=img_arrays,
            instruction=instruction,
            robot_state=robot_state,
        )

    # GT poses are always 3D positions from the dataset
    gt_poses = gt_actions.reshape(action_horizon, 17, 3)

    # Handle predicted poses based on mode
    if use_joint_angles:
        # Model outputs joint angles, convert to 3D positions for visualization
        pred_angles = pred_actions.reshape(action_horizon, JOINT_ANGLES_DIM)
        pred_poses = np.zeros((action_horizon, 17, 3), dtype=np.float32)

        # Use first GT pose as reference for bone lengths
        reference_pose = gt_poses[0]

        for t in range(action_horizon):
            pred_poses[t] = joint_angles_to_pose3d(pred_angles[t], reference_pose=reference_pose)

        # Compute MSE in joint angle space (convert GT to joint angles)
        gt_angles = np.array([pose3d_to_joint_angles(gt_poses[t]) for t in range(action_horizon)])
        mse = np.mean((gt_angles - pred_angles) ** 2)
        print(f"  Joint angle MSE: {mse:.6f} rad² ({np.degrees(np.sqrt(mse)):.2f}° RMSE)")
    else:
        # Model outputs 3D positions directly
        pred_poses = pred_actions.reshape(action_horizon, 17, 3)
        mse = np.mean((gt_poses - pred_poses) ** 2)
        print(f"  Position MSE: {mse:.6f}")

    # Determine frame range
    # Load 2D keypoints for visualization (these are in pixel coordinates - no drift)
    keypoints2d = data["keypoints2d"].astype(np.float32)  # [F, 17, 2]
    scores2d = data["scores2d"].astype(np.float32)  # [F, 17]

    if args.full_video:
        num_frames = len(bboxes)
        frame_range = range(num_frames)
        all_video_indices = pose_indices
        all_keypoints2d = keypoints2d
        all_scores2d = scores2d
    else:
        num_frames = action_horizon
        frame_range = range(num_frames)
        all_video_indices = action_video_indices
        all_keypoints2d = keypoints2d[start_frame:end_frame]
        all_scores2d = scores2d[start_frame:end_frame]
        # Pad if needed
        if len(all_keypoints2d) < action_horizon:
            pad_len = action_horizon - len(all_keypoints2d)
            all_keypoints2d = np.concatenate([all_keypoints2d, np.repeat(all_keypoints2d[-1:], pad_len, axis=0)])
            all_scores2d = np.concatenate([all_scores2d, np.repeat(all_scores2d[-1:], pad_len, axis=0)])

    print(f"  Rendering {num_frames} frames using 2D keypoints...")

    # Setup video writer - 10fps to match pose data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, 10, (out_w, out_h))

    # Load and render each frame
    cap = cv2.VideoCapture(str(clip.video_path))

    for t in frame_range:
        video_frame_idx = int(all_video_indices[t])
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ok, frame = cap.read()

        if not ok:
            frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        # Resize frame if needed
        if not args.native_res:
            frame = cv2.resize(frame, (out_w, out_h))

        # Draw future GT poses with decreasing opacity (action_horizon frames ahead)
        for future_offset in range(action_horizon - 1, -1, -1):  # Draw farthest first, current last
            future_t = t + future_offset
            if future_t >= len(all_keypoints2d):
                continue

            # Get 2D keypoints for future frame
            future_kp2d = all_keypoints2d[future_t].copy()

            # Scale keypoints if frame was resized
            if not args.native_res:
                future_kp2d[:, 0] *= out_w / orig_w
                future_kp2d[:, 1] *= out_h / orig_h

            # Opacity decreases for frames further in the future
            # Current frame (offset=0) = 0.9, furthest = 0.2
            if future_offset == 0:
                alpha = 0.9
            else:
                # Linear interpolation from 0.2 (furthest) to 0.6 (nearest non-current)
                alpha = 0.2 + (0.4 * (1.0 - future_offset / action_horizon))

            frame = draw_skeleton(frame, future_kp2d, gt_color, alpha=alpha)

        # Project predicted poses using Procrustes alignment to GT skeleton
        if not args.full_video:
            # Get current frame GT keypoints as alignment target
            gt_kp2d = all_keypoints2d[t].copy()
            if not args.native_res:
                gt_kp2d[:, 0] *= out_w / orig_w
                gt_kp2d[:, 1] *= out_h / orig_h

            pred_idx = t
            if pred_idx < action_horizon:
                pred_3d = pred_poses[pred_idx]

                # Project 3D pose to 2D (using X and Y, ignoring Z depth)
                # This gives us the frontal projection of the predicted pose
                pred_2d_raw = pred_3d[:, :2].copy()  # [17, 2] - just X, Y

                # Use Procrustes alignment to optimally rotate/scale/translate
                # the predicted 2D projection to match the GT 2D keypoints
                # This removes the hip rotation noise while preserving joint angle differences
                pred_2d_aligned = procrustes_align_2d(
                    source=pred_2d_raw,
                    target=gt_kp2d,
                    allow_scale=True,  # Allow scale to match GT skeleton size
                )

                frame = draw_skeleton(frame, pred_2d_aligned, pred_color, alpha=0.9)

        # Add legend
        font_scale = 0.7 if args.native_res else 0.5
        cv2.putText(frame, "GT (now+future)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, gt_color, 2)
        if not args.full_video:
            cv2.putText(frame, "Pred (now)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, pred_color, 2)
        cv2.putText(frame, f"f{t+1}", (out_w - 50, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
