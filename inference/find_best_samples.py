#!/usr/bin/env python3
"""
Find the best samples for visualization based on:
1. Large person size (head-to-toe length on screen)
2. Low prediction loss

Samples N videos, scores them, and generates rolling inference videos for top K.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.kinetics_dataset import (
    KineticsPoseDataset, joint_angles_to_pose3d, pose3d_to_joint_angles, JOINT_ANGLES_DIM
)
from training.vla_model import VLAModel, VLAConfig


@dataclass
class SampleScore:
    sample_idx: int
    clip_idx: int
    clip_id: str
    action_class: str
    person_height_ratio: float  # Person height as fraction of video height
    min_edge_margin: float  # Min distance from any joint to edge (fraction)
    fully_visible_ratio: float  # Fraction of frames where person is fully visible
    loss: float  # Joint angle RMSE in degrees
    combined_score: float  # Higher is better
    is_valid: bool  # Meets filtering criteria


def parse_args():
    ap = argparse.ArgumentParser(description="Find best samples for visualization")
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    ap.add_argument("--config", required=True, help="Path to training config YAML")
    ap.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate")
    ap.add_argument("--top-k", type=int, default=3, help="Number of best samples to visualize")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--device", default="cuda:0", help="Device for inference")
    ap.add_argument("--output-dir", default=str(Path(__file__).parent), help="Output directory for videos")
    ap.add_argument("--horizon-sec", type=float, default=0.8, help="Horizon in seconds")
    ap.add_argument("--inference-interval-sec", type=float, default=0.8, help="Inference interval in seconds")
    # Filtering criteria
    ap.add_argument("--min-height-ratio", type=float, default=0.30, help="Min person height as fraction of video height")
    ap.add_argument("--min-edge-margin", type=float, default=0.05, help="Min margin from edge as fraction (0.05 = 5%%)")
    ap.add_argument("--min-visible-ratio", type=float, default=0.80, help="Min fraction of frames where person is fully visible")
    ap.add_argument("--action-class", type=str, default=None, help="Filter to specific action class (e.g., 'eating spaghetti')")
    ap.add_argument("--exclude-clips", type=str, nargs='*', default=[], help="Clip IDs to exclude (for finding alternatives)")
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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    print("Model loaded!")
    return model


def compute_person_metrics(keypoints2d: np.ndarray, video_width: int, video_height: int) -> dict:
    """
    Compute person metrics for filtering good samples.

    Returns dict with:
    - person_height_ratio: person height as fraction of video height
    - min_edge_margin: minimum distance from any joint to any edge (as fraction)
    - fully_visible_ratio: fraction of frames where person is fully visible

    keypoints2d: [F, 17, 2] - 2D keypoints for all frames

    H36M joint indices:
    0: Pelvis, 1: R_Hip, 2: R_Knee, 3: R_Ankle, 4: L_Hip, 5: L_Knee, 6: L_Ankle
    7: Spine, 8: Thorax, 9: Neck, 10: Head
    11: L_Shoulder, 12: L_Elbow, 13: L_Wrist
    14: R_Shoulder, 15: R_Elbow, 16: R_Wrist
    """
    HEAD_IDX = 10
    L_ANKLE_IDX = 6
    R_ANKLE_IDX = 3

    # Edge margin threshold (fraction of dimension)
    EDGE_MARGIN_THRESHOLD = 0.05  # 5% from edge

    height_ratios = []
    edge_margins = []
    fully_visible_frames = 0
    valid_frames = 0

    for t in range(len(keypoints2d)):
        kp = keypoints2d[t]  # [17, 2]

        # Skip frames with too many NaN joints
        valid_joints = ~np.isnan(kp).any(axis=1)
        if valid_joints.sum() < 10:
            continue

        valid_frames += 1

        # Get valid keypoints
        valid_kp = kp[valid_joints]

        # Compute bounding box of person
        min_x, min_y = valid_kp.min(axis=0)
        max_x, max_y = valid_kp.max(axis=0)

        # Person height ratio
        person_height = max_y - min_y
        height_ratio = person_height / video_height
        height_ratios.append(height_ratio)

        # Edge margins (as fraction of respective dimension)
        margin_left = min_x / video_width
        margin_right = (video_width - max_x) / video_width
        margin_top = min_y / video_height
        margin_bottom = (video_height - max_y) / video_height

        min_margin = min(margin_left, margin_right, margin_top, margin_bottom)
        edge_margins.append(min_margin)

        # Check if fully visible (all margins > threshold)
        if min_margin > EDGE_MARGIN_THRESHOLD:
            fully_visible_frames += 1

    if valid_frames == 0:
        return {
            'person_height_ratio': 0.0,
            'min_edge_margin': 0.0,
            'fully_visible_ratio': 0.0,
            'avg_edge_margin': 0.0,
        }

    return {
        'person_height_ratio': np.mean(height_ratios),
        'min_edge_margin': np.min(edge_margins) if edge_margins else 0.0,
        'avg_edge_margin': np.mean(edge_margins) if edge_margins else 0.0,
        'fully_visible_ratio': fully_visible_frames / valid_frames,
    }


def evaluate_sample(
    model: VLAModel,
    val_dataset: KineticsPoseDataset,
    sample_idx: int,
    config: dict,
    device: str,
    min_height_ratio: float = 0.30,
    min_edge_margin: float = 0.05,
    min_visible_ratio: float = 0.80,
) -> SampleScore:
    """Evaluate a single sample for visibility and prediction loss."""
    clip_idx, start_frame = val_dataset.samples[sample_idx]
    clip = val_dataset.clips[clip_idx]

    # Load pose data
    data = np.load(clip.pose_path, allow_pickle=True)
    keypoints2d = data["keypoints2d"].astype(np.float32)
    pose3d = data["pose3d"].astype(np.float32)

    # Get video dimensions from the clip
    cap = cv2.VideoCapture(str(clip.video_path))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Compute person metrics
    metrics = compute_person_metrics(keypoints2d, video_width, video_height)

    # Check if sample meets filtering criteria
    is_valid = (
        metrics['person_height_ratio'] >= min_height_ratio and
        metrics['min_edge_margin'] >= min_edge_margin and
        metrics['fully_visible_ratio'] >= min_visible_ratio
    )

    # Get sample for inference
    sample = val_dataset[sample_idx]
    images = sample['images']
    instruction = sample['instruction']
    gt_actions = sample['actions']
    robot_state = sample['robot_state']

    action_horizon = config['model_config'].get('action_horizon', 16)
    action_dim = config['model_config'].get('action_dim', 51)
    use_joint_angles = action_dim == JOINT_ANGLES_DIM

    # Convert robot_state if needed
    if use_joint_angles:
        robot_state_3d = robot_state.reshape(17, 3)
        robot_state = pose3d_to_joint_angles(robot_state_3d)

    # Run inference
    img_arrays = [np.array(img) for img in images]
    with torch.no_grad():
        pred_actions = model.get_action(
            images=img_arrays,
            instruction=instruction,
            robot_state=robot_state,
        )

    # Compute loss
    gt_poses = gt_actions.reshape(action_horizon, 17, 3)

    if use_joint_angles:
        pred_angles = pred_actions.reshape(action_horizon, JOINT_ANGLES_DIM)
        gt_angles = np.array([pose3d_to_joint_angles(gt_poses[t]) for t in range(action_horizon)])
        mse = np.mean((gt_angles - pred_angles) ** 2)
        rmse_deg = np.degrees(np.sqrt(mse))
    else:
        mse = np.mean((gt_poses - pred_actions.reshape(action_horizon, 17, 3)) ** 2)
        rmse_deg = np.sqrt(mse) * 100  # Scale for comparison

    # Combined score: prioritize low loss for valid samples
    # For valid samples: higher score = lower loss
    # For invalid samples: score = 0 (filtered out)
    if is_valid:
        combined_score = 100.0 / (rmse_deg + 1.0)  # Higher for lower loss
    else:
        combined_score = 0.0

    return SampleScore(
        sample_idx=sample_idx,
        clip_idx=clip_idx,
        clip_id=clip.clip_id,
        action_class=clip.action_class,
        person_height_ratio=metrics['person_height_ratio'],
        min_edge_margin=metrics['min_edge_margin'],
        fully_visible_ratio=metrics['fully_visible_ratio'],
        loss=rmse_deg,
        combined_score=combined_score,
        is_valid=is_valid,
    )


def generate_rolling_video(
    args,
    config: dict,
    model: VLAModel,
    val_dataset: KineticsPoseDataset,
    score: SampleScore,
    output_path: Path,
):
    """Generate rolling inference video for a sample."""
    from compare_gt_pred import run_rolling_inference

    clip = val_dataset.clips[score.clip_idx]

    # Create a namespace-like object with required attributes
    class Args:
        pass

    video_args = Args()
    video_args.gt_color = "0,200,255"
    video_args.pred_color = "255,100,100"
    video_args.native_res = False
    video_args.output = str(output_path)
    video_args.horizon_sec = args.horizon_sec
    video_args.inference_interval_sec = args.inference_interval_sec

    run_rolling_inference(video_args, config, model, clip, val_dataset)


def main():
    args = parse_args()
    config = load_config(args.config)

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    model = load_model(args.checkpoint, config, device)

    action_horizon = config['model_config'].get('action_horizon', 16)
    num_frames = config['model_config'].get('num_frames', 4)
    dc = config['dataset']

    print("Loading validation dataset...")
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
        use_joint_angles=False,
        seed=dc.get('seed', 42),
    )

    print(f"Validation dataset: {len(val_dataset)} samples")

    # Filter indices by action class if specified
    all_indices = np.arange(len(val_dataset))

    if args.action_class:
        # Filter to samples from the specified action class
        filtered_indices = []
        for idx in all_indices:
            clip_idx, _ = val_dataset.samples[idx]
            clip = val_dataset.clips[clip_idx]
            if clip.action_class.lower() == args.action_class.lower():
                # Also check if clip should be excluded
                if clip.clip_id not in args.exclude_clips:
                    filtered_indices.append(idx)
        all_indices = np.array(filtered_indices)
        print(f"Filtered to {len(all_indices)} samples from action class: '{args.action_class}'")
        if args.exclude_clips:
            print(f"Excluding clips: {args.exclude_clips}")

    sample_indices = np.random.choice(all_indices, size=min(args.num_samples, len(all_indices)), replace=False)

    print(f"\nEvaluating {len(sample_indices)} samples (seed={args.seed})...")
    print(f"Filtering criteria:")
    print(f"  - Person height >= {args.min_height_ratio*100:.0f}% of video height")
    print(f"  - Edge margin >= {args.min_edge_margin*100:.0f}% (no joints near edges)")
    print(f"  - Fully visible in >= {args.min_visible_ratio*100:.0f}% of frames")

    scores: List[SampleScore] = []
    for i, idx in enumerate(sample_indices):
        try:
            score = evaluate_sample(
                model, val_dataset, idx, config, device,
                min_height_ratio=args.min_height_ratio,
                min_edge_margin=args.min_edge_margin,
                min_visible_ratio=args.min_visible_ratio,
            )
            scores.append(score)

            if (i + 1) % 10 == 0:
                valid_count = sum(1 for s in scores if s.is_valid)
                print(f"  Evaluated {i+1}/{len(sample_indices)} samples ({valid_count} valid so far)")
        except Exception as e:
            print(f"  Error evaluating sample {idx}: {e}")
            continue

    # Filter to valid samples and sort by loss (lower is better)
    valid_scores = [s for s in scores if s.is_valid]
    valid_scores.sort(key=lambda s: s.loss)  # Sort by loss ascending

    print(f"\n{'='*80}")
    print(f"Results: {len(valid_scores)}/{len(scores)} samples passed filtering criteria")
    print(f"{'='*80}")

    if len(valid_scores) == 0:
        print("\nNo samples passed filtering! Try relaxing criteria:")
        print("  --min-height-ratio (lower = smaller person ok)")
        print("  --min-edge-margin (lower = closer to edge ok)")
        print("  --min-visible-ratio (lower = more cropped frames ok)")

        # Show stats anyway
        print(f"\nStatistics across all {len(scores)} evaluated samples:")
        print(f"  Height ratio: min={min(s.person_height_ratio for s in scores):.2f}, "
              f"max={max(s.person_height_ratio for s in scores):.2f}, "
              f"mean={np.mean([s.person_height_ratio for s in scores]):.2f}")
        print(f"  Min edge margin: min={min(s.min_edge_margin for s in scores):.2f}, "
              f"max={max(s.min_edge_margin for s in scores):.2f}, "
              f"mean={np.mean([s.min_edge_margin for s in scores]):.2f}")
        print(f"  Fully visible ratio: min={min(s.fully_visible_ratio for s in scores):.2f}, "
              f"max={max(s.fully_visible_ratio for s in scores):.2f}, "
              f"mean={np.mean([s.fully_visible_ratio for s in scores]):.2f}")
        return

    print(f"\nTop {min(args.top_k, len(valid_scores))} valid samples (sorted by loss):")

    for i, score in enumerate(valid_scores[:args.top_k]):
        print(f"\n{i+1}. {score.action_class}/{score.clip_id}")
        print(f"   Sample index: {score.sample_idx}")
        print(f"   Loss: {score.loss:.2f}° RMSE")
        print(f"   Height ratio: {score.person_height_ratio*100:.1f}%")
        print(f"   Min edge margin: {score.min_edge_margin*100:.1f}%")
        print(f"   Fully visible: {score.fully_visible_ratio*100:.1f}%")

    # Statistics for valid samples
    print(f"\n{'='*80}")
    print(f"Statistics across {len(valid_scores)} valid samples:")
    print(f"  Loss: min={min(s.loss for s in valid_scores):.2f}°, "
          f"max={max(s.loss for s in valid_scores):.2f}°, "
          f"mean={np.mean([s.loss for s in valid_scores]):.2f}°")
    print(f"  Height ratio: min={min(s.person_height_ratio for s in valid_scores)*100:.1f}%, "
          f"max={max(s.person_height_ratio for s in valid_scores)*100:.1f}%, "
          f"mean={np.mean([s.person_height_ratio for s in valid_scores])*100:.1f}%")

    # Generate videos for top K valid samples
    print(f"\n{'='*80}")
    print(f"Generating rolling inference videos for top {min(args.top_k, len(valid_scores))} samples...")
    print(f"{'='*80}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, score in enumerate(valid_scores[:args.top_k]):
        output_path = output_dir / f"best_{i+1}_{score.action_class.replace(' ', '_')}_{score.clip_id}.mp4"
        print(f"\nGenerating video {i+1}/{min(args.top_k, len(valid_scores))}: {output_path.name}")
        print(f"  Action: {score.action_class}, Loss: {score.loss:.2f}°, Height: {score.person_height_ratio*100:.1f}%")

        generate_rolling_video(args, config, model, val_dataset, score, output_path)

    print(f"\nDone! Videos saved to {output_dir}")


if __name__ == "__main__":
    main()
