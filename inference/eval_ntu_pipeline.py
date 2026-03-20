#!/usr/bin/env python3
"""
Evaluate our 3D pose extraction pipeline against NTU RGB+D 120 ground truth skeletons.

Computes MPJPE, PA-MPJPE, and PCK metrics on a sample of NTU videos by:
1. Running our pipeline (ViTPose 2D → MotionAGFormer 3D) on NTU RGB videos
2. Parsing NTU Kinect skeleton ground truth
3. Mapping between NTU 25-joint and H3.6M 17-joint formats
4. Computing standard pose estimation metrics

Usage:
    python inference/eval_ntu_pipeline.py --device cuda:0 --num-samples 200
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "MotionAGFormer"))


# ─── NTU skeleton parsing ────────────────────────────────────────────────────

def parse_ntu_skeleton(skeleton_path: str) -> np.ndarray:
    """
    Parse an NTU RGB+D .skeleton file.

    NTU format (per frame):
      line 0: num_bodies
      per body:
        line 0: body info (10 fields)
        line 1: num_joints (25)
        lines 2-26: joint data (x y z depthX depthY colorX colorY qw qx qy qz trackingState)

    Returns:
        poses: (F, 25, 3) float32 array of 3D joint positions (x, y, z in meters)
        Returns empty array if parsing fails.
    """
    try:
        with open(skeleton_path, "r") as f:
            lines = [l.strip() for l in f.readlines()]
    except Exception:
        return np.zeros((0, 25, 3), dtype=np.float32)

    idx = 0
    num_frames = int(lines[idx]); idx += 1
    all_frames = []

    for _ in range(num_frames):
        num_bodies = int(lines[idx]); idx += 1

        best_body = None
        best_tracking = -1

        for _ in range(num_bodies):
            body_info = lines[idx].split(); idx += 1
            num_joints = int(lines[idx]); idx += 1

            joints = np.zeros((num_joints, 3), dtype=np.float32)
            tracking_sum = 0
            for j in range(num_joints):
                parts = lines[idx].split(); idx += 1
                joints[j, 0] = float(parts[0])  # x
                joints[j, 1] = float(parts[1])  # y
                joints[j, 2] = float(parts[2])  # z
                tracking_sum += int(parts[-1])  # tracking state

            # Select body with best tracking quality
            if tracking_sum > best_tracking:
                best_tracking = tracking_sum
                best_body = joints

        if best_body is not None:
            all_frames.append(best_body)

    if len(all_frames) == 0:
        return np.zeros((0, 25, 3), dtype=np.float32)

    return np.stack(all_frames, axis=0)


# ─── Joint mapping: NTU 25 → H3.6M 17 ──────────────────────────────────────

# NTU Kinect v2 joint indices:
#  0: base spine (pelvis)   1: mid spine       2: neck          3: head
#  4: left shoulder          5: left elbow      6: left wrist    7: left hand
#  8: right shoulder         9: right elbow    10: right wrist  11: right hand
# 12: left hip              13: left knee      14: left ankle   15: left foot
# 16: right hip             17: right knee     18: right ankle  19: right foot
# 20: spine shoulder        21: left hand tip  22: left thumb
# 23: right hand tip        24: right thumb

# H3.6M 17-joint format:
#  0: pelvis    1: right hip   2: right knee   3: right ankle
#  4: left hip  5: left knee   6: left ankle   7: spine
#  8: neck      9: jaw/nose   10: head top
# 11: left shoulder  12: left elbow  13: left wrist
# 14: right shoulder 15: right elbow 16: right wrist

NTU_TO_H36M = {
    0: 0,    # pelvis → pelvis
    16: 1,   # right hip → right hip
    17: 2,   # right knee → right knee
    18: 3,   # right ankle → right ankle
    12: 4,   # left hip → left hip
    13: 5,   # left knee → left knee
    14: 6,   # left ankle → left ankle
    20: 7,   # spine shoulder → spine (closest match)
    2: 8,    # neck → neck
    3: 9,    # head → jaw/nose (approximate)
    # 10: head top — no exact NTU match, use head (3) again
    4: 11,   # left shoulder → left shoulder
    5: 12,   # left elbow → left elbow
    6: 13,   # left wrist → left wrist
    8: 14,   # right shoulder → right shoulder
    9: 15,   # right elbow → right elbow
    10: 16,  # right wrist → right wrist
}

# Joints to evaluate (skip head top = index 10, since NTU has no separate head top)
EVAL_H36M_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]  # 16 of 17


def ntu_to_h36m(ntu_poses: np.ndarray) -> np.ndarray:
    """Convert NTU 25-joint poses to H3.6M 17-joint format.

    Args:
        ntu_poses: (F, 25, 3) array
    Returns:
        h36m_poses: (F, 17, 3) array
    """
    F = ntu_poses.shape[0]
    h36m = np.zeros((F, 17, 3), dtype=np.float32)
    for ntu_idx, h36m_idx in NTU_TO_H36M.items():
        h36m[:, h36m_idx] = ntu_poses[:, ntu_idx]
    # Head top (10): use head joint as approximation
    h36m[:, 10] = ntu_poses[:, 3]
    return h36m


# ─── Metrics ─────────────────────────────────────────────────────────────────

def mpjpe(pred: np.ndarray, gt: np.ndarray, joint_mask=None) -> float:
    """Mean Per Joint Position Error in mm.

    Args:
        pred: (F, J, 3) predicted poses
        gt: (F, J, 3) ground truth poses
        joint_mask: optional list of joint indices to evaluate
    """
    if joint_mask is not None:
        pred = pred[:, joint_mask]
        gt = gt[:, joint_mask]
    errors = np.linalg.norm(pred - gt, axis=-1)  # (F, J)
    return float(np.mean(errors) * 1000)  # Convert to mm


def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Procrustes alignment (rotation, translation, scale) per frame.

    Args:
        pred: (J, 3) predicted joints
        gt: (J, 3) ground truth joints
    Returns:
        aligned_pred: (J, 3)
    """
    mu_pred = pred.mean(axis=0, keepdims=True)
    mu_gt = gt.mean(axis=0, keepdims=True)

    pred_c = pred - mu_pred
    gt_c = gt - mu_gt

    # Scale
    s_pred = np.sqrt(np.sum(pred_c ** 2))
    s_gt = np.sqrt(np.sum(gt_c ** 2))

    if s_pred < 1e-8 or s_gt < 1e-8:
        return pred

    pred_n = pred_c / s_pred
    gt_n = gt_c / s_gt

    # Rotation via SVD
    H = pred_n.T @ gt_n
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    scale = s_gt * np.sum(S)  # optimal scale
    aligned = (pred_c @ R) * (scale / s_pred) + mu_gt
    return aligned


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray, joint_mask=None) -> float:
    """Procrustes-Aligned MPJPE in mm."""
    if joint_mask is not None:
        pred = pred[:, joint_mask]
        gt = gt[:, joint_mask]

    errors = []
    for f in range(pred.shape[0]):
        aligned = procrustes_align(pred[f], gt[f])
        err = np.linalg.norm(aligned - gt[f], axis=-1)
        errors.append(err)
    errors = np.concatenate(errors)
    return float(np.mean(errors) * 1000)


def joint_angle_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Joint angle RMSE in degrees between predicted and GT 3D poses.

    Converts both pose sequences to 22-DoF joint angles via inverse kinematics,
    then computes RMSE in degrees.

    Args:
        pred: (F, 17, 3) predicted poses (root-centered)
        gt: (F, 17, 3) ground truth poses (root-centered)
    Returns:
        RMSE in degrees
    """
    from training.kinetics_dataset import pose3d_to_joint_angles
    F = pred.shape[0]
    pred_angles = np.array([pose3d_to_joint_angles(pred[f]) for f in range(F)])
    gt_angles = np.array([pose3d_to_joint_angles(gt[f]) for f in range(F)])
    rmse_rad = np.sqrt(np.mean((pred_angles - gt_angles) ** 2))
    return float(np.degrees(rmse_rad))


def pck(pred: np.ndarray, gt: np.ndarray, threshold_mm=150, joint_mask=None) -> float:
    """Percentage of Correct Keypoints within threshold (mm)."""
    if joint_mask is not None:
        pred = pred[:, joint_mask]
        gt = gt[:, joint_mask]
    threshold = threshold_mm / 1000.0  # Convert to meters (NTU uses meters)
    errors = np.linalg.norm(pred - gt, axis=-1)  # (F, J)
    return float(np.mean(errors < threshold) * 100)


# ─── Pipeline runner ─────────────────────────────────────────────────────────

def load_pipeline_models(device: str):
    """Load all pipeline models (YOLO, ViTPose, MotionAGFormer)."""
    from ultralytics import YOLO
    from transformers import AutoImageProcessor, VitPoseForPoseEstimation
    from data_prep.pose3d import build_motionagformer
    from model.MotionAGFormer import MotionAGFormer

    print("Loading YOLOv8x...")
    det_model = YOLO("yolov8x.pt")

    print("Loading ViTPose-Large...")
    vitpose_model_id = "usyd-community/vitpose-plus-large"
    vitpose_processor = AutoImageProcessor.from_pretrained(
        vitpose_model_id, trust_remote_code=True
    )
    vitpose_model = VitPoseForPoseEstimation.from_pretrained(
        vitpose_model_id, trust_remote_code=True, torch_dtype=torch.float32
    )
    vitpose_model.to(device)
    vitpose_model.eval()

    print("Loading MotionAGFormer...")
    ckpt_path = PROJECT_ROOT / "models" / "motionagformer-b-h36m.pth.tr"
    model_3d = build_motionagformer(MotionAGFormer, device, ckpt_path)

    return det_model, vitpose_processor, vitpose_model, model_3d


def run_pipeline_on_video(
    video_path: str,
    det_model,
    vitpose_processor,
    vitpose_model,
    model_3d,
    device: str,
) -> np.ndarray:
    """Run our pipeline on a video and return 3D poses (F, 17, 3).

    Calls components directly: YOLO detection → ViTPose 2D → MotionAGFormer 3D.
    Returns empty array on failure.
    """
    from data_prep.fast_video_loader import probe_video_meta_fast, read_frames_batch_fast, sample_indices_for_fps
    from data_prep.vitpose import infer_sequence
    from data_prep.keypoints import h36m_coco_format, flip_magformer
    from data_prep.pose3d import lift_sequence_to_3d
    from torchvision.transforms.functional import to_tensor

    try:
        # 1. Load video frames at native FPS (NTU is 30fps)
        meta = probe_video_meta_fast(video_path)
        target_fps = min(meta["fps"], 30.0)
        idxs = sample_indices_for_fps(meta["frames"], meta["fps"], target_fps=target_fps)
        frames = read_frames_batch_fast(video_path, idxs, target_fps=target_fps)
        if frames.shape[0] < 10:
            return np.zeros((0, 17, 3), dtype=np.float32)

        # 2. Run YOLO detection per frame (simple: take largest person box)
        T = frames.shape[0]
        boxes_xyxy = np.full((T, 4), np.nan, dtype=np.float32)

        batch_size = 16
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            batch_frames = frames[start:end]
            # YOLO expects BGR numpy arrays
            batch_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in batch_frames]
            results = det_model(batch_bgr, classes=[0], verbose=False)
            for j, result in enumerate(results):
                if result.boxes is not None and len(result.boxes) > 0:
                    person_boxes = result.boxes.xyxy.cpu().numpy()
                    # Take largest box (most prominent person)
                    areas = (person_boxes[:, 2] - person_boxes[:, 0]) * \
                            (person_boxes[:, 3] - person_boxes[:, 1])
                    best = np.argmax(areas)
                    boxes_xyxy[start + j] = person_boxes[best]

        # Fill NaN boxes with nearest valid box
        valid_mask = ~np.isnan(boxes_xyxy[:, 0])
        if valid_mask.sum() < T * 0.5:
            return np.zeros((0, 17, 3), dtype=np.float32)
        if not valid_mask.all():
            valid_indices = np.where(valid_mask)[0]
            for t in range(T):
                if not valid_mask[t]:
                    nearest = valid_indices[np.argmin(np.abs(valid_indices - t))]
                    boxes_xyxy[t] = boxes_xyxy[nearest]

        # 3. Run ViTPose for 2D keypoints
        kpts, scrs = infer_sequence(
            raw_video_path=video_path,
            image_processor=vitpose_processor,
            model=vitpose_model,
            device=torch.device(device),
            idxs=idxs,
            boxes_xyxy=boxes_xyxy,
            frames=frames,
            batch_size=16,
        )

        # 4. Convert COCO → H3.6M format
        h36m_kpts, h36m_scores, valid_frames = h36m_coco_format(kpts, scrs)
        if h36m_kpts.shape[0] == 0:
            return np.zeros((0, 17, 3), dtype=np.float32)

        seq_k = h36m_kpts[0]   # (F, 17, 2)
        seq_s = h36m_scores[0]  # (F, 17)

        # 5. Lift to 3D via MotionAGFormer
        y3d = lift_sequence_to_3d(
            seq_k[None, ...], seq_s[None, ...],
            meta["width"], meta["height"],
            model_3d, device
        )
        return y3d  # (F, 17, 3)

    except Exception as e:
        print(f"  Pipeline error: {e}")
        return np.zeros((0, 17, 3), dtype=np.float32)


# ─── Main evaluation ─────────────────────────────────────────────────────────

def find_ntu_samples(
    video_dir: str,
    skeleton_dir: str,
    num_samples: int = 200,
    seed: int = 42,
) -> list:
    """Find paired NTU video + skeleton files."""
    skel_dir = Path(skeleton_dir)
    vid_base = Path(video_dir)

    # Get all skeleton files
    skel_files = sorted(skel_dir.glob("*.skeleton"))
    print(f"Found {len(skel_files)} skeleton files")

    # Find matching videos
    pairs = []
    for skel_path in skel_files:
        # S001C001P001R001A001.skeleton → S001C001P001R001A001_rgb.avi
        stem = skel_path.stem
        session = stem[1:4]  # e.g., "001"

        # Try both possible locations
        vid_path = vid_base / f"nturgb+d_rgb_s{session}" / "nturgb+d_rgb" / f"{stem}_rgb.avi"
        if not vid_path.exists():
            vid_path = vid_base / "nturgb+d_rgb" / f"{stem}_rgb.avi"
        if not vid_path.exists():
            continue

        pairs.append((str(vid_path), str(skel_path)))

    print(f"Found {len(pairs)} video-skeleton pairs")

    # Random sample
    rng = np.random.RandomState(seed)
    if len(pairs) > num_samples:
        indices = rng.choice(len(pairs), num_samples, replace=False)
        pairs = [pairs[i] for i in sorted(indices)]

    return pairs


def align_sequences(
    pred_3d: np.ndarray,
    gt_3d: np.ndarray,
) -> tuple:
    """Align predicted and GT sequences to common length and root.

    Both are root-centered (pelvis at origin).
    Predicted may have different frame count than GT due to FPS differences.

    Returns:
        pred_aligned: (F, 17, 3)
        gt_aligned: (F, 17, 3)
    """
    F_pred = pred_3d.shape[0]
    F_gt = gt_3d.shape[0]

    # Use shorter sequence length
    F = min(F_pred, F_gt)
    if F == 0:
        return np.zeros((0, 17, 3)), np.zeros((0, 17, 3))

    # If lengths differ significantly, resample the longer one
    if F_pred != F_gt:
        # Resample pred to match GT length (GT is authoritative)
        indices = np.linspace(0, F_pred - 1, F_gt).astype(int)
        pred_3d = pred_3d[indices]
        F = F_gt

    pred = pred_3d[:F]
    gt = gt_3d[:F]

    # Root-center both (pelvis = joint 0)
    pred = pred - pred[:, 0:1, :]
    gt = gt - gt[:, 0:1, :]

    return pred, gt


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline against NTU RGB+D 120")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--video-dir", default="/root/movement/data/extracted")
    parser.add_argument("--skeleton-dir",
                        default="/root/movement/data/extracted_skeletons/nturgb+d_skeletons")
    parser.add_argument("--output", default="/root/movement/inference/ntu_eval_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"=== NTU RGB+D Pipeline Evaluation ===")
    print(f"Device: {args.device}")
    print(f"Samples: {args.num_samples}")

    # Find sample pairs
    pairs = find_ntu_samples(
        args.video_dir, args.skeleton_dir, args.num_samples, args.seed
    )
    if not pairs:
        print("ERROR: No video-skeleton pairs found!")
        return

    print(f"Evaluating {len(pairs)} samples")

    # Load models
    det_model, vitpose_proc, vitpose_model, model_3d = load_pipeline_models(args.device)
    print("All models loaded\n")

    # Run evaluation
    all_mpjpe = []
    all_pa_mpjpe = []
    all_pck150 = []
    all_angle_rmse = []
    per_joint_errors = []
    failures = 0
    t_start = time.time()

    for i, (vid_path, skel_path) in enumerate(pairs):
        stem = Path(vid_path).stem.replace("_rgb", "")
        elapsed = time.time() - t_start
        rate = (i + 1) / max(elapsed, 1)
        eta = (len(pairs) - i - 1) / max(rate, 0.01)

        print(f"[{i+1}/{len(pairs)}] {stem}  ({rate:.1f} samples/s, ETA {eta:.0f}s)", end="")

        # Parse GT skeleton
        ntu_gt = parse_ntu_skeleton(skel_path)  # (F, 25, 3)
        if ntu_gt.shape[0] < 10:
            print("  SKIP (GT too short)")
            failures += 1
            continue

        # Convert GT to H3.6M format
        gt_h36m = ntu_to_h36m(ntu_gt)  # (F, 17, 3)

        # Run our pipeline
        pred_h36m = run_pipeline_on_video(
            vid_path, det_model, vitpose_proc, vitpose_model, model_3d, args.device
        )
        if pred_h36m.shape[0] < 10:
            print("  FAIL (pipeline returned too few frames)")
            failures += 1
            continue

        # Align sequences
        pred_aligned, gt_aligned = align_sequences(pred_h36m, gt_h36m)
        if pred_aligned.shape[0] < 10:
            print("  FAIL (alignment too short)")
            failures += 1
            continue

        # Compute metrics on eval joints (exclude head top)
        m = mpjpe(pred_aligned, gt_aligned, EVAL_H36M_INDICES)
        pa = pa_mpjpe(pred_aligned, gt_aligned, EVAL_H36M_INDICES)
        p = pck(pred_aligned, gt_aligned, threshold_mm=150, joint_mask=EVAL_H36M_INDICES)

        # Joint angle RMSE
        ja = joint_angle_rmse(pred_aligned, gt_aligned)

        all_mpjpe.append(m)
        all_pa_mpjpe.append(pa)
        all_pck150.append(p)
        all_angle_rmse.append(ja)

        # Per-joint errors for this sample
        per_frame_errors = np.linalg.norm(
            pred_aligned[:, EVAL_H36M_INDICES] - gt_aligned[:, EVAL_H36M_INDICES], axis=-1
        ) * 1000  # (F, 16) in mm
        per_joint_errors.append(np.mean(per_frame_errors, axis=0))

        print(f"  MPJPE={m:.1f}mm  PA-MPJPE={pa:.1f}mm  AngleRMSE={ja:.1f}°  PCK@150={p:.1f}%")

    elapsed = time.time() - t_start

    # Aggregate results
    if len(all_mpjpe) == 0:
        print("\nNo successful evaluations!")
        return

    per_joint_mean = np.mean(per_joint_errors, axis=0)  # (16,)
    joint_names = [
        "Pelvis", "R.Hip", "R.Knee", "R.Ankle",
        "L.Hip", "L.Knee", "L.Ankle", "Spine",
        "Neck", "Head",
        "L.Shoulder", "L.Elbow", "L.Wrist",
        "R.Shoulder", "R.Elbow", "R.Wrist",
    ]

    results = {
        "num_samples": len(all_mpjpe),
        "num_failures": failures,
        "total_time_s": elapsed,
        "mpjpe_mm": {
            "mean": float(np.mean(all_mpjpe)),
            "median": float(np.median(all_mpjpe)),
            "std": float(np.std(all_mpjpe)),
        },
        "pa_mpjpe_mm": {
            "mean": float(np.mean(all_pa_mpjpe)),
            "median": float(np.median(all_pa_mpjpe)),
            "std": float(np.std(all_pa_mpjpe)),
        },
        "pck_at_150mm": {
            "mean": float(np.mean(all_pck150)),
        },
        "joint_angle_rmse_deg": {
            "mean": float(np.mean(all_angle_rmse)),
            "median": float(np.median(all_angle_rmse)),
            "std": float(np.std(all_angle_rmse)),
        },
        "per_joint_mpjpe_mm": {
            name: float(err) for name, err in zip(joint_names, per_joint_mean)
        },
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS ({results['num_samples']} samples, {failures} failures)")
    print(f"{'='*60}")
    print(f"  MPJPE:      {results['mpjpe_mm']['mean']:.1f} ± {results['mpjpe_mm']['std']:.1f} mm  (median {results['mpjpe_mm']['median']:.1f})")
    print(f"  PA-MPJPE:   {results['pa_mpjpe_mm']['mean']:.1f} ± {results['pa_mpjpe_mm']['std']:.1f} mm  (median {results['pa_mpjpe_mm']['median']:.1f})")
    print(f"  PCK@150mm:  {results['pck_at_150mm']['mean']:.1f}%")
    print(f"  Angle RMSE: {results['joint_angle_rmse_deg']['mean']:.1f} ± {results['joint_angle_rmse_deg']['std']:.1f}°  (median {results['joint_angle_rmse_deg']['median']:.1f}°)")
    print(f"\n  Per-joint MPJPE (mm):")
    for name, err in zip(joint_names, per_joint_mean):
        print(f"    {name:15s}  {err:.1f}")
    print(f"\n  Time: {elapsed:.0f}s ({elapsed/len(pairs):.1f}s/sample)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
