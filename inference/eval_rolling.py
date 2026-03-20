#!/usr/bin/env python3
"""
Evaluate model vs baselines with rolling inference at different step sizes.

For each step size S, all methods get re-initialized with ground truth every S
frames. This is a fair comparison: the model, static baseline, and linear
extrapolation baseline all receive the same information at the same times.

Metrics are computed over all 16 frames (including boundary frames where GT
is provided). With smaller step sizes, all methods benefit equally from more
frequent re-initialization.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.kinetics_dataset import (
    pose3d_to_joint_angles, joint_angles_to_sincos, sincos_to_joint_angles,
    _parse_description,
)
from inference.paper_figure import (
    load_model, load_val_dataset, CHECKPOINT, DATASET_CONFIG, MODEL_CONFIG,
)

ACTION_HORIZON = 16
NUM_FRAMES = MODEL_CONFIG["num_frames"]  # 4
RESIZE = DATASET_CONFIG["image_size"]    # 224
STEP_SIZES = [1, 2, 3, 5, 8]
NUM_SAMPLES = 500
SEED = 42


def load_video_frames(clip, poses3d, pose_indices, start_frame):
    """Pre-load all video frames needed for rolling inference.

    For each offset t in 0..15, we need NUM_FRAMES images spanning from
    (start_frame+t) to (start_frame+t+remaining-1). Pre-read all unique
    video frames once to avoid repeated seeks.
    """
    # Collect all unique video frame indices we'll need
    needed_vid_indices = set()
    for t in range(ACTION_HORIZON):
        curr = start_frame + t
        remaining = min(ACTION_HORIZON, len(poses3d) - curr)
        if remaining <= 0:
            continue
        fi = np.linspace(curr, curr + remaining - 1, NUM_FRAMES, dtype=int)
        for f_idx in fi:
            pi = min(int(f_idx), len(pose_indices) - 1)
            needed_vid_indices.add(int(pose_indices[pi]))

    # Read all needed video frames at once
    frame_cache = {}
    cap = cv2.VideoCapture(str(clip.video_path))
    for vid_idx in sorted(needed_vid_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
        ok, fr = cap.read()
        if ok:
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            frame_cache[vid_idx] = np.array(Image.fromarray(rgb).resize(
                (RESIZE, RESIZE), Image.BILINEAR))
    cap.release()
    return frame_cache


def get_images_at(frame_cache, poses3d, pose_indices, start_offset):
    """Get model input images from pre-loaded cache."""
    remaining = min(ACTION_HORIZON, len(poses3d) - start_offset)
    fi = np.linspace(start_offset, start_offset + remaining - 1, NUM_FRAMES, dtype=int)

    imgs = []
    fallback = np.zeros((RESIZE, RESIZE, 3), dtype=np.uint8)
    for f_idx in fi:
        pi = min(int(f_idx), len(pose_indices) - 1)
        vid_idx = int(pose_indices[pi])
        img = frame_cache.get(vid_idx, fallback)
        imgs.append(img)

    while len(imgs) < NUM_FRAMES:
        imgs.append(imgs[-1] if imgs else fallback)

    return [Image.fromarray(im) for im in imgs]


def get_instruction(clip):
    """Get the text instruction for a clip."""
    try:
        desc_body = _parse_description(clip.desc_path)
        return f"Task: {clip.action_class}. Instruction: {desc_body}"
    except Exception:
        return f"A person performing {clip.action_class}"


def run_model_at_offset(model, frame_cache, poses3d, pose_indices,
                        instruction, start, offset, device):
    """Run model inference at (start + offset), return pred angles (16, 22)."""
    curr = start + offset
    imgs = get_images_at(frame_cache, poses3d, pose_indices, curr)

    rs = poses3d[curr].copy() - poses3d[curr][0:1]
    rs_a = pose3d_to_joint_angles(rs)
    rs_sc = joint_angles_to_sincos(rs_a.reshape(1, -1))[0]
    rs_t = torch.from_numpy(rs_sc).float().unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.forward([imgs], [instruction], robot_state=rs_t,
                                compute_loss=False)
        pred_sc = out["actions"].squeeze(0).cpu().numpy()

    return np.array([sincos_to_joint_angles(pred_sc[t])
                     for t in range(pred_sc.shape[0])])


def assemble_rolling_model(model_cache, step_size):
    """Assemble rolling prediction from cached model outputs."""
    pred = np.zeros((ACTION_HORIZON, 22))
    t = 0
    while t < ACTION_HORIZON:
        out = model_cache[t]  # (16, 22) predictions starting from offset t
        for s in range(min(step_size, ACTION_HORIZON - t)):
            pred[t + s] = out[s]
        t += step_size
    return pred


def rolling_static(gt_angles, step_size):
    """Rolling static baseline: copy GT at each step boundary."""
    pred = np.zeros_like(gt_angles)
    t = 0
    while t < ACTION_HORIZON:
        for s in range(min(step_size, ACTION_HORIZON - t)):
            pred[t + s] = gt_angles[t]
        t += step_size
    return pred


def rolling_linear(gt_angles, step_size, prev_angles=None):
    """Rolling linear extrapolation baseline."""
    pred = np.zeros_like(gt_angles)
    t = 0
    while t < ACTION_HORIZON:
        current = gt_angles[t]
        if t > 0:
            velocity = gt_angles[t] - gt_angles[t - 1]
        elif prev_angles is not None:
            velocity = gt_angles[0] - prev_angles
        else:
            velocity = np.zeros_like(current)

        for s in range(min(step_size, ACTION_HORIZON - t)):
            pred[t + s] = current + velocity * s
        t += step_size
    return pred


def compute_motion(gt_angles):
    """Total angular displacement across the action horizon."""
    return float(np.sum(np.abs(np.diff(gt_angles, axis=0))))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--step-sizes", type=int, nargs="+", default=None,
                        help="Override step sizes (default: 1 2 3 5 8)")
    parser.add_argument("--output", default="inference/eval_rolling_results.json")
    args = parser.parse_args()

    step_sizes = args.step_sizes if args.step_sizes else STEP_SIZES

    np.random.seed(SEED)
    device = args.device

    print("Loading dataset...", flush=True)
    dataset = load_val_dataset()
    total_val = len(dataset)
    print(f"Validation set: {total_val} samples", flush=True)

    num_samples = min(args.num_samples, total_val)
    indices = np.random.choice(total_val, size=num_samples, replace=False)
    indices.sort()
    print(f"Evaluating {num_samples} samples with step sizes {step_sizes}",
          flush=True)

    print("Loading model...", flush=True)
    model = load_model(CHECKPOINT, device)

    # Find all unique offsets needed across all step sizes
    needed_offsets = set()
    for S in step_sizes:
        t = 0
        while t < ACTION_HORIZON:
            needed_offsets.add(t)
            t += S
    needed_offsets = sorted(needed_offsets)
    print(f"Unique model call offsets: {needed_offsets} "
          f"({len(needed_offsets)} calls per sample)", flush=True)

    # Storage: errors[step_size][method] = list of (16, 22) error arrays
    errors = {s: {"model": [], "static": [], "linear": []} for s in step_sizes}
    motions = []
    failures = 0
    total_model_calls = 0

    t0 = time.time()
    for i, idx in enumerate(indices):
        idx = int(idx)
        clip_idx, start_frame = dataset.samples[idx]
        clip = dataset.clips[clip_idx]

        try:
            data = np.load(clip.pose_path, allow_pickle=True)
            poses3d = data["pose3d"].astype(np.float32)
            pose_indices_arr = data["indices"].astype(np.int32)
        except Exception as e:
            failures += 1
            if failures <= 5:
                print(f"  Sample {idx} failed to load: {e}", flush=True)
            continue

        end = start_frame + ACTION_HORIZON
        if end > len(poses3d) or end > len(pose_indices_arr):
            failures += 1
            continue

        # GT angles
        gt_norm = poses3d[start_frame:end].copy() - poses3d[start_frame][0:1]
        gt_angles = np.array([pose3d_to_joint_angles(gt_norm[t])
                              for t in range(ACTION_HORIZON)])

        # Previous frame for linear baseline
        if start_frame >= 1:
            prev_norm = (poses3d[start_frame - 1].copy()
                         - poses3d[start_frame - 1][0:1])
            prev_angles = pose3d_to_joint_angles(prev_norm)
        else:
            prev_angles = None

        instruction = get_instruction(clip)
        motions.append(compute_motion(gt_angles))

        # Pre-load video frames for this sample
        frame_cache = load_video_frames(
            clip, poses3d, pose_indices_arr, start_frame)

        # Run model at all needed offsets (cached)
        try:
            model_cache = {}
            for offset in needed_offsets:
                model_cache[offset] = run_model_at_offset(
                    model, frame_cache, poses3d, pose_indices_arr,
                    instruction, start_frame, offset, device)
                total_model_calls += 1
        except Exception as e:
            failures += 1
            if failures <= 10:
                print(f"  Sample {idx} model failed: {e}", flush=True)
            continue

        # Assemble predictions for each step size
        for S in step_sizes:
            pred_model = assemble_rolling_model(model_cache, S)
            errors[S]["model"].append(gt_angles - pred_model)

            pred_static = rolling_static(gt_angles, S)
            errors[S]["static"].append(gt_angles - pred_static)

            pred_linear = rolling_linear(gt_angles, S, prev_angles)
            errors[S]["linear"].append(gt_angles - pred_linear)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate
            n_ok = len(errors[step_sizes[0]]["model"])
            print(f"  [{i+1}/{num_samples}] {rate:.2f} samples/s, "
                  f"ETA {eta/60:.1f}min, {n_ok} ok, {failures} fail, "
                  f"{total_model_calls} model calls", flush=True)

    elapsed = time.time() - t0
    N = len(errors[step_sizes[0]]["model"])
    print(f"\nDone: {N}/{num_samples} successful in {elapsed:.0f}s "
          f"({failures} failures, {total_model_calls} model calls)",
          flush=True)

    motions = np.array(motions[:N])

    # High-motion subset: top quartile
    motion_threshold = np.percentile(motions, 75)
    high_motion_mask = motions >= motion_threshold
    n_high = int(high_motion_mask.sum())
    print(f"High-motion subset: {n_high} samples "
          f"(threshold: {motion_threshold:.1f} rad)", flush=True)

    # Compute and print results
    results = {"num_samples": N, "num_failures": failures,
               "elapsed_seconds": elapsed, "step_sizes": step_sizes,
               "high_motion_count": n_high}

    print(f"\n{'='*75}")
    hdr = (f"  {'Step':>4s} | {'Method':<8s} | {'All RMSE':>8s} | "
           f"{'High-Motion':>11s} | {'Future-Only':>11s}")
    print(hdr)
    print(f"{'='*75}")

    for S in step_sizes:
        step_results = {}
        for method in ["model", "static", "linear"]:
            err_all = np.array(errors[S][method])  # (N, 16, 22)

            all_rmse = float(np.degrees(np.sqrt(np.mean(err_all ** 2))))

            err_hm = err_all[high_motion_mask]
            hm_rmse = float(np.degrees(np.sqrt(np.mean(err_hm ** 2))))

            boundaries = set(range(0, ACTION_HORIZON, S))
            future_idx = [t for t in range(ACTION_HORIZON)
                          if t not in boundaries]
            if future_idx:
                future_rmse = float(np.degrees(
                    np.sqrt(np.mean(err_all[:, future_idx, :] ** 2))))
            else:
                future_rmse = None

            step_results[method] = {
                "all_rmse_deg": round(all_rmse, 1),
                "high_motion_rmse_deg": round(hm_rmse, 1),
                "future_only_rmse_deg": (round(future_rmse, 1)
                                         if future_rmse else None),
            }

            fut_str = f"{future_rmse:.1f}" if future_rmse else "n/a"
            print(f"  {S:>4d} | {method:<8s} | {all_rmse:>7.1f}° | "
                  f"{hm_rmse:>10.1f}° | {fut_str:>10s}°")

        results[f"step_{S}"] = step_results
        print(f"  {'-'*69}")

    # Distribution stats
    print(f"\n{'='*75}")
    print("  Per-sample RMSE distribution (model):")
    print(f"{'='*75}")
    for S in step_sizes:
        err = np.array(errors[S]["model"])
        ps = np.degrees(np.sqrt(np.mean(err ** 2, axis=(1, 2))))
        print(f"  Step {S}: median={np.median(ps):.1f}°, "
              f"p25={np.percentile(ps, 25):.1f}°, "
              f"p75={np.percentile(ps, 75):.1f}°")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
