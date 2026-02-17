#!/usr/bin/env python3
"""
Generate three-panel figure for paper:
  Panel 1: Current video frame (model input)
  Panel 2: Predicted future poses on black background (temporal opacity gradient)
  Panel 3: Actual video frame 1.6s in the future

Two-phase approach:
  1. GT-only scan: Find intervals with highest motion (no model needed)
  2. Inference: Run model on top motion intervals, rank by RMSE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_prep.constants import H36M_I, H36M_J
from training.kinetics_dataset import (
    KineticsPoseDataset, joint_angles_to_pose3d, pose3d_to_joint_angles,
    JOINT_ANGLES_DIM, _parse_description,
)
from training.vla_model import VLAModel, VLAConfig

# ── L48 config (hardcoded, independent of current training config) ──────────

L48_MODEL_CONFIG = dict(
    qwen_model_name="/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17",
    qwen_hidden_size=2560,
    use_intermediate_hidden=True,
    hidden_layer_fraction=0.5,
    hidden_layer_index=18,
    use_early_exit=True,
    use_deepstack_features=True,
    use_flash_attention=False,
    projection_dim=1024,
    action_dim=22,
    diffusion_hidden_dim=1536,
    num_diffusion_layers=48,
    num_diffusion_heads=24,
    num_future_tokens=4,
    action_horizon=16,
    num_frames=4,
    use_lora=True,
    lora_rank=128,
    lora_alpha=128,
    lora_dropout=0.05,
    freeze_vision_encoder=True,
    freeze_qwen_layers=0,
    use_thinking_mode=False,
    diffusion_steps=2,
    init_from_current_pose=False,
)

L48_DATASET_CONFIG = dict(
    pose_dir="data/kinetics_processed",
    desc_dir="data/kinetics_full_output/descriptions",
    video_dir="data/kinetics-dataset/k700-2020",
    val_split=0.02,
    image_size=224,
    normalize_pose=True,
    sample_stride=16,
    seed=42,
)

L48_CHECKPOINT = "checkpoints/kinetics_vla/checkpoint_step_34500.pth"

FITNESS_KEYWORDS = {
    "push up", "pull ups", "deadlifting", "jumping jacks", "squat", "lunge",
    "sit up", "burpee", "exercising", "planking", "bench pressing",
    "clean and jerk", "snatch weight lifting", "kettlebell",
    "yoga", "contorting", "head stand",
    "backflip", "cartwheel", "cartwheeling", "gymnastics", "somersault", "somersaulting", "handstand",
    "front flip", "tumbling", "bouncing on trampoline",
    "capoeira", "punching bag", "kickboxing", "boxing", "punching person",
    "high jump", "javelin throw", "pole vault", "hurdling",
    "long jump", "shot put", "climbing", "skipping rope",
    "jumping", "zumba", "stretching arm", "stretching leg",
    "dancing", "krumping", "robot dancing", "tai chi",
    "wrestling", "judo", "karate", "martial arts",
    "ice climbing", "climbing a rope", "climbing tree",
    "surfing", "windsurfing", "skateboarding",
    "kicking soccer ball", "kicking field goal",
    "catching or throwing", "throwing",
    "swinging", "swinging on something",
    "base jumping", "bungee jumping", "diving",
    "running", "sprinting", "jogging",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def is_fitness(action_class: str) -> bool:
    ac = action_class.lower()
    return any(kw in ac for kw in FITNESS_KEYWORDS)


def load_model(checkpoint_path: str, device: str) -> VLAModel:
    vla_config = VLAConfig(**L48_MODEL_CONFIG)
    print("Loading L48 model...")
    model = VLAModel(vla_config)
    print(f"Loading checkpoint {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    clean = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(clean, strict=False)
    del ckpt, state_dict, clean
    import gc; gc.collect()
    model = model.to(device)
    model.eval()
    print("Model ready.")
    return model


def load_val_dataset() -> KineticsPoseDataset:
    dc = L48_DATASET_CONFIG
    mc = L48_MODEL_CONFIG
    return KineticsPoseDataset(
        pose_dir=dc["pose_dir"], desc_dir=dc["desc_dir"],
        video_dir=dc["video_dir"], split="val", val_split=dc["val_split"],
        action_horizon=mc["action_horizon"], num_frames=mc["num_frames"],
        sample_stride=dc["sample_stride"], resize=dc["image_size"],
        normalize_pose=dc["normalize_pose"], use_joint_angles=False, seed=dc["seed"],
    )


def procrustes_align_2d(source: np.ndarray, target: np.ndarray,
                        allow_scale: bool = True) -> np.ndarray:
    valid = ~(np.isnan(source).any(1) | np.isnan(target).any(1))
    if valid.sum() < 3:
        return source
    s, t = source[valid], target[valid]
    sc, tc = s.mean(0), t.mean(0)
    H = (s - sc).T @ (t - tc)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    scale = S.sum() / max((( s - sc) ** 2).sum(), 1e-8) if allow_scale else 1.0
    return (source - sc) @ R.T * scale + tc


def draw_skeleton(canvas, joints_xy, color, alpha=0.8, radius=3, thickness=2):
    overlay = canvas.copy()
    h, w = canvas.shape[:2]
    ok = lambda p: not np.any(np.isnan(p)) and -w < p[0] < 2*w and -h < p[1] < 2*h
    pt = lambda p: (int(np.clip(round(p[0]), 0, w-1)), int(np.clip(round(p[1]), 0, h-1)))
    for i, j in zip(H36M_I, H36M_J):
        if i < len(joints_xy) and j < len(joints_xy) and ok(joints_xy[i]) and ok(joints_xy[j]):
            cv2.line(overlay, pt(joints_xy[i]), pt(joints_xy[j]), color, thickness, cv2.LINE_AA)
    for p in joints_xy:
        if ok(p):
            cv2.circle(overlay, pt(p), radius, (255,255,255), -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, canvas, 1-alpha, 0)


def compute_visibility(kp2d, vw, vh):
    margins, heights, visible, total = [], [], 0, 0
    for kp in kp2d:
        valid = ~np.isnan(kp).any(1)
        if valid.sum() < 10: continue
        total += 1
        pts = kp[valid]
        mn_x, mn_y = pts.min(0); mx_x, mx_y = pts.max(0)
        heights.append((mx_y - mn_y) / vh)
        m = min(mn_x/vw, (vw-mx_x)/vw, mn_y/vh, (vh-mx_y)/vh)
        margins.append(m)
        if m > 0.03: visible += 1
    if total == 0: return dict(height=0, margin=0, visible_ratio=0)
    return dict(height=float(np.mean(heights)), margin=float(np.min(margins)),
                visible_ratio=visible/total)


def get_video_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def get_video_dims(video_path):
    cap = cv2.VideoCapture(str(video_path))
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    return w, h


def compute_path_length(poses3d, start, end):
    """Sum of per-frame mean joint displacement (captures cyclical motion)."""
    return float(sum(
        np.mean(np.linalg.norm(poses3d[t] - poses3d[t-1], axis=1))
        for t in range(start+1, end)
    ))


def compute_visual_displacement(kp2d, start, end, vh):
    """Mean 2D keypoint displacement between first and last frame, normalized by person height.
    This measures how VISUALLY different the person looks at the end vs start."""
    kp_start = kp2d[start]
    kp_end = kp2d[end - 1]
    # Only use joints that are valid in both frames
    valid_s = (np.abs(kp_start).sum(1) > 1.0)
    valid_e = (np.abs(kp_end).sum(1) > 1.0)
    valid = valid_s & valid_e
    if valid.sum() < 5:
        return 0.0
    disp = np.linalg.norm(kp_start[valid] - kp_end[valid], axis=1)
    # Normalize by person height in pixels
    pts = kp_start[valid_s]
    person_h = max(pts[:, 1].max() - pts[:, 1].min(), 1.0)
    return float(np.mean(disp) / person_h)


# ── Phase 1: GT-only scan ───────────────────────────────────────────────────

def find_best_intervals(dataset, action_horizon=16, min_path=0.4,
                        min_height=0.50, min_visual_disp=0.15,
                        stride=4, top_n=100):
    """Find intervals with highest VISUAL motion across ALL clips. No inference."""
    candidates = []
    for ci, clip in enumerate(dataset.clips):
        if not is_fitness(clip.action_class):
            continue
        try:
            data = np.load(clip.pose_path, allow_pickle=True)
            kp2d = data["keypoints2d"].astype(np.float32)
            poses3d = data["pose3d"].astype(np.float32)
            n = len(poses3d)
            if n < action_horizon + 1:
                continue
            vw, vh = get_video_dims(clip.video_path)

            for start in range(0, n - action_horizon, stride):
                end = start + action_horizon
                # All frames valid 2D?
                ok = True
                for ft in range(start, end):
                    if ft >= len(kp2d) or (np.abs(kp2d[ft]).sum(1) > 1.0).sum() < 10:
                        ok = False; break
                if not ok:
                    continue
                # Visual displacement: how different does the person LOOK?
                vis_disp = compute_visual_displacement(kp2d, start, end, vh)
                if vis_disp < min_visual_disp:
                    continue
                path_len = compute_path_length(poses3d, start, end)
                vis = compute_visibility(kp2d[start:end], vw, vh)
                if vis["height"] < min_height or vis["margin"] < 0.03 or vis["visible_ratio"] < 0.90:
                    continue
                candidates.append(dict(
                    clip_idx=ci, start_frame=start,
                    action_class=clip.action_class, clip_id=clip.clip_id,
                    path_length=path_len, height=vis["height"],
                    visual_disp=vis_disp,
                ))
        except Exception:
            continue
        if (ci+1) % 500 == 0:
            print(f"  {ci+1}/{len(dataset.clips)} clips, {len(candidates)} intervals")

    # Deduplicate: keep only the best interval per clip (by visual displacement)
    best_per_clip = {}
    for c in candidates:
        cid = c["clip_id"]
        if cid not in best_per_clip or c["visual_disp"] > best_per_clip[cid]["visual_disp"]:
            best_per_clip[cid] = c
    deduped = sorted(best_per_clip.values(), key=lambda c: -c["visual_disp"])

    print(f"\nFound {len(candidates)} intervals with path >= {min_path} "
          f"({len(deduped)} unique clips)")
    if deduped:
        print(f"Top path_length: {deduped[0]['path_length']:.3f} "
              f"({deduped[0]['action_class']})")
    return deduped[:top_n]


# ── Phase 2: Inference ──────────────────────────────────────────────────────

def run_inference_on_intervals(model, dataset, candidates, device="cuda:0"):
    """Run inference on pre-filtered intervals (arbitrary start frames)."""
    action_horizon = L48_MODEL_CONFIG["action_horizon"]
    num_input_frames = L48_MODEL_CONFIG["num_frames"]
    resize = L48_DATASET_CONFIG["image_size"]
    normalize = L48_DATASET_CONFIG["normalize_pose"]
    results = []

    for i, cand in enumerate(candidates):
        try:
            clip = dataset.clips[cand["clip_idx"]]
            data = np.load(clip.pose_path, allow_pickle=True)
            poses3d = data["pose3d"].astype(np.float32)
            pose_indices = data["indices"].astype(np.int32)
            start = cand["start_frame"]

            # Build images: num_input_frames evenly spaced
            frame_idxs = np.linspace(start, start + action_horizon - 1,
                                     num_input_frames, dtype=int)
            cap = cv2.VideoCapture(str(clip.video_path))
            images = []
            for fi in frame_idxs:
                vi = int(pose_indices[fi])
                cap.set(cv2.CAP_PROP_POS_FRAMES, vi)
                ok, frame = cap.read()
                if not ok: raise RuntimeError(f"Can't read frame {vi}")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(np.array(Image.fromarray(rgb).resize((resize, resize), Image.BILINEAR)))
            cap.release()

            # Robot state
            rs = poses3d[start].copy()
            if normalize: rs = rs - rs[0:1]
            robot_state = pose3d_to_joint_angles(rs)

            # GT actions
            gt = poses3d[start:start+action_horizon].copy()
            if normalize: gt = gt - poses3d[start][0:1]

            try:
                desc_body = _parse_description(clip.desc_path)
                instruction = f"Task: {clip.action_class}. Instruction: {desc_body}"
            except Exception:
                instruction = f"A person performing {clip.action_class}"

            with torch.no_grad():
                pred = model.get_action(images=images, instruction=instruction,
                                        robot_state=robot_state)

            pred_angles = pred.reshape(action_horizon, JOINT_ANGLES_DIM)
            gt_angles = np.array([pose3d_to_joint_angles(gt[t]) for t in range(action_horizon)])
            rmse = float(np.degrees(np.sqrt(np.mean((gt_angles - pred_angles)**2))))

            results.append(dict(**cand, rmse_deg=rmse))
            if (i+1) % 10 == 0:
                best = min(r["rmse_deg"] for r in results)
                print(f"  Inference {i+1}/{len(candidates)}, best RMSE: {best:.1f}°")
        except Exception as e:
            print(f"  Skip {cand['action_class']}/{cand['clip_id']}: {e}")
            continue

    results.sort(key=lambda r: r["rmse_deg"])
    return results


# ── Panel generation ────────────────────────────────────────────────────────

def generate_panels(model, dataset, cand, output_dir, device="cuda:0"):
    """Generate panels for an interval (arbitrary start frame)."""
    action_horizon = L48_MODEL_CONFIG["action_horizon"]
    num_input_frames = L48_MODEL_CONFIG["num_frames"]
    resize = L48_DATASET_CONFIG["image_size"]
    normalize = L48_DATASET_CONFIG["normalize_pose"]

    clip = dataset.clips[cand["clip_idx"]]
    start = cand["start_frame"]
    data = np.load(clip.pose_path, allow_pickle=True)
    poses3d = data["pose3d"].astype(np.float32)
    kp2d = data["keypoints2d"].astype(np.float32)
    pose_indices = data["indices"].astype(np.int32)
    vw, vh = get_video_dims(clip.video_path)
    end = start + action_horizon

    # Panel 1 & 3: video frames
    frame_now = get_video_frame(clip.video_path, int(pose_indices[start]))
    frame_future = get_video_frame(clip.video_path, int(pose_indices[min(end-1, len(pose_indices)-1)]))
    if frame_now is None or frame_future is None:
        raise RuntimeError("Cannot read video frames")

    # Build images for inference
    frame_idxs = np.linspace(start, start + action_horizon - 1, num_input_frames, dtype=int)
    cap = cv2.VideoCapture(str(clip.video_path))
    images = []
    for fi in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pose_indices[fi]))
        ok, fr = cap.read()
        images.append(np.array(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize(
            (resize, resize), Image.BILINEAR)))
    cap.release()

    rs = poses3d[start].copy()
    if normalize: rs = rs - rs[0:1]
    robot_state = pose3d_to_joint_angles(rs)
    try:
        desc_body = _parse_description(clip.desc_path)
        instruction = f"Task: {clip.action_class}. Instruction: {desc_body}"
    except Exception:
        instruction = f"A person performing {clip.action_class}"

    with torch.no_grad():
        pred = model.get_action(images=images, instruction=instruction, robot_state=robot_state)

    # Convert to 3D
    pred_angles = pred.reshape(action_horizon, JOINT_ANGLES_DIM)
    ref_pose = poses3d[start]
    pred_3d = np.array([joint_angles_to_pose3d(pred_angles[t], reference_pose=ref_pose)
                        for t in range(action_horizon)])

    # Per-frame Procrustes alignment
    aligned_pred, gt_2d_list = [], []
    for t in range(action_horizon):
        gt2d = kp2d[start + t].copy()
        gt_2d_list.append(gt2d)
        aligned_pred.append(procrustes_align_2d(pred_3d[t][:, :2].copy(), gt2d))

    # Centering transform (using all aligned points)
    all_pts = np.concatenate(aligned_pred + gt_2d_list, axis=0)
    valid = (~np.isnan(all_pts).any(1)) & (np.abs(all_pts).sum(1) > 1.0)
    pts = all_pts[valid]
    center = pts.mean(0)
    extent = max(pts[:, 0].max()-pts[:, 0].min(), pts[:, 1].max()-pts[:, 1].min())
    scale = min(vw, vh) * 0.85 / max(extent, 1e-6)
    cc = np.array([vw/2, vh/2])

    pred_color, gt_color = (0, 220, 220), (0, 180, 0)
    canvas_pred = np.zeros((vh, vw, 3), np.uint8)
    canvas_gt = np.zeros((vh, vw, 3), np.uint8)
    canvas_cmp = np.zeros((vh, vw, 3), np.uint8)

    num_draw = 5
    draw_idx = np.linspace(0, action_horizon-1, num_draw, dtype=int)

    for pos, t in enumerate(draw_idx):
        frac = pos / max(num_draw-1, 1)
        a = 0.20 + 0.80*frac
        r = max(2, int(2 + 2*frac))
        th = max(1, int(1 + 2*frac))

        pp = (aligned_pred[t] - center) * scale + cc
        canvas_pred = draw_skeleton(canvas_pred, pp, pred_color, a, r, th)

        gk = gt_2d_list[t]
        if (np.abs(gk).sum(1) > 1.0).sum() >= 10:
            gp = (gk - center) * scale + cc
            canvas_gt = draw_skeleton(canvas_gt, gp, gt_color, a, r, th)

        # Comparison
        canvas_cmp = draw_skeleton(canvas_cmp, pp, pred_color, a, r, th)
        if (np.abs(gk).sum(1) > 1.0).sum() >= 10:
            canvas_cmp = draw_skeleton(canvas_cmp, gp, gt_color, a*0.7, r, th)

    # Overlay on future frame
    overlay = frame_future.copy()
    overlay = draw_skeleton(overlay, aligned_pred[-1], pred_color, 0.85, 6, 4)

    # Save individual panels
    tag = f"{cand['action_class'].replace(' ', '_')}_{cand['clip_id']}_f{start}"
    sub = output_dir / tag
    sub.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sub / "panel1_current.png"), frame_now)
    cv2.imwrite(str(sub / "panel2_predicted.png"), canvas_pred)
    cv2.imwrite(str(sub / "panel2b_gt.png"), canvas_gt)
    cv2.imwrite(str(sub / "panel2c_compare.png"), canvas_cmp)
    cv2.imwrite(str(sub / "panel3_future.png"), frame_future)
    cv2.imwrite(str(sub / "panel4_overlay.png"), overlay)

    # Create composite: [current | predicted | GT | future]
    # Resize all panels to same height
    target_h = vh
    panels = [frame_now, canvas_pred, canvas_gt, frame_future]
    resized = []
    for p in panels:
        ph, pw = p.shape[:2]
        scale_f = target_h / ph
        new_w = int(pw * scale_f)
        resized.append(cv2.resize(p, (new_w, target_h)))
    # Add thin white separator
    sep = np.ones((target_h, 3, 3), np.uint8) * 255
    parts = []
    for i, r in enumerate(resized):
        parts.append(r)
        if i < len(resized) - 1:
            parts.append(sep)
    composite = np.concatenate(parts, axis=1)
    cv2.imwrite(str(sub / "composite_4panel.png"), composite)

    # Also 3-panel: [current | predicted | future]
    panels3 = [frame_now, canvas_pred, frame_future]
    resized3 = []
    for p in panels3:
        ph, pw = p.shape[:2]
        scale_f = target_h / ph
        new_w = int(pw * scale_f)
        resized3.append(cv2.resize(p, (new_w, target_h)))
    parts3 = []
    for i, r in enumerate(resized3):
        parts3.append(r)
        if i < len(resized3) - 1:
            parts3.append(sep)
    composite3 = np.concatenate(parts3, axis=1)
    cv2.imwrite(str(sub / "composite_3panel.png"), composite3)

    print(f"  Saved to {sub}/")
    return sub


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min-path", type=float, default=0.0, help="Min GT path length")
    ap.add_argument("--min-height", type=float, default=0.40)
    ap.add_argument("--min-visual-disp", type=float, default=0.20, help="Min visual displacement (fraction of person height)")
    ap.add_argument("--top-gt", type=int, default=100, help="Top GT intervals for inference")
    ap.add_argument("--top-panels", type=int, default=5, help="Generate panels for top N")
    ap.add_argument("--output-dir", default="inference/paper_figures_v2")
    ap.add_argument("--stride", type=int, default=4, help="Sliding window stride")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_val_dataset()
    print(f"Val dataset: {len(dataset.clips)} clips, {len(dataset)} samples")

    # Phase 1: GT-only scan
    print(f"\n=== Phase 1: Finding high-motion intervals (visual_disp >= {args.min_visual_disp}) ===")
    intervals = find_best_intervals(
        dataset, min_path=args.min_path, min_height=args.min_height,
        min_visual_disp=args.min_visual_disp,
        stride=args.stride, top_n=args.top_gt,
    )
    if not intervals:
        print("No intervals found!"); return

    print(f"\nTop 20 by visual displacement:")
    print(f"{'#':>3} {'VDisp':>6} {'Path':>6} {'Ht%':>5} {'Action':<30} {'Clip ID':<35} {'Start'}")
    for i, c in enumerate(intervals[:20]):
        print(f"{i+1:>3} {c['visual_disp']:>6.3f} {c['path_length']:>6.3f} {c['height']:>4.0%} "
              f"{c['action_class']:<30} {c['clip_id']:<35} f{c['start_frame']}")

    # Phase 2: Inference on top intervals
    print(f"\n=== Phase 2: Running inference on top {len(intervals)} intervals ===")
    model = load_model(L48_CHECKPOINT, args.device)
    results = run_inference_on_intervals(model, dataset, intervals, args.device)

    print(f"\nTop 15 by RMSE:")
    print(f"{'#':>3} {'RMSE°':>6} {'Path':>6} {'Ht%':>5} {'Action':<30} {'Clip':<35} {'Start'}")
    for i, r in enumerate(results[:15]):
        print(f"{i+1:>3} {r['rmse_deg']:>6.1f} {r['path_length']:>6.3f} {r['height']:>4.0%} "
              f"{r['action_class']:<30} {r['clip_id']:<35} f{r['start_frame']}")

    # Generate panels
    print(f"\n=== Phase 3: Generating panels for top {args.top_panels} ===")
    for i, r in enumerate(results[:args.top_panels]):
        print(f"\n[{i+1}] {r['action_class']} (RMSE={r['rmse_deg']:.1f}°, path={r['path_length']:.3f})")
        try:
            generate_panels(model, dataset, r, output_dir, args.device)
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
