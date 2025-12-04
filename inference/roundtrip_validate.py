#!/usr/bin/env python3
"""
Round-trip validator: render a pose trace with Blender, then re-extract 2D/3D
keypoints from the rendered video using ViTPose + MotionAGFormer to measure
drift versus the original trace. This is a debugging tool to quickly sanity-
check the rendering/retargeting pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, VitPoseForPoseEstimation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "MotionAGFormer"))

from data_prep.vitpose import infer_sequence  # noqa: E402
from data_prep.pose3d import build_motionagformer, lift_sequence_to_3d  # noqa: E402
from MotionAGFormer.model.MotionAGFormer import MotionAGFormer  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render a trace then re-extract keypoints to validate round-trip.")
    ap.add_argument("--trace", required=True, help="Path to the source .npz trace (pose3d/bboxes/meta)")
    ap.add_argument("--output-dir", default="movement/inference/renders/roundtrip",
                    help="Directory for rendered video and outputs")
    ap.add_argument("--limit", type=int, default=120, help="Frames to render/evaluate (default: 120)")
    ap.add_argument("--resolution", default="960x540", help="Render resolution (WIDTHxHEIGHT)")
    ap.add_argument("--samples", type=int, default=8, help="Eevee samples for quick renders")
    ap.add_argument("--background-image", default=None, help="Optional background image for render_trace.py")
    ap.add_argument("--scene-style", default="indoor", choices=["indoor", "outdoor", "studio"],
                    help="Scene style for render_trace.py")
    ap.add_argument("--axis-mode", default="x_negz_y",
                    choices=[
                        "x_negz_y", "x_z_y", "x_z_negy", "x_negy_z",
                        "negx_negz_y", "x_negz_negy",
                        "y_negz_x", "y_z_x", "y_negz_negx", "y_z_negx"
                    ],
                    help="Axis mapping passed to render_trace.py")
    ap.add_argument("--engine", default="BLENDER_EEVEE", choices=["BLENDER_EEVEE", "CYCLES"],
                    help="Render engine to request (Eevee will be auto-selected if Cycles unavailable)")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value for both Blender and torch")
    ap.add_argument("--skip-render", action="store_true", help="Reuse existing rendered video if present")
    ap.add_argument("--render-binary", default="blender", help="Path to Blender executable")
    ap.add_argument("--video-name", default=None, help="Override output video name (defaults to trace stem)")
    return ap.parse_args()


def load_trace(npz_path: Path, limit: int | None) -> Tuple[np.ndarray, np.ndarray | None, dict]:
    data = np.load(npz_path, allow_pickle=True)
    pose3d = np.array(data["pose3d"], dtype=np.float32)
    bboxes = data["bboxes"] if "bboxes" in data else None
    meta = {}
    if "meta" in data:
        meta_arr = data["meta"]
        if meta_arr.size >= 4:
            fps, _, w, h = meta_arr[:4]
            meta = {"fps": float(fps), "width": int(w), "height": int(h)}
    if limit:
        pose3d = pose3d[:limit]
        if bboxes is not None:
            bboxes = bboxes[:limit]
    return pose3d, bboxes, meta


def maybe_render_video(args: argparse.Namespace, trace_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = args.video_name or trace_path.stem
    video_path = output_dir / f"{video_name}.mp4"
    if args.skip_render and video_path.exists():
        return video_path

    cmd = [
        args.render_binary,
        "--background",
        "--python",
        str(PROJECT_ROOT / "inference" / "render_trace.py"),
        "--",
        "--trace",
        str(trace_path),
        "--output-dir",
        str(output_dir),
        "--limit",
        str(args.limit),
        "--resolution",
        args.resolution,
        "--samples",
        str(args.samples),
        "--scene-style",
        args.scene_style,
        "--engine",
        args.engine,
        "--clothing-style",
        "procedural",
        "--axis-mode",
        args.axis_mode,
    ]
    if args.background_image:
        cmd += ["--background-image", args.background_image]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    subprocess.run(cmd, check=True, env=env)
    return video_path


def read_video_frames(video_path: Path, max_frames: int | None = None) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    count = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(frame_bgr[:, :, ::-1])  # RGB
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    cap.release()
    return frames


def prepare_boxes(frames: List[np.ndarray], bboxes: np.ndarray | None) -> np.ndarray:
    if bboxes is not None and len(bboxes) >= len(frames):
        return np.asarray(bboxes[: len(frames)], dtype=np.float32)
    h, w = frames[0].shape[:2]
    full = np.array([0, 0, w, h], dtype=np.float32)
    return np.tile(full, (len(frames), 1))


def run_vitpose_and_lift(frames: List[np.ndarray], boxes_xyxy: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    vitpose_id = "usyd-community/vitpose-plus-large"
    processor = AutoImageProcessor.from_pretrained(vitpose_id, trust_remote_code=True)
    model = VitPoseForPoseEstimation.from_pretrained(
        vitpose_id, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    idxs = np.arange(len(frames))
    keypoints2d, scores2d = infer_sequence(
        raw_video_path="",
        image_processor=processor,
        model=model,
        device=device,
        idxs=idxs,
        boxes_xyxy=boxes_xyxy,
        frames=np.array(frames),
        batch_size=8,
    )

    width = frames[0].shape[1]
    height = frames[0].shape[0]
    model_3d = build_motionagformer(MotionAGFormer, device, PROJECT_ROOT / "models" / "motionagformer-b-h36m.pth.tr")
    pose3d_pred = lift_sequence_to_3d(
        seq_keypoints=keypoints2d,
        seq_scores=scores2d,
        width=width,
        height=height,
        model_3d=model_3d,
        device=device,
        use_overlap=True,
    )
    return keypoints2d, pose3d_pred


def compute_metrics(src_pose3d: np.ndarray, pred_pose3d: np.ndarray) -> dict:
    T = min(len(src_pose3d), len(pred_pose3d))
    src = src_pose3d[:T]
    pred = pred_pose3d[:T]
    src_rooted = src - src[:, :1, :]
    pred_rooted = pred - pred[:, :1, :]
    l2 = np.linalg.norm(src_rooted - pred_rooted, axis=2)  # (T,17)
    return {
        "mpjpe_mean": float(np.mean(l2)),
        "mpjpe_median": float(np.median(l2)),
        "mpjpe_per_joint": np.mean(l2, axis=0).tolist(),
    }


def main():
    args = parse_args()
    trace_path = Path(args.trace).resolve()
    output_dir = Path(args.output_dir).resolve()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pose3d_src, bboxes, meta = load_trace(trace_path, args.limit)
    video_path = maybe_render_video(args, trace_path, output_dir)

    frames = read_video_frames(video_path, max_frames=args.limit)
    if not frames:
        raise RuntimeError(f"Failed to read any frames from {video_path}")

    boxes_xyxy = prepare_boxes(frames, bboxes)
    keypoints2d, pose3d_pred = run_vitpose_and_lift(frames, boxes_xyxy, device)
    metrics = compute_metrics(pose3d_src, pose3d_pred)

    result_path = output_dir / f"{trace_path.stem}_roundtrip.npz"
    np.savez_compressed(
        result_path,
        keypoints2d=keypoints2d,
        pose3d_pred=pose3d_pred,
        pose3d_src=pose3d_src[: len(pose3d_pred)],
        bboxes=boxes_xyxy,
        meta=meta,
        metrics=np.array(metrics, dtype=object),
    )

    metrics_path = output_dir / f"{trace_path.stem}_roundtrip_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Round-trip saved to {result_path}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
