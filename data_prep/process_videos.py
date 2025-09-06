import argparse
import concurrent.futures as futures
import json
import os
from pathlib import Path
from typing import List
import torch

from data_prep.pipeline.pipeline import process_video
from transformers import AutoProcessor, VitPoseForPoseEstimation
import torchvision


def find_videos(root: Path) -> List[Path]:
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    vids = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            vids.append(p)
    vids.sort()
    return vids


def shard_indices(n_items: int, n_shards: int, shard_id: int):
    base = n_items // n_shards
    rem = n_items % n_shards
    start = shard_id * base + min(shard_id, rem)
    end = start + base + (1 if shard_id < rem else 0)
    return start, end


def process_one(
    path: Path,
    target_fps: float,
    out_dir: Path,
    device: str,
    score_thresh: float,
    debug: bool,
    save_frames: bool,
    model3d: str | None,
    ckpt3d: str | None,
    det_2d_model,
    vitpose_processor,
    vitpose_model,
):
    result = process_video(
        video_path=str(path),
        out_dir=out_dir,
        target_fps=target_fps,
        device=device,
        score_thresh=score_thresh,
        debug=debug,
        save_frames=save_frames,
        model3d=model3d,
        ckpt3d=ckpt3d,
        det_2d_model=det_2d_model,
        vitpose_processor=vitpose_processor,
        vitpose_model=vitpose_model,
    )
    return result.get("npz", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--target_fps", type=float, default=20.0)
    ap.add_argument("--score_thresh", type=float, default=0.5)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--save_frames", action="store_true")
    ap.add_argument("--model3d", type=str, default=None, help="Module:Class path for MotionAGFormer")
    ap.add_argument("--ckpt3d", type=str, default=None, help="Checkpoint path for MotionAGFormer")
    ap.add_argument("--vitpose_repo", type=str, required=True, help="HuggingFace repo id for ViTPose")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num_gpus", type=int, default=4)
    ap.add_argument("--gpu_id", type=int, default=None, help="If set, process only this shard (0..num_gpus-1)")
    ap.add_argument("--max_workers", type=int, default=8)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(data_root)
    if args.limit is not None:
        videos = videos[: args.limit]

    if args.gpu_id is not None:
        start, end = shard_indices(len(videos), args.num_gpus, args.gpu_id)
        videos = videos[start:end]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else ("cuda" if torch.cuda.is_available() else "cpu")
    # Build shared models once
    det_2d_model = None
    vitpose_processor = None
    vitpose_model = None
    vitpose_processor = AutoProcessor.from_pretrained(args.vitpose_repo)
    vitpose_model = VitPoseForPoseEstimation.from_pretrained(args.vitpose_repo).to(device).eval()
    det_2d_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device).eval()

    with futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        tasks = [
            ex.submit(
                process_one,
                v,
                args.target_fps,
                out_dir,
                device,
                args.score_thresh,
                args.debug,
                args.save_frames,
                args.model3d,
                args.ckpt3d,
                det_2d_model,
                vitpose_processor,
                vitpose_model,
            )
            for v in videos
        ]
        for t in tasks:
            try:
                path_out = t.result()
                print(f"Wrote {path_out}")
            except Exception as e:
                print(f"Failed: {e}")


if __name__ == "__main__":
    main()


