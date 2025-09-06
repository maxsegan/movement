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


# Global worker state (initialized in each process once)
_DEVICE = None
_DET_MODEL = None
_VITPOSE_PROCESSOR = None
_VITPOSE_MODEL = None


def _init_worker(vitpose_repo: str, gpu_id: int):
    global _DEVICE, _DET_MODEL, _VITPOSE_PROCESSOR, _VITPOSE_MODEL
    _DEVICE = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    _VITPOSE_PROCESSOR = AutoProcessor.from_pretrained(vitpose_repo)
    _VITPOSE_MODEL = VitPoseForPoseEstimation.from_pretrained(vitpose_repo).to(_DEVICE).eval()
    _DET_MODEL = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(_DEVICE).eval()


def _process_video_task(
    video_path: str,
    out_dir: str,
    data_root: str,
    target_fps: float,
    score_thresh: float,
    debug: bool,
    save_frames: bool,
    model3d: str | None,
    ckpt3d: str | None,
):
    # Uses globals initialized in _init_worker
    # Mirror input directory structure under out_dir to avoid name collisions
    vp = Path(video_path)
    dr = Path(data_root)
    try:
        rel = vp.relative_to(dr)
        save_dir = Path(out_dir) / rel.parent
    except Exception:
        save_dir = Path(out_dir)
    result = process_video(
        video_path=video_path,
        out_dir=save_dir,
        target_fps=target_fps,
        device=_DEVICE,
        score_thresh=score_thresh,
        debug=debug,
        save_frames=save_frames,
        model3d=model3d,
        ckpt3d=ckpt3d,
        det_2d_model=_DET_MODEL,
        vitpose_processor=_VITPOSE_PROCESSOR,
        vitpose_model=_VITPOSE_MODEL,
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
    ap.add_argument("--workers_per_gpu", type=int, default=16)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(data_root)
    if args.limit is not None:
        videos = videos[: args.limit]

    # Create one process pool per GPU so each worker can initialize its own models on that GPU
    executors = []
    for gpu_id in range(args.num_gpus):
        ex = futures.ProcessPoolExecutor(
            max_workers=args.workers_per_gpu,
            initializer=_init_worker,
            initargs=(args.vitpose_repo, gpu_id),
        )
        executors.append(ex)

    # Submit videos round-robin across GPU pools
    pending = []
    for i, v in enumerate(videos):
        pool = executors[i % args.num_gpus]
        fut = pool.submit(
            _process_video_task,
            str(v),
            str(out_dir),
            str(data_root),
            args.target_fps,
            args.score_thresh,
            args.debug,
            args.save_frames,
            args.model3d,
            args.ckpt3d,
        )
        pending.append(fut)

    for fut in pending:
        try:
            path_out = fut.result()
            print(f"Wrote {path_out}")
        except Exception as e:
            print(f"Failed: {e}")

    for ex in executors:
        ex.shutdown(wait=True)


if __name__ == "__main__":
    main()


