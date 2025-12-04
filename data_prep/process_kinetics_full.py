#!/usr/bin/env python3
"""
Process the entire Kinetics dataset using existing NPZ files.
Generates bounding box videos and VLM descriptions using GPUs 1, 2, and 3.
"""

import os
import sys
import cv2
import json
import torch
import shutil
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty
from multiprocessing import Process, Queue as MPQueue, cpu_count
from typing import List, Dict, Tuple, Optional
from PIL import Image
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_prep.multi_vlm_loader import load_vlm_model, generate_description_with_model


def create_bbox_video(
    video_path: Path,
    npz_path: Path,
    output_path: Path,
    max_frames: int = 300,
    target_fps: float = 10.0
) -> bool:
    """Create video with bounding box overlay from NPZ data."""
    try:
        # Load pose data
        data = np.load(npz_path, allow_pickle=True)

        if 'bboxes' not in data:
            logger.warning(f"No bboxes in {npz_path}")
            return False

        indices = data.get('indices', np.arange(data['bboxes'].shape[0]))
        bboxes = data['bboxes']

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resize to 480p for efficiency
        target_height = 480
        aspect_ratio = orig_width / orig_height
        target_width = int(target_height * aspect_ratio)
        if target_width % 2 != 0:
            target_width += 1

        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            min(target_fps, fps),
            (target_width, target_height)
        )

        # Calculate scaling
        scale_x = target_width / orig_width
        scale_y = target_height / orig_height

        frame_idx = 0
        processed = 0

        while cap.isOpened() and processed < min(max_frames, len(indices)):
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame
            frame = cv2.resize(frame, (target_width, target_height))

            # Draw bounding box if available for this frame
            if frame_idx in indices:
                pose_idx = np.where(indices == frame_idx)[0]
                if len(pose_idx) > 0:
                    pose_idx = pose_idx[0]

                    if pose_idx < len(bboxes) and not np.any(np.isnan(bboxes[pose_idx])):
                        x1, y1, x2, y2 = bboxes[pose_idx]
                        x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
                        x2, y2 = int(x2 * scale_x), int(y2 * scale_y)

                        # Draw green bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                        # Add label
                        label = f"Person (Frame {frame_idx})"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

                        # Label background
                        cv2.rectangle(frame,
                                    (x1, y1 - label_size[1] - 10),
                                    (x1 + label_size[0] + 10, y1),
                                    (0, 255, 0), -1)

                        # Label text
                        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    processed += 1

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        return True

    except Exception as e:
        logger.error(f"Error creating bbox video: {e}")
        return False


def extract_frames_for_vlm(video_path: Path, max_frames: int = 8) -> List[Image.Image]:
    """Extract frames from video for VLM processing."""
    frames = []
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return frames

    # Sample frames evenly
    sample_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB and to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def process_worker(
    worker_id: int,
    gpu_id: int,
    task_queue: MPQueue,
    result_queue: MPQueue,
    output_base_dir: Path,
    enable_vlm: bool,
    max_frames_video: int,
    skip_existing: bool
):
    """Worker process for a specific GPU."""

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = f"cuda:0"  # Since we're setting CUDA_VISIBLE_DEVICES, always use cuda:0

    logger.info(f"Worker {worker_id} starting on GPU {gpu_id}")

    # Load VLM model if enabled
    vlm_model = None
    vlm_processor = None

    if enable_vlm:
        logger.info(f"Worker {worker_id}: Loading Qwen3-VL-32B on GPU {gpu_id}...")
        config = {
            "name": "qwen3vl-32b",
            "model_id": "Qwen/Qwen3-VL-32B-Instruct",
            "use_4bit": True,
        }

        try:
            vlm_model, vlm_processor = load_vlm_model(config, device=device)
            logger.info(f"Worker {worker_id}: VLM loaded successfully")
        except Exception as e:
            logger.error(f"Worker {worker_id}: Failed to load VLM: {e}")
            enable_vlm = False

    # Process tasks
    while True:
        try:
            task = task_queue.get(timeout=5)
            if task is None:  # Poison pill
                break

            npz_path = Path(task['npz_path'])
            video_path = Path(task['video_path'])
            action_class = task['action_class']

            # Create output directories
            bbox_video_dir = output_base_dir / "bbox_videos" / action_class
            descriptions_dir = output_base_dir / "descriptions" / action_class
            bbox_video_dir.mkdir(parents=True, exist_ok=True)
            descriptions_dir.mkdir(parents=True, exist_ok=True)

            video_stem = npz_path.stem
            bbox_video_path = bbox_video_dir / f"{video_stem}_bbox.mp4"
            description_path = descriptions_dir / f"{video_stem}.txt"

            result = {
                'worker_id': worker_id,
                'video': str(video_path),
                'npz': str(npz_path),
                'action_class': action_class,
                'bbox_video_created': False,
                'description_generated': False,
                'errors': []
            }

            # Skip if outputs exist and skip_existing is True
            if skip_existing:
                if bbox_video_path.exists() and description_path.exists():
                    result['bbox_video_created'] = True
                    result['description_generated'] = True
                    result['skipped'] = True
                    result_queue.put(result)
                    continue

            # Create bounding box video
            if not bbox_video_path.exists() or not skip_existing:
                logger.info(f"Worker {worker_id}: Creating bbox video for {video_stem}")
                bbox_success = create_bbox_video(
                    video_path,
                    npz_path,
                    bbox_video_path,
                    max_frames=max_frames_video
                )
                result['bbox_video_created'] = bbox_success
                if bbox_success:
                    result['bbox_video_path'] = str(bbox_video_path)
                else:
                    result['errors'].append("Failed to create bbox video")
            else:
                result['bbox_video_created'] = True
                result['bbox_video_path'] = str(bbox_video_path)

            # Generate description using VLM
            if enable_vlm and vlm_model is not None:
                if not description_path.exists() or not skip_existing:
                    logger.info(f"Worker {worker_id}: Generating description for {video_stem}")

                    # Extract frames from original video
                    frames = extract_frames_for_vlm(video_path)

                    if frames:
                        # Use the optimized robotic prompt
                        prompt = """Rewrite only the physical motion as a sequence of imperative commands. Do not mention a person, actor, bounding box, body type, clothing, scene, environment, animals, spectators, or camera. Mention only objects that the action physically interacts with. Do not narrate, summarize, or describe anything. Use only imperative verbs. Start immediately with a command. Include fine-grained movement details (body position, limb use, sequence of steps). The output must be a concise instruction that someone could follow to reproduce the motion precisely." Format requirement: 1–3 short sentences, each in pure imperative form. Examples of desired output: "Slide into a split, roll forward across the floor, and push up into a low crouch.", "Position the metal sheet under the press brake, lower the tooling in one smooth motion, then lift the bent piece and place it aside.","Kneel beside the tree, lift the ornament with your right hand, and hook it onto the mid-level branch while stabilizing the tree with your left.", "Raise the object overhead, step forward, and swing it downward in a controlled arc."""

                        try:
                            description = generate_description_with_model(
                                "qwen3vl-32b",
                                vlm_model,
                                vlm_processor,
                                frames,
                                prompt
                            )

                            if description and not description.startswith("Error"):
                                # Save description
                                with open(description_path, 'w') as f:
                                    f.write(f"Video: {video_stem}\n")
                                    f.write(f"Action Class: {action_class}\n")
                                    f.write(f"Timestamp: {datetime.now()}\n")
                                    f.write(f"\nDescription:\n{description}\n")

                                result['description_generated'] = True
                                result['description'] = description
                            else:
                                result['errors'].append(f"VLM error: {description}")

                        except Exception as e:
                            result['errors'].append(f"VLM exception: {e}")
                    else:
                        result['errors'].append("Failed to extract frames")
                else:
                    result['description_generated'] = True
                    result['description_path'] = str(description_path)

            result_queue.put(result)

        except Empty:
            continue
        except Exception as e:
            logger.error(f"Worker {worker_id}: Error processing task: {e}")
            result_queue.put({
                'worker_id': worker_id,
                'error': str(e)
            })

    # Clean up VLM model
    if vlm_model is not None:
        del vlm_model
        del vlm_processor
        torch.cuda.empty_cache()

    logger.info(f"Worker {worker_id} finished")


def collect_all_tasks(npz_base_dir: Path, video_base_dir: Path, start_idx: int = 0) -> List[Dict]:
    """Collect all NPZ files and corresponding videos."""
    tasks = []

    # Get all NPZ files
    all_npz_files = sorted(list(npz_base_dir.rglob("*.npz")))
    logger.info(f"Found {len(all_npz_files)} NPZ files")

    # Skip to start_idx if resuming
    if start_idx > 0:
        all_npz_files = all_npz_files[start_idx:]
        logger.info(f"Resuming from index {start_idx}, processing {len(all_npz_files)} remaining files")

    for npz_path in all_npz_files:
        # Extract action class and video name from NPZ path
        action_class = npz_path.parent.name
        video_name = npz_path.stem + ".mp4"

        # Find corresponding video
        video_path = video_base_dir / action_class / video_name

        if not video_path.exists():
            # Try without action class subdirectory
            video_path = video_base_dir / video_name

        if video_path.exists():
            tasks.append({
                'npz_path': str(npz_path),
                'video_path': str(video_path),
                'action_class': action_class
            })
        else:
            logger.warning(f"Video not found for NPZ: {npz_path}")

    logger.info(f"Collected {len(tasks)} valid tasks")
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Process entire Kinetics dataset with existing NPZ files")

    parser.add_argument('--npz_dir', type=str,
                        default='/root/movement/data/kinetics_processed',
                        help='Directory containing processed NPZ files')
    parser.add_argument('--video_dir', type=str,
                        default='/root/movement/data/kinetics-dataset/k700-2020/train',
                        help='Directory containing original videos')
    parser.add_argument('--output_dir', type=str,
                        default='/root/movement/data/kinetics_full_output',
                        help='Output directory for results')
    parser.add_argument('--gpus', type=str, default='1,2,3',
                        help='Comma-separated GPU IDs to use')
    parser.add_argument('--workers_per_gpu', type=int, default=1,
                        help='Number of worker processes per GPU')
    parser.add_argument('--enable_vlm', action='store_true',
                        help='Enable VLM description generation')
    parser.add_argument('--max_frames_video', type=int, default=300,
                        help='Maximum frames per output video')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip processing if outputs already exist')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index for resuming processing')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Number of videos to process before saving checkpoint')

    args = parser.parse_args()

    # Parse GPUs
    gpu_ids = [int(g) for g in args.gpus.split(',')]
    num_workers = len(gpu_ids) * args.workers_per_gpu

    logger.info(f"Starting Kinetics full processing")
    logger.info(f"NPZ directory: {args.npz_dir}")
    logger.info(f"Video directory: {args.video_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Using GPUs: {gpu_ids}")
    logger.info(f"Workers per GPU: {args.workers_per_gpu}")
    logger.info(f"Total workers: {num_workers}")
    logger.info(f"VLM enabled: {args.enable_vlm}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all tasks
    tasks = collect_all_tasks(
        Path(args.npz_dir),
        Path(args.video_dir),
        args.start_idx
    )

    if not tasks:
        logger.error("No tasks found!")
        return

    # Create queues
    task_queue = MPQueue(maxsize=num_workers * 10)
    result_queue = MPQueue()

    # Start worker processes
    workers = []
    worker_id = 0
    for gpu_id in gpu_ids:
        for _ in range(args.workers_per_gpu):
            p = Process(
                target=process_worker,
                args=(
                    worker_id,
                    gpu_id,
                    task_queue,
                    result_queue,
                    output_dir,
                    args.enable_vlm,
                    args.max_frames_video,
                    args.skip_existing
                )
            )
            p.start()
            workers.append(p)
            worker_id += 1

    # Add tasks to queue
    task_thread = Process(
        target=lambda: [task_queue.put(t) for t in tasks] + [task_queue.put(None) for _ in range(num_workers)]
    )
    task_thread.start()

    # Collect results
    results = []
    stats = {
        'total': len(tasks),
        'processed': 0,
        'bbox_videos_created': 0,
        'descriptions_generated': 0,
        'errors': 0,
        'skipped': 0
    }

    start_time = time.time()
    last_checkpoint = 0

    logger.info(f"Processing {len(tasks)} videos...")

    while stats['processed'] < len(tasks):
        try:
            result = result_queue.get(timeout=30)
            results.append(result)
            stats['processed'] += 1

            if result.get('bbox_video_created'):
                stats['bbox_videos_created'] += 1
            if result.get('description_generated'):
                stats['descriptions_generated'] += 1
            if result.get('errors'):
                stats['errors'] += 1
            if result.get('skipped'):
                stats['skipped'] += 1

            # Progress update
            if stats['processed'] % 100 == 0:
                elapsed = time.time() - start_time
                rate = stats['processed'] / elapsed
                eta = (len(tasks) - stats['processed']) / rate if rate > 0 else 0

                logger.info(f"Progress: {stats['processed']}/{len(tasks)} "
                          f"({stats['processed']*100/len(tasks):.1f}%) "
                          f"Rate: {rate:.1f} videos/sec, ETA: {eta/3600:.1f}h")
                logger.info(f"  BBox videos: {stats['bbox_videos_created']}, "
                          f"Descriptions: {stats['descriptions_generated']}, "
                          f"Errors: {stats['errors']}, Skipped: {stats['skipped']}")

            # Save checkpoint
            if stats['processed'] - last_checkpoint >= args.batch_size:
                checkpoint = {
                    'timestamp': datetime.now().isoformat(),
                    'last_index': args.start_idx + stats['processed'],
                    'stats': stats,
                    'elapsed_time': time.time() - start_time
                }

                checkpoint_file = output_dir / "processing_checkpoint.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f, indent=2)

                logger.info(f"Checkpoint saved at index {checkpoint['last_index']}")
                last_checkpoint = stats['processed']

        except Empty:
            # Check if workers are still alive
            alive = sum(1 for w in workers if w.is_alive())
            if alive == 0:
                break
            logger.debug(f"Waiting for results... {alive} workers still active")

    # Wait for workers to finish
    task_thread.join()
    for w in workers:
        w.join(timeout=30)
        if w.is_alive():
            w.terminate()

    # Save final results
    final_stats = {
        'timestamp': datetime.now().isoformat(),
        'total_processed': stats['processed'],
        'bbox_videos_created': stats['bbox_videos_created'],
        'descriptions_generated': stats['descriptions_generated'],
        'errors': stats['errors'],
        'skipped': stats['skipped'],
        'total_time_seconds': time.time() - start_time,
        'processing_rate': stats['processed'] / (time.time() - start_time)
    }

    stats_file = output_dir / "final_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(final_stats, f, indent=2)

    # Save detailed results
    if results:
        results_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

    logger.info("\n" + "="*80)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Total processed: {stats['processed']}/{len(tasks)}")
    logger.info(f"BBox videos created: {stats['bbox_videos_created']}")
    logger.info(f"Descriptions generated: {stats['descriptions_generated']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Skipped: {stats['skipped']}")
    logger.info(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
    logger.info(f"Processing rate: {final_stats['processing_rate']:.2f} videos/second")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()