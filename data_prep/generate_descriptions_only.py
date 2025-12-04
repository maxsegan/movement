#!/usr/bin/env python3
"""
Generate descriptions for already-created bounding box videos.
Uses the existing bbox videos to generate VLM descriptions.
"""

import os
import sys
import torch
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty
from multiprocessing import Process, Queue as MPQueue
from typing import List, Dict
from PIL import Image
import cv2
import numpy as np
import time
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_prep.multi_vlm_loader import load_vlm_model, generate_description_with_model


def extract_frames_for_vlm(video_path: Path, max_frames: int = 20) -> List[Image.Image]:
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
    skip_existing: bool
):
    """Worker process for a specific GPU."""

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = f"cuda:0"  # Since we're setting CUDA_VISIBLE_DEVICES, always use cuda:0

    logger.info(f"Worker {worker_id} starting on GPU {gpu_id}")

    # Load VLM model
    vlm_model = None
    vlm_processor = None
    vlm_loaded = False

    logger.info(f"Worker {worker_id}: Loading Qwen3-VL-32B on GPU {gpu_id}...")
    config = {
        "name": "qwen3vl-32b",
        "model_id": "Qwen/Qwen3-VL-32B-Instruct",
        "use_4bit": True,
    }

    try:
        vlm_model, vlm_processor = load_vlm_model(config, device=device)

        if vlm_model is None or vlm_processor is None:
            logger.error(f"Worker {worker_id}: VLM model or processor is None")
            vlm_loaded = False
        else:
            logger.info(f"Worker {worker_id}: VLM loaded successfully")
            # Test the model with a simple inference
            test_img = Image.new('RGB', (224, 224), color='white')
            test_desc = generate_description_with_model(
                "qwen3vl-32b",
                vlm_model,
                vlm_processor,
                [test_img],
                "Test"
            )
            logger.info(f"Worker {worker_id}: VLM test inference result: {test_desc[:50]}...")
            vlm_loaded = True

    except Exception as e:
        logger.error(f"Worker {worker_id}: Failed to load VLM: {e}")
        logger.error(f"Worker {worker_id}: Traceback: {traceback.format_exc()}")
        vlm_loaded = False

    if not vlm_loaded:
        logger.error(f"Worker {worker_id}: Exiting due to VLM load failure")
        # Put error messages in queue
        for _ in range(100):  # Send multiple error messages
            result_queue.put({
                'worker_id': worker_id,
                'error': f"VLM failed to load on GPU {gpu_id}",
                'description_generated': False
            })
        return

    # Process tasks
    processed_count = 0
    while True:
        try:
            task = task_queue.get(timeout=5)
            if task is None:  # Poison pill
                break

            bbox_video_path = Path(task['bbox_video_path'])
            action_class = task['action_class']
            video_stem = task['video_stem']

            # Create output directory
            descriptions_dir = output_base_dir / "descriptions" / action_class
            descriptions_dir.mkdir(parents=True, exist_ok=True)

            description_path = descriptions_dir / f"{video_stem}.txt"

            result = {
                'worker_id': worker_id,
                'video': str(bbox_video_path),
                'action_class': action_class,
                'description_generated': False,
                'errors': []
            }

            # Skip if exists
            if skip_existing and description_path.exists():
                result['description_generated'] = True
                result['skipped'] = True
                result_queue.put(result)
                continue

            # Extract frames from bbox video
            logger.debug(f"Worker {worker_id}: Extracting frames from {video_stem}")
            frames = extract_frames_for_vlm(bbox_video_path)

            if not frames:
                result['errors'].append("Failed to extract frames")
                result_queue.put(result)
                continue

            # Use the optimized robotic prompt
            prompt = """Rewrite only the physical motion as a sequence of imperative commands. Do not mention a person, actor, bounding box, body type, clothing, scene, environment, animals, spectators, or camera. Mention only objects that the action physically interacts with. Do not narrate, summarize, or describe anything. Use only imperative verbs. Start immediately with a command. Include fine-grained movement details (body position, limb use, sequence of steps). The output must be a concise instruction that someone could follow to reproduce the motion precisely." Format requirement: 1–3 short sentences, each in pure imperative form. Examples of desired output: "Slide into a split, roll forward across the floor, and push up into a low crouch.", "Position the metal sheet under the press brake, lower the tooling in one smooth motion, then lift the bent piece and place it aside.","Kneel beside the tree, lift the ornament with your right hand, and hook it onto the mid-level branch while stabilizing the tree with your left.", "Raise the object overhead, step forward, and swing it downward in a controlled arc."""

            try:
                logger.debug(f"Worker {worker_id}: Generating description for {video_stem}")
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
                        f.write(f"Source: {bbox_video_path.name}\n")
                        f.write(f"Timestamp: {datetime.now()}\n")
                        f.write(f"\nDescription:\n{description}\n")

                    result['description_generated'] = True
                    result['description'] = description
                    processed_count += 1

                    if processed_count % 10 == 0:
                        logger.info(f"Worker {worker_id}: Processed {processed_count} descriptions")
                else:
                    result['errors'].append(f"VLM error: {description}")
                    logger.error(f"Worker {worker_id}: VLM returned error for {video_stem}: {description}")

            except Exception as e:
                result['errors'].append(f"VLM exception: {str(e)}")
                logger.error(f"Worker {worker_id}: Exception for {video_stem}: {e}")
                logger.error(f"Worker {worker_id}: Traceback: {traceback.format_exc()}")

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

    logger.info(f"Worker {worker_id} finished. Processed {processed_count} descriptions.")


def collect_bbox_videos(bbox_video_dir: Path, start_idx: int = 0) -> List[Dict]:
    """Collect all bbox videos that need descriptions."""
    tasks = []

    # Get all bbox videos
    all_videos = sorted(list(bbox_video_dir.rglob("*_bbox.mp4")))
    logger.info(f"Found {len(all_videos)} bbox videos")

    # Skip to start_idx if resuming
    if start_idx > 0:
        all_videos = all_videos[start_idx:]
        logger.info(f"Resuming from index {start_idx}, processing {len(all_videos)} remaining videos")

    for video_path in all_videos:
        action_class = video_path.parent.name
        # Remove _bbox.mp4 suffix to get original name
        video_stem = video_path.stem[:-5]  # Remove "_bbox"

        tasks.append({
            'bbox_video_path': str(video_path),
            'action_class': action_class,
            'video_stem': video_stem
        })

    logger.info(f"Collected {len(tasks)} tasks")
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Generate descriptions for existing bbox videos")

    parser.add_argument('--bbox_video_dir', type=str,
                        default='/root/movement/data/kinetics_full_output/bbox_videos',
                        help='Directory containing bbox videos')
    parser.add_argument('--output_dir', type=str,
                        default='/root/movement/data/kinetics_full_output',
                        help='Output directory for descriptions')
    parser.add_argument('--gpus', type=str, default='1,2,3',
                        help='Comma-separated GPU IDs to use')
    parser.add_argument('--workers_per_gpu', type=int, default=1,
                        help='Number of worker processes per GPU')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip if description already exists')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index for resuming')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Checkpoint save interval')

    args = parser.parse_args()

    # Parse GPUs
    gpu_ids = [int(g) for g in args.gpus.split(',')]
    num_workers = len(gpu_ids) * args.workers_per_gpu

    logger.info(f"Starting description generation")
    logger.info(f"BBox video directory: {args.bbox_video_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Using GPUs: {gpu_ids}")
    logger.info(f"Workers per GPU: {args.workers_per_gpu}")
    logger.info(f"Total workers: {num_workers}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all tasks
    tasks = collect_bbox_videos(Path(args.bbox_video_dir), args.start_idx)

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

            # Check for worker load failures
            if 'error' in result and 'VLM failed to load' in result.get('error', ''):
                logger.error(f"Critical: {result['error']}")
                stats['errors'] += 1
                stats['processed'] += 1
                continue

            results.append(result)
            stats['processed'] += 1

            if result.get('description_generated'):
                stats['descriptions_generated'] += 1
            if result.get('errors'):
                stats['errors'] += 1
            if result.get('skipped'):
                stats['skipped'] += 1

            # Progress update
            if stats['processed'] % 100 == 0:
                elapsed = time.time() - start_time
                rate = stats['processed'] / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - stats['processed']) / rate if rate > 0 else 0

                logger.info(f"Progress: {stats['processed']}/{len(tasks)} "
                          f"({stats['processed']*100/len(tasks):.1f}%) "
                          f"Rate: {rate:.1f} videos/sec, ETA: {eta/3600:.1f}h")
                logger.info(f"  Descriptions: {stats['descriptions_generated']}, "
                          f"Errors: {stats['errors']}, Skipped: {stats['skipped']}")

            # Save checkpoint
            if stats['processed'] - last_checkpoint >= args.batch_size:
                checkpoint = {
                    'timestamp': datetime.now().isoformat(),
                    'last_index': args.start_idx + stats['processed'],
                    'stats': stats,
                    'elapsed_time': time.time() - start_time
                }

                checkpoint_file = output_dir / "description_checkpoint.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f, indent=2)

                logger.info(f"Checkpoint saved at index {checkpoint['last_index']}")
                last_checkpoint = stats['processed']

        except Empty:
            # Check if workers are still alive
            alive = sum(1 for w in workers if w.is_alive())
            if alive == 0:
                logger.warning("All workers have died!")
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
        'descriptions_generated': stats['descriptions_generated'],
        'errors': stats['errors'],
        'skipped': stats['skipped'],
        'total_time_seconds': time.time() - start_time,
        'processing_rate': stats['processed'] / (time.time() - start_time) if time.time() - start_time > 0 else 0
    }

    stats_file = output_dir / "description_final_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(final_stats, f, indent=2)

    # Save detailed results
    if results:
        results_file = output_dir / f"description_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

    logger.info("\n" + "="*80)
    logger.info("DESCRIPTION GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total processed: {stats['processed']}/{len(tasks)}")
    logger.info(f"Descriptions generated: {stats['descriptions_generated']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Skipped: {stats['skipped']}")
    logger.info(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
    logger.info(f"Processing rate: {final_stats['processing_rate']:.2f} videos/second")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()