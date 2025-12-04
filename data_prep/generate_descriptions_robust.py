#!/usr/bin/env python3
"""
Robust version of generate_descriptions_only.py with better error handling and retry logic.
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
from multiprocessing import get_context
from multiprocessing.connection import Connection, wait
from typing import List, Dict
from PIL import Image
import cv2
import numpy as np
import time
import traceback
import random

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


def load_model_with_retry(config, device, max_retries=3, worker_id=0):
    """Load VLM model with retry logic."""
    for attempt in range(max_retries):
        try:
            # Add random delay to avoid concurrent loading issues
            if attempt > 0:
                delay = random.uniform(1, 5) * attempt
                logger.info(f"Worker {worker_id}: Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}")
                time.sleep(delay)

            logger.info(f"Worker {worker_id}: Loading attempt {attempt + 1}/{max_retries}")

            # Set environment variable to avoid conflicts
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'

            vlm_model, vlm_processor = load_vlm_model(config, device=device)

            if vlm_model is None or vlm_processor is None:
                raise ValueError("Model or processor returned None")

            # Test the model
            test_img = Image.new('RGB', (224, 224), color='white')
            test_desc = generate_description_with_model(
                config["name"],
                vlm_model,
                vlm_processor,
                [test_img],
                "Test"
            )

            if not test_desc or test_desc.startswith("Error"):
                raise ValueError(f"Model test failed: {test_desc}")

            logger.info(f"Worker {worker_id}: Model loaded successfully on attempt {attempt + 1}")
            return vlm_model, vlm_processor

        except Exception as e:
            logger.error(f"Worker {worker_id}: Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Worker {worker_id}: All {max_retries} attempts failed")
                raise

            # Clear CUDA cache before retry
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return None, None


def process_worker(
    worker_id: int,
    gpu_id: int,
    tasks: List[Dict],
    output_base_dir: Path,
    skip_existing: bool,
    result_conn: Connection,
    start_delay: float = 0
):
    """Worker process for a specific GPU."""

    worker_start = time.time()

    # Add initial delay to stagger worker startup
    if start_delay > 0:
        logger.info(f"Worker {worker_id}: Waiting {start_delay:.1f}s before starting")
        time.sleep(start_delay)

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = "cuda:0"  # Visible device remapped per worker

    logger.info(f"Worker {worker_id} starting on GPU {gpu_id} with {len(tasks)} tasks")

    # Load VLM model with retry
    config = {
        "name": "qwen3vl-32b",
        "model_id": "Qwen/Qwen3-VL-32B-Instruct",
        "use_4bit": True,
    }

    processed_count = 0
    error_count = 0

    try:
        vlm_model, vlm_processor = load_model_with_retry(config, device, worker_id=worker_id)
        vlm_loaded = True
    except Exception as e:
        logger.error(f"Worker {worker_id}: Failed to load VLM after retries: {e}")
        vlm_loaded = False

    if not vlm_loaded:
        logger.error(f"Worker {worker_id}: Exiting due to VLM load failure")
        result_conn.send({
            'message_type': 'load_error',
            'worker_id': worker_id,
            'error': f"VLM failed to load on GPU {gpu_id}"
        })
        # Ensure cleanup message still happens via finally block
        return

    try:
        for task in tasks:
            bbox_video_path = Path(task['bbox_video_path'])
            action_class = task['action_class']
            video_stem = task['video_stem']

            # Create output directory
            descriptions_dir = output_base_dir / "descriptions" / action_class
            descriptions_dir.mkdir(parents=True, exist_ok=True)

            description_path = descriptions_dir / f"{video_stem}.txt"

            result = {
                'message_type': 'result',
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
                result_conn.send(result)
                continue

            # Extract frames from bbox video
            frames = extract_frames_for_vlm(bbox_video_path)

            if not frames:
                result['errors'].append("Failed to extract frames")
                result_conn.send(result)
                error_count += 1
                continue

            # Use the optimized robotic prompt
            prompt = """Rewrite only the physical motion as a sequence of imperative commands. Do not mention a person, actor, bounding box, body type, clothing, scene, environment, animals, spectators, or camera. Mention only objects that the action physically interacts with. Do not narrate, summarize, or describe anything. Use only imperative verbs. Start immediately with a command. Include fine-grained movement details (body position, limb use, sequence of steps). The output must be a concise instruction that someone could follow to reproduce the motion precisely." Format requirement: 1–3 short sentences, each in pure imperative form. Examples of desired output: "Slide into a split, roll forward across the floor, and push up into a low crouch.", "Position the metal sheet under the press brake, lower the tooling in one smooth motion, then lift the bent piece and place it aside.","Kneel beside the tree, lift the ornament with your right hand, and hook it onto the mid-level branch while stabilizing the tree with your left.", "Raise the object overhead, step forward, and swing it downward in a controlled arc."""

            try:
                description = generate_description_with_model(
                    config["name"],
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
                        logger.info(f"Worker {worker_id}: Processed {processed_count} descriptions, {error_count} errors")
                else:
                    result['errors'].append(f"VLM error: {description}")
                    error_count += 1

            except Exception as e:
                result['errors'].append(f"VLM exception: {str(e)}")
                error_count += 1
                logger.error(f"Worker {worker_id}: Exception for {video_stem}: {e}")

            result_conn.send(result)

    except Exception as e:
        logger.error(f"Worker {worker_id}: Fatal error: {e}")
        result_conn.send({
            'message_type': 'worker_exception',
            'worker_id': worker_id,
            'error': str(e)
        })
        traceback.print_exc()
    finally:
        # Clean up VLM model
        if 'vlm_model' in locals() and vlm_model is not None:
            del vlm_model
        if 'vlm_processor' in locals() and vlm_processor is not None:
            del vlm_processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Worker {worker_id} finished. Processed {processed_count} descriptions, {error_count} errors.")
        result_conn.send({
            'message_type': 'done',
            'worker_id': worker_id,
            'processed_count': processed_count,
            'error_count': error_count,
            'duration': time.time() - worker_start
        })
        result_conn.close()


def collect_bbox_videos(bbox_video_dir: Path, start_idx: int = 0, limit: int = None) -> List[Dict]:
    """Collect all bbox videos that need descriptions."""
    tasks = []

    # Get all bbox videos
    all_videos = sorted(list(bbox_video_dir.rglob("*_bbox.mp4")))
    logger.info(f"Found {len(all_videos)} bbox videos")

    # Skip to start_idx if resuming
    if start_idx > 0:
        all_videos = all_videos[start_idx:]
        logger.info(f"Resuming from index {start_idx}, processing {len(all_videos)} remaining videos")

    # Apply limit if specified
    if limit:
        all_videos = all_videos[:limit]
        logger.info(f"Limiting to {limit} videos for testing")

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
    parser = argparse.ArgumentParser(description="Generate descriptions for existing bbox videos with robust error handling")

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
    parser.add_argument('--test_mode', action='store_true',
                        help='Test mode - process only first 100 videos')
    parser.add_argument('--stagger_delay', type=float, default=10.0,
                        help='Delay in seconds between starting each worker')

    args = parser.parse_args()

    # Parse GPUs
    gpu_ids = [int(g) for g in args.gpus.split(',')]
    num_workers = len(gpu_ids) * args.workers_per_gpu

    logger.info(f"Starting robust description generation")
    logger.info(f"BBox video directory: {args.bbox_video_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Using GPUs: {gpu_ids}")
    logger.info(f"Workers per GPU: {args.workers_per_gpu}")
    logger.info(f"Total workers: {num_workers}")
    logger.info(f"Worker stagger delay: {args.stagger_delay}s")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all tasks
    limit = 100 if args.test_mode else None
    tasks = collect_bbox_videos(Path(args.bbox_video_dir), args.start_idx, limit)

    if not tasks:
        logger.error("No tasks found!")
        return

    # Set multiprocessing context that works in restricted environments
    ctx = get_context("spawn")

    # Evenly distribute tasks across workers to avoid runtime schedulers
    task_splits: List[List[Dict]] = [[] for _ in range(num_workers)]
    for idx, task in enumerate(tasks):
        task_splits[idx % num_workers].append(task)

    workers = []
    worker_connections: Dict[int, Connection] = {}
    conn_to_worker: Dict[Connection, int] = {}
    worker_id = 0

    for gpu_id in gpu_ids:
        for gpu_worker_idx in range(args.workers_per_gpu):
            if worker_id >= num_workers:
                break

            worker_tasks = task_splits[worker_id]
            if not worker_tasks:
                logger.info(f"No tasks assigned to worker {worker_id}, skipping launch")
                worker_id += 1
                continue

            # Calculate delay for this worker
            start_delay = worker_id * args.stagger_delay

            parent_conn, child_conn = ctx.Pipe(duplex=False)
            p = ctx.Process(
                target=process_worker,
                args=(
                    worker_id,
                    gpu_id,
                    worker_tasks,
                    output_dir,
                    args.skip_existing,
                    child_conn,
                    start_delay
                )
            )
            p.start()
            child_conn.close()

            workers.append((worker_id, p))
            worker_connections[worker_id] = parent_conn
            conn_to_worker[parent_conn] = worker_id

            logger.info(f"Started worker {worker_id} on GPU {gpu_id} with {start_delay:.1f}s delay "
                        f"handling {len(worker_tasks)} videos")
            worker_id += 1

    if not workers:
        logger.error("No workers started; verify GPU configuration.")
        return

    # Collect results via pipes
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
    worker_failures = 0
    worker_summaries = {}
    active_workers = set(worker_connections.keys())

    logger.info(f"Processing {len(tasks)} videos...")

    while active_workers:
        ready = wait([worker_connections[wid] for wid in active_workers], timeout=60)

        if not ready:
            alive = sum(1 for _, proc in workers if proc.is_alive())
            if alive == 0:
                logger.warning("All workers have exited but some summaries were not received.")
                break
            logger.debug(f"Waiting for results... {alive} workers still active")
            continue

        for conn in ready:
            try:
                message = conn.recv()
            except EOFError:
                wid = conn_to_worker.get(conn)
                if wid is not None:
                    active_workers.discard(wid)
                    logger.error(f"Worker {wid} connection closed unexpectedly")
                continue

            message_type = message.get('message_type', 'result')

            if message_type == 'result':
                result = {k: v for k, v in message.items() if k != 'message_type'}
                results.append(result)
                stats['processed'] += 1

                if result.get('description_generated'):
                    stats['descriptions_generated'] += 1
                if result.get('errors'):
                    stats['errors'] += 1
                if result.get('skipped'):
                    stats['skipped'] += 1

                # Progress update
                if stats['processed'] % 100 == 0 or stats['processed'] == len(tasks):
                    elapsed = time.time() - start_time
                    rate = stats['processed'] / elapsed if elapsed > 0 else 0
                    remaining = max(len(tasks) - stats['processed'], 0)
                    eta = remaining / rate if rate > 0 else 0

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

            elif message_type in {'load_error', 'worker_exception'}:
                worker_failures += 1
                logger.error(f"Critical worker issue (worker {message.get('worker_id')}, {message_type}): "
                             f"{message.get('error')}")

            elif message_type == 'done':
                worker_summary = {k: v for k, v in message.items() if k != 'message_type'}
                wid = worker_summary.get('worker_id')
                if wid is not None:
                    worker_summaries[wid] = worker_summary
                    active_workers.discard(wid)
                    conn.close()
            else:
                logger.warning(f"Unknown message type received: {message_type}")

    # Wait for workers to finish
    for wid, proc in workers:
        proc.join(timeout=30)
        if proc.is_alive():
            logger.warning(f"Worker {wid} still alive after timeout, terminating")
            proc.terminate()

    # Save final results
    final_stats = {
        'timestamp': datetime.now().isoformat(),
        'total_processed': stats['processed'],
        'descriptions_generated': stats['descriptions_generated'],
        'errors': stats['errors'],
        'skipped': stats['skipped'],
        'worker_failures': worker_failures,
        'worker_summaries': worker_summaries,
        'total_workers_launched': len(workers),
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
    logger.info(f"Worker failures: {worker_failures}")
    logger.info(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
    logger.info(f"Processing rate: {final_stats['processing_rate']:.2f} videos/second")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
