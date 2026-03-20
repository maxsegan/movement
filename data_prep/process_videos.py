#!/usr/bin/env python3
"""
Batch processing script for Kinetics-700 dataset.
Processes videos with YOLOv8x TensorRT detection and pose estimation.
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Process, Queue, Manager, set_start_method
import queue

import numpy as np
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def setup_logging(log_dir: Path, worker_id: int = 0):
    """Setup logging configuration for worker."""
    log_file = log_dir / f"processing_log_worker_{worker_id}.txt"

    logger = logging.getLogger(f"worker_{worker_id}")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if worker_id == 0:  # Only main worker logs to console
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def find_videos(data_root: Path, start_idx: int = 0, limit: Optional[int] = None) -> List[Path]:
    """Find all video files in Kinetics dataset."""
    # Find all MP4 files in Kinetics structure
    all_videos = sorted(list(data_root.rglob("*.mp4")))

    # Apply start index and limit
    if start_idx > 0:
        all_videos = all_videos[start_idx:]

    if limit is not None:
        all_videos = all_videos[:limit]

    return all_videos


def process_video_worker(
    worker_id: int,
    gpu_id: int,
    video_queue: Queue,
    result_queue: Queue,
    output_dir: Path,
    target_fps: float,
    save_debug_videos: bool,
    verbose: bool = False
):
    """
    Worker process that runs on a specific GPU.
    Processes videos from the queue.
    """
    # Set CUDA device for this worker
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = "cuda:0"  # Since we set CUDA_VISIBLE_DEVICES, always use cuda:0

    # Setup logging for this worker
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_dir, worker_id)
    logger.info(f"Worker {worker_id} started on GPU {gpu_id}")

    # Import here to avoid loading models in main process
    from data_prep.pipeline.pipeline import process_video
    from transformers import AutoImageProcessor, VitPoseForPoseEstimation
    from data_prep.pose3d import build_motionagformer

    # Add MotionAGFormer to path and import
    sys.path.append(str(Path(__file__).parent.parent / 'MotionAGFormer'))
    from model.MotionAGFormer import MotionAGFormer

    try:
        # Load models for this worker
        logger.info(f"Worker {worker_id}: Loading models...")

        # ViTPose model - use float32 for stability
        vitpose_model_id = "usyd-community/vitpose-plus-large"
        vitpose_processor = AutoImageProcessor.from_pretrained(
            vitpose_model_id,
            trust_remote_code=True
        )

        # Use float32 to avoid dtype mismatch issues
        vitpose_model = VitPoseForPoseEstimation.from_pretrained(
            vitpose_model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        vitpose_model.to(device)
        vitpose_model.eval()

        # Enable mixed precision training for speed (autocast will handle it)
        torch.backends.cudnn.benchmark = True

        # MotionAGFormer for 3D pose lifting
        logger.info("Loading MotionAGFormer for 3D pose lifting...")
        ckpt_path = Path('/root/movement/models/motionagformer-b-h36m.pth.tr')
        if ckpt_path.exists():
            model_3d = build_motionagformer(MotionAGFormer, device, ckpt_path)
            logger.info("MotionAGFormer loaded successfully")
        else:
            logger.warning(f"MotionAGFormer checkpoint not found at {ckpt_path}, 3D lifting disabled")
            model_3d = None

        logger.info(f"Worker {worker_id}: Models loaded successfully")

        # Create output directories based on action classes
        # Videos will be organized by action class subdirectories

        # Process videos from queue
        while True:
            try:
                video_path = video_queue.get(timeout=5)

                if video_path is None:  # Poison pill to stop worker
                    break

                start_time = time.time()

                # Get video info
                action_class = video_path.parent.name
                video_name = video_path.stem

                # Create output directory for this action class
                action_output_dir = output_dir / action_class
                action_output_dir.mkdir(parents=True, exist_ok=True)

                if verbose:
                    logger.info(f"Worker {worker_id}: Processing {action_class}/{video_name}")

                try:
                    # Process video
                    result = process_video(
                        str(video_path),
                        action_output_dir,
                        target_fps=target_fps,
                        device=device,
                        vitpose_processor=vitpose_processor,
                        vitpose_model=vitpose_model,
                        debug=save_debug_videos,
                        model3d="MotionAGFormer.model.MotionAGFormer:MotionAGFormer" if model_3d else None,
                        ckpt3d=str(ckpt_path) if model_3d else None,
                        verbose=verbose,
                    )

                    elapsed = time.time() - start_time

                    if result and 'npz' in result:
                        # Load NPZ to validate BEFORE considering it a success
                        npz_path = Path(result['npz'])
                        if npz_path.exists():
                            data = np.load(npz_path, allow_pickle=True)

                            from data_prep.clip_filtering import validate_clip_improved

                            keypoints = data.get('keypoints2d', np.array([]))
                            scores = data.get('scores2d', np.ones_like(keypoints[..., 0]))
                            bboxes = data.get('bboxes', np.full((keypoints.shape[0], 4), np.nan))

                            is_valid, issues, classification = validate_clip_improved(
                                keypoints, scores, bboxes,
                                video_path=str(video_path),
                                min_confidence=0.3,
                                min_frames=20,
                                verbose=False
                            )

                            if classification == "VALID":
                                if verbose:
                                    logger.info(f"Worker {worker_id}: ✅ VALID: {action_class}/{video_name} ({elapsed:.1f}s)")
                                    logger.info(f"  Quality: {data.get('quality', [0])[0]:.3f}")

                                result_queue.put({
                                    'status': 'valid',
                                    'video': str(video_path),
                                    'action_class': action_class,
                                    'time': elapsed,
                                    'worker_id': worker_id,
                                    'npz_path': str(result['npz']),
                                    'quality': float(data.get('quality', [0])[0])
                                })
                            else:
                                if verbose:
                                    logger.info(f"Worker {worker_id}: ⚠️ {classification}: {action_class}/{video_name} ({elapsed:.1f}s)")
                                    logger.info(f"  Issues: {', '.join(issues)}, Quality: {data.get('quality', [0])[0]:.3f}")
                                    logger.info(f"  Deleting NPZ file...")

                                npz_path.unlink()

                                result_queue.put({
                                    'status': 'filtered',
                                    'classification': classification,
                                    'video': str(video_path),
                                    'action_class': action_class,
                                    'time': elapsed,
                                    'worker_id': worker_id,
                                    'issues': issues
                                })
                        else:
                            if verbose:
                                logger.warning(f"Worker {worker_id}: ⚠️ NPZ not found: {action_class}/{video_name}")
                            result_queue.put({
                                'status': 'failed',
                                'video': str(video_path),
                                'action_class': action_class,
                                'error': 'NPZ file not found',
                                'worker_id': worker_id
                            })
                    else:
                        if verbose:
                            logger.warning(f"Worker {worker_id}: ⚠️ Failed: {action_class}/{video_name} - No output")
                        result_queue.put({
                            'status': 'failed',
                            'video': str(video_path),
                            'action_class': action_class,
                            'error': 'No output',
                            'worker_id': worker_id
                        })

                except Exception as e:
                    logger.error(f"Worker {worker_id}: ❌ Error processing {action_class}/{video_name}: {e}")
                    result_queue.put({
                        'status': 'error',
                        'video': str(video_path),
                        'action_class': action_class,
                        'error': str(e),
                        'worker_id': worker_id
                    })

            except queue.Empty:
                continue
            except BrokenPipeError:
                logger.warning(f"Worker {worker_id}: Queue closed, shutting down")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id}: Unexpected error: {e}")
                continue

    except Exception as e:
        logger.error(f"Worker {worker_id} fatal error: {e}")

    logger.info(f"Worker {worker_id} shutting down")


def result_collector(result_queue: Queue, total_videos: int, output_dir: Path, num_workers: int):
    """Collects results from all workers and saves statistics."""
    results = []
    valid_count = 0
    filtered_count = 0
    failed_count = 0
    error_count = 0

    pbar = tqdm(total=total_videos, desc="Processing videos")

    # Statistics tracking
    action_class_stats = {}
    processing_times = []
    start_time = time.time()  # Track wall-clock time

    while len(results) < total_videos:
        try:
            result = result_queue.get(timeout=60)  # Longer timeout for robustness
            results.append(result)

            # Update counts
            if result['status'] == 'valid':
                valid_count += 1
                processing_times.append(result['time'])
            elif result['status'] == 'filtered':
                filtered_count += 1
            elif result['status'] == 'failed':
                failed_count += 1
            else:
                error_count += 1

            # Track by action class
            action_class = result['action_class']
            if action_class not in action_class_stats:
                action_class_stats[action_class] = {'valid': 0, 'filtered': 0, 'failed': 0, 'error': 0}

            status = result['status'] if result['status'] in ['valid', 'filtered', 'failed'] else 'error'
            action_class_stats[action_class][status] += 1

            # Update progress
            pbar.update(1)
            pbar.set_postfix({
                'valid': valid_count,
                'filtered': filtered_count,
                'failed': failed_count,
                'error': error_count
            })

            # Save intermediate results and log stats every 100 videos
            if len(results) % 100 == 0:
                wall_clock_time = time.time() - start_time
                save_statistics(results, action_class_stats, processing_times, output_dir, num_workers, wall_clock_time)

                # Log aggregate stats
                avg_time = np.mean(processing_times) if processing_times else 0
                throughput = (len(results) / (wall_clock_time / 60)) if wall_clock_time > 0 else 0
                print(f"\n📊 Progress Update: {len(results)}/{total_videos} videos processed")
                print(f"   ✅ Valid: {valid_count} ({valid_count/len(results)*100:.1f}%)")
                print(f"   ⚠️  Filtered: {filtered_count} ({filtered_count/len(results)*100:.1f}%)")
                print(f"   ⚠️  Failed: {failed_count} ({failed_count/len(results)*100:.1f}%)")
                print(f"   ❌ Error: {error_count} ({error_count/len(results)*100:.1f}%)")
                print(f"   ⏱️  Avg time: {avg_time:.1f}s/video")
                print(f"   🚀 Throughput: {throughput:.1f} videos/minute")
                print(f"   ⏰ Elapsed: {wall_clock_time/3600:.2f} hours\n")

        except queue.Empty:
            if len(results) < total_videos:
                print(f"Warning: Result collection timeout. Got {len(results)}/{total_videos} results")
                print(f"Continuing to wait...")
                continue  # Keep waiting instead of breaking

    pbar.close()

    # Save final statistics
    wall_clock_time = time.time() - start_time
    save_statistics(results, action_class_stats, processing_times, output_dir, num_workers, wall_clock_time)

    return results


def save_statistics(results, action_class_stats, processing_times, output_dir, num_workers, wall_clock_time):
    """Save processing statistics to file."""
    stats_file = output_dir / "processing_stats.json"

    total = len(results)
    valid = sum(1 for r in results if r['status'] == 'valid')
    filtered = sum(1 for r in results if r['status'] == 'filtered')
    failed = sum(1 for r in results if r['status'] == 'failed')
    error = total - valid - filtered - failed

    cumulative_time = sum(processing_times) if processing_times else 0

    stats = {
        'summary': {
            'total_processed': total,
            'valid': valid,
            'filtered': filtered,
            'failed': failed,
            'error': error,
            'valid_rate': (valid / total * 100) if total > 0 else 0
        },
        'by_action_class': action_class_stats,
        'performance': {
            'avg_time_per_video': np.mean(processing_times) if processing_times else 0,
            'median_time_per_video': np.median(processing_times) if processing_times else 0,
            'cumulative_processing_time_seconds': cumulative_time,
            'wall_clock_time_seconds': wall_clock_time,
            'num_workers': num_workers,
            'throughput_videos_per_minute': (total / (wall_clock_time / 60)) if wall_clock_time > 0 else 0,
            'per_worker_throughput': (total / num_workers / (wall_clock_time / 60)) if wall_clock_time > 0 and num_workers > 0 else 0
        },
        'timestamp': datetime.now().isoformat()
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Create summary markdown
    summary_file = output_dir / "README.md"
    with open(summary_file, 'w') as f:
        f.write("# Kinetics-700 Processing Results\n\n")
        f.write(f"**Total Videos Processed:** {total}\n\n")
        f.write("## Summary\n\n")
        f.write(f"| Status | Count | Percentage |\n")
        f.write(f"|--------|-------|------------|\n")
        f.write(f"| ✅ Valid | {valid} | {valid/total*100:.1f}% |\n")
        f.write(f"| ⚠️ Filtered | {filtered} | {filtered/total*100:.1f}% |\n")
        f.write(f"| ⚠️ Failed | {failed} | {failed/total*100:.1f}% |\n")
        f.write(f"| ❌ Error | {error} | {error/total*100:.1f}% |\n\n")

        if processing_times:
            f.write("## Performance\n\n")
            f.write(f"- Average processing time: {np.mean(processing_times):.2f} seconds/video\n")
            f.write(f"- Median processing time: {np.median(processing_times):.2f} seconds/video\n")
            f.write(f"- Cumulative processing time: {cumulative_time/3600:.2f} hours\n")
            f.write(f"- Wall-clock time: {wall_clock_time/3600:.2f} hours\n")
            f.write(f"- Number of workers: {num_workers}\n")
            f.write(f"- Throughput: {(total / (wall_clock_time / 60)) if wall_clock_time > 0 else 0:.1f} videos/minute\n")
            f.write(f"- Per-worker throughput: {(total / num_workers / (wall_clock_time / 60)) if wall_clock_time > 0 and num_workers > 0 else 0:.1f} videos/minute\n\n")

        f.write("## Action Classes Processed\n\n")
        f.write(f"Total unique action classes: {len(action_class_stats)}\n\n")

        # Sort action classes by total videos
        sorted_classes = sorted(action_class_stats.items(),
                              key=lambda x: sum(x[1].values()),
                              reverse=True)

        if len(sorted_classes) <= 20:
            # Show all if not too many
            f.write("| Action Class | Valid | Filtered | Failed | Error | Total |\n")
            f.write("|--------------|-------|----------|--------|-------|-------|\n")
            for action, stats in sorted_classes:
                total_class = sum(stats.values())
                f.write(f"| {action} | {stats.get('valid', 0)} | {stats.get('filtered', 0)} | {stats.get('failed', 0)} | {stats.get('error', 0)} | {total_class} |\n")
        else:
            # Show top 10 action classes
            f.write("### Top 10 Action Classes\n\n")
            f.write("| Action Class | Valid | Filtered | Failed | Error | Total |\n")
            f.write("|--------------|-------|----------|--------|-------|-------|\n")
            for action, stats in sorted_classes[:10]:
                total_class = sum(stats.values())
                f.write(f"| {action} | {stats.get('valid', 0)} | {stats.get('filtered', 0)} | {stats.get('failed', 0)} | {stats.get('error', 0)} | {total_class} |\n")

            f.write(f"\n*... and {len(sorted_classes) - 10} more action classes*\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Process Kinetics-700 videos with YOLOv8x TensorRT detection'
    )

    # Dataset arguments
    parser.add_argument('--data_root', type=str,
                        default='/root/movement/data/kinetics-dataset/k700-2020/train',
                        help='Root directory of Kinetics dataset')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for processed files')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index for processing (for resuming)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of videos to process')

    # Processing settings
    parser.add_argument('--target_fps', type=float, default=30.0,
                        help='Target FPS for processing')
    parser.add_argument('--debug', action='store_true',
                        help='Save debug visualization videos')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose per-video logging')

    # Parallelization settings
    parser.add_argument('--workers_per_gpu', type=int, default=2,
                        help='Number of worker processes per GPU')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='Comma-separated list of GPU IDs to use')

    args = parser.parse_args()

    # Parse GPU list
    gpu_ids = [int(g) for g in args.gpus.split(',')]
    num_workers = len(gpu_ids) * args.workers_per_gpu

    # Setup directories
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup main logger
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_dir, 0)

    # Print configuration
    logger.info("="*60)
    logger.info("Kinetics-700 Video Processing with YOLOv8x TensorRT")
    logger.info("="*60)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Target FPS: {args.target_fps}")
    logger.info(f"Debug videos: {args.debug}")
    logger.info(f"Verbose logging: {args.verbose}")
    logger.info(f"Using {len(gpu_ids)} GPUs with {args.workers_per_gpu} workers each = {num_workers} total workers")
    logger.info("="*60)

    # Find videos
    if not data_root.exists():
        logger.error(f"Data root does not exist: {data_root}")
        sys.exit(1)

    videos = find_videos(data_root, args.start_idx, args.limit)
    logger.info(f"Found {len(videos)} videos to process")

    # Count by action class
    action_classes = {}
    for video in videos:
        action = video.parent.name
        action_classes[action] = action_classes.get(action, 0) + 1
    logger.info(f"Covering {len(action_classes)} unique action classes")

    if not videos:
        logger.error("No videos found!")
        sys.exit(1)

    # Create queues
    manager = Manager()
    video_queue = manager.Queue()
    result_queue = manager.Queue()

    # Add videos to queue
    for video in videos:
        video_queue.put(video)

    # Add poison pills for workers
    for _ in range(num_workers):
        video_queue.put(None)

    # Start worker processes
    workers = []
    worker_id = 0
    for gpu_id in gpu_ids:
        for _ in range(args.workers_per_gpu):
            p = Process(target=process_video_worker, args=(
                worker_id, gpu_id, video_queue, result_queue,
                out_dir, args.target_fps, args.debug, args.verbose
            ))
            p.start()
            workers.append(p)
            worker_id += 1
            time.sleep(2)  # Stagger worker starts to avoid memory spikes

    logger.info(f"Started {num_workers} worker processes")

    # Collect results in main process
    results = result_collector(result_queue, len(videos), out_dir, num_workers)

    # Wait for all workers to finish
    logger.info("Waiting for workers to complete...")
    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            logger.warning(f"Worker {p.pid} did not finish in time, terminating...")
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                logger.warning(f"Worker {p.pid} did not terminate, killing...")
                p.kill()

    # Print final summary
    logger.info("="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)

    valid = sum(1 for r in results if r['status'] == 'valid')
    filtered = sum(1 for r in results if r['status'] == 'filtered')
    failed = sum(1 for r in results if r['status'] == 'failed')
    error = sum(1 for r in results if r['status'] == 'error')

    if results:
        logger.info(f"Total processed: {len(results)}/{len(videos)}")
        logger.info(f"✅ Valid: {valid} ({valid/len(results)*100:.1f}%)")
        logger.info(f"⚠️ Filtered: {filtered} ({filtered/len(results)*100:.1f}%)")
        logger.info(f"⚠️ Failed: {failed} ({failed/len(results)*100:.1f}%)")
        logger.info(f"❌ Error: {error} ({error/len(results)*100:.1f}%)")

        # Performance stats - read from saved statistics
        stats_file = out_dir / "processing_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)

            perf = stats.get('performance', {})
            logger.info("="*60)
            logger.info("PERFORMANCE METRICS")
            logger.info(f"Average time per video: {perf.get('avg_time_per_video', 0):.2f} seconds")
            logger.info(f"Cumulative processing time: {perf.get('cumulative_processing_time_seconds', 0)/3600:.2f} hours")
            logger.info(f"Wall-clock time: {perf.get('wall_clock_time_seconds', 0)/3600:.2f} hours")
            logger.info(f"Throughput: {perf.get('throughput_videos_per_minute', 0):.1f} videos/minute")
            logger.info(f"Per-worker throughput: {perf.get('per_worker_throughput', 0):.1f} videos/minute")

    logger.info("="*60)
    logger.info(f"📁 Results saved to: {out_dir}")
    logger.info(f"   - Processed files organized by action class")
    logger.info(f"   - Statistics: {out_dir}/processing_stats.json")
    logger.info(f"   - Summary: {out_dir}/README.md")
    logger.info(f"   - Logs: {out_dir}/logs/")


if __name__ == "__main__":
    # Set spawn method for CUDA multiprocessing
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass  # Already set

    main()