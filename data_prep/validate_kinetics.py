#!/usr/bin/env python3
"""
Unified parallelized validation script for Kinetics-700 dataset.
Combines multi-GPU processing with debug video generation.
"""

import sys
import os
import random
import json
import logging
import time
import shutil
import cv2
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from multiprocessing import Process, Queue, Manager, set_start_method
import queue

import numpy as np
import torch
from tqdm import tqdm

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))


def setup_logging(output_dir: Path, worker_id: int = 0):
    """Setup logging configuration for worker."""
    log_file = output_dir / f"validation_log_worker_{worker_id}.txt"

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


def sample_videos(data_root: Path, num_samples: int = 256, seed: int = 42) -> List[Path]:
    """
    Randomly sample videos from all Kinetics subdirectories with deterministic ordering.

    Using seed=42 by default ensures the same videos are sampled in the same order
    every time, making results reproducible and allowing for consistent testing.
    """
    # Set all random seeds for complete reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Find all video files and sort them for deterministic ordering
    all_videos = sorted(list(data_root.rglob("*.mp4")))

    if len(all_videos) < num_samples:
        print(f"Warning: Only found {len(all_videos)} videos, using all of them")
        return all_videos

    # Sample randomly with the fixed seed
    sampled = random.sample(all_videos, num_samples)

    # Sort by action class, then by filename for complete determinism
    sampled.sort(key=lambda x: (x.parent.name, x.name))

    return sampled


# Import the improved validation function
try:
    from data_prep.clip_filtering import validate_clip_improved
    use_improved = True
except ImportError:
    from data_prep.clip_filtering_strict import get_validation_status_strict
    use_improved = False

def get_validation_status(npz_data: dict, video_path: str = None) -> Tuple[str, List[str]]:
    """Determine validation status using improved validation criteria."""
    if use_improved:
        # Use improved activity-aware validation
        keypoints = npz_data.get('keypoints2d', np.array([]))
        scores = npz_data.get('scores2d', np.ones_like(keypoints[..., 0]))
        bboxes = npz_data.get('bboxes', np.full((keypoints.shape[0], 4), np.nan))
        has_hard_cuts = bool(npz_data.get('has_hard_cuts', [False])[0]) if 'has_hard_cuts' in npz_data else False
        hard_cut_frames = list(npz_data.get('hard_cut_frames', [])) if 'hard_cut_frames' in npz_data else []
        tracking_switches = list(npz_data.get('tracking_switches', [])) if 'tracking_switches' in npz_data else []

        is_valid, issues, classification = validate_clip_improved(
            keypoints, scores, bboxes,
            video_path=video_path,
            min_confidence=0.3,
            min_frames=20,  # More lenient
            verbose=False,
            has_hard_cuts=has_hard_cuts,
            hard_cut_frames=hard_cut_frames,
            tracking_switches=tracking_switches
        )
        return classification, issues
    else:
        # Fallback to strict validation
        return get_validation_status_strict(npz_data)


def create_debug_video(
    video_path: Path,
    npz_path: Path,
    output_path: Path,
    max_frames: int = 150
) -> bool:
    """Create debug video with skeleton overlay."""
    try:
        # Load pose data
        data = np.load(npz_path, allow_pickle=True)

        if 'keypoints2d' not in data:
            return False

        keypoints = data['keypoints2d']
        scores = data.get('scores2d', np.ones_like(keypoints[..., 0]))
        indices = data.get('indices', np.arange(keypoints.shape[0]))
        bboxes = data.get('bboxes', np.full((keypoints.shape[0], 4), np.nan))

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resize to 480p while maintaining aspect ratio
        target_height = 480
        aspect_ratio = orig_width / orig_height
        target_width = int(target_height * aspect_ratio)
        # Make width even for codec compatibility
        if target_width % 2 != 0:
            target_width += 1

        # Create output video at 480p
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, min(fps / 3, 10), (target_width, target_height))

        # H36M skeleton connections
        skeleton = [
            [0, 1], [1, 2], [2, 3],  # Right leg
            [0, 4], [4, 5], [5, 6],  # Left leg
            [0, 7], [7, 8], [8, 9], [9, 10],  # Spine to head
            [8, 11], [11, 12], [12, 13],  # Right arm
            [8, 14], [14, 15], [15, 16],  # Left arm
        ]

        # Calculate scaling factor for keypoints
        scale_x = target_width / orig_width
        scale_y = target_height / orig_height

        frame_idx = 0
        processed = 0

        while cap.isOpened() and processed < min(max_frames, len(indices)):
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to 480p
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            # Check if this frame has pose data
            if frame_idx in indices:
                pose_idx = np.where(indices == frame_idx)[0]
                if len(pose_idx) > 0:
                    pose_idx = pose_idx[0]

                    # Draw bounding box if available (scale coordinates)
                    if pose_idx < len(bboxes) and not np.any(np.isnan(bboxes[pose_idx])):
                        x1, y1, x2, y2 = bboxes[pose_idx]
                        x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)

                        # Draw main tracking box (solid green)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "Tracked", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.5, (0, 255, 0), 1, cv2.LINE_AA)

                        # Draw buffered box used for pose detection (dashed blue)
                        buffer_ratio = 0.2
                        width = x2 - x1
                        height = y2 - y1
                        buffer_x = int(width * buffer_ratio)
                        buffer_y = int(height * buffer_ratio)

                        x1_buf = max(0, x1 - buffer_x)
                        y1_buf = max(0, y1 - buffer_y)
                        x2_buf = min(target_width, x2 + buffer_x)
                        y2_buf = min(target_height, y2 + buffer_y)

                        # Draw buffered box with dashed line effect
                        cv2.rectangle(frame, (x1_buf, y1_buf), (x2_buf, y2_buf), (255, 200, 0), 1)
                        cv2.putText(frame, "Detection Area", (x1_buf, y1_buf-5), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.4, (255, 200, 0), 1, cv2.LINE_AA)

                    # Draw skeleton (scale keypoints)
                    kpts = keypoints[pose_idx].copy()
                    kpts[:, 0] *= scale_x
                    kpts[:, 1] *= scale_y
                    scrs = scores[pose_idx]

                    # Color based on score
                    for j, (x, y) in enumerate(kpts):
                        if x > 0 and y > 0:
                            color = (0, 255, 0) if scrs[j] > 0.5 else (0, 255, 255) if scrs[j] > 0.3 else (0, 0, 255)
                            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
                            cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), 1)

                    # Draw bones
                    for connection in skeleton:
                        if len(connection) == 2:
                            j1, j2 = connection
                            if j1 < len(kpts) and j2 < len(kpts):
                                if (scrs[j1] > 0.3 and scrs[j2] > 0.3 and
                                    kpts[j1][0] > 0 and kpts[j2][0] > 0):
                                    pt1 = (int(kpts[j1][0]), int(kpts[j1][1]))
                                    pt2 = (int(kpts[j2][0]), int(kpts[j2][1]))
                                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

                    processed += 1

            # Add status text
            status, reasons = get_validation_status(data, str(video_path))

            # Status bar background
            cv2.rectangle(frame, (0, 0), (target_width, 100), (0, 0, 0), -1)

            # Status text
            text = f"Frame {frame_idx}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} | Status: {status}"
            color = (0, 255, 0) if status == "VALID" else (0, 0, 255) if status == "INVALID" else (0, 255, 255)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, color, 2, cv2.LINE_AA)

            if reasons:
                reason_text = f"Issues: {', '.join(reasons)}"
                cv2.putText(frame, reason_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Add quality score if available
            if 'quality' in data:
                quality_text = f"Quality: {float(data['quality'][0]):.3f}"
                cv2.putText(frame, quality_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (255, 255, 255), 1, cv2.LINE_AA)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        return True

    except Exception as e:
        logging.error(f"Error creating debug video: {e}")
        return False


def save_description(npz_path: Path, output_path: Path) -> bool:
    """Extract and save VLM description to text file."""
    try:
        data = np.load(npz_path, allow_pickle=True)

        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"Video Analysis: {npz_path.stem}\n")
            f.write("=" * 60 + "\n\n")

            # Basic info
            if 'meta' in data:
                meta = data['meta']
                f.write(f"Resolution: {meta[2]:.0f}x{meta[3]:.0f}\n")
                f.write(f"Frames: {meta[1]:.0f} @ {meta[0]:.1f} FPS\n\n")

            # Validation status
            status, reasons = get_validation_status(data, str(video_path))
            f.write(f"Validation Status: {status}\n")
            if reasons:
                f.write(f"Issues: {', '.join(reasons)}\n")
            f.write("\n")

            # Quality metrics
            if 'quality' in data:
                f.write(f"Quality Score: {float(data['quality'][0]):.3f}\n")
            if 'density_ok' in data:
                f.write(f"Density OK: {data['density_ok'][0]}\n")
            if 'dynamic_ok' in data:
                f.write(f"Dynamic OK: {data['dynamic_ok'][0]}\n")
            f.write("\n")

            # VLM descriptions
            if 'action_descriptions' in data:
                desc_json = str(data['action_descriptions'][0])
                if desc_json != '[]':
                    descriptions = json.loads(desc_json)
                    if descriptions:
                        f.write("-" * 40 + "\n")
                        f.write("ACTION DESCRIPTIONS\n")
                        f.write("-" * 40 + "\n\n")

                        for i, desc in enumerate(descriptions, 1):
                            f.write(f"Segment {i}:\n")
                            if 'frames' in desc:
                                frames = desc['frames']
                                if isinstance(frames, list) and len(frames) > 0:
                                    f.write(f"  Frames: {frames[0]} to {frames[-1]}\n")

                            if 'description' in desc:
                                text = desc['description']
                                if text.startswith('"') and text.endswith('"'):
                                    text = text[1:-1]
                                f.write(f"\n  Action:\n")
                                wrapped = textwrap.wrap(text, width=70, initial_indent="    ",
                                                       subsequent_indent="    ")
                                f.write('\n'.join(wrapped) + "\n")

                            if 'confidence' in desc:
                                f.write(f"\n  Confidence: {desc['confidence']:.2f}\n")

                            f.write("\n")

        return True

    except Exception as e:
        logging.error(f"Error saving description: {e}")
        return False


def process_video_worker(
    worker_id: int,
    gpu_id: int,
    video_queue: Queue,
    result_queue: Queue,
    output_dir: Path,
    target_fps: float,
    enable_vlm: bool,
    create_debug: bool,
    max_frames: int
):
    """
    Worker process that runs on a specific GPU.
    Processes videos from the queue and creates debug videos.
    """
    # Set CUDA device for this worker
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = "cuda:0"  # Since we set CUDA_VISIBLE_DEVICES, always use cuda:0

    # Setup logging for this worker
    logger = setup_logging(output_dir, worker_id)
    logger.info(f"Worker {worker_id} started on GPU {gpu_id}")

    # Import here to avoid loading models in main process
    from data_prep.pipeline.pipeline import process_video
    from data_prep.video_action_description import FastVideoActionDescriber as VideoActionDescriber
    from torchvision import models
    from transformers import AutoImageProcessor, VitPoseForPoseEstimation

    try:
        # Load models for this worker
        logger.info(f"Worker {worker_id}: Loading models...")

        # Detection model
        det_model = models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        det_model.to(device)
        det_model.eval()

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

        # VLM if enabled
        action_describer = None
        if enable_vlm:
            action_describer = VideoActionDescriber(device=device)

        logger.info(f"Worker {worker_id}: Models loaded successfully")

        # Create output directories
        npz_dir = output_dir / "npz_files"
        video_dir = output_dir / "debug_videos"
        desc_dir = output_dir / "descriptions"

        npz_dir.mkdir(exist_ok=True, parents=True)
        video_dir.mkdir(exist_ok=True, parents=True)
        desc_dir.mkdir(exist_ok=True, parents=True)

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

                logger.info(f"Worker {worker_id}: Processing {action_class}/{video_name}")

                try:
                    # Process video
                    result = process_video(
                        str(video_path),
                        npz_dir,
                        target_fps=target_fps,
                        device=device,
                        det_2d_model=det_model,
                        vitpose_processor=vitpose_processor,
                        vitpose_model=vitpose_model,
                        action_describer=action_describer,
                        enable_action_description=enable_vlm,
                        debug=create_debug  # Enable debug mode in pipeline if debug videos requested
                    )

                    if result and 'npz' in result:
                        # Load NPZ data for validation
                        npz_path = Path(result['npz'])
                        data = np.load(npz_path, allow_pickle=True)

                        # Get validation status
                        status, reasons = get_validation_status(data, str(video_path))

                        # Create descriptive filename
                        reason_str = "_".join(reasons[:2]) if reasons else ""  # Limit reasons
                        new_name = f"{status}_{action_class.replace(' ', '_')}_{video_name}"
                        if reason_str:
                            new_name = f"{new_name}_{reason_str}"
                        # Truncate if too long
                        if len(new_name) > 200:
                            new_name = new_name[:200]

                        # Move NPZ to permanent location with new name
                        new_npz_path = npz_dir / f"{new_name}.npz"
                        shutil.move(str(npz_path), str(new_npz_path))

                        # Use debug video from pipeline if it was created
                        debug_created = False
                        debug_path = None
                        if create_debug and 'debug' in result and result['debug']:
                            # Pipeline created a debug video - copy it to our output dir
                            pipeline_debug_path = Path(result['debug'])
                            if pipeline_debug_path.exists():
                                debug_path = video_dir / f"{new_name}.mp4"
                                shutil.copy2(pipeline_debug_path, debug_path)
                                debug_created = True
                                logger.info(f"Worker {worker_id}: Using pipeline debug video: {debug_path.name}")

                        # Save description
                        desc_created = False
                        if enable_vlm:
                            desc_path = desc_dir / f"{new_name}.txt"
                            desc_created = save_description(new_npz_path, desc_path)
                            if desc_created:
                                logger.info(f"Worker {worker_id}: Saved description: {desc_path.name}")

                        # Create result
                        result_data = {
                            'video': str(video_path),
                            'action_class': action_class,
                            'status': status,
                            'reasons': reasons,
                            'quality': float(data['quality'][0]) if 'quality' in data else 0.0,
                            'density_ok': bool(data['density_ok'][0]) if 'density_ok' in data else False,
                            'dynamic_ok': bool(data['dynamic_ok'][0]) if 'dynamic_ok' in data else False,
                            'npz_path': str(new_npz_path),
                            'debug_video': str(debug_path) if debug_path and debug_created else None,
                            'description': str(desc_path) if desc_created else None,
                            'processing_time': time.time() - start_time,
                            'worker_id': worker_id
                        }

                        result_queue.put(result_data)
                    else:
                        logger.warning(f"Worker {worker_id}: Failed to process {video_name}")
                        result_queue.put({
                            'video': str(video_path),
                            'action_class': action_class,
                            'status': 'FAILED',
                            'reasons': ['PROCESSING_ERROR'],
                            'worker_id': worker_id
                        })

                except Exception as e:
                    logger.error(f"Worker {worker_id}: Error processing {video_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    result_queue.put({
                        'video': str(video_path),
                        'action_class': action_class,
                        'status': 'FAILED',
                        'reasons': [str(e)],
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


def result_collector(result_queue: Queue, output_dir: Path, total_videos: int, max_frames: int = 150, target_fps: float = 10.0, processes: list = None):
    """
    Collects results from all workers and saves them.
    """
    results = []
    valid_count = 0
    invalid_count = 0
    partial_count = 0
    failed_count = 0

    pbar = tqdm(total=total_videos, desc="Processing videos")

    timeout_count = 0
    max_timeouts = 1  # Only 1 retry as requested
    timeout_seconds = 300  # 5 minutes timeout

    while len(results) < total_videos and timeout_count < max_timeouts:
        try:
            result = result_queue.get(timeout=timeout_seconds)
            results.append(result)
            timeout_count = 0  # Reset timeout counter on success

            # Update counts
            if result['status'] == "VALID":
                valid_count += 1
            elif result['status'] == "INVALID":
                invalid_count += 1
            elif result['status'] == "PARTIAL":
                partial_count += 1
            else:
                failed_count += 1

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'valid': valid_count,
                'invalid': invalid_count,
                'partial': partial_count,
                'failed': failed_count
            })

            # Save intermediate results every 50 videos
            if len(results) % 50 == 0:
                save_results(results, output_dir, valid_count, invalid_count, partial_count, failed_count)

        except queue.Empty:
            timeout_count += 1
            print(f"Warning: Result collector timeout #{timeout_count}. Got {len(results)}/{total_videos} results")
            print(f"Timeout was {timeout_seconds}s. Workers may be taking longer with max_frames={max_frames}, target_fps={target_fps}")

            # Check if workers are still alive
            alive_workers = sum(1 for p in processes if p.is_alive())
            print(f"Workers still alive: {alive_workers}/{len(processes)}")

            if timeout_count >= max_timeouts:
                print("Max timeouts reached, stopping collection")
                print("This usually means workers crashed or processing is taking too long.")
                print("Try reducing --max_frames or --target_fps, or increasing worker timeout.")

                # Kill any remaining workers
                print("Terminating remaining worker processes...")
                if processes:
                    for p in processes:
                        if p.is_alive():
                            print(f"  Terminating worker process {p.pid}")
                            p.terminate()
                    time.sleep(2)  # Give them time to terminate gracefully
                    for p in processes:
                        if p.is_alive():
                            print(f"  Force killing worker process {p.pid}")
                            p.kill()
                break

    pbar.close()

    # Save final results
    save_results(results, output_dir, valid_count, invalid_count, partial_count, failed_count)

    return results


def save_results(results, output_dir, valid_count, invalid_count, partial_count, failed_count):
    """Save results to JSON and create summary."""
    report_path = output_dir / "validation_report.json"

    total = len(results)
    if total == 0:
        return

    with open(report_path, 'w') as f:
        json.dump({
            'summary': {
                'total': total,
                'valid': valid_count,
                'invalid': invalid_count,
                'partial': partial_count,
                'failed': failed_count,
                'valid_percentage': (valid_count / total * 100)
            },
            'results': results
        }, f, indent=2)

    # Create summary markdown
    summary_path = output_dir / "README.md"
    with open(summary_path, 'w') as f:
        f.write("# Kinetics-700 Validation Results (Parallelized)\n\n")
        f.write(f"**Samples Processed:** {total}\n\n")
        f.write("## Summary\n\n")
        f.write(f"| Status | Count | Percentage |\n")
        f.write(f"|--------|-------|------------|\n")
        f.write(f"| ✅ Valid | {valid_count} | {valid_count/total*100:.1f}% |\n")
        f.write(f"| ❌ Invalid | {invalid_count} | {invalid_count/total*100:.1f}% |\n")
        f.write(f"| ⚠️  Partial | {partial_count} | {partial_count/total*100:.1f}% |\n")
        f.write(f"| 🔥 Failed | {failed_count} | {failed_count/total*100:.1f}% |\n\n")

        # Common issues
        f.write("## Common Issues\n\n")
        issue_counts = {}
        for result in results:
            for reason in result.get('reasons', []):
                issue_counts[reason] = issue_counts.get(reason, 0) + 1

        if issue_counts:
            f.write("| Issue | Count | Percentage |\n")
            f.write("|-------|-------|------------|\n")
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"| {issue} | {count} | {count/total*100:.1f}% |\n")

        f.write(f"\n## Files\n\n")
        f.write(f"- **Debug Videos:** `debug_videos/` - Visual overlays showing skeleton tracking\n")
        f.write(f"- **Descriptions:** `descriptions/` - Text files with VLM analysis and validation details\n")
        f.write(f"- **NPZ Files:** `npz_files/` - Raw pose data\n")
        f.write(f"- **Full Report:** `validation_report.json` - Detailed JSON results\n\n")

        # Processing speed stats
        processing_times = [r.get('processing_time', 0) for r in results if 'processing_time' in r]
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            f.write(f"## Performance\n\n")
            f.write(f"- Average processing time: {avg_time:.2f} seconds/video\n")
            f.write(f"- Total wall time: {max(processing_times):.1f} seconds\n")
            f.write(f"- Effective throughput: {len(processing_times)/max(processing_times)*60:.1f} videos/minute\n\n")

        f.write(f"## How to Review\n\n")
        f.write(f"1. Check `debug_videos/` for visual validation of pose tracking\n")
        f.write(f"2. Files are named: `[STATUS]_[ACTION]_[VIDEO]_[ISSUES].mp4`\n")
        f.write(f"3. Green skeletons = high confidence, Yellow = medium, Red = low\n")
        f.write(f"4. Review `descriptions/` for VLM analysis and validation details\n")


def main():
    parser = argparse.ArgumentParser(description='Unified parallel Kinetics validation with debug videos')
    parser.add_argument('--num_samples', type=int, default=256,
                        help='Number of videos to sample')
    parser.add_argument('--data_root', type=str,
                        default='/root/movement/data/kinetics-dataset/k700-2020/train',
                        help='Root directory of Kinetics dataset')
    parser.add_argument('--output_dir', type=str,
                        default='/root/movement/data/kinetics_validation',
                        help='Output directory for validation results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    parser.add_argument('--enable_vlm', action='store_true',
                        help='Enable VLM descriptions')
    parser.add_argument('--target_fps', type=float, default=10.0,
                        help='Target FPS for processing')
    parser.add_argument('--workers_per_gpu', type=int, default=2,
                        help='Number of worker processes per GPU')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--max_frames', type=int, default=150,
                        help='Maximum frames per debug video')
    parser.add_argument('--skip_debug_videos', action='store_true',
                        help='Skip generating debug videos')

    args = parser.parse_args()

    # Parse GPU list
    gpu_ids = [int(g) for g in args.gpus.split(',')]
    num_workers = len(gpu_ids) * args.workers_per_gpu

    # Setup directories
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup main logger
    logger = setup_logging(output_dir, 0)
    logger.info(f"Starting unified parallel Kinetics validation")
    logger.info(f"Using {len(gpu_ids)} GPUs with {args.workers_per_gpu} workers each = {num_workers} total workers")
    logger.info(f"Debug videos: {'DISABLED' if args.skip_debug_videos else 'ENABLED'}")
    logger.info(f"VLM descriptions: {'ENABLED' if args.enable_vlm else 'DISABLED'}")
    logger.info(f"Output directory: {output_dir}")

    # Sample videos
    logger.info("Sampling videos...")
    videos = sample_videos(data_root, args.num_samples, args.seed)
    logger.info(f"Sampled {len(videos)} videos from {len(set(v.parent for v in videos))} action classes")

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
                output_dir, args.target_fps, args.enable_vlm,
                not args.skip_debug_videos, args.max_frames
            ))
            p.start()
            workers.append(p)
            worker_id += 1
            time.sleep(2)  # Stagger worker starts to avoid memory spikes

    logger.info(f"Started {num_workers} worker processes")

    # Collect results in main process
    results = result_collector(result_queue, output_dir, len(videos),
                              max_frames=args.max_frames, target_fps=args.target_fps,
                              processes=workers)

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
    logger.info("=" * 60)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 60)

    valid = sum(1 for r in results if r['status'] == 'VALID')
    invalid = sum(1 for r in results if r['status'] == 'INVALID')
    partial = sum(1 for r in results if r['status'] == 'PARTIAL')
    failed = sum(1 for r in results if r['status'] == 'FAILED')

    if results:
        logger.info(f"Total processed: {len(results)}/{args.num_samples}")
        logger.info(f"✅ Valid: {valid} ({valid/len(results)*100:.1f}%)")
        logger.info(f"❌ Invalid: {invalid} ({invalid/len(results)*100:.1f}%)")
        logger.info(f"⚠️  Partial: {partial} ({partial/len(results)*100:.1f}%)")
        logger.info(f"🔥 Failed: {failed} ({failed/len(results)*100:.1f}%)")

        # Performance stats
        processing_times = [r.get('processing_time', 0) for r in results if 'processing_time' in r]
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            wall_time = max(processing_times)
            throughput = len(results) / wall_time * 60

            logger.info("=" * 60)
            logger.info("PERFORMANCE METRICS")
            logger.info(f"Average time per video: {avg_time:.2f} seconds")
            logger.info(f"Total wall time: {wall_time:.1f} seconds")
            logger.info(f"Effective throughput: {throughput:.1f} videos/minute")
            logger.info(f"Speedup vs serial: ~{num_workers:.1f}x")

    logger.info("=" * 60)
    logger.info(f"📁 Results saved to: {output_dir}")
    logger.info(f"   - Debug videos: {output_dir}/debug_videos/")
    logger.info(f"   - Descriptions: {output_dir}/descriptions/")
    logger.info(f"   - NPZ files: {output_dir}/npz_files/")
    logger.info(f"   - Report: {output_dir}/validation_report.json")
    logger.info(f"   - Summary: {output_dir}/README.md")


if __name__ == "__main__":
    # Set spawn method for CUDA multiprocessing
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass  # Already set

    main()