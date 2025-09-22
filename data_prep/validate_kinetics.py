#!/usr/bin/env python3
"""
Validation script for Kinetics-700 dataset.
Processes 256 random samples with debug visualizations and validation metrics.
"""

import sys
import os
import random
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import numpy as np
import cv2
from tqdm import tqdm

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from data_prep.pipeline.pipeline import process_video
from data_prep.video_action_description import VideoActionDescriber
from data_prep.keypoints import h36m_coco_format
import data_prep.clip_filtering as filt


def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    log_file = output_dir / "validation_log.txt"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def sample_videos(data_root: Path, num_samples: int = 256, seed: int = 42) -> List[Path]:
    """
    Randomly sample videos from all Kinetics subdirectories.

    Args:
        data_root: Root directory of Kinetics dataset
        num_samples: Number of videos to sample
        seed: Random seed for reproducibility

    Returns:
        List of sampled video paths
    """
    random.seed(seed)

    # Find all video files
    all_videos = list(data_root.rglob("*.mp4"))

    if len(all_videos) < num_samples:
        print(f"Warning: Only found {len(all_videos)} videos, using all of them")
        return all_videos

    # Sample randomly
    sampled = random.sample(all_videos, num_samples)

    # Sort by action class for better organization
    sampled.sort(key=lambda x: x.parent.name)

    return sampled


def get_validation_status(npz_data: dict) -> Tuple[str, List[str]]:
    """
    Determine validation status and reasons from processed data.

    Args:
        npz_data: Loaded NPZ data

    Returns:
        Tuple of (status, reasons) where status is 'VALID', 'INVALID', or 'PARTIAL'
    """
    reasons = []

    # Check density
    if 'density_ok' in npz_data and not npz_data['density_ok'][0]:
        reasons.append("LOW_DENSITY")

    # Check motion
    if 'dynamic_ok' in npz_data and not npz_data['dynamic_ok'][0]:
        reasons.append("NO_MOTION")

    # Check quality score
    if 'quality' in npz_data:
        quality = float(npz_data['quality'][0])
        if quality < 0.3:
            reasons.append("LOW_QUALITY")
        elif quality < 0.5:
            reasons.append("MEDIUM_QUALITY")

    # Check pose detection
    if 'keypoints2d' in npz_data:
        kpts = npz_data['keypoints2d']
        if kpts.shape[0] == 0:
            reasons.append("NO_DETECTION")
        else:
            # Check for tracking loss
            valid_frames = np.sum(np.any(kpts > 0, axis=(1, 2)))
            total_frames = kpts.shape[0]
            if valid_frames < total_frames * 0.5:
                reasons.append("TRACKING_LOSS")

    # Determine overall status
    if not reasons:
        status = "VALID"
    elif "NO_DETECTION" in reasons or "TRACKING_LOSS" in reasons:
        status = "INVALID"
    else:
        status = "PARTIAL"

    return status, reasons


def create_debug_video(
    video_path: Path,
    npz_path: Path,
    output_path: Path,
    max_frames: int = 300
) -> bool:
    """
    Create debug video with skeleton overlay.

    Args:
        video_path: Original video path
        npz_path: Path to NPZ file with pose data
        output_path: Output path for debug video
        max_frames: Maximum frames to process

    Returns:
        True if successful
    """
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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # H36M skeleton connections
        skeleton = [
            [0, 1], [1, 2], [2, 3],  # Right leg
            [0, 4], [4, 5], [5, 6],  # Left leg
            [0, 7], [7, 8], [8, 9], [9, 10],  # Spine to head
            [8, 11], [11, 12], [12, 13],  # Right arm
            [8, 14], [14, 15], [15, 16],  # Left arm
        ]

        frame_idx = 0
        processed = 0

        while cap.isOpened() and processed < min(max_frames, len(indices)):
            ret, frame = cap.read()
            if not ret:
                break

            # Check if this frame has pose data
            if frame_idx in indices:
                pose_idx = np.where(indices == frame_idx)[0]
                if len(pose_idx) > 0:
                    pose_idx = pose_idx[0]

                    # Draw bounding box if available
                    if pose_idx < len(bboxes) and not np.any(np.isnan(bboxes[pose_idx])):
                        x1, y1, x2, y2 = bboxes[pose_idx].astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw skeleton
                    kpts = keypoints[pose_idx]
                    scrs = scores[pose_idx]

                    # Draw joints
                    for j, (x, y) in enumerate(kpts):
                        if scrs[j] > 0.3 and x > 0 and y > 0:
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

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
            status, reasons = get_validation_status(data)
            text = f"Frame {frame_idx} | Status: {status}"
            if reasons:
                text += f" | {', '.join(reasons)}"

            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Add quality score if available
            if 'quality' in data:
                quality_text = f"Quality: {float(data['quality'][0]):.3f}"
                cv2.putText(frame, quality_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 255, 255), 2, cv2.LINE_AA)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        return True

    except Exception as e:
        logging.error(f"Error creating debug video: {e}")
        return False


def save_description(npz_path: Path, output_dir: Path) -> Optional[Path]:
    """
    Extract and save VLM description to text file.

    Args:
        npz_path: Path to NPZ file
        output_dir: Output directory for text files

    Returns:
        Path to saved text file or None
    """
    try:
        data = np.load(npz_path, allow_pickle=True)

        if 'action_descriptions' not in data:
            return None

        desc_json = str(data['action_descriptions'][0])
        if desc_json == '[]':
            return None

        descriptions = json.loads(desc_json)
        if not descriptions:
            return None

        # Create text file
        stem = npz_path.stem.replace('_rgb', '')
        text_path = output_dir / f"{stem}_description.txt"

        with open(text_path, 'w') as f:
            f.write(f"Video: {stem}\n")
            f.write("=" * 60 + "\n\n")

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
                    f.write(f"  Action: {text}\n")

                if 'confidence' in desc:
                    f.write(f"  Confidence: {desc['confidence']:.2f}\n")

                f.write("\n")

        return text_path

    except Exception as e:
        logging.error(f"Error saving description: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Validate Kinetics-700 pose extraction')
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
    parser.add_argument('--enable_vlm', action='store_true', default=True,
                        help='Enable VLM descriptions')
    parser.add_argument('--debug_videos', action='store_true', default=True,
                        help='Generate debug videos with skeleton overlay')
    parser.add_argument('--max_frames', type=int, default=300,
                        help='Maximum frames per debug video')

    args = parser.parse_args()

    # Setup directories
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    npz_dir = output_dir / "npz_files"
    video_dir = output_dir / "debug_videos"
    desc_dir = output_dir / "descriptions"

    npz_dir.mkdir(exist_ok=True)
    video_dir.mkdir(exist_ok=True)
    desc_dir.mkdir(exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting Kinetics validation with {args.num_samples} samples")

    # Sample videos
    logger.info("Sampling videos...")
    videos = sample_videos(data_root, args.num_samples, args.seed)
    logger.info(f"Sampled {len(videos)} videos")

    # Initialize models
    logger.info("Initializing models...")

    # Load models once
    import torch
    from data_prep.vitpose import load_vitpose_model
    from torchvision import models

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Detection model
    det_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    det_model.to(device)
    det_model.eval()

    # ViTPose model
    vitpose_processor, vitpose_model = load_vitpose_model(device=device)

    # VLM if enabled
    action_describer = None
    if args.enable_vlm:
        action_describer = VideoActionDescriber(device=device)

    # Process videos
    results = []
    valid_count = 0
    invalid_count = 0
    partial_count = 0

    for i, video_path in enumerate(tqdm(videos, desc="Processing videos")):
        try:
            # Get video info
            action_class = video_path.parent.name
            video_name = video_path.stem

            logger.info(f"[{i+1}/{len(videos)}] Processing {action_class}/{video_name}")

            # Process video
            result = process_video(
                str(video_path),
                npz_dir,
                target_fps=10.0,  # Use fast settings
                device=device,
                det_2d_model=det_model,
                vitpose_processor=vitpose_processor,
                vitpose_model=vitpose_model,
                action_describer=action_describer,
                enable_action_description=args.enable_vlm,
                debug=False  # We'll create our own debug videos
            )

            if not result or 'npz' not in result:
                logger.warning(f"Failed to process {video_name}")
                continue

            # Load NPZ data
            npz_path = Path(result['npz'])
            data = np.load(npz_path, allow_pickle=True)

            # Get validation status
            status, reasons = get_validation_status(data)

            # Update counts
            if status == "VALID":
                valid_count += 1
            elif status == "INVALID":
                invalid_count += 1
            else:
                partial_count += 1

            # Create descriptive filename
            reason_str = "_".join(reasons) if reasons else ""
            new_name = f"{status}_{action_class}_{video_name}"
            if reason_str:
                new_name += f"_{reason_str}"

            # Move/rename NPZ file
            new_npz_path = npz_dir / f"{new_name}.npz"
            shutil.move(str(npz_path), str(new_npz_path))

            # Create debug video if requested
            if args.debug_videos:
                debug_path = video_dir / f"{new_name}.mp4"
                success = create_debug_video(
                    video_path, new_npz_path, debug_path, args.max_frames
                )
                if success:
                    logger.info(f"Created debug video: {debug_path.name}")

            # Save VLM description
            if args.enable_vlm:
                desc_path = save_description(new_npz_path, desc_dir)
                if desc_path:
                    # Also rename description file
                    new_desc_path = desc_dir / f"{new_name}_description.txt"
                    shutil.move(str(desc_path), str(new_desc_path))

            # Record result
            results.append({
                'video': str(video_path.relative_to(data_root)),
                'action_class': action_class,
                'status': status,
                'reasons': reasons,
                'quality': float(data['quality'][0]) if 'quality' in data else 0.0,
                'density_ok': bool(data['density_ok'][0]) if 'density_ok' in data else False,
                'dynamic_ok': bool(data['dynamic_ok'][0]) if 'dynamic_ok' in data else False
            })

        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            continue

    # Save summary report
    report_path = output_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            'summary': {
                'total': len(results),
                'valid': valid_count,
                'invalid': invalid_count,
                'partial': partial_count,
                'valid_percentage': (valid_count / len(results) * 100) if results else 0
            },
            'results': results
        }, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total videos processed: {len(results)}")
    logger.info(f"Valid: {valid_count} ({valid_count/len(results)*100:.1f}%)")
    logger.info(f"Invalid: {invalid_count} ({invalid_count/len(results)*100:.1f}%)")
    logger.info(f"Partial: {partial_count} ({partial_count/len(results)*100:.1f}%)")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - NPZ files: {npz_dir}")
    logger.info(f"  - Debug videos: {video_dir}")
    logger.info(f"  - Descriptions: {desc_dir}")
    logger.info(f"  - Report: {report_path}")


if __name__ == "__main__":
    main()