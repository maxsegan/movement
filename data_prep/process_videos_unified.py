#!/usr/bin/env python3
"""
Unified video processing script for all datasets with optional validation and statistics.

Usage:
    # Basic processing (any videos)
    python process_videos_unified.py

    # NTU RGB+D with validation
    python process_videos_unified.py --dataset ntu --validate

    # Process with statistics
    python process_videos_unified.py --stats

    # Process specific directory
    python process_videos_unified.py --data_root /path/to/videos
"""

import argparse
import concurrent.futures as futures
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_prep.pipeline.pipeline import process_video
from transformers import AutoProcessor, VitPoseForPoseEstimation
import torchvision

# Import validation modules conditionally
try:
    from data_prep.skeleton_reader import read_ntu_skeleton, find_matching_skeleton, convert_ntu_to_h36m_joints
    from data_prep.pose_similarity import compare_pose_sequences, detect_pose_anomalies
    from data_prep.comparison_visualization import create_comparison_summary_plot
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    print("Warning: Validation modules not available")


class VideoProcessor:
    """Unified video processor with dataset-specific handling."""

    def __init__(self, args):
        self.args = args
        self.logger = self._setup_logging()
        self.stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'validated': 0,
            'quality_passed': 0,
            'processing_times': [],
            'errors': [],
            'quality_scores': [],
            'similarity_scores': []
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('video_processor')
        logger.setLevel(logging.INFO if not self.args.debug else logging.DEBUG)
        logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

        # File handler if specified
        if self.args.log_file:
            log_path = Path(self.args.out_dir) / self.args.log_file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def find_videos(self, root: Path) -> List[Path]:
        """Find videos based on dataset type."""
        if self.args.dataset == 'ntu':
            # NTU RGB+D specific pattern
            pattern = "*_rgb.avi"
            vids = list(root.rglob(pattern))
        else:
            # Generic video extensions
            exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
            vids = []
            for p in root.rglob("*"):
                if p.suffix.lower() in exts:
                    vids.append(p)

        vids.sort()

        if self.args.limit > 0:
            vids = vids[:self.args.limit]

        return vids

    def process_video_task(
        self,
        video_path: str,
        det_2d_model,
        vitpose_processor,
        vitpose_model,
        device: str
    ) -> Dict:
        """Process a single video with optional validation."""
        start_time = time.time()

        # Setup paths
        vp = Path(video_path)
        dr = Path(self.args.data_root)

        try:
            rel = vp.relative_to(dr)
            save_dir = Path(self.args.out_dir) / rel.parent
        except:
            save_dir = Path(self.args.out_dir)

        try:
            # Process video
            result = process_video(
                video_path=video_path,
                out_dir=save_dir,
                target_fps=self.args.target_fps,
                device=device,
                score_thresh=self.args.score_thresh,
                debug=self.args.debug,
                save_frames=self.args.save_frames,
                model3d=self.args.model3d,
                ckpt3d=self.args.ckpt3d,
                det_2d_model=det_2d_model,
                vitpose_processor=vitpose_processor,
                vitpose_model=vitpose_model,
            )

            # Collect statistics
            stats_dict = {
                'video_path': video_path,
                'npz_path': result.get('npz', ''),
                'success': True,
                'processing_time': time.time() - start_time
            }

            # Load NPZ for quality metrics
            if result.get('npz') and Path(result['npz']).exists():
                data = np.load(result['npz'])
                stats_dict.update({
                    'frames': int(data['meta'][1]),
                    'fps': float(data['meta'][0]),
                    'resolution': (int(data['meta'][2]), int(data['meta'][3])),
                    'density_ok': bool(data['density_ok'][0]),
                    'dynamic_ok': bool(data['dynamic_ok'][0]),
                    'quality': float(data['quality'][0]),
                    'has_3d': data['pose3d'].shape[0] > 0
                })

                # Validation for NTU dataset
                if self.args.validate and self.args.dataset == 'ntu' and VALIDATION_AVAILABLE:
                    validation_result = self._validate_ntu_video(vp, data, save_dir)
                    if validation_result:
                        stats_dict['validation'] = validation_result

            return stats_dict

        except Exception as e:
            self.logger.error(f"Failed processing {video_path}: {e}")
            return {
                'video_path': video_path,
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }

    def _validate_ntu_video(self, video_path: Path, pose_data: dict, save_dir: Path) -> Optional[Dict]:
        """Validate NTU video against ground truth skeleton."""
        if not VALIDATION_AVAILABLE:
            return None

        try:
            # Find matching skeleton file
            skeleton_path = find_matching_skeleton(
                video_path,
                Path(self.args.skeleton_root) if self.args.skeleton_root else None
            )

            if not skeleton_path or not skeleton_path.exists():
                return None

            # Read ground truth
            skeleton_data = read_ntu_skeleton(str(skeleton_path))
            gt_h36m = convert_ntu_to_h36m_joints(skeleton_data)

            # Compare poses
            pred_2d = pose_data['keypoints2d']
            similarity_result = compare_pose_sequences(
                pred_2d,
                gt_h36m['joint_positions'][0, :pred_2d.shape[0]]
            )

            # Generate visualization if requested
            if self.args.save_comparison_plots:
                plot_path = save_dir / f"{video_path.stem}_comparison.png"
                create_comparison_summary_plot(
                    pred_2d,
                    gt_h36m['joint_positions'][0],
                    similarity_result,
                    str(plot_path)
                )

            return {
                'similarity_score': similarity_result['overall_similarity'],
                'temporal_consistency': similarity_result.get('temporal_consistency', 0),
                'anomalies_detected': len(similarity_result.get('anomalous_frames', [])),
                'skeleton_file': str(skeleton_path)
            }

        except Exception as e:
            self.logger.warning(f"Validation failed for {video_path}: {e}")
            return None

    def print_statistics(self):
        """Print processing statistics."""
        if not self.args.stats:
            return

        print("\n" + "="*60)
        print("PROCESSING STATISTICS")
        print("="*60)

        print(f"\nTotal videos: {self.stats['total']}")
        print(f"  ├─ Successful: {self.stats['successful']}")
        print(f"  └─ Failed: {self.stats['failed']}")

        if self.stats['quality_scores']:
            print(f"\nQuality Metrics:")
            print(f"  ├─ Passed quality check: {self.stats['quality_passed']}/{self.stats['successful']}")
            print(f"  └─ Average quality score: {np.mean(self.stats['quality_scores']):.3f}")

        if self.stats['similarity_scores']:
            print(f"\nValidation Metrics:")
            print(f"  ├─ Videos validated: {self.stats['validated']}")
            print(f"  └─ Average similarity: {np.mean(self.stats['similarity_scores']):.3f}")

        if self.stats['processing_times']:
            print(f"\nPerformance:")
            avg_time = np.mean(self.stats['processing_times'])
            total_time = sum(self.stats['processing_times'])
            print(f"  ├─ Average time/video: {avg_time:.1f}s")
            print(f"  └─ Total processing: {total_time/60:.1f} minutes")

        if self.stats['errors']:
            print(f"\n❌ Errors ({len(self.stats['errors'])}):")
            for err in self.stats['errors'][:5]:
                print(f"  • {err}")
            if len(self.stats['errors']) > 5:
                print(f"  ... and {len(self.stats['errors'])-5} more")

        # Save detailed stats
        if self.args.stats_file:
            stats_path = Path(self.args.out_dir) / self.args.stats_file
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
            print(f"\n📊 Detailed stats saved to: {stats_path}")

    def run(self):
        """Main processing loop."""
        # Find videos
        data_root = Path(self.args.data_root)
        videos = self.find_videos(data_root)

        if not videos:
            self.logger.error(f"No videos found in {data_root}")
            return

        self.logger.info(f"Found {len(videos)} videos to process")
        self.stats['total'] = len(videos)

        # Auto-select videos for debug output (2% or at least 1)
        debug_sample_size = max(1, int(len(videos) * self.args.debug_sample_percent / 100))
        debug_indices = np.random.choice(len(videos), debug_sample_size, replace=False)
        debug_videos = set(videos[i] for i in debug_indices)

        if self.args.auto_debug:
            self.logger.info(f"Auto-debug enabled: Will generate debug videos for {debug_sample_size} samples ({self.args.debug_sample_percent}%)")

        # Initialize models
        self.logger.info("Initializing models...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = AutoProcessor.from_pretrained(self.args.vitpose_repo, use_fast=True)
        model = VitPoseForPoseEstimation.from_pretrained(self.args.vitpose_repo).to(device).eval()
        det_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device).eval()

        # Process videos
        for i, video in enumerate(videos, 1):
            # Enable debug for sampled videos
            enable_debug = self.args.debug or (self.args.auto_debug and video in debug_videos)
            enable_comparison = self.args.save_comparison_plots or (self.args.auto_debug and video in debug_videos and self.args.validate)

            if enable_debug:
                self.logger.info(f"[{i}/{len(videos)}] Processing with debug: {video}")
            else:
                self.logger.info(f"[{i}/{len(videos)}] Processing: {video}")

            # Temporarily override debug settings for this video
            original_debug = self.args.debug
            original_comparison = self.args.save_comparison_plots
            self.args.debug = enable_debug
            self.args.save_comparison_plots = enable_comparison

            result = self.process_video_task(
                str(video),
                det_model,
                processor,
                model,
                device
            )

            # Restore original settings
            self.args.debug = original_debug
            self.args.save_comparison_plots = original_comparison

            # Update statistics
            if result['success']:
                self.stats['successful'] += 1
                self.stats['processing_times'].append(result['processing_time'])

                if 'quality' in result:
                    self.stats['quality_scores'].append(result['quality'])
                    if result.get('density_ok') and result.get('dynamic_ok'):
                        self.stats['quality_passed'] += 1

                if 'validation' in result and result['validation']:
                    self.stats['validated'] += 1
                    self.stats['similarity_scores'].append(
                        result['validation']['similarity_score']
                    )

                print(f"  ✓ Saved: {result['npz_path']}")
            else:
                self.stats['failed'] += 1
                self.stats['errors'].append(f"{video}: {result.get('error', 'Unknown')}")
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")

        # Print final statistics
        self.print_statistics()


# Global worker state for parallel processing
_DEVICE = None
_DET_MODEL = None
_VITPOSE_PROCESSOR = None
_VITPOSE_MODEL = None


def _init_worker(vitpose_repo: str, gpu_id: int):
    """Initialize models in worker process."""
    global _DEVICE, _DET_MODEL, _VITPOSE_PROCESSOR, _VITPOSE_MODEL
    _DEVICE = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    _VITPOSE_PROCESSOR = AutoProcessor.from_pretrained(vitpose_repo, use_fast=True)
    _VITPOSE_MODEL = VitPoseForPoseEstimation.from_pretrained(vitpose_repo).to(_DEVICE).eval()
    _DET_MODEL = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(_DEVICE).eval()


def main():
    parser = argparse.ArgumentParser(description="Unified video processing with optional validation")

    # Dataset selection
    parser.add_argument("--dataset", choices=['generic', 'ntu', 'kinetics'], default='generic',
                       help="Dataset type for specific handling")

    # Paths
    parser.add_argument("--data_root", type=str, default="/root/movement/data/extracted",
                       help="Root directory containing videos")
    parser.add_argument("--out_dir", type=str, default="/root/movement/data/pose_processed",
                       help="Output directory for processed results")
    parser.add_argument("--skeleton_root", type=str, default="/root/movement/data/nturgb+d_skeletons",
                       help="Directory containing NTU skeleton files (for validation)")

    # Processing options
    parser.add_argument("--target_fps", type=float, default=20.0,
                       help="Target FPS for video processing")
    parser.add_argument("--score_thresh", type=float, default=0.5,
                       help="Score threshold for pose detection")
    parser.add_argument("--limit", type=int, default=256,
                       help="Limit number of videos (-1 for all)")

    # Model options
    parser.add_argument("--model3d", type=str, default="MotionAGFormer.model.MotionAGFormer:MotionAGFormer",
                       help="3D pose model path")
    parser.add_argument("--ckpt3d", type=str, default="/root/movement/models/motionagformer-b-h36m.pth.tr",
                       help="3D model checkpoint")
    parser.add_argument("--vitpose_repo", type=str, default="usyd-community/vitpose-plus-large",
                       help="ViTPose model repository")

    # Features
    parser.add_argument("--validate", action="store_true",
                       help="Enable validation against ground truth (NTU only)")
    parser.add_argument("--stats", action="store_true",
                       help="Print detailed statistics after processing")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with visualization")
    parser.add_argument("--auto_debug", action="store_true",
                       help="Automatically generate debug videos for a sample of videos")
    parser.add_argument("--debug_sample_percent", type=float, default=2.0,
                       help="Percentage of videos to generate debug output for (default: 2%)")
    parser.add_argument("--save_frames", action="store_true",
                       help="Save individual frames")
    parser.add_argument("--save_comparison_plots", action="store_true",
                       help="Save comparison plots (requires --validate)")

    # Logging
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log file name (saved in out_dir)")
    parser.add_argument("--stats_file", type=str, default="stats.json",
                       help="Statistics JSON file name")

    # Performance
    parser.add_argument("--parallel", action="store_true",
                       help="Enable parallel processing (experimental)")
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs for parallel processing")
    parser.add_argument("--workers_per_gpu", type=int, default=4,
                       help="Workers per GPU for parallel processing")

    args = parser.parse_args()

    # Create output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Run processor
    processor = VideoProcessor(args)
    processor.run()


if __name__ == "__main__":
    main()