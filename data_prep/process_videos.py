#!/usr/bin/env python3
"""
Fast video processing script with VLM optimization.
Reduces resolution and optimizes VLM processing for faster results.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_prep.process_videos_old import main as original_main


def parse_args_fast():
    """Parse arguments with fast defaults."""
    parser = argparse.ArgumentParser(description='Fast video processing with VLM')

    # Core arguments
    parser.add_argument('--dataset', type=str, default='ntu',
                        choices=['generic', 'ntu', 'kinetics'])
    parser.add_argument('--data_root', type=str,
                        default='/root/movement/data/extracted')
    parser.add_argument('--out_dir', type=str,
                        default='/root/movement/data/pose_processed')
    parser.add_argument('--limit', type=int, default=2,
                        help='Number of videos to process (default: 2)')

    # Optimization settings
    parser.add_argument('--target_fps', type=float, default=10.0,  # Reduced from 20
                        help='Target FPS for processing (default: 10)')
    parser.add_argument('--max_resolution', type=int, default=480,  # New: max resolution
                        help='Maximum resolution (height) for VLM processing')
    parser.add_argument('--vlm_max_frames', type=int, default=4,  # Reduced from 8
                        help='Maximum frames to sample for VLM (default: 4)')
    parser.add_argument('--vlm_segment_duration', type=float, default=5.0,  # Increased from 3
                        help='Duration of each segment for VLM in seconds')

    # Model settings
    parser.add_argument('--enable_vlm', action='store_true', default=True,
                        help='Enable VLM action descriptions (default: True)')
    parser.add_argument('--vlm_model', type=str,
                        default='Qwen/Qwen2.5-VL-3B-Instruct')

    # Other settings
    parser.add_argument('--debug', action='store_true',
                        help='Save debug visualization videos')
    parser.add_argument('--parallel', action='store_true', default=True,
                        help='Use parallel processing (default: True)')
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--workers_per_gpu', type=int, default=1)

    # Hidden/advanced settings with defaults
    parser.add_argument('--skeleton_root', type=str, default=None)
    parser.add_argument('--score_thresh', type=float, default=0.15)
    parser.add_argument('--model3d', type=str, default=None)
    parser.add_argument('--ckpt3d', type=str, default=None)
    parser.add_argument('--vitpose_repo', type=str,
                        default='/root/movement/third-party/ViTPose')
    parser.add_argument('--validate', action='store_true', default=False)
    parser.add_argument('--stats', action='store_true', default=False)
    parser.add_argument('--auto_debug', action='store_true', default=False)
    parser.add_argument('--debug_sample_percent', type=float, default=5.0)
    parser.add_argument('--save_frames', action='store_true', default=False)
    parser.add_argument('--save_comparison_plots', action='store_true', default=False)
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--stats_file', type=str, default=None)

    return parser.parse_args()


def optimize_vlm_settings():
    """Monkey-patch VLM settings for faster processing."""
    import data_prep.video_action_description as vad

    # Store original methods
    original_sample = vad.VideoActionDescriber.sample_frames_for_description

    # Create optimized version
    def sample_frames_fast(self, frames, bboxes, indices, max_frames=4):  # Reduced default
        """Optimized frame sampling with lower resolution."""
        import cv2
        import numpy as np

        # First do normal sampling
        sampled_frames, sampled_indices = original_sample(
            self, frames, bboxes, indices, max_frames
        )

        # Then resize frames for faster VLM processing
        resized_frames = []
        for frame in sampled_frames:
            h, w = frame.shape[:2]
            if h > 480:  # Resize if larger than 480p
                scale = 480 / h
                new_w = int(w * scale)
                new_h = 480
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                resized_frames.append(resized)
            else:
                resized_frames.append(frame)

        return resized_frames, sampled_indices

    # Apply the patch
    vad.VideoActionDescriber.sample_frames_for_description = sample_frames_fast

    # Also patch the segment duration in process_video_with_tracking
    original_process = vad.VideoActionDescriber.process_video_with_tracking

    def process_video_fast(self, video_path, keypoints, bboxes, indices,
                            segment_duration=5.0, fps=10.0):  # Increased segment, reduced fps
        """Process with optimized settings."""
        return original_process(self, video_path, keypoints, bboxes, indices,
                                 segment_duration, fps)

    vad.VideoActionDescriber.process_video_with_tracking = process_video_fast


def main():
    """Main function with fast settings."""
    print("=" * 60)
    print("FAST Video Processing with VLM Optimizations")
    print("=" * 60)
    print("Optimizations applied:")
    print("  - Reduced FPS: 10 (from 20)")
    print("  - Reduced resolution: 480p max for VLM")
    print("  - Fewer VLM frames: 4 (from 8)")
    print("  - Longer segments: 5s (from 3s)")
    print("  - Output to: /root/movement/data/pose_processed")
    print("=" * 60)

    # Apply optimizations
    optimize_vlm_settings()

    # Parse arguments with fast defaults
    args = parse_args_fast()

    # Set args in sys.argv format for original main
    import sys
    sys.argv = ['process_videos.py']

    # Add arguments to sys.argv
    sys.argv.extend(['--dataset', args.dataset])
    sys.argv.extend(['--data_root', args.data_root])
    sys.argv.extend(['--out_dir', args.out_dir])
    sys.argv.extend(['--limit', str(args.limit)])
    sys.argv.extend(['--target_fps', str(args.target_fps)])

    if args.enable_vlm:
        sys.argv.append('--enable_vlm')
    if args.debug:
        sys.argv.append('--debug')
    if args.parallel:
        sys.argv.append('--parallel')

    sys.argv.extend(['--num_gpus', str(args.num_gpus)])
    sys.argv.extend(['--workers_per_gpu', str(args.workers_per_gpu)])

    # Run the original main
    original_main()


if __name__ == "__main__":
    main()