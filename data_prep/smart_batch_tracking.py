"""
Smart batch tracking that only detects on frames where pose will be computed.

Key principle: Only detect on the sampled frames (for pose), not all frames.
This avoids interpolation issues while still being efficient.
"""

import numpy as np
import cv2
import torch
from typing import List, Tuple, Optional, Dict
from .bytetrack import ByteTracker, Track


def smart_batch_track(
    det_model,
    frames: np.ndarray,
    device,
    to_tensor_fn,
    detection_batch_size: int = 32,
    track_thresh: float = 0.5,
    match_thresh: float = 0.3,
    min_box_area: float = 100,
    hard_cuts: Optional[List[int]] = None,
    downscale_detection: bool = False,  # Keep default as False for compatibility
    detection_height: int = 480,  # Target height when downscaling
) -> Tuple[np.ndarray, List[Track], Dict]:
    """
    Smart tracking that detects on every sampled frame (no interpolation needed).

    This is the RIGHT way to optimize:
    1. We only detect on frames where we'll compute pose
    2. We batch the detection for GPU efficiency
    3. We can optionally downscale for speed, but default to full res for accuracy

    Args:
        det_model: Detection model (Faster R-CNN)
        frames: (N, H, W, 3) array of SAMPLED frames (these are the only frames we need boxes for)
        device: torch device
        to_tensor_fn: Function to convert frame to tensor
        detection_batch_size: Batch size for detection inference
        track_thresh: High confidence threshold for tracking
        match_thresh: IoU threshold for matching
        min_box_area: Minimum box area
        hard_cuts: List of frame indices with hard cuts (reset tracker)
        downscale_detection: Whether to downscale for detection (False = full quality)
        detection_height: If downscaling, target height for detection

    Returns:
        boxes_xyxy: (N, 4) array of bounding boxes (one per input frame)
        all_tracks: List of tracks per frame (for visualization)
        stats: Dictionary with performance statistics
    """
    import time

    stats = {
        'num_frames': len(frames),
        'prep_time': 0.0,  # Time to prepare frames (resize, etc)
        'inference_time': 0.0,  # Pure model inference time
        'postprocess_time': 0.0,  # Time to process detection results
        'tracking_time': 0.0,  # ByteTracker time
        'original_resolution': f"{frames.shape[2]}x{frames.shape[1]}",
        'detection_resolution': f"{frames.shape[2]}x{frames.shape[1]}",  # Will be updated if downscaling
        'actual_downscale': False,
        'batch_size': detection_batch_size,
    }

    num_frames = len(frames)
    boxes_xyxy = np.full((num_frames, 4), np.nan, dtype=np.float32)
    all_tracks_per_frame = []

    if hard_cuts is None:
        hard_cuts = []
    hard_cuts_set = set(hard_cuts)

    # Initialize tracker
    tracker = ByteTracker(
        track_thresh=track_thresh,
        track_buffer=30,
        match_thresh=match_thresh,
        min_box_area=min_box_area
    )
    main_track_id = None

    # Prepare frames for detection (optionally downscale)
    t_prep = time.time()
    detection_frames = []
    scale_factors = []

    for frame in frames:
        if downscale_detection and frame.shape[0] > detection_height:
            h, w = frame.shape[:2]
            scale = detection_height / h
            new_width = int(w * scale)
            new_height = detection_height

            # Make width even for codec compatibility
            if new_width % 2 != 0:
                new_width += 1

            downsampled = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            detection_frames.append(downsampled)
            scale_factors.append(1.0 / scale)  # Inverse scale for mapping back
        else:
            detection_frames.append(frame)
            scale_factors.append(1.0)

    if downscale_detection and len(detection_frames) > 0:
        avg_h = np.mean([f.shape[0] for f in detection_frames])
        avg_w = np.mean([f.shape[1] for f in detection_frames])
        stats['detection_resolution'] = f"{int(avg_w)}x{int(avg_h)}"
        stats['actual_downscale'] = True

    stats['prep_time'] = time.time() - t_prep

    # Run batched detection on all frames
    detection_results = {}  # frame_idx -> (boxes, scores)

    t_inference_total = 0.0
    t_postprocess_total = 0.0

    num_batches = (num_frames + detection_batch_size - 1) // detection_batch_size
    print(f"  Processing {num_frames} frames in {num_batches} batches of up to {detection_batch_size}")

    with torch.no_grad():
        for batch_num, batch_start in enumerate(range(0, num_frames, detection_batch_size)):
            batch_end = min(batch_start + detection_batch_size, num_frames)
            batch_frames = detection_frames[batch_start:batch_end]
            batch_scales = scale_factors[batch_start:batch_end]
            actual_batch_size = len(batch_frames)

            # Convert to tensors
            t_tensor = time.time()
            batch_tensors = [to_tensor_fn(frame).to(device) for frame in batch_frames]
            t_tensor_elapsed = time.time() - t_tensor

            # Run detection (pure inference)
            t_inf = time.time()
            batch_outputs = det_model(batch_tensors)
            torch.cuda.synchronize() if torch.cuda.is_available() else None  # Ensure GPU completes
            batch_inference_time = time.time() - t_inf
            t_inference_total += batch_inference_time

            # Log if batch is slow
            if batch_inference_time > 5.0:
                print(f"  WARNING: Batch {batch_num+1}/{num_batches} took {batch_inference_time:.2f}s for {actual_batch_size} frames!")

            # Post-process results
            t_post = time.time()
            for i, (out, scale) in enumerate(zip(batch_outputs, batch_scales)):
                frame_idx = batch_start + i

                if out["boxes"].numel() > 0:
                    labels = out["labels"].detach().cpu().numpy()
                    scores = out["scores"].detach().cpu().numpy()
                    boxes = out["boxes"].detach().cpu().numpy()

                    # Scale boxes back to original resolution if downscaled
                    if scale != 1.0:
                        boxes *= scale

                    # CRITICAL: Filter for persons only (class 1 in COCO) AND by score
                    person_mask = (labels == 1) & (scores >= 0.5)
                    if person_mask.any():
                        detection_results[frame_idx] = (
                            boxes[person_mask],
                            scores[person_mask]
                        )
                    else:
                        detection_results[frame_idx] = (
                            np.empty((0, 4)),
                            np.empty(0)
                        )
                else:
                    detection_results[frame_idx] = (
                        np.empty((0, 4)),
                        np.empty(0)
                    )
            t_postprocess_total += (time.time() - t_post)

    stats['inference_time'] = t_inference_total
    stats['postprocess_time'] = t_postprocess_total

    # Run tracking on all frames
    t_tracking = time.time()

    for i in range(num_frames):
        # Reset tracker at hard cuts
        if i in hard_cuts_set:
            tracker = ByteTracker(
                track_thresh=track_thresh,
                track_buffer=30,
                match_thresh=match_thresh,
                min_box_area=min_box_area
            )
            main_track_id = None

        # Get detections for this frame
        person_boxes, person_scores = detection_results.get(i, (np.empty((0, 4)), np.empty(0)))

        # Update tracker
        active_tracks = tracker.update(person_boxes, person_scores)
        all_tracks_per_frame.append(active_tracks)

        # Select main track
        if active_tracks:
            if main_track_id is None:
                # First frame with tracks: pick the best one
                main_track = tracker.get_main_track()
                if main_track:
                    main_track_id = main_track.track_id
                    boxes_xyxy[i] = main_track.bbox.astype(np.float32)
            else:
                # Find our main track
                main_track = None
                for track in active_tracks:
                    if track.track_id == main_track_id:
                        main_track = track
                        break

                if main_track:
                    # Found our tracked person
                    boxes_xyxy[i] = main_track.bbox.astype(np.float32)
                else:
                    # Lost main track, pick new one
                    main_track = tracker.get_main_track()
                    if main_track:
                        main_track_id = main_track.track_id
                        boxes_xyxy[i] = main_track.bbox.astype(np.float32)

    stats['tracking_time'] = time.time() - t_tracking

    # Calculate total detection time
    total_detection_time = stats['prep_time'] + stats['inference_time'] + stats['postprocess_time']

    # Print detailed statistics
    print(f"\n=== Smart Batch Tracking Statistics ===")
    print(f"Frames: {stats['num_frames']} | Batch size: {stats['batch_size']}")
    print(f"Original resolution: {stats['original_resolution']}")
    if stats['actual_downscale']:
        print(f"Detection resolution: {stats['detection_resolution']} (downscaled from {stats['original_resolution']})")
    else:
        print(f"Detection resolution: {stats['detection_resolution']} (no downscaling)")

    print(f"\nTiming breakdown:")
    print(f"  Frame preparation: {stats['prep_time']:.3f}s")
    if stats['actual_downscale']:
        print(f"    (includes resizing {stats['num_frames']} frames to {stats['detection_resolution']})")
    print(f"  Model inference: {stats['inference_time']:.3f}s")
    print(f"    ({stats['num_frames']/stats['inference_time']:.1f} frames/sec)")
    print(f"  Post-processing: {stats['postprocess_time']:.3f}s")
    print(f"  Tracking (ByteTracker): {stats['tracking_time']:.3f}s")
    print(f"  ──────────────────────")
    print(f"  Total: {total_detection_time + stats['tracking_time']:.3f}s")
    print(f"========================================\n")

    return boxes_xyxy, all_tracks_per_frame, stats