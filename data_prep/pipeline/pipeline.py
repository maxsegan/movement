from pathlib import Path
from typing import Dict
import time
import logging

import cv2
import numpy as np
import torch

from data_prep.keypoints import h36m_coco_format
from data_prep.vitpose import infer_sequence
# Use fast video loader for improved performance
from data_prep.fast_video_loader import probe_video_meta_fast, read_frames_batch_fast, sample_indices_for_fps
from data_prep.pose3d import load_motionagformer_from_path, lift_sequence_to_3d
from data_prep import clip_filtering as filt

logger = logging.getLogger(__name__)


def detect_hard_cuts(frames, threshold=0.4):
    """Detect hard cuts in video by comparing histogram differences between consecutive frames."""
    import cv2

    hard_cuts = []
    prev_hist = None

    for i, frame in enumerate(frames):
        # Convert to grayscale for histogram comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            # Compare histograms using correlation method
            correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)

            # Low correlation indicates a hard cut
            if correlation < threshold:
                hard_cuts.append(i)

        prev_hist = hist

    return hard_cuts

def process_video(
    video_path: str,
    out_dir: Path,
    target_fps: float = 20.0,
    device: str = "cuda",
    score_thresh: float = 0.5,
    debug: bool = False,
    save_frames: bool = False,
    model3d: str | None = None,
    ckpt3d: str | None = None,
    # injected models/functions to avoid reloading per video
    vitpose_processor: object | None = None,
    vitpose_model: object | None = None,
    verbose: bool = False,  # Whether to print detailed processing stats
) -> Dict[str, str]:
    t_start = time.time()
    timings = {}

    t0 = time.time()
    meta = probe_video_meta_fast(video_path)
    idxs = sample_indices_for_fps(meta["frames"], meta["fps"], target_fps=target_fps)
    frames = read_frames_batch_fast(video_path, idxs, target_fps=target_fps)
    # Ensure frames are in RGB format (ffmpeg returns RGB, OpenCV returns BGR)
    timings['video_load'] = time.time() - t0

    # For debug mode, we need ALL frames for proper visualization
    all_frames = None
    if debug:
        all_frame_indices = np.arange(meta["frames"])
        all_frames = read_frames_batch_fast(video_path, all_frame_indices)

    # ViTPose path using existing module (requires precomputed boxes per frame)
    # Detect hard cuts first (using sampled frames is fine for this)
    t0 = time.time()
    hard_cuts = detect_hard_cuts(frames)
    has_hard_cuts = len(hard_cuts) > 0
    timings['hard_cut_detection'] = time.time() - t0

    boxes_xyxy = np.full((frames.shape[0], 4), np.nan, dtype=np.float32)

    t0 = time.time()
    if vitpose_processor is None or vitpose_model is None:
        raise RuntimeError("ViTPose processor/model must be provided to process_video")

    # Pass pre-loaded frames to avoid re-reading from disk
    kpts, scrs = infer_sequence(
        raw_video_path=video_path,
        image_processor=vitpose_processor,
        model=vitpose_model,
        device=torch.device(device),
        idxs=idxs,
        boxes_xyxy=boxes_xyxy,
        frames=frames,  # Use already-loaded frames
        batch_size=16,  # Batch processing for GPU efficiency
    )
    timings['pose_2d'] = time.time() - t0
    
    det = dict(
        keypoints=kpts,
        scores=scrs,
        bboxes=boxes_xyxy,
        det_scores=np.zeros(frames.shape[0], dtype=np.float32),
        indices=idxs.astype(np.int32),
        has_hard_cuts=has_hard_cuts,
        hard_cut_frames=hard_cuts,
    )

    # Convert COCO->H36M
    h36m_kpts, h36m_scores, valid_frames = h36m_coco_format(det["keypoints"], det["scores"])
    if h36m_kpts.shape[0] == 0:
        # no detections; create empty placeholders
        h36m_kpts = np.zeros((1, frames.shape[0], 17, 2), dtype=np.float32)
        h36m_scores = np.zeros((1, frames.shape[0], 17), dtype=np.float32)

    seq_k = h36m_kpts[0]
    seq_s = h36m_scores[0]

    # Filtering metrics - set defaults for now (actual validation happens in validate_kinetics.py)
    density_ok = True
    dynamic_ok = True
    quality = 1.0

    # Optional 3D lifting
    y3d = None
    if model3d and ckpt3d and frames.shape[0] > 0:
        try:
            model_3d = load_motionagformer_from_path(model3d, ckpt3d, device)
            y3d = lift_sequence_to_3d(seq_k[None, ...], seq_s[None, ...], meta["width"], meta["height"], model_3d, device)
        except Exception as e:
            y3d = None
            timings['pose_3d'] = 0.0
    
    if 'pose_3d' not in timings:
        timings['pose_3d'] = time.time() - t0
    
    t0 = time.time()

    # Save outputs
    vp = Path(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{vp.stem}.npz"

    np.savez_compressed(
        npz_path,
        indices=idxs.astype(np.int32),
        bboxes=det["bboxes"].astype(np.float32),
        keypoints2d=seq_k.astype(np.float32),
        scores2d=seq_s.astype(np.float32),
        pose3d=(y3d.astype(np.float32) if y3d is not None else np.zeros((0, 17, 3), dtype=np.float32)),
        meta=np.array([meta["fps"], meta["frames"], meta["width"], meta["height"]], dtype=np.float32),
        density_ok=np.array([density_ok], dtype=bool),
        dynamic_ok=np.array([dynamic_ok], dtype=bool),
        quality=np.array([quality], dtype=np.float32),
        has_hard_cuts=np.array([det["has_hard_cuts"]], dtype=bool),
        hard_cut_frames=np.array(det["hard_cut_frames"], dtype=np.int32),
    )

    # Optional frame dump
    if save_frames and frames.shape[0] > 0:
        frames_dir = out_dir / f"{vp.stem}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i in range(frames.shape[0]):
            p = frames_dir / f"{i:06d}.png"
            cv2.imwrite(str(p), cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))

    debug_path = None
    if debug and frames.shape[0] > 0:
        debug_path = out_dir / f"{vp.stem}_debug.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Use original FPS for debug video if we have all frames
        debug_fps = meta["fps"] if all_frames is not None else target_fps
        debug_frames = all_frames if all_frames is not None else frames
        writer = cv2.VideoWriter(str(debug_path), fourcc, debug_fps, (debug_frames.shape[2], debug_frames.shape[1]))

        # H36M skeleton connections for visualization
        h36m_connections = [
            (0, 1), (1, 2), (2, 3),  # Right leg
            (0, 4), (4, 5), (5, 6),  # Left leg
            (0, 7), (7, 8), (8, 9), (9, 10),  # Spine to head
            (8, 11), (11, 12), (12, 13),  # Right arm
            (8, 14), (14, 15), (15, 16),  # Left arm
        ]

        # Create mapping from all frames to sampled keypoints
        keypoint_mapping = {}
        for idx, frame_no in enumerate(idxs):
            keypoint_mapping[int(frame_no)] = idx

        for t in range(debug_frames.shape[0]):
            img = cv2.cvtColor(debug_frames[t], cv2.COLOR_RGB2BGR)

            # Draw skeleton for main person (only on sampled frames)
            if t in keypoint_mapping:
                kpt_idx = keypoint_mapping[t]
                bb = det["bboxes"][kpt_idx]
                if np.all(np.isfinite(bb)) and np.any(bb != 0):
                    # Draw main person's skeleton
                    pts = seq_k[kpt_idx]
                    sc = seq_s[kpt_idx]

                    # Draw skeleton connections
                    for conn in h36m_connections:
                        if sc[conn[0]] >= 0.15 and sc[conn[1]] >= 0.15:
                            pt1 = (int(round(pts[conn[0], 0])), int(round(pts[conn[0], 1])))
                            pt2 = (int(round(pts[conn[1], 0])), int(round(pts[conn[1], 1])))
                            cv2.line(img, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

                    # Draw joints
                    for j in range(pts.shape[0]):
                        if sc[j] >= 0.15:
                            p = (int(round(pts[j, 0])), int(round(pts[j, 1])))
                            cv2.circle(img, p, 3, (0, 255, 255), -1, cv2.LINE_AA)
                            cv2.circle(img, p, 4, (0, 0, 0), 1, cv2.LINE_AA)  # Border

            # Add frame info
            cv2.putText(img, f"Frame {t}/{debug_frames.shape[0]-1}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if t in hard_cuts:
                cv2.putText(img, "HARD CUT", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            writer.write(img)
        writer.release()
    
    timings['debug_video'] = time.time() - t0

    timings['total'] = time.time() - t_start
    
    # Log detailed timing breakdown
    import sys
    timing_msg = (f"Pipeline timing - Load: {timings.get('video_load', 0):.1f}s, " +
                f"Pose2D: {timings.get('pose_2d', 0):.1f}s, " +
                f"Pose3D: {timings.get('pose_3d', 0):.1f}s, " +
                f"Debug: {timings.get('debug_video', 0):.1f}s, " +
                f"Total: {timings['total']:.1f}s")
    print(timing_msg, file=sys.stderr, flush=True)
    
    return {"npz": str(npz_path), "debug": str(debug_path) if debug_path else ""}


