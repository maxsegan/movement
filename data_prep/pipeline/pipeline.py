from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from data_prep.keypoints import h36m_coco_format
from data_prep.vitpose import infer_sequence
from data_prep.pipeline.sampling import probe_video_meta, read_frames_by_indices, sample_indices_for_fps
from data_prep.pose3d import load_motionagformer_from_path, lift_sequence_to_3d
from data_prep import clip_filtering as filt
from data_prep.video_action_description import FastVideoActionDescriber as VideoActionDescriber, ActionDescription
from data_prep.bytetrack import ByteTracker, Track


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
    det_2d_model: object | None = None,
    vitpose_processor: object | None = None,
    vitpose_model: object | None = None,
    # VLM for action description
    action_describer: Optional[VideoActionDescriber] = None,
    enable_action_description: bool = False,
) -> Dict[str, str]:
    meta = probe_video_meta(video_path)
    idxs = sample_indices_for_fps(meta["frames"], meta["fps"], target_fps=target_fps)
    frames = read_frames_by_indices(video_path, idxs)

    # For debug mode, we need ALL frames for proper visualization
    all_frames = None
    if debug:
        all_frame_indices = np.arange(meta["frames"])
        all_frames = read_frames_by_indices(video_path, all_frame_indices)

    # ViTPose path using existing module (requires precomputed boxes per frame)
    # Detect hard cuts first (using sampled frames is fine for this)
    hard_cuts = detect_hard_cuts(frames)
    has_hard_cuts = len(hard_cuts) > 0

    # Run tracking on ALL frames if in debug mode, otherwise just sampled frames
    frames_to_track = all_frames if (debug and all_frames is not None) else frames

    boxes_xyxy = np.full((frames.shape[0], 4), np.nan, dtype=np.float32)
    boxes_xyxy_all_frames = None  # For debug visualization

    from torchvision.transforms.functional import to_tensor

    # Initialize ByteTracker for multi-object tracking
    tracker = ByteTracker(
        track_thresh=0.5,     # High confidence threshold
        track_buffer=30,      # Keep lost tracks for 30 frames
        match_thresh=0.3,     # Lower IoU threshold for better tracking during posture changes
        min_box_area=100      # Minimum box area
    )

    # Store all tracks for visualization
    all_tracks_per_frame = []
    main_track_id = None
    tracking_switches = []

    if det_2d_model is not None:
        if debug and all_frames is not None:
            # In debug mode: track ALL frames for smooth visualization
            boxes_xyxy_all_frames = np.full((all_frames.shape[0], 4), np.nan, dtype=np.float32)

        with torch.no_grad():
            for i in range(frames_to_track.shape[0]):
                img = to_tensor(frames_to_track[i]).to(device)
                out = det_2d_model([img])[0]

                # Reset tracker at hard cuts
                if i in hard_cuts:
                    tracker = ByteTracker(
                        track_thresh=0.5,
                        track_buffer=30,
                        match_thresh=0.3,  # Lower IoU threshold
                        min_box_area=100
                    )
                    main_track_id = None

                # Get all person detections
                person_boxes = []
                person_scores = []

                if out["boxes"].numel() > 0:
                    labels = out["labels"].detach().cpu().numpy()
                    scores = out["scores"].detach().cpu().numpy()
                    boxes = out["boxes"].detach().cpu().numpy()

                    # Filter for persons only (class 1 in COCO) with higher confidence
                    mask = (labels == 1) & (scores >= 0.5)  # Increased from 0.3 to avoid false positives
                    idx = np.flatnonzero(mask)

                    if idx.size > 0:
                        person_boxes = boxes[idx]
                        person_scores = scores[idx]
                    else:
                        person_boxes = np.empty((0, 4))
                        person_scores = np.empty(0)
                else:
                    person_boxes = np.empty((0, 4))
                    person_scores = np.empty(0)

                # Update tracker with all detections
                active_tracks = tracker.update(person_boxes, person_scores)
                all_tracks_per_frame.append(active_tracks)

                # Select main track to follow
                if active_tracks:
                    # First frame with tracks: pick the best one
                    if main_track_id is None:
                        main_track = tracker.get_main_track()
                        if main_track:
                            main_track_id = main_track.track_id
                            if debug and all_frames is not None:
                                boxes_xyxy_all_frames[i] = main_track.bbox.astype(np.float32)
                            # Map to sampled frame if needed
                            if i in idxs:
                                sampled_idx = np.where(idxs == i)[0][0]
                                boxes_xyxy[sampled_idx] = main_track.bbox.astype(np.float32)
                    else:
                        # Subsequent frames: stick with the same track ID
                        main_track = None
                        for track in active_tracks:
                            if track.track_id == main_track_id:
                                main_track = track
                                break

                        if main_track:
                            # Found our tracked person
                            if debug and all_frames is not None:
                                boxes_xyxy_all_frames[i] = main_track.bbox.astype(np.float32)
                            # Map to sampled frame if needed
                            if i in idxs:
                                sampled_idx = np.where(idxs == i)[0][0]
                                boxes_xyxy[sampled_idx] = main_track.bbox.astype(np.float32)
                        else:
                            # Lost our main track - pick a new one
                            main_track = tracker.get_main_track()
                            if main_track:
                                # Only count as a switch if we had to pick a different person
                                tracking_switches.append(i)
                                main_track_id = main_track.track_id
                                if debug and all_frames is not None:
                                    boxes_xyxy_all_frames[i] = main_track.bbox.astype(np.float32)
                                # Map to sampled frame if needed
                                if i in idxs:
                                    sampled_idx = np.where(idxs == i)[0][0]
                                    boxes_xyxy[sampled_idx] = main_track.bbox.astype(np.float32)

    if vitpose_processor is None or vitpose_model is None:
        raise RuntimeError("ViTPose processor/model must be provided to process_video")

    kpts, scrs = infer_sequence(
        raw_video_path=video_path,
        image_processor=vitpose_processor,
        model=vitpose_model,
        device=torch.device(device),
        idxs=idxs,
        boxes_xyxy=boxes_xyxy,
    )
    # Store tracking data for debug visualization
    tracking_data = {
        'all_tracks_per_frame': all_tracks_per_frame if 'all_tracks_per_frame' in locals() else [],
        'main_track_id': main_track_id if 'main_track_id' in locals() else None,
        'boxes_all_frames': boxes_xyxy_all_frames if debug and boxes_xyxy_all_frames is not None else None,
        'all_frames': all_frames if debug and all_frames is not None else None,
        'sampled_indices': idxs
    }

    det = dict(
        keypoints=kpts,
        scores=scrs,
        bboxes=boxes_xyxy,
        det_scores=np.zeros(frames.shape[0], dtype=np.float32),  # Placeholder
        indices=idxs.astype(np.int32),
        has_hard_cuts=has_hard_cuts if 'has_hard_cuts' in locals() else False,
        hard_cut_frames=hard_cuts if 'hard_cuts' in locals() else [],
        tracking_switches=tracking_switches if 'tracking_switches' in locals() else [],
        tracking_data=tracking_data if 'tracking_data' in locals() else {},
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

    # Generate action descriptions if enabled
    action_descriptions = []
    if enable_action_description and action_describer and frames.shape[0] > 0:
        try:
            # Process video with tracking data to get descriptions
            descriptions = action_describer.process_video_with_tracking(
                video_path=video_path,
                keypoints=seq_k,
                bboxes=det["bboxes"],
                indices=idxs,
                segment_duration=3.0,
                fps=target_fps
            )

            # Convert to serializable format
            for desc in descriptions:
                action_descriptions.append({
                    'frames': desc.frame_indices,
                    'description': desc.description,
                    'confidence': desc.confidence
                })
        except Exception as e:
            print(f"Error generating action descriptions: {e}")

    # Save outputs
    vp = Path(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{vp.stem}.npz"

    # Prepare action descriptions for saving
    if action_descriptions:
        # Convert dataclass to dictionary for JSON serialization
        import json
        from dataclasses import asdict

        # Convert ActionDescription objects to dictionaries
        if hasattr(action_descriptions[0], '__dataclass_fields__'):
            # It's a dataclass, convert to dict
            action_dicts = [asdict(desc) for desc in action_descriptions]
        else:
            # Already a dict or other format
            action_dicts = action_descriptions

        action_json = json.dumps(action_dicts, default=str)  # Use default=str for any remaining numpy types
    else:
        action_json = "[]"

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
        action_descriptions=np.array([action_json], dtype=object),
        has_hard_cuts=np.array([det.get("has_hard_cuts", False)], dtype=bool),
        hard_cut_frames=np.array(det.get("hard_cut_frames", []), dtype=np.int32),
        tracking_confidence=det.get("det_scores", np.zeros(frames.shape[0], dtype=np.float32)),
        tracking_switches=np.array(det.get("tracking_switches", []), dtype=np.int32),
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

            # Draw ALL detected people with colored boxes
            if 'tracking_data' in det and t < len(det['tracking_data'].get('all_tracks_per_frame', [])):
                tracks = det['tracking_data']['all_tracks_per_frame'][t]
                main_id = det['tracking_data'].get('main_track_id')

                for track in tracks:
                    x1, y1, x2, y2 = [int(v) for v in track.bbox]
                    color = track.color
                    thickness = 3 if track.track_id == main_id else 1

                    # Draw box with track ID
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

                    # Draw track ID
                    label = f"ID:{track.track_id}"
                    if track.track_id == main_id:
                        label += " [MAIN]"

                    # Background for text
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1-text_h-4), (x1+text_w+4, y1), color, -1)
                    cv2.putText(img, label, (x1+2, y1-2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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

            if t in tracking_switches:
                cv2.putText(img, "TRACKING SWITCH", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

            writer.write(img)
        writer.release()

    return {"npz": str(npz_path), "debug": str(debug_path) if debug_path else ""}


