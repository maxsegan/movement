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
from data_prep.video_action_description import VideoActionDescriber, ActionDescription


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

    # ViTPose path using existing module (requires precomputed boxes per frame)
    # Detect hard cuts first
    hard_cuts = detect_hard_cuts(frames)
    has_hard_cuts = len(hard_cuts) > 0

    boxes_xyxy = np.full((frames.shape[0], 4), np.nan, dtype=np.float32)
    from torchvision.transforms.functional import to_tensor

    # Import IoU function for tracking
    from data_prep.boxes import iou_xyxy

    if det_2d_model is not None:
        # SIMPLE SINGLE-PERSON LOCK TRACKING
        # Pick ONE person at the start and NEVER switch unless completely lost
        target_box = None
        target_initialized = False
        consecutive_lost_frames = 0
        max_lost_frames = 15  # Be patient before giving up

        with torch.no_grad():
            for i in range(frames.shape[0]):
                img = to_tensor(frames[i]).to(device)
                out = det_2d_model([img])[0]

                chosen_box = None

                if out["boxes"].numel() > 0:
                    labels = out["labels"].detach().cpu().numpy()
                    scores = out["scores"].detach().cpu().numpy()
                    boxes = out["boxes"].detach().cpu().numpy()

                    # Filter for persons only
                    mask = (labels == 1) & (scores >= 0.3)  # Lower threshold to not lose people
                    idx = np.flatnonzero(mask)

                    if idx.size > 0:
                        person_boxes = boxes[idx]
                        person_scores = scores[idx]

                        if not target_initialized:
                            # FIRST FRAME: Pick the most prominent person
                            frame_h, frame_w = frames[i].shape[:2]
                            best_idx = 0
                            best_score = -1

                            for j, box in enumerate(person_boxes):
                                # Prefer larger, more central, higher confidence persons
                                area = (box[2] - box[0]) * (box[3] - box[1])
                                size_score = area / (frame_w * frame_h)

                                center_x = (box[0] + box[2]) / 2
                                center_y = (box[1] + box[3]) / 2
                                center_dist = np.sqrt((center_x - frame_w/2)**2 + (center_y - frame_h/2)**2)
                                center_score = 1.0 - (center_dist / (frame_w/2))

                                # Combined score: confidence + size + centrality
                                total_score = person_scores[j] * 0.4 + size_score * 5.0 + center_score * 0.2

                                if total_score > best_score:
                                    best_score = total_score
                                    best_idx = j

                            # Lock onto this person
                            target_box = person_boxes[best_idx].copy()
                            chosen_box = target_box.copy()
                            target_initialized = True
                            consecutive_lost_frames = 0

                        else:
                            # TRACKING: Find the same person we locked onto
                            # Calculate IoU with all candidates
                            ious = np.array([iou_xyxy(target_box, box) for box in person_boxes])

                            # RULE 1: Any IoU overlap > 0 means it's likely our person
                            overlap_indices = np.where(ious > 0)[0]

                            if len(overlap_indices) > 0:
                                # Among overlapping boxes, pick the one with highest IoU
                                best_overlap_idx = overlap_indices[np.argmax(ious[overlap_indices])]
                                chosen_box = person_boxes[best_overlap_idx].copy()
                                target_box = chosen_box.copy()  # Update target to latest position
                                consecutive_lost_frames = 0

                            else:
                                # No overlap - find closest box by center distance
                                target_center = np.array([(target_box[0] + target_box[2])/2,
                                                         (target_box[1] + target_box[3])/2])

                                min_dist = float('inf')
                                closest_idx = -1

                                for j, box in enumerate(person_boxes):
                                    center = np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])
                                    dist = np.linalg.norm(center - target_center)

                                    if dist < min_dist:
                                        min_dist = dist
                                        closest_idx = j

                                # Only accept if reasonably close (within ~10% of frame width)
                                frame_w = frames[i].shape[1]
                                if closest_idx >= 0 and min_dist < frame_w * 0.15:
                                    chosen_box = person_boxes[closest_idx].copy()
                                    target_box = chosen_box.copy()
                                    consecutive_lost_frames = 0
                                else:
                                    # Too far - we've lost them
                                    consecutive_lost_frames += 1
                    else:
                        # No person detections at all
                        if target_initialized:
                            consecutive_lost_frames += 1
                else:
                    # No detections at all
                    if target_initialized:
                        consecutive_lost_frames += 1

                # Check for hard cuts - reset tracking
                if i in hard_cuts:
                    target_box = None
                    target_initialized = False
                    consecutive_lost_frames = 0

                # If lost for too long, reset (but not at hard cuts)
                elif consecutive_lost_frames > max_lost_frames:
                    target_box = None
                    target_initialized = False
                    consecutive_lost_frames = 0

                # Store the chosen box
                if chosen_box is not None:
                    boxes_xyxy[i] = chosen_box.astype(np.float32)

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
    # Prepare tracking confidence array (all zeros since we removed confidence tracking)
    tracking_conf_array = np.zeros((frames.shape[0],), dtype=np.float32)

    det = dict(
        keypoints=kpts,
        scores=scrs,
        bboxes=boxes_xyxy,
        det_scores=tracking_conf_array,
        indices=idxs.astype(np.int32),
        has_hard_cuts=has_hard_cuts if 'has_hard_cuts' in locals() else False,
        hard_cut_frames=hard_cuts if 'hard_cuts' in locals() else [],
    )

    # Convert COCO->H36M
    h36m_kpts, h36m_scores, valid_frames = h36m_coco_format(det["keypoints"], det["scores"])
    if h36m_kpts.shape[0] == 0:
        # no detections; create empty placeholders
        h36m_kpts = np.zeros((1, frames.shape[0], 17, 2), dtype=np.float32)
        h36m_scores = np.zeros((1, frames.shape[0], 17), dtype=np.float32)

    seq_k = h36m_kpts[0]
    seq_s = h36m_scores[0]

    # Filtering metrics (tag only for now)
    density_ok = filt.has_sufficient_keypoint_density(seq_k)
    dynamic_ok = filt.is_sequence_dynamic(seq_k)
    quality = filt.sequence_quality_score(seq_k)

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
        writer = cv2.VideoWriter(str(debug_path), fourcc, target_fps, (frames.shape[2], frames.shape[1]))
        # Draw simple COCO-like points (use available data)
        for t in range(frames.shape[0]):
            img = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
            bb = det["bboxes"][t]
            if np.all(np.isfinite(bb)) and np.any(bb != 0):
                x1, y1, x2, y2 = [int(v) for v in bb]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            pts = seq_k[t]
            sc = seq_s[t]
            for j in range(pts.shape[0]):
                if sc[j] >= 0.15:
                    p = (int(round(pts[j, 0])), int(round(pts[j, 1])))
                    cv2.circle(img, p, 2, (0, 255, 255), -1, cv2.LINE_AA)
            writer.write(img)
        writer.release()

    return {"npz": str(npz_path), "debug": str(debug_path) if debug_path else ""}


