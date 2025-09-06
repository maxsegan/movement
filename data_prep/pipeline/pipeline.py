from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch

from data_prep.keypoints import h36m_coco_format
from data_prep.vitpose import infer_sequence
from data_prep.pipeline.sampling import probe_video_meta, read_frames_by_indices, sample_indices_for_fps
from data_prep.pose3d import load_motionagformer_from_path, lift_sequence_to_3d
from data_prep import clip_filtering as filt


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
) -> Dict[str, str]:
    meta = probe_video_meta(video_path)
    idxs = sample_indices_for_fps(meta["frames"], meta["fps"], target_fps=target_fps)
    frames = read_frames_by_indices(video_path, idxs)

    # ViTPose path using existing module (requires precomputed boxes per frame)
    boxes_xyxy = np.full((frames.shape[0], 4), np.nan, dtype=np.float32)
    from torchvision.transforms.functional import to_tensor
    if det_2d_model is not None:
        with torch.no_grad():
            for i in range(frames.shape[0]):
                img = to_tensor(frames[i]).to(device)
                out = det_2d_model([img])[0]
                if out["boxes"].numel() == 0:
                    continue
                labels = out["labels"].detach().cpu().numpy()
                scores = out["scores"].detach().cpu().numpy()
                boxes = out["boxes"].detach().cpu().numpy()
                mask = (labels == 1) & (scores >= 0.7)
                idx = np.flatnonzero(mask)
                if idx.size == 0:
                    continue
                j = int(idx[np.argmax(scores[idx])])
                boxes_xyxy[i] = boxes[j].astype(np.float32)

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
    det = dict(
        keypoints=kpts,
        scores=scrs,
        bboxes=boxes_xyxy,
        det_scores=np.zeros((frames.shape[0],), dtype=np.float32),
        indices=idxs.astype(np.int32),
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


