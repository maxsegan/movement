from typing import List, Tuple
import numpy as np


def track_person_iou(
    raw_video_path: str,
    model_det,
    transform,
    iou_thresh: float = 0.3,
    score_thresh: float = 0.7,
) -> Tuple[List[int], List[List[float]]]:
    """
    Track a single person across frames using IoU continuity.

    Returns:
        frame_idxs: list of frame indices where a detection was chosen
        boxes_xyxy: list of [x1,y1,x2,y2] boxes per chosen frame
    """
    import cv2
    import torch

    from .boxes import iou_xyxy

    frame_idxs: List[int] = []
    boxes_xyxy: List[List[float]] = []
    last_box = None

    cap = cv2.VideoCapture(raw_video_path)
    i = 0
    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = transform(frame_rgb)
            out = model_det([img])[0]

            chosen = None
            if out["boxes"].numel() > 0:
                labels = out["labels"].detach().cpu().numpy()
                scores = out["scores"].detach().cpu().numpy()
                boxes = out["boxes"].detach().cpu().numpy()  # xyxy
                mask = (labels == 1) & (scores >= score_thresh)
                cand_idx = np.flatnonzero(mask)
                if cand_idx.size > 0:
                    if last_box is not None:
                        ious = [iou_xyxy(last_box, boxes[j]) for j in cand_idx]
                        j_rel = int(np.argmax(ious))
                        if ious[j_rel] >= iou_thresh:
                            chosen = boxes[cand_idx[j_rel]]
                    if chosen is None:
                        j_rel = int(np.argmax(scores[cand_idx]))
                        chosen = boxes[cand_idx[j_rel]]
            if chosen is not None:
                x1, y1, x2, y2 = map(float, chosen.tolist())
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                w, h = (x2 - x1), (y2 - y1)
                scale = 1.15
                nw, nh = w * scale, h * scale
                x1e, y1e = max(0.0, cx - nw / 2), max(0.0, cy - nh / 2)
                x2e, y2e = cx + nw / 2, cy + nh / 2
                frame_idxs.append(i)
                boxes_xyxy.append([x1e, y1e, x2e, y2e])
                last_box = np.array([x1e, y1e, x2e, y2e], dtype=np.float32)
            i += 1
    cap.release()
    return frame_idxs, boxes_xyxy


