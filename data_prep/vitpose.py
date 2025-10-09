from typing import Tuple
import numpy as np
import cv2
import torch
import PIL
import json
from pathlib import Path


def infer_sequence(
    raw_video_path: str,
    image_processor,
    model,
    device,
    idxs,
    boxes_xyxy,
    frames: np.ndarray = None,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run ViTPose across specific frame indices using per-frame person boxes.

    Args:
        raw_video_path: Path to video (used only if frames is None)
        image_processor: ViTPose image processor
        model: ViTPose model
        device: torch device
        idxs: Frame indices to process
        boxes_xyxy: Bounding boxes per frame
        frames: Optional pre-loaded frames (RGB format). If provided, avoids re-reading video.
        batch_size: Batch size for inference (default 16)

    Returns:
        all_keypoints: (1, T, 17, 2)
        all_scores:    (1, T, 17)
    """
    T = len(idxs)
    all_keypoints = np.zeros((1, T, 17, 2), dtype=np.float32)
    all_scores = np.zeros((1, T, 17), dtype=np.float32)

    # Prepare all crops and metadata first
    crops_to_process = []
    crop_metadata = []  # Store (t, x1_buffered, y1_buffered, person_boxes_coco)

    cap = None
    if frames is None:
        cap = cv2.VideoCapture(raw_video_path)

    for t, frame_no in enumerate(idxs):
        # Get frame
        if frames is not None:
            rgb_frame = frames[t]
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            rgb_frame = frame_bgr[:, :, ::-1]

        bbox_xyxy = np.array(boxes_xyxy[t], dtype=np.float32)

        # Skip frames with invalid bounding boxes
        if np.any(np.isnan(bbox_xyxy)) or np.any(np.isinf(bbox_xyxy)):
            continue

        # Check for valid bbox dimensions
        width = bbox_xyxy[2] - bbox_xyxy[0]
        height = bbox_xyxy[3] - bbox_xyxy[1]
        if width <= 0 or height <= 0:
            continue

        # Add buffer around bounding box (20% on each side)
        buffer_ratio = 0.2
        buffer_x = width * buffer_ratio
        buffer_y = height * buffer_ratio

        # Calculate buffered box coordinates
        x1_buffered = max(0, int(bbox_xyxy[0] - buffer_x))
        y1_buffered = max(0, int(bbox_xyxy[1] - buffer_y))
        x2_buffered = min(rgb_frame.shape[1], int(bbox_xyxy[2] + buffer_x))
        y2_buffered = min(rgb_frame.shape[0], int(bbox_xyxy[3] + buffer_y))

        # Crop the image to the buffered bounding box
        cropped_frame = rgb_frame[y1_buffered:y2_buffered, x1_buffered:x2_buffered]

        # If crop is too small, skip
        if cropped_frame.shape[0] < 10 or cropped_frame.shape[1] < 10:
            continue

        # Adjust bbox coordinates relative to the cropped image
        person_boxes_coco = np.expand_dims(
            np.array([
                bbox_xyxy[0] - x1_buffered,  # x relative to crop
                bbox_xyxy[1] - y1_buffered,  # y relative to crop
                width,
                height
            ], dtype=np.float32),
            axis=0,
        )

        crops_to_process.append(cropped_frame)
        crop_metadata.append((t, x1_buffered, y1_buffered, person_boxes_coco))

    if cap is not None:
        cap.release()

    # Process crops in batches
    with torch.no_grad():
        for batch_start in range(0, len(crops_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(crops_to_process))
            batch_crops = crops_to_process[batch_start:batch_end]
            batch_meta = crop_metadata[batch_start:batch_end]

            # Prepare batch inputs
            batch_pixel_values = []
            batch_boxes = []

            for crop, (t, x1_buf, y1_buf, boxes_coco) in zip(batch_crops, batch_meta):
                image = PIL.Image.fromarray(crop)
                pixel_values = image_processor(image, boxes=[boxes_coco], return_tensors="pt").pixel_values
                batch_pixel_values.append(pixel_values)
                batch_boxes.append(boxes_coco)

            # Stack into single batch tensor
            batch_tensor = torch.cat(batch_pixel_values, dim=0).to(device=device)
            dataset_indices = torch.zeros(batch_tensor.shape[0], dtype=torch.long, device=device)

            # Run batched inference
            outputs = model(batch_tensor, dataset_index=dataset_indices)

            # Post-process entire batch at once
            all_boxes = [meta[3] for meta in batch_meta]
            pose_results = image_processor.post_process_pose_estimation(outputs, boxes=all_boxes)

            # Extract results for each frame
            for i, (t, x1_buffered, y1_buffered, boxes_coco) in enumerate(batch_meta):
                image_pose_result = pose_results[i][0]

                # Get keypoints relative to cropped image
                keypoints_crop = image_pose_result["keypoints"].cpu().numpy()

                # Transform keypoints back to original image coordinates
                keypoints_orig = keypoints_crop.copy()
                keypoints_orig[:, 0] += x1_buffered  # Add back x offset
                keypoints_orig[:, 1] += y1_buffered  # Add back y offset

                all_keypoints[0, t] = keypoints_orig
                all_scores[0, t] = image_pose_result["scores"].cpu().numpy()

    return all_keypoints, all_scores


def infer_sequence_from_tracking(
    raw_video_path: str,
    tracking_path: Path | str,
    image_processor,
    model,
    device,
    person_id: str | int = "1",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load tracking json, run ViTPose across its frames, return (keypoints, scores, idxs).
    """
    with open(Path(tracking_path), "r") as f:
        tracking = json.load(f)
    idxs = np.array(tracking[str(person_id)]["frame_idxs"])  # (T,)
    boxes_xyxy = tracking[str(person_id)]["box"]
    all_keypoints, all_scores = infer_sequence(
        raw_video_path=raw_video_path,
        image_processor=image_processor,
        model=model,
        device=device,
        idxs=idxs,
        boxes_xyxy=boxes_xyxy,
    )
    return all_keypoints, all_scores, idxs


