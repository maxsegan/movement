from typing import Tuple
import numpy as np
import cv2
import torch
import PIL


def infer_sequence(
    raw_video_path: str,
    image_processor,
    model,
    device,
    idxs,
    boxes_xyxy,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run ViTPose across specific frame indices using per-frame person boxes.

    Returns:
        all_keypoints: (1, T, 17, 2)
        all_scores:    (1, T, 17)
    """
    T = len(idxs)
    all_keypoints = np.zeros((1, T, 17, 2), dtype=np.float32)
    all_scores = np.zeros((1, T, 17), dtype=np.float32)

    cap = cv2.VideoCapture(raw_video_path)
    for t, frame_no in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        rgb_frame = frame_bgr[:, :, ::-1]
        bbox_xyxy = np.array(boxes_xyxy[t], dtype=np.float32)
        person_boxes_coco = np.expand_dims(
            np.array([bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1]], dtype=np.float32),
            axis=0,
        )

        image = PIL.Image.fromarray(rgb_frame)
        pixel_values = image_processor(image, boxes=[person_boxes_coco], return_tensors="pt").pixel_values
        dataset_index = torch.tensor([0], device=device)

        pixel_values = pixel_values.to(device=device)
        with torch.no_grad():
            outputs = model(pixel_values, dataset_index=dataset_index)
        pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes_coco])
        image_pose_result = pose_results[0][0]
        all_keypoints[0, t] = image_pose_result["keypoints"].cpu().numpy()
        all_scores[0, t] = image_pose_result["scores"].cpu().numpy()

    cap.release()
    return all_keypoints, all_scores


