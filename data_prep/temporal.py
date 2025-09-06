import numpy as np


def resample(n_frames: int, target: int = 243) -> np.ndarray:
    even = np.linspace(0, n_frames, num=target, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result


def to_clips(arr: np.ndarray, boxes_xyxy: np.ndarray, target: int = 243):
    """
    arr: (N,F,17,3)  (x,y,conf) in pixels + conf (BEFORE bbox-norm)
    boxes_xyxy: (F,4)
    Returns: arr_clips (list), boxes_clips (list), idx_maps (list or Nones)
    """
    N, F, J, D = arr.shape
    arr_cs, box_cs, idx_maps = [], [], []
    if F <= target:
        idx = np.floor(np.linspace(0, F-1, target)).astype(np.int32)
        arr_cs.append(arr[:, idx])
        box_cs.append(boxes_xyxy[idx])
        idx_maps.append(np.unique(idx, return_index=True)[1])
    else:
        for s in range(0, F, target):
            chunk = arr[:, s:s+target]
            bchunk = boxes_xyxy[s:s+target]
            if chunk.shape[1] < target:
                idx = np.floor(np.linspace(0, chunk.shape[1]-1, target)).astype(np.int32)
                arr_cs.append(chunk[:, idx])
                box_cs.append(bchunk[idx])
                idx_maps.append(np.unique(idx, return_index=True)[1])
            else:
                arr_cs.append(chunk)
                box_cs.append(bchunk)
                idx_maps.append(None)
    return arr_cs, box_cs, idx_maps


def turn_into_clips(keypoints: np.ndarray):
    clips = []
    n_frames = keypoints.shape[1]
    if n_frames <= 243:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, 243):
            keypoints_clip = keypoints[:, start_idx:start_idx + 243, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != 243:
                new_indices = resample(clip_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)
                downsample = None
    return clips, downsample


