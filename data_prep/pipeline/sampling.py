import cv2
import numpy as np


def probe_video_meta(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return dict(fps=fps, frames=frames, width=width, height=height)


def sample_indices_for_fps(total_frames: int, src_fps: float, target_fps: float = 20.0) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros((0,), dtype=np.int32)
    duration = total_frames / max(1e-6, float(src_fps))
    target_count = int(np.floor(duration * float(target_fps)))
    target_count = max(1, target_count)
    times = np.linspace(0.0, duration, num=target_count, endpoint=False)
    idxs = np.floor(times * float(src_fps)).astype(np.int32)
    idxs = np.clip(idxs, 0, max(0, total_frames - 1))
    return idxs


def read_frames_by_indices(path: str, indices: np.ndarray) -> np.ndarray:
    frames = []
    for i in indices.tolist():
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not frames:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


