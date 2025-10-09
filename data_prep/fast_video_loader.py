"""
Optimized video loading using ffmpeg-python for faster frame extraction.
"""

import numpy as np
import ffmpeg
import cv2
from pathlib import Path
from typing import Tuple, Optional


def probe_video_meta_fast(video_path: str) -> dict:
    """Fast video metadata extraction using ffprobe."""
    try:
        probe = ffmpeg.probe(video_path, select_streams='v:0')
        video_stream = probe['streams'][0]

        width = int(video_stream['width'])
        height = int(video_stream['height'])

        # Get FPS
        if 'r_frame_rate' in video_stream:
            fps_str = video_stream['r_frame_rate']
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den != 0 else 30.0
            else:
                fps = float(fps_str)
        else:
            fps = 30.0

        # Get frame count
        if 'nb_frames' in video_stream:
            frames = int(video_stream['nb_frames'])
        else:
            frames = int(float(video_stream.get('duration', 0)) * fps)

        return {
            'width': width,
            'height': height,
            'fps': fps,
            'frames': frames
        }
    except:
        # Fallback to OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return {
            'width': width,
            'height': height,
            'fps': fps,
            'frames': frames
        }


def read_frames_batch_fast(
    video_path: str,
    indices: np.ndarray,
    target_fps: Optional[float] = None
) -> np.ndarray:
    """
    Fast batch frame reading using ffmpeg.

    This is much faster than OpenCV for reading many frames.
    """
    meta = probe_video_meta_fast(video_path)
    width = meta['width']
    height = meta['height']
    fps = meta['fps']

    # Sort indices for sequential reading
    sorted_indices = np.sort(indices)

    # Use ffmpeg to extract frames
    out, _ = (
        ffmpeg
        .input(video_path)
        .filter('select', f'gte(n,{sorted_indices[0]})*lte(n,{sorted_indices[-1]})')
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, quiet=True)
    )

    # Convert to numpy array
    video = np.frombuffer(out, np.uint8)

    # Calculate expected frames
    n_frames_extracted = len(video) // (height * width * 3)

    if n_frames_extracted > 0:
        video = video.reshape((n_frames_extracted, height, width, 3))

        # Select only the frames we need
        frame_map = {}
        frame_idx = 0
        for i in range(sorted_indices[0], sorted_indices[-1] + 1):
            if i in sorted_indices:
                frame_map[i] = frame_idx
            frame_idx += 1

        # Build output in original order
        frames = []
        for idx in indices:
            if idx in frame_map and frame_map[idx] < len(video):
                frames.append(video[frame_map[idx]])

        if frames:
            return np.array(frames)

    # Fallback to OpenCV if ffmpeg fails
    return read_frames_opencv_fallback(video_path, indices)


def read_frames_opencv_fallback(video_path: str, indices: np.ndarray) -> np.ndarray:
    """Fallback to OpenCV for frame reading."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return np.array(frames) if frames else np.array([])


def sample_indices_for_fps(total_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    """Calculate frame indices for target FPS sampling."""
    if target_fps >= source_fps:
        # Sample all frames
        return np.arange(total_frames)

    # Sample at target FPS
    frame_interval = source_fps / target_fps
    indices = []
    current_frame = 0.0

    while current_frame < total_frames:
        indices.append(int(current_frame))
        current_frame += frame_interval

    return np.array(indices, dtype=np.int32)