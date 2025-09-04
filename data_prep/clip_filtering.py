from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict
import numpy as np


# ------------------------------------------------------------
# Utilities & masking
# ------------------------------------------------------------
def _as_np(a) -> np.ndarray:
    return np.asarray(a)

def _is_valid_xy(points: np.ndarray) -> np.ndarray:
    # Valid if both coords are non-zero and not NaN.
    return np.isfinite(points[..., 0]) & np.isfinite(points[..., 1]) & (points[..., 0] != 0) & (points[..., 1] != 0)

def _valid_mask_with_conf(poses: np.ndarray, min_confidence: float) -> np.ndarray:
    has_conf = poses.shape[-1] >= 3
    xy_valid = _is_valid_xy(poses[..., :2])
    if has_conf:
        conf_ok = np.isfinite(poses[..., 2]) & (poses[..., 2] >= min_confidence)
        return xy_valid & conf_ok
    return xy_valid

def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return int(x.shape[0])


# ------------------------------------------------------------
# Segment-based visibility (mid-sequence occlusion handling)
# ------------------------------------------------------------
def split_visible_segments(
    poses: np.ndarray,
    min_joints: int = 12,
    min_len: int = 1,
) -> List[Tuple[int, int]]:
    """
    Return (start, end) index pairs (Python slice-style end exclusive) for
    contiguous runs where each frame has >= min_joints visible joints.

    Useful to keep only fully-visible chunks and drop occluded gaps.
    """
    arr = _as_np(poses)
    if arr.ndim != 3:
        raise ValueError("poses must be (T, J, C)")
    valid = _is_valid_xy(arr[..., :2])
    counts = np.sum(valid, axis=1)

    T = _safe_len(arr)
    segments: List[Tuple[int, int]] = []
    in_run = False
    run_start = 0
    for t in range(T):
        ok = counts[t] >= min_joints
        if ok and not in_run:
            in_run = True
            run_start = t
        elif not ok and in_run:
            if t - run_start >= min_len:
                segments.append((run_start, t))
            in_run = False
    if in_run and T - run_start >= min_len:
        segments.append((run_start, T))
    return segments


def trim_obfuscated_portions(poses: np.ndarray, min_joints: int = 12) -> np.ndarray:
    """
    Backwards-compatible head/tail trim (kept from your version) but NaN-safe.
    """
    arr = _as_np(poses)
    valid = _is_valid_xy(arr[..., :2])
    counts = np.sum(valid, axis=1)
    T = _safe_len(arr)

    start_idx = 0
    end_idx = T
    for i in range(T):
        if counts[i] >= min_joints:
            start_idx = i
            break
    for i in range(T - 1, -1, -1):
        if counts[i] >= min_joints:
            end_idx = i + 1
            break
    return arr[start_idx:end_idx]


# ------------------------------------------------------------
# Density / confidence
# ------------------------------------------------------------
def has_sufficient_keypoint_density(
    poses: np.ndarray,
    min_joints: int = 12,
    min_frame_percentage: float = 0.8,
) -> bool:
    arr = _as_np(poses)
    if _safe_len(arr) == 0:
        return False
    valid = _is_valid_xy(arr[..., :2])
    counts = np.sum(valid, axis=1)
    ok_frames = np.sum(counts >= min_joints)
    return (ok_frames / _safe_len(arr)) >= float(min_frame_percentage)


def per_frame_confidence_stats(poses: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Returns per-frame stats over *valid* joints only (non-zero xy, finite).
      - mean_conf, median_conf, valid_joint_count
    If no confidence channel, mean/median_conf will be NaN arrays.
    """
    arr = _as_np(poses)
    T, J = arr.shape[0], arr.shape[1]
    valid = _is_valid_xy(arr[..., :2])
    has_conf = arr.shape[-1] >= 3

    valid_counts = np.sum(valid, axis=1).astype(np.int32)
    mean_conf = np.full((T,), np.nan, dtype=np.float32)
    median_conf = np.full((T,), np.nan, dtype=np.float32)
    if has_conf:
        for t in range(T):
            if valid_counts[t] > 0:
                vals = arr[t, valid[t], 2]
                mean_conf[t] = float(np.nanmean(vals))
                median_conf[t] = float(np.nanmedian(vals))
    return dict(mean_conf=mean_conf, median_conf=median_conf, valid_joint_count=valid_counts)


# ------------------------------------------------------------
# Movement metrics (raw, camera-robust, and scale-normalized)
# ------------------------------------------------------------
def _pairwise_dists(points: np.ndarray) -> np.ndarray:
    # points: (K, 2)
    diffs = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diffs, axis=-1)

def _rigid_align(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Rigid Procrustes (no scaling). Returns R (2x2), t (ignored), and alignment error.
    Used to mitigate camera motion by aligning frame t+1 -> t.
    """
    # center
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    # SVD for rotation
    H = Bc.T @ Ac
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        # fix reflection
        Vt[-1, :] *= -1
        R = U @ Vt
    # translation not used directly; we align and measure residual
    B_aligned = (Bc @ R)
    err = float(np.mean(np.linalg.norm(B_aligned - Ac, axis=1)))
    return R, err

def _scale_proxy(points: np.ndarray, joint_pairs: Optional[List[Tuple[int, int]]] = None) -> float:
    """
    Estimate a person-relative scale. Defaults to max pairwise distance among valid joints
    if joint_pairs is None. If pairs are provided (e.g., shoulders/hips), use their median.
    """
    if points.shape[0] < 2:
        return 1.0
    if joint_pairs:
        dists = []
        for a, b in joint_pairs:
            if a < points.shape[0] and b < points.shape[0]:
                d = np.linalg.norm(points[a] - points[b])
                if np.isfinite(d):
                    dists.append(d)
        if dists:
            return float(np.median(dists)) or 1.0
    # Fallback: max pairwise (robust-ish to noise)
    D = _pairwise_dists(points)
    return float(np.nanmax(D)) or 1.0


def calculate_sequence_movement(
    poses: np.ndarray, min_confidence: float = 0.5
) -> float:
    arr = _as_np(poses)
    if arr.shape[-1] < 2:
        return 0.0
    mask = _valid_mask_with_conf(arr, min_confidence=min_confidence)

    total = 0.0
    pairs = 0
    for i in range(_safe_len(arr) - 1):
        joint_valid = mask[i] & mask[i + 1]
        if np.any(joint_valid):
            p0 = arr[i, joint_valid, :2]
            p1 = arr[i + 1, joint_valid, :2]
            d = np.linalg.norm(p1 - p0, axis=1)
            total += float(np.mean(d))
            pairs += 1
    return total / pairs if pairs else 0.0


def calculate_sequence_movement_camera_robust(
    poses: np.ndarray, min_confidence: float = 0.5
) -> float:
    """
    Camera-motion-robust by aligning t+1 to t with a rigid transform and
    measuring residual error (also robust to pure translation/rotation of camera).
    """
    arr = _as_np(poses)
    if arr.shape[-1] < 2:
        return 0.0
    mask = _valid_mask_with_conf(arr, min_confidence=min_confidence)

    total = 0.0
    pairs = 0
    for i in range(_safe_len(arr) - 1):
        joint_valid = mask[i] & mask[i + 1]
        if np.any(joint_valid):
            A = arr[i, joint_valid, :2]      # reference
            B = arr[i + 1, joint_valid, :2]  # next
            # Align B -> A to factor out rigid camera motion
            R, _ = _rigid_align(A, B)
            Bc = B - B.mean(axis=0, keepdims=True)
            Ac = A - A.mean(axis=0, keepdims=True)
            Baligned = (Bc @ R)
            residual = np.linalg.norm(Baligned - Ac, axis=1)  # per-joint residual
            total += float(np.mean(residual))
            pairs += 1
    return total / pairs if pairs else 0.0


def calculate_sequence_movement_normalized(
    poses: np.ndarray,
    min_confidence: float = 0.5,
    joint_pairs_for_scale: Optional[List[Tuple[int, int]]] = None,
    camera_robust: bool = True,
) -> float:
    """
    Movement divided by a per-frame scale proxy (e.g., shoulder width).
    Makes thresholds more portable across resolutions and person sizes.
    """
    arr = _as_np(poses)
    if arr.shape[-1] < 2:
        return 0.0
    mask = _valid_mask_with_conf(arr, min_confidence=min_confidence)

    total = 0.0
    pairs = 0
    for i in range(_safe_len(arr) - 1):
        joint_valid = mask[i] & mask[i + 1]
        if np.any(joint_valid):
            A = arr[i, joint_valid, :2]
            B = arr[i + 1, joint_valid, :2]
            if camera_robust:
                R, _ = _rigid_align(A, B)
                A0 = A - A.mean(axis=0, keepdims=True)
                B0 = B - B.mean(axis=0, keepdims=True)
                B_al = B0 @ R
                frame_motion = float(np.mean(np.linalg.norm(B_al - A0, axis=1)))
            else:
                frame_motion = float(np.mean(np.linalg.norm(B - A, axis=1)))
            # scale from reference frame A (valid joints)
            scale = _scale_proxy(A, joint_pairs_for_scale)
            total += frame_motion / max(scale, 1e-6)
            pairs += 1
    return total / pairs if pairs else 0.0


# ------------------------------------------------------------
# Dynamic decision + duration
# ------------------------------------------------------------
def is_sequence_dynamic(
    poses: Sequence[np.ndarray],
    movement_threshold: float = 0.15,     # normalized units (~15% of shoulder width per step)
    min_confidence: float = 0.5,
    min_valid_frames: int = 2,
    camera_robust: bool = True,
    joint_pairs_for_scale: Optional[List[Tuple[int, int]]] = None,
    use_percentile: Optional[float] = 90.0,  # use P90 over per-pair motions for robustness
) -> bool:
    arr = _as_np(poses)
    if _safe_len(arr) < min_valid_frames:
        return False

    # Compute per-pair normalized motion so we can take a percentile if desired
    arr = _as_np(poses)
    mask = _valid_mask_with_conf(arr, min_confidence=min_confidence)
    motions = []
    for i in range(_safe_len(arr) - 1):
        joint_valid = mask[i] & mask[i + 1]
        if np.any(joint_valid):
            A = arr[i, joint_valid, :2]
            B = arr[i + 1, joint_valid, :2]
            if camera_robust:
                R, _ = _rigid_align(A, B)
                A0 = A - A.mean(axis=0, keepdims=True)
                B0 = B - B.mean(axis=0, keepdims=True)
                B_al = B0 @ R
                m = float(np.mean(np.linalg.norm(B_al - A0, axis=1)))
            else:
                m = float(np.mean(np.linalg.norm(B - A, axis=1)))
            scale = _scale_proxy(A, joint_pairs_for_scale)
            motions.append(m / max(scale, 1e-6))

    if not motions:
        return False

    if use_percentile is not None:
        score = float(np.percentile(motions, use_percentile))
    else:
        score = float(np.mean(motions))
    return score >= movement_threshold


def is_min_duration(frame_indices: List[int], fps: float, min_seconds: float = 2.5) -> bool:
    return len(frame_indices) >= int(round(fps * min_seconds))


# ------------------------------------------------------------
# IOU-based occlusion (optional)
# ------------------------------------------------------------
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    a, b: [x1, y1, x2, y2]
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0

def flag_occluded_frames_by_iou(
    bboxes: np.ndarray,                      # (T,4) xyxy for this track
    other_tracks_bboxes: List[np.ndarray],   # list of (T,4) for others; can include NaNs when absent
    iou_thresh: float = 0.4,
) -> np.ndarray:
    """
    Returns boolean mask (T,) where True means 'occluded' because another box overlaps with IOU>=thr.
    """
    T = _safe_len(bboxes)
    occluded = np.zeros((T,), dtype=bool)
    for t in range(T):
        bb = bboxes[t]
        if not np.all(np.isfinite(bb)):  # no bbox at this frame
            continue
        for ob in other_tracks_bboxes:
            obb = ob[t]
            if not np.all(np.isfinite(obb)):
                continue
            if iou_xyxy(bb, obb) >= iou_thresh:
                occluded[t] = True
                break
    return occluded


# ------------------------------------------------------------
# Simple composite quality score
# ------------------------------------------------------------
@dataclass
class QualityWeights:
    density_w: float = 0.4
    confidence_w: float = 0.3
    stability_w: float = 0.2
    blur_w: float = 0.1  # hook for external per-frame blur scores (0=blurry,1=sharp)

def sequence_quality_score(
    poses: np.ndarray,
    min_joints: int = 12,
    min_confidence: float = 0.5,
    blur_scores_per_frame: Optional[np.ndarray] = None,  # normalized 0..1, optional
    weights: QualityWeights = QualityWeights(),
) -> float:
    """
    Produces a single sequence score in [0,1] (higher is better) combining:
      - density: fraction of frames with >= min_joints
      - confidence: mean of per-frame mean confidence (clipped to [min_confidence,1])
      - stability: low jitter after camera-robust alignment (less residual -> higher score)
      - blur: mean of provided blur scores (if available)
    """
    arr = _as_np(poses)
    T = _safe_len(arr)
    if T == 0:
        return 0.0

    # density
    valid = _is_valid_xy(arr[..., :2])
    counts = np.sum(valid, axis=1)
    density = float(np.mean(counts >= min_joints))

    # confidence
    stats = per_frame_confidence_stats(arr)
    mc = stats["mean_conf"]
    if np.all(np.isnan(mc)):
        conf_score = 1.0  # no confidence channel; don't penalize
    else:
        # Map [min_confidence .. 1.0] -> [0 .. 1], clip below to 0
        conf_norm = np.clip((mc - min_confidence) / max(1e-6, (1.0 - min_confidence)), 0.0, 1.0)
        conf_score = float(np.nanmean(conf_norm)) if np.any(np.isfinite(conf_norm)) else 0.0

    # stability (inverse of per-pair normalized residual)
    # We map residual r to score 1/(1 + r_norm) where r_norm uses scale proxy
    mask = _valid_mask_with_conf(arr, min_confidence=min_confidence)
    per_pair_scores = []
    for i in range(T - 1):
        joint_valid = mask[i] & mask[i + 1]
        if np.any(joint_valid):
            A = arr[i, joint_valid, :2]
            B = arr[i + 1, joint_valid, :2]
            R, _ = _rigid_align(A, B)
            A0 = A - A.mean(axis=0, keepdims=True)
            B0 = B - B.mean(axis=0, keepdims=True)
            residual = np.linalg.norm((B0 @ R) - A0, axis=1)
            r = float(np.mean(residual))
            s = _scale_proxy(A)
            r_norm = r / max(s, 1e-6)
            per_pair_scores.append(1.0 / (1.0 + r_norm))
    stability = float(np.mean(per_pair_scores)) if per_pair_scores else 0.0

    # blur
    if blur_scores_per_frame is not None and len(blur_scores_per_frame) == T:
        blur_score = float(np.nanmean(np.clip(blur_scores_per_frame, 0.0, 1.0)))
    else:
        blur_score = 1.0  # default to neutral if not provided

    w = weights
    score = (
        w.density_w * density
        + w.confidence_w * conf_score
        + w.stability_w * stability
        + w.blur_w * blur_score
    )
    return float(np.clip(score, 0.0, 1.0))
