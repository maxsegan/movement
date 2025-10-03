"""
Clip validation based purely on pose quality and movement metrics.
No activity-specific logic - all decisions based on detected poses.
"""

import numpy as np
from typing import Tuple, List, Optional
from data_prep.boxes import iou_xyxy


def check_minimum_body_coverage(
    keypoints: np.ndarray,
    scores: np.ndarray,
    min_confidence: float = 0.3,
    min_valid_ratio: float = 0.4
) -> Tuple[bool, str]:
    """
    Check if enough of the body is visible in enough frames.

    Args:
        keypoints: (T, 17, 2) array of 2D keypoints
        scores: (T, 17) array of confidence scores
        min_confidence: Minimum confidence for a keypoint to be considered valid
        min_valid_ratio: Minimum ratio of frames that need valid body coverage

    Returns:
        (is_valid, reason_if_not)
    """
    if keypoints.shape[0] == 0:
        return False, "NO_KEYPOINTS"

    # Core joints that indicate a person is present
    # Nose(0), shoulders(5,6), hips(11,12)
    core_joints = {0, 5, 6, 11, 12}
    min_core_joints = 3

    valid_frames = 0
    total_frames = keypoints.shape[0]

    for t in range(total_frames):
        frame_kpts = keypoints[t]
        frame_scores = scores[t] if scores is not None else np.ones(17)

        # Count valid keypoints
        valid_mask = (frame_kpts[:, 0] > 0) & (frame_kpts[:, 1] > 0) & (frame_scores > min_confidence)

        # Check if core joints are present
        core_present = sum(valid_mask[j] for j in core_joints)

        if core_present >= min_core_joints:
            valid_frames += 1

    ratio = valid_frames / total_frames
    if ratio < min_valid_ratio:
        return False, f"INSUFFICIENT_COVERAGE_{int(ratio*100)}%"

    return True, "OK"


def check_motion(
    keypoints: np.ndarray,
    movement_threshold: float = 0.02,
    variance_threshold: float = 5.0
) -> Tuple[bool, str]:
    """
    Check if there is sufficient motion OR pose variation in the clip.

    Two types of valid motion:
    1. Continuous movement (person moving)
    2. Pose variation (person holding different poses)

    Args:
        keypoints: (T, 17, 2) array of 2D keypoints
        movement_threshold: Minimum average movement to be considered motion
        variance_threshold: Minimum pose variance to be considered variation

    Returns:
        (has_motion_or_variation, reason_if_not)
    """
    if keypoints.shape[0] < 2:
        return False, "TOO_SHORT_FOR_MOTION_CHECK"

    # Calculate frame-to-frame movement
    total_movement = 0
    valid_comparisons = 0
    pose_configurations = []

    for t in range(keypoints.shape[0] - 1):
        curr_kpts = keypoints[t]
        next_kpts = keypoints[t + 1]

        # Valid joints in both frames
        valid_mask = (curr_kpts[:, 0] > 0) & (curr_kpts[:, 1] > 0) & \
                    (next_kpts[:, 0] > 0) & (next_kpts[:, 1] > 0)

        if np.sum(valid_mask) < 3:
            continue

        # Calculate movement for valid joints
        movement = np.sqrt(np.sum((next_kpts[valid_mask] - curr_kpts[valid_mask])**2, axis=1))

        # Normalize by rough image size
        normalized_movement = np.mean(movement) / 100
        total_movement += normalized_movement
        valid_comparisons += 1

        # Store pose configuration for variation check
        pose_config = curr_kpts[valid_mask].flatten()
        pose_configurations.append(pose_config)

    if valid_comparisons == 0:
        return False, "NO_VALID_COMPARISONS"

    # Check average movement
    avg_movement = total_movement / valid_comparisons

    # Check pose variation (are poses different even if not moving much?)
    has_variation = False
    if len(pose_configurations) > 1:
        # Align all pose configs to same length
        min_len = min(len(p) for p in pose_configurations)
        if min_len > 0:
            try:
                aligned_configs = [p[:min_len] for p in pose_configurations]
                pose_array = np.array(aligned_configs)
                if pose_array.ndim >= 2:
                    # Calculate standard deviation across time
                    pose_std = np.std(pose_array, axis=0)
                    has_variation = np.mean(pose_std) > variance_threshold
            except:
                has_variation = False

    # Valid if either moving OR changing poses
    if avg_movement > movement_threshold or has_variation:
        return True, "OK"
    else:
        return False, "NO_MOTION"


def check_tracking_consistency(
    keypoints: np.ndarray,
    bboxes: np.ndarray,
    max_jump_ratio: float = 2.0,
    max_iou_breaks: float = 0.1
) -> Tuple[bool, str]:
    """
    Check if tracking is consistent throughout the clip.

    Args:
        keypoints: (T, 17, 2) array of 2D keypoints
        bboxes: (T, 4) array of bounding boxes
        max_jump_ratio: Maximum allowed ratio of bbox size for frame-to-frame jumps
        max_iou_breaks: Maximum ratio of frames with poor IoU overlap

    Returns:
        (is_consistent, reason_if_not)
    """
    if keypoints.shape[0] < 2:
        return True, "OK"

    iou_breaks = 0
    valid_transitions = 0

    for t in range(len(bboxes) - 1):
        bbox1 = bboxes[t]
        bbox2 = bboxes[t + 1]

        if np.any(np.isnan(bbox1)) or np.any(np.isnan(bbox2)):
            continue

        # Calculate IoU
        iou = iou_xyxy(bbox1, bbox2)

        # Calculate center distance
        center1 = [(bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2]
        center2 = [(bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2]

        # Calculate size change
        size1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        size2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        if size1 > 0 and size2 > 0:
            size_ratio = max(size1/size2, size2/size1)
        else:
            size_ratio = max_jump_ratio + 1

        valid_transitions += 1

        # Check for tracking breaks
        if iou < 0.1 and size_ratio > max_jump_ratio:
            iou_breaks += 1

    if valid_transitions == 0:
        return True, "OK"

    break_ratio = iou_breaks / valid_transitions
    if break_ratio > max_iou_breaks:
        return False, f"TRACKING_BREAKS_{int(break_ratio*100)}%"

    return True, "OK"


def validate_clip_improved(
    keypoints: np.ndarray,
    scores: np.ndarray,
    bboxes: np.ndarray,
    video_path: Optional[str] = None,
    min_confidence: float = 0.3,
    min_frames: int = 20,
    verbose: bool = False,
    has_hard_cuts: bool = False,
    hard_cut_frames: Optional[List[int]] = None,
    tracking_switches: Optional[List[int]] = None
) -> Tuple[bool, List[str], str]:
    """
    Validate clip based purely on pose quality and movement.

    Returns:
        (is_valid, list_of_issues, classification)
        classification is one of: VALID, INVALID, PARTIAL
    """
    issues = []

    # Check for hard cuts (scene transitions)
    if has_hard_cuts:
        issues.append("HARD_CUTS_DETECTED")
        if hard_cut_frames and len(hard_cut_frames) > 0:
            issues.append(f"SCENE_TRANSITIONS_AT_FRAMES_{hard_cut_frames[:3]}")

    # Check tracking consistency
    if tracking_switches and len(tracking_switches) > max(5, len(bboxes) * 0.05):
        issues.append(f"EXCESSIVE_TRACKING_SWITCHES_{len(tracking_switches)}_FRAMES")
    else:
        tracking_consistent, tracking_msg = check_tracking_consistency(keypoints, bboxes)
        if not tracking_consistent:
            issues.append(tracking_msg)

    # Check minimum frames
    if keypoints.shape[0] < min_frames:
        issues.append(f"TOO_SHORT_{keypoints.shape[0]}_frames")

    # Check body coverage
    has_coverage, reason = check_minimum_body_coverage(
        keypoints, scores, min_confidence
    )
    if not has_coverage:
        issues.append(reason)

    # Check motion/variation
    has_motion, reason = check_motion(keypoints)
    if not has_motion:
        issues.append(reason)

    # Determine classification
    if len(issues) == 0:
        return True, issues, "VALID"

    # Critical issues that make it INVALID
    critical_issues = {
        "NO_KEYPOINTS", "NO_VALID_BODY", "NO_DETECTION",
        "TOO_SHORT_FOR_MOTION_CHECK", "NO_VALID_COMPARISONS"
    }

    # Check for critical issues
    has_critical = any(
        issue in critical_issues or
        (issue.startswith("TOO_SHORT_") and int(issue.split('_')[2]) < 15)
        for issue in issues
    )

    if has_critical:
        return False, issues, "INVALID"

    # Check for issues that make it PARTIAL
    partial_triggers = {"NO_MOTION", "HARD_CUTS_DETECTED", "EXCESSIVE_TRACKING_SWITCHES"}
    has_partial_trigger = any(
        any(trigger in issue for trigger in partial_triggers)
        for issue in issues
    )

    if has_partial_trigger:
        return False, issues, "PARTIAL"

    # Default to INVALID for other issues
    return False, issues, "INVALID"


# Keep the old function for compatibility
def validate_clip(
    keypoints: np.ndarray,
    scores: np.ndarray = None,
    min_confidence: float = 0.3,
    min_frames: int = 25,
    verbose: bool = False
) -> Tuple[bool, List[str]]:
    """Legacy validation function for backward compatibility."""
    if scores is None:
        scores = np.ones((keypoints.shape[0], 17))

    # Create dummy bboxes
    bboxes = np.zeros((keypoints.shape[0], 4))

    is_valid, issues, _ = validate_clip_improved(
        keypoints, scores, bboxes, None,
        min_confidence, min_frames, verbose
    )

    return is_valid, issues