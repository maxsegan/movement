import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SimilarityMetrics:
    """Container for similarity metrics between poses."""
    overall_similarity: float
    per_joint_similarity: np.ndarray
    temporal_consistency: float
    outlier_frames: List[int]
    statistics: Dict[str, float]

def normalize_pose(pose: np.ndarray, reference_joint: int = 0) -> np.ndarray:
    """
    Normalize pose by centering on reference joint (typically hip/root).

    Args:
        pose: (..., num_joints, 3) pose data
        reference_joint: Joint index to use as reference point

    Returns:
        Normalized pose with reference joint at origin
    """
    if pose.shape[-2] <= reference_joint:
        logger.warning(f"Reference joint {reference_joint} not available in pose with {pose.shape[-2]} joints")
        return pose

    reference_pos = pose[..., reference_joint:reference_joint+1, :]
    return pose - reference_pos

def calculate_bone_lengths(pose: np.ndarray, bone_connections: List[Tuple[int, int]]) -> np.ndarray:
    """
    Calculate bone lengths for pose validation.

    Args:
        pose: (..., num_joints, 3) pose data
        bone_connections: List of (parent_joint, child_joint) tuples

    Returns:
        Array of bone lengths (..., num_bones)
    """
    bone_lengths = []
    for parent_idx, child_idx in bone_connections:
        if parent_idx < pose.shape[-2] and child_idx < pose.shape[-2]:
            bone_vector = pose[..., child_idx, :] - pose[..., parent_idx, :]
            bone_length = np.linalg.norm(bone_vector, axis=-1)
            bone_lengths.append(bone_length)

    return np.stack(bone_lengths, axis=-1) if bone_lengths else np.array([])

def calculate_joint_similarity(pred_pose: np.ndarray, gt_pose: np.ndarray,
                             valid_joints: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate per-joint similarity using normalized euclidean distance.

    Args:
        pred_pose: (..., num_joints, 3) predicted pose
        gt_pose: (..., num_joints, 3) ground truth pose
        valid_joints: (..., num_joints) boolean mask for valid joints

    Returns:
        Per-joint similarity scores (0-1, higher is better)
    """
    # Normalize both poses
    pred_norm = normalize_pose(pred_pose)
    gt_norm = normalize_pose(gt_pose)

    # Calculate euclidean distances
    distances = np.linalg.norm(pred_norm - gt_norm, axis=-1)

    # Convert to similarity (using exponential decay)
    # Scale factor chosen empirically - smaller means more sensitive to errors
    scale_factor = 0.1
    similarities = np.exp(-distances / scale_factor)

    # Apply validity mask if provided
    if valid_joints is not None:
        similarities = similarities * valid_joints

    return similarities

def calculate_temporal_consistency(pose_sequence: np.ndarray,
                                 max_velocity_threshold: float = 1.0) -> Tuple[float, List[int]]:
    """
    Calculate temporal consistency by checking for impossible movements.

    Args:
        pose_sequence: (num_frames, num_joints, 3) pose sequence
        max_velocity_threshold: Maximum allowed joint velocity between frames

    Returns:
        Tuple of (consistency_score, outlier_frame_indices)
    """
    if pose_sequence.shape[0] < 2:
        return 1.0, []

    # Calculate frame-to-frame velocities
    velocities = np.diff(pose_sequence, axis=0)  # (num_frames-1, num_joints, 3)
    velocity_magnitudes = np.linalg.norm(velocities, axis=-1)  # (num_frames-1, num_joints)

    # Find outlier frames (any joint moving too fast)
    outlier_mask = np.any(velocity_magnitudes > max_velocity_threshold, axis=-1)
    outlier_frames = np.where(outlier_mask)[0] + 1  # +1 because diff reduces frame count

    # Calculate consistency score
    consistency_ratio = 1.0 - (len(outlier_frames) / (pose_sequence.shape[0] - 1))

    return consistency_ratio, outlier_frames.tolist()

def calculate_bone_length_consistency(pose_sequence: np.ndarray,
                                    bone_connections: List[Tuple[int, int]],
                                    tolerance: float = 0.1) -> float:
    """
    Calculate consistency of bone lengths across frames.

    Args:
        pose_sequence: (num_frames, num_joints, 3) pose sequence
        bone_connections: List of bone connections
        tolerance: Relative tolerance for bone length variation

    Returns:
        Bone length consistency score (0-1)
    """
    if pose_sequence.shape[0] < 2:
        return 1.0

    bone_lengths = calculate_bone_lengths(pose_sequence, bone_connections)
    if bone_lengths.size == 0:
        return 1.0

    # Calculate coefficient of variation for each bone
    mean_lengths = np.mean(bone_lengths, axis=0)
    std_lengths = np.std(bone_lengths, axis=0)

    # Avoid division by zero
    cv = np.divide(std_lengths, mean_lengths,
                   out=np.zeros_like(std_lengths), where=mean_lengths!=0)

    # Convert to consistency scores
    consistency_scores = np.maximum(0, 1 - cv / tolerance)

    return np.mean(consistency_scores)

def compare_pose_sequences(predicted_poses: np.ndarray,
                          ground_truth_poses: np.ndarray,
                          confidence_scores: Optional[np.ndarray] = None,
                          bone_connections: Optional[List[Tuple[int, int]]] = None) -> SimilarityMetrics:
    """
    Comprehensive comparison between predicted and ground truth pose sequences.

    Args:
        predicted_poses: (num_frames, num_joints, 3) predicted poses
        ground_truth_poses: (num_frames, num_joints, 3) ground truth poses
        confidence_scores: (num_frames, num_joints) confidence scores for predictions
        bone_connections: List of bone connections for validation

    Returns:
        SimilarityMetrics object containing all comparison results
    """
    # Ensure sequences have same length
    min_frames = min(predicted_poses.shape[0], ground_truth_poses.shape[0])
    pred_poses = predicted_poses[:min_frames]
    gt_poses = ground_truth_poses[:min_frames]

    if confidence_scores is not None:
        conf_scores = confidence_scores[:min_frames]
        # Create validity mask based on confidence threshold
        valid_joints = conf_scores > 0.5
    else:
        valid_joints = None

    # Calculate per-joint similarities
    per_joint_sim = calculate_joint_similarity(pred_poses, gt_poses, valid_joints)

    # Overall similarity (weighted by confidence if available)
    if valid_joints is not None:
        # Weight by confidence and validity
        weights = conf_scores * valid_joints
        overall_sim = np.average(per_joint_sim, weights=weights.flatten())
    else:
        overall_sim = np.mean(per_joint_sim)

    # Temporal consistency for predicted poses
    temporal_consistency, outlier_frames = calculate_temporal_consistency(pred_poses)

    # Bone length consistency if connections provided
    bone_consistency = 1.0
    if bone_connections:
        bone_consistency = calculate_bone_length_consistency(pred_poses, bone_connections)

    # Calculate statistics
    per_frame_similarities = np.mean(per_joint_sim, axis=-1)
    statistics = {
        'mean_similarity': float(np.mean(per_frame_similarities)),
        'median_similarity': float(np.median(per_frame_similarities)),
        'std_similarity': float(np.std(per_frame_similarities)),
        'min_similarity': float(np.min(per_frame_similarities)),
        'max_similarity': float(np.max(per_frame_similarities)),
        'q25_similarity': float(np.percentile(per_frame_similarities, 25)),
        'q75_similarity': float(np.percentile(per_frame_similarities, 75)),
        'temporal_consistency': temporal_consistency,
        'bone_consistency': bone_consistency,
        'num_outlier_frames': len(outlier_frames),
        'outlier_percentage': len(outlier_frames) / min_frames * 100
    }

    return SimilarityMetrics(
        overall_similarity=overall_sim,
        per_joint_similarity=per_joint_sim,
        temporal_consistency=temporal_consistency,
        outlier_frames=outlier_frames,
        statistics=statistics
    )

# H36M bone connections for validation
H36M_BONE_CONNECTIONS = [
    (0, 1), (0, 4),  # Hip to left/right hip
    (1, 2), (2, 3),  # Right leg
    (4, 5), (5, 6),  # Left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # Spine to head
    (8, 11), (11, 12), (12, 13),  # Left arm
    (8, 14), (14, 15), (15, 16),  # Right arm
]

def detect_pose_anomalies(pose_sequence: np.ndarray,
                         similarity_threshold: float = 0.7,
                         velocity_threshold: float = 1.0) -> Dict[str, List[int]]:
    """
    Detect various types of anomalies in pose sequences.

    Args:
        pose_sequence: (num_frames, num_joints, 3) pose sequence
        similarity_threshold: Threshold for frame-to-frame similarity
        velocity_threshold: Threshold for joint velocity

    Returns:
        Dictionary of anomaly types and frame indices
    """
    anomalies = {
        'high_velocity': [],
        'low_similarity': [],
        'missing_joints': [],
        'bone_length_inconsistency': []
    }

    if pose_sequence.shape[0] < 2:
        return anomalies

    # High velocity detection
    _, high_vel_frames = calculate_temporal_consistency(pose_sequence, velocity_threshold)
    anomalies['high_velocity'] = high_vel_frames

    # Frame-to-frame similarity
    for i in range(1, pose_sequence.shape[0]):
        prev_frame = pose_sequence[i-1:i]
        curr_frame = pose_sequence[i:i+1]

        frame_sim = calculate_joint_similarity(curr_frame, prev_frame)
        if np.mean(frame_sim) < similarity_threshold:
            anomalies['low_similarity'].append(i)

    # Missing joints (all zeros)
    for i, frame in enumerate(pose_sequence):
        zero_joints = np.all(frame == 0, axis=-1)
        if np.any(zero_joints):
            anomalies['missing_joints'].append(i)

    # Bone length inconsistency
    bone_consistency = calculate_bone_length_consistency(pose_sequence, H36M_BONE_CONNECTIONS)
    if bone_consistency < 0.8:  # Threshold for bone consistency
        # For simplicity, mark all frames as having bone inconsistency
        # In practice, you might want to identify specific problematic frames
        anomalies['bone_length_inconsistency'] = list(range(pose_sequence.shape[0]))

    return anomalies