"""
Kinetics Pose Dataset for VLA training.

Each video produces multiple training samples by sliding a window across the video.
Images are sampled from BEFORE the action window to provide temporal context
(motion direction) without leaking future information.
Bounding boxes are rendered on frames to indicate the target person.
"""

import hashlib
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class KineticsSample:
    """Metadata for a video clip (not a training sample - one video = many samples)."""
    pose_path: Path
    desc_path: Path
    video_path: Path
    action_class: str
    clip_id: str


def _deterministic_hash(text: str) -> float:
    """Stable hash to split train/val without storing manifest."""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _parse_description(desc_path: Path) -> str:
    """Extract the free-form description body from the txt file."""
    lines = desc_path.read_text().strip().splitlines()
    desc_lines: List[str] = []
    found = False
    for line in lines:
        if line.strip().lower().startswith("description"):
            found = True
            continue
        if found:
            stripped = line.strip()
            # Skip 'assistant' artifact from VLM output
            if stripped.lower() == "assistant":
                continue
            if stripped:
                desc_lines.append(stripped)
    if not desc_lines:
        # Fallback to the last non-empty line
        for line in reversed(lines):
            stripped = line.strip()
            if stripped and stripped.lower() != "assistant":
                return stripped
        return ""
    return " ".join(desc_lines)


# =============================================================================
# Joint Angle Conversion (3D positions -> joint angles)
# =============================================================================

def _normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize vector, handling zero length."""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.array([0, 1, 0], dtype=np.float32)
    return v / norm


def _compute_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle between two vectors in radians."""
    v1_n = _normalize_vector(v1)
    v2_n = _normalize_vector(v2)
    cos_angle = np.clip(np.dot(v1_n, v2_n), -1.0, 1.0)
    return np.arccos(cos_angle)


def _rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Find rotation matrix that aligns vec1 to vec2."""
    a = _normalize_vector(vec1)
    b = _normalize_vector(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.linalg.norm(v) < 1e-8:
        if c > 0:
            return np.eye(3, dtype=np.float32)
        else:
            return -np.eye(3, dtype=np.float32)

    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], dtype=np.float32)
    return np.eye(3, dtype=np.float32) + kmat + kmat @ kmat * ((1 - c) / (s ** 2 + 1e-8))


def _rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert rotation matrix to euler angles (pitch, roll, yaw)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        roll = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        roll = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return pitch, roll, yaw


def pose3d_to_joint_angles(pose3d: np.ndarray) -> np.ndarray:
    """
    Convert 3D skeleton positions to joint angles.

    H36M joint indices:
    0: Hip, 1: R_Hip, 2: R_Knee, 3: R_Ankle, 4: L_Hip, 5: L_Knee, 6: L_Ankle
    7: Spine, 8: Thorax, 9: Neck, 10: Head
    11: L_Shoulder, 12: L_Elbow, 13: L_Wrist, 14: R_Shoulder, 15: R_Elbow, 16: R_Wrist

    Returns 22 joint angles:
    - Spine (3): pitch, roll, yaw
    - L_Hip (3), L_Knee (1), R_Hip (3), R_Knee (1)
    - L_Shoulder (3), L_Elbow (1), R_Shoulder (3), R_Elbow (1)
    - Neck (3)
    """
    HIP, R_HIP, R_KNEE, R_ANKLE = 0, 1, 2, 3
    L_HIP, L_KNEE, L_ANKLE = 4, 5, 6
    SPINE, THORAX, NECK, HEAD = 7, 8, 9, 10
    L_SHOULDER, L_ELBOW, L_WRIST = 11, 12, 13
    R_SHOULDER, R_ELBOW, R_WRIST = 14, 15, 16

    angles = []
    up = np.array([0, 1, 0], dtype=np.float32)

    # 1. SPINE (3 DOF)
    pelvis_to_spine = pose3d[SPINE] - pose3d[HIP]
    R_spine = _rotation_matrix_from_vectors(up, pelvis_to_spine)
    angles.extend(_rotation_matrix_to_euler(R_spine))

    # 2. LEFT HIP (3 DOF)
    hip_to_knee_l = pose3d[L_KNEE] - pose3d[L_HIP]
    R_hip_l = _rotation_matrix_from_vectors(-up, hip_to_knee_l)
    angles.extend(_rotation_matrix_to_euler(R_hip_l))

    # 3. LEFT KNEE (1 DOF)
    knee_to_ankle_l = pose3d[L_ANKLE] - pose3d[L_KNEE]
    knee_angle_l = np.pi - _compute_angle(hip_to_knee_l, knee_to_ankle_l)
    angles.append(knee_angle_l)

    # 4. RIGHT HIP (3 DOF)
    hip_to_knee_r = pose3d[R_KNEE] - pose3d[R_HIP]
    R_hip_r = _rotation_matrix_from_vectors(-up, hip_to_knee_r)
    angles.extend(_rotation_matrix_to_euler(R_hip_r))

    # 5. RIGHT KNEE (1 DOF)
    knee_to_ankle_r = pose3d[R_ANKLE] - pose3d[R_KNEE]
    knee_angle_r = np.pi - _compute_angle(hip_to_knee_r, knee_to_ankle_r)
    angles.append(knee_angle_r)

    # 6. LEFT SHOULDER (3 DOF)
    shoulder_to_elbow_l = pose3d[L_ELBOW] - pose3d[L_SHOULDER]
    thorax_to_shoulder_l = pose3d[L_SHOULDER] - pose3d[THORAX]
    R_shoulder_l = _rotation_matrix_from_vectors(thorax_to_shoulder_l, shoulder_to_elbow_l)
    angles.extend(_rotation_matrix_to_euler(R_shoulder_l))

    # 7. LEFT ELBOW (1 DOF)
    elbow_to_wrist_l = pose3d[L_WRIST] - pose3d[L_ELBOW]
    elbow_angle_l = np.pi - _compute_angle(shoulder_to_elbow_l, elbow_to_wrist_l)
    angles.append(elbow_angle_l)

    # 8. RIGHT SHOULDER (3 DOF)
    shoulder_to_elbow_r = pose3d[R_ELBOW] - pose3d[R_SHOULDER]
    thorax_to_shoulder_r = pose3d[R_SHOULDER] - pose3d[THORAX]
    R_shoulder_r = _rotation_matrix_from_vectors(thorax_to_shoulder_r, shoulder_to_elbow_r)
    angles.extend(_rotation_matrix_to_euler(R_shoulder_r))

    # 9. RIGHT ELBOW (1 DOF)
    elbow_to_wrist_r = pose3d[R_WRIST] - pose3d[R_ELBOW]
    elbow_angle_r = np.pi - _compute_angle(shoulder_to_elbow_r, elbow_to_wrist_r)
    angles.append(elbow_angle_r)

    # 10. NECK (3 DOF)
    neck_to_head = pose3d[HEAD] - pose3d[NECK]
    R_neck = _rotation_matrix_from_vectors(up, neck_to_head)
    angles.extend(_rotation_matrix_to_euler(R_neck))

    return np.array(angles, dtype=np.float32)


# Number of joint angles output by pose3d_to_joint_angles
JOINT_ANGLES_DIM = 22


# =============================================================================
# Horizontal Flip Augmentation Helpers
# =============================================================================

# H36M left/right joint swap pairs (left_idx, right_idx)
_H36M_LR_PAIRS = [(4, 1), (5, 2), (6, 3), (11, 14), (12, 15), (13, 16)]


def flip_pose3d(pose3d: np.ndarray) -> np.ndarray:
    """Horizontally flip 3D pose: negate X-axis and swap left/right joints.

    Args:
        pose3d: [..., 17, 3] array of 3D joint positions.
    Returns:
        Flipped pose3d with same shape.
    """
    flipped = pose3d.copy()
    flipped[..., 0] *= -1  # Negate X-axis for mirror
    for l_idx, r_idx in _H36M_LR_PAIRS:
        tmp = flipped[..., l_idx, :].copy()
        flipped[..., l_idx, :] = flipped[..., r_idx, :]
        flipped[..., r_idx, :] = tmp
    return flipped


def flip_instruction_text(text: str) -> str:
    """Swap 'left'<->'right' in instruction text (case-preserving)."""
    # Use placeholder to prevent double-swap
    def _placeholder(m):
        return f'\x00{m.group(0)}\x00'
    text = re.sub(r'\b(left|right|Left|Right|LEFT|RIGHT)\b', _placeholder, text)
    swap = {'left': 'right', 'right': 'left', 'Left': 'Right', 'Right': 'Left',
            'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
    text = re.sub(r'\x00(left|right|Left|Right|LEFT|RIGHT)\x00',
                  lambda m: swap[m.group(1)], text)
    return text


def _euler_to_rotation_matrix(pitch: float, roll: float, yaw: float) -> np.ndarray:
    """Convert euler angles (pitch, roll, yaw) to rotation matrix. Inverse of _rotation_matrix_to_euler."""
    # ZYX convention
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(roll), np.sin(roll)
    cz, sz = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy * cz, -cy * sz, sy],
        [cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy],
        [sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy]
    ], dtype=np.float32)
    return R


def joint_angles_to_pose3d(
    angles: np.ndarray,
    reference_pose: Optional[np.ndarray] = None,
    bone_lengths: Optional[dict] = None,
) -> np.ndarray:
    """
    Convert joint angles back to 3D skeleton positions (Forward Kinematics).

    Args:
        angles: [22] array of joint angles in radians
        reference_pose: [17, 3] reference pose to extract bone lengths from.
                       If None, uses default normalized bone lengths.
        bone_lengths: Optional dict of bone lengths. If provided, overrides reference_pose.

    Returns:
        pose3d: [17, 3] array of 3D joint positions
    """
    # Joint indices
    HIP, R_HIP, R_KNEE, R_ANKLE = 0, 1, 2, 3
    L_HIP, L_KNEE, L_ANKLE = 4, 5, 6
    SPINE, THORAX, NECK, HEAD = 7, 8, 9, 10
    L_SHOULDER, L_ELBOW, L_WRIST = 11, 12, 13
    R_SHOULDER, R_ELBOW, R_WRIST = 14, 15, 16

    # Extract or use default bone lengths
    if bone_lengths is None:
        if reference_pose is not None:
            bone_lengths = {
                'hip_to_spine': np.linalg.norm(reference_pose[SPINE] - reference_pose[HIP]),
                'spine_to_thorax': np.linalg.norm(reference_pose[THORAX] - reference_pose[SPINE]),
                'thorax_to_neck': np.linalg.norm(reference_pose[NECK] - reference_pose[THORAX]),
                'neck_to_head': np.linalg.norm(reference_pose[HEAD] - reference_pose[NECK]),
                'hip_to_l_hip': np.linalg.norm(reference_pose[L_HIP] - reference_pose[HIP]),
                'hip_to_r_hip': np.linalg.norm(reference_pose[R_HIP] - reference_pose[HIP]),
                'l_thigh': np.linalg.norm(reference_pose[L_KNEE] - reference_pose[L_HIP]),
                'l_shin': np.linalg.norm(reference_pose[L_ANKLE] - reference_pose[L_KNEE]),
                'r_thigh': np.linalg.norm(reference_pose[R_KNEE] - reference_pose[R_HIP]),
                'r_shin': np.linalg.norm(reference_pose[R_ANKLE] - reference_pose[R_KNEE]),
                'thorax_to_l_shoulder': np.linalg.norm(reference_pose[L_SHOULDER] - reference_pose[THORAX]),
                'thorax_to_r_shoulder': np.linalg.norm(reference_pose[R_SHOULDER] - reference_pose[THORAX]),
                'l_upper_arm': np.linalg.norm(reference_pose[L_ELBOW] - reference_pose[L_SHOULDER]),
                'l_forearm': np.linalg.norm(reference_pose[L_WRIST] - reference_pose[L_ELBOW]),
                'r_upper_arm': np.linalg.norm(reference_pose[R_ELBOW] - reference_pose[R_SHOULDER]),
                'r_forearm': np.linalg.norm(reference_pose[R_WRIST] - reference_pose[R_ELBOW]),
                # Directional offsets for hips and shoulders (from reference)
                'l_hip_offset': reference_pose[L_HIP] - reference_pose[HIP],
                'r_hip_offset': reference_pose[R_HIP] - reference_pose[HIP],
                'l_shoulder_offset': reference_pose[L_SHOULDER] - reference_pose[THORAX],
                'r_shoulder_offset': reference_pose[R_SHOULDER] - reference_pose[THORAX],
            }
        else:
            # Default normalized bone lengths (roughly human proportions, ~1 unit height)
            bone_lengths = {
                'hip_to_spine': 0.12,
                'spine_to_thorax': 0.15,
                'thorax_to_neck': 0.08,
                'neck_to_head': 0.10,
                'hip_to_l_hip': 0.10,
                'hip_to_r_hip': 0.10,
                'l_thigh': 0.40,
                'l_shin': 0.38,
                'r_thigh': 0.40,
                'r_shin': 0.38,
                'thorax_to_l_shoulder': 0.18,
                'thorax_to_r_shoulder': 0.18,
                'l_upper_arm': 0.28,
                'l_forearm': 0.25,
                'r_upper_arm': 0.28,
                'r_forearm': 0.25,
                'l_hip_offset': np.array([-0.10, 0, 0], dtype=np.float32),
                'r_hip_offset': np.array([0.10, 0, 0], dtype=np.float32),
                'l_shoulder_offset': np.array([-0.18, 0, 0], dtype=np.float32),
                'r_shoulder_offset': np.array([0.18, 0, 0], dtype=np.float32),
            }

    # Initialize pose array
    pose3d = np.zeros((17, 3), dtype=np.float32)

    # Start from hip at origin
    pose3d[HIP] = np.array([0, 0, 0], dtype=np.float32)

    # Parse angles
    spine_angles = angles[0:3]      # pitch, roll, yaw
    l_hip_angles = angles[3:6]      # pitch, roll, yaw
    l_knee_angle = angles[6]        # flexion
    r_hip_angles = angles[7:10]     # pitch, roll, yaw
    r_knee_angle = angles[10]       # flexion
    l_shoulder_angles = angles[11:14]  # pitch, roll, yaw
    l_elbow_angle = angles[14]      # flexion
    r_shoulder_angles = angles[15:18]  # pitch, roll, yaw
    r_elbow_angle = angles[18]      # flexion
    neck_angles = angles[19:22]     # pitch, roll, yaw

    # Reference directions
    up = np.array([0, 1, 0], dtype=np.float32)
    down = np.array([0, -1, 0], dtype=np.float32)

    # Build kinematic chains

    # 1. SPINE chain: Hip -> Spine -> Thorax -> Neck -> Head
    R_spine = _euler_to_rotation_matrix(*spine_angles)
    spine_dir = R_spine @ up
    pose3d[SPINE] = pose3d[HIP] + spine_dir * bone_lengths['hip_to_spine']
    pose3d[THORAX] = pose3d[SPINE] + spine_dir * bone_lengths['spine_to_thorax']
    pose3d[NECK] = pose3d[THORAX] + up * bone_lengths['thorax_to_neck']

    # Neck orientation for head
    R_neck = _euler_to_rotation_matrix(*neck_angles)
    head_dir = R_neck @ up
    pose3d[HEAD] = pose3d[NECK] + head_dir * bone_lengths['neck_to_head']

    # 2. LEFT LEG: L_Hip -> L_Knee -> L_Ankle
    if isinstance(bone_lengths.get('l_hip_offset'), np.ndarray):
        pose3d[L_HIP] = pose3d[HIP] + bone_lengths['l_hip_offset']
    else:
        pose3d[L_HIP] = pose3d[HIP] + np.array([-bone_lengths['hip_to_l_hip'], 0, 0], dtype=np.float32)

    R_l_hip = _euler_to_rotation_matrix(*l_hip_angles)
    l_thigh_dir = R_l_hip @ down  # Legs point down by default
    pose3d[L_KNEE] = pose3d[L_HIP] + l_thigh_dir * bone_lengths['l_thigh']

    # Knee bends in the plane of the thigh
    # Simple planar knee: rotate thigh direction by knee angle around the perpendicular
    l_knee_bend = np.pi - l_knee_angle  # Convert from joint angle convention
    # Create rotation around X axis (assumes frontal plane)
    c, s = np.cos(l_knee_bend), np.sin(l_knee_bend)
    l_shin_dir = R_l_hip @ np.array([0, -c, s], dtype=np.float32)
    pose3d[L_ANKLE] = pose3d[L_KNEE] + l_shin_dir * bone_lengths['l_shin']

    # 3. RIGHT LEG: R_Hip -> R_Knee -> R_Ankle
    if isinstance(bone_lengths.get('r_hip_offset'), np.ndarray):
        pose3d[R_HIP] = pose3d[HIP] + bone_lengths['r_hip_offset']
    else:
        pose3d[R_HIP] = pose3d[HIP] + np.array([bone_lengths['hip_to_r_hip'], 0, 0], dtype=np.float32)

    R_r_hip = _euler_to_rotation_matrix(*r_hip_angles)
    r_thigh_dir = R_r_hip @ down
    pose3d[R_KNEE] = pose3d[R_HIP] + r_thigh_dir * bone_lengths['r_thigh']

    r_knee_bend = np.pi - r_knee_angle
    c, s = np.cos(r_knee_bend), np.sin(r_knee_bend)
    r_shin_dir = R_r_hip @ np.array([0, -c, s], dtype=np.float32)
    pose3d[R_ANKLE] = pose3d[R_KNEE] + r_shin_dir * bone_lengths['r_shin']

    # 4. LEFT ARM: L_Shoulder -> L_Elbow -> L_Wrist
    if isinstance(bone_lengths.get('l_shoulder_offset'), np.ndarray):
        pose3d[L_SHOULDER] = pose3d[THORAX] + bone_lengths['l_shoulder_offset']
    else:
        pose3d[L_SHOULDER] = pose3d[THORAX] + np.array([-bone_lengths['thorax_to_l_shoulder'], 0, 0], dtype=np.float32)

    # Shoulder direction from euler angles
    R_l_shoulder = _euler_to_rotation_matrix(*l_shoulder_angles)
    l_shoulder_ref = _normalize_vector(bone_lengths['l_shoulder_offset']) if isinstance(bone_lengths.get('l_shoulder_offset'), np.ndarray) else np.array([-1, 0, 0], dtype=np.float32)
    l_upper_arm_dir = R_l_shoulder @ l_shoulder_ref
    pose3d[L_ELBOW] = pose3d[L_SHOULDER] + l_upper_arm_dir * bone_lengths['l_upper_arm']

    # Elbow bends in the plane of the upper arm
    l_elbow_bend = np.pi - l_elbow_angle
    c, s = np.cos(l_elbow_bend), np.sin(l_elbow_bend)
    l_forearm_dir = R_l_shoulder @ np.array([-c, -s, 0], dtype=np.float32)
    pose3d[L_WRIST] = pose3d[L_ELBOW] + l_forearm_dir * bone_lengths['l_forearm']

    # 5. RIGHT ARM: R_Shoulder -> R_Elbow -> R_Wrist
    if isinstance(bone_lengths.get('r_shoulder_offset'), np.ndarray):
        pose3d[R_SHOULDER] = pose3d[THORAX] + bone_lengths['r_shoulder_offset']
    else:
        pose3d[R_SHOULDER] = pose3d[THORAX] + np.array([bone_lengths['thorax_to_r_shoulder'], 0, 0], dtype=np.float32)

    R_r_shoulder = _euler_to_rotation_matrix(*r_shoulder_angles)
    r_shoulder_ref = _normalize_vector(bone_lengths['r_shoulder_offset']) if isinstance(bone_lengths.get('r_shoulder_offset'), np.ndarray) else np.array([1, 0, 0], dtype=np.float32)
    r_upper_arm_dir = R_r_shoulder @ r_shoulder_ref
    pose3d[R_ELBOW] = pose3d[R_SHOULDER] + r_upper_arm_dir * bone_lengths['r_upper_arm']

    r_elbow_bend = np.pi - r_elbow_angle
    c, s = np.cos(r_elbow_bend), np.sin(r_elbow_bend)
    r_forearm_dir = R_r_shoulder @ np.array([c, -s, 0], dtype=np.float32)
    pose3d[R_WRIST] = pose3d[R_ELBOW] + r_forearm_dir * bone_lengths['r_forearm']

    return pose3d


def _draw_bbox_on_frame(frame: np.ndarray, bbox: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 3) -> np.ndarray:
    """Draw bounding box on frame. bbox is [x1, y1, x2, y2] in pixel coords."""
    if bbox is None or np.any(np.isnan(bbox)):
        return frame

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)

    # Clamp to frame bounds
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    frame = frame.copy()
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


class KineticsPoseDataset(Dataset):
    """
    Dataset that pairs Kinetics pose3D traces with generated descriptions and videos.

    Key features:
    - Multiple samples per video: slides a window at ~4Hz (stride=2-3 frames at 10fps)
    - Temporal context: images are sampled from BEFORE the action window to provide
      motion direction context without leaking future poses
    - Bbox rendering: draws bounding box on frames to indicate target person
    """

    def __init__(
        self,
        pose_dir: str,
        desc_dir: str,
        video_dir: str,  # Original video dir (not bbox_videos)
        split: str = "train",
        val_split: float = 0.02,
        action_horizon: int = 16,
        num_frames: int = 4,
        sample_stride: int = 3,  # Stride between samples (~3.3Hz at 10fps)
        resize: Optional[int] = 224,
        max_samples_per_class: Optional[int] = None,
        normalize_pose: bool = True,
        use_joint_angles: bool = True,  # Convert 3D positions to joint angles
        include_temporal_context: bool = False,  # Add temporal position to prompt
        action_focus_prompt: bool = False,  # Add prompt to focus on immediate next movements
        video_fps: float = 10.0,  # FPS of pose data for temporal context
        augment_flip: bool = False,  # Double data with horizontal flip augmentation
        seed: int = 42,
    ):
        self.pose_dir = Path(pose_dir)
        self.desc_dir = Path(desc_dir)
        self.video_dir = Path(video_dir)
        self.split = split
        self.val_split = val_split
        self.action_horizon = action_horizon
        self.num_frames = num_frames
        self.sample_stride = sample_stride
        self.resize = resize
        self.max_samples_per_class = max_samples_per_class
        self.normalize_pose = normalize_pose
        self.use_joint_angles = use_joint_angles
        self.include_temporal_context = include_temporal_context
        self.action_focus_prompt = action_focus_prompt
        self.video_fps = video_fps
        self.augment_flip = augment_flip
        self.rng = random.Random(seed)

        # Build index of video clips
        self.clips: List[KineticsSample] = []
        self.class_counts: Dict[str, int] = {}
        self._build_clip_index()

        # Build index of all training samples (clip_idx, start_frame)
        self.samples: List[Tuple[int, int]] = []
        self._build_sample_index()

        if not self.samples:
            raise ValueError(f"No samples found for split={split}")

        print(f"KineticsPoseDataset: {len(self.clips)} clips -> {len(self.samples)} samples ({split})")

    def _build_clip_index(self):
        """Build index of video clips."""
        by_class: Dict[str, List[KineticsSample]] = {}

        for desc_path in self.desc_dir.rglob("*.txt"):
            action_class = desc_path.parent.name
            clip_id = desc_path.stem
            pose_path = self.pose_dir / action_class / f"{clip_id}.npz"

            # Find original video (try both train and val subdirs)
            video_path = None
            for subdir in ["train", "val", "test"]:
                candidate = self.video_dir / subdir / action_class / f"{clip_id}.mp4"
                if candidate.exists():
                    video_path = candidate
                    break

            if not pose_path.exists() or video_path is None:
                continue

            # Train/val split based on clip_id hash
            hval = _deterministic_hash(clip_id)
            want_val = hval < self.val_split
            if (self.split == "val" and not want_val) or (self.split == "train" and want_val):
                continue

            sample = KineticsSample(
                pose_path=pose_path,
                desc_path=desc_path,
                video_path=video_path,
                action_class=action_class,
                clip_id=clip_id,
            )
            by_class.setdefault(action_class, []).append(sample)

        for action_class, clips in by_class.items():
            self.rng.shuffle(clips)
            if self.max_samples_per_class:
                clips = clips[: self.max_samples_per_class]
            self.clips.extend(clips)
            self.class_counts[action_class] = len(clips)

        # Deterministic order
        self.clips.sort(key=lambda s: (s.action_class, s.clip_id))

    def _build_sample_index(self):
        """Build index of (clip_idx, start_frame) pairs for all training samples."""
        for clip_idx, clip in enumerate(self.clips):
            # Load pose to get frame count
            try:
                data = np.load(clip.pose_path, allow_pickle=True)
                num_pose_frames = data["pose3d"].shape[0]
            except Exception:
                continue

            # Generate samples by sliding window
            # Each sample needs:
            # - num_frames of history (before action window)
            # - action_horizon frames (the action window itself)
            min_start = self.num_frames  # Need history frames before start
            max_start = num_pose_frames - self.action_horizon

            if max_start < min_start:
                # Video too short for proper history + action window
                # Skip this clip entirely to avoid information leakage
                continue
            else:
                # Slide window with stride, starting after we have enough history
                for start in range(min_start, max_start + 1, self.sample_stride):
                    self.samples.append((clip_idx, start))

        # Shuffle samples for training
        if self.split == "train":
            self.rng.shuffle(self.samples)

    def __len__(self):
        n = len(self.samples)
        return 2 * n if self.augment_flip else n

    def _load_video_frames(
        self,
        video_path: Path,
        frame_indices: List[int],
        bboxes: np.ndarray,
        pose_indices: np.ndarray
    ) -> List[Image.Image]:
        """Load specific frames from video and render bboxes."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            # Return black frames as fallback
            return [Image.fromarray(np.zeros((self.resize or 224, self.resize or 224, 3), dtype=np.uint8))
                    for _ in frame_indices]

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()

            if not ok:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find bbox for this frame
            # pose_indices maps pose frame -> video frame
            # We need to find pose_idx where pose_indices[pose_idx] == frame_idx
            pose_idx = np.where(pose_indices == frame_idx)[0]
            if len(pose_idx) > 0 and pose_idx[0] < len(bboxes):
                bbox = bboxes[pose_idx[0]]
                frame = _draw_bbox_on_frame(frame, bbox)

            # Resize
            if self.resize:
                frame = cv2.resize(frame, (self.resize, self.resize))

            frames.append(Image.fromarray(frame))

        cap.release()
        return frames

    def __getitem__(self, idx: int) -> Dict[str, object]:
        # Determine if this is a flipped sample (indices N..2N-1)
        should_flip = False
        if self.augment_flip and idx >= len(self.samples):
            should_flip = True
            idx = idx - len(self.samples)

        clip_idx, start_frame = self.samples[idx]
        clip = self.clips[clip_idx]

        # Load pose data
        data = np.load(clip.pose_path, allow_pickle=True)
        pose3d = data["pose3d"].astype(np.float32)  # [F, J, 3]
        bboxes = data["bboxes"].astype(np.float32)  # [F, 4]
        pose_indices = data["indices"].astype(np.int32)  # [F] - maps pose frame to video frame

        # Extract action sequence
        end_frame = start_frame + self.action_horizon
        action_seq = pose3d[start_frame:end_frame]

        # Pad if necessary (for short videos)
        if action_seq.shape[0] < self.action_horizon:
            pad = np.repeat(action_seq[-1:], self.action_horizon - action_seq.shape[0], axis=0)
            action_seq = np.concatenate([action_seq, pad], axis=0)

        # Horizontal flip augmentation: flip 3D pose before joint angle conversion
        if should_flip:
            action_seq = flip_pose3d(action_seq)

        if self.use_joint_angles:
            # Convert 3D positions to joint angles (22 DOF per frame)
            # Do this before normalization since joint angles are computed from relative positions
            joint_angles = np.stack([pose3d_to_joint_angles(action_seq[t]) for t in range(self.action_horizon)])
            action_seq = joint_angles  # [H, 22]
            robot_state = action_seq[0]  # Current joint angles as proprioception
        else:
            # Legacy: use normalized 3D positions
            if self.normalize_pose:
                root = action_seq[:, :1, :]
                action_seq = action_seq - root
                scale = np.linalg.norm(action_seq.reshape(-1, 3), axis=1).mean() + 1e-6
                action_seq = action_seq / scale
            action_seq = action_seq.reshape(self.action_horizon, -1)  # [H, J*3]
            robot_state = action_seq[0]  # Current pose as proprioception

        # Sample video frames from BEFORE the action window (history frames)
        # This provides temporal context (motion direction) without leaking future poses
        # Map pose frames to video frames using indices
        history_start = start_frame - self.num_frames
        history_end = start_frame - 1  # Last frame before action window

        # Get video frame indices for history window
        history_start_idx = max(0, history_start)
        history_end_idx = max(0, history_end)

        video_start = pose_indices[history_start_idx] if history_start_idx < len(pose_indices) else 0
        video_end = pose_indices[history_end_idx] if history_end_idx < len(pose_indices) else pose_indices[0]

        # Sample num_frames evenly across the history window
        video_frame_indices = np.linspace(video_start, video_end, self.num_frames, dtype=np.int32)

        # Load frames with bbox rendering
        images = self._load_video_frames(clip.video_path, video_frame_indices, bboxes, pose_indices)

        # Flip images if augmenting
        if should_flip:
            images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]

        # Instruction text
        desc_body = _parse_description(clip.desc_path)
        instruction = f"Task: {clip.action_class}. Instruction: {desc_body}"

        # Flip instruction text (swap left/right references)
        if should_flip:
            instruction = flip_instruction_text(instruction)

        # Add temporal context if enabled
        if self.include_temporal_context:
            total_frames = len(pose3d)
            current_time = start_frame / self.video_fps
            total_time = total_frames / self.video_fps
            progress_pct = (start_frame / total_frames) * 100
            instruction += f" Progress: {current_time:.1f}s/{total_time:.1f}s ({progress_pct:.0f}%)"

        # Add action focus prompt if enabled
        if self.action_focus_prompt:
            action_duration = self.action_horizon / self.video_fps
            instruction += f" Based on the current body position in these frames, predict the precise movements for the next {action_duration:.1f} seconds."

        return {
            "images": images,
            "instruction": instruction,
            "actions": action_seq,
            "robot_state": robot_state,
            "seq_name": f"{clip.action_class}/{clip.clip_id}",
            "action_class": clip.action_class,
        }
