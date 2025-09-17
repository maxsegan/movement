import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def read_ntu_skeleton(skeleton_path: str) -> Dict:
    """
    Read NTU RGB+D skeleton file format.

    Returns:
        Dictionary containing:
        - frames: List of frame data
        - num_bodies_per_frame: List of body counts per frame
        - joint_positions: (N_frames, max_bodies, 25, 3) array of xyz positions
        - joint_confidences: (N_frames, max_bodies, 25) array of confidence scores
    """
    with open(skeleton_path, 'r') as f:
        lines = f.readlines()

    line_idx = 0
    num_frames = int(lines[line_idx].strip())
    line_idx += 1

    frames_data = []
    all_joint_positions = []
    all_joint_confidences = []
    num_bodies_per_frame = []

    for frame_idx in range(num_frames):
        num_bodies = int(lines[line_idx].strip())
        line_idx += 1
        num_bodies_per_frame.append(num_bodies)

        frame_joint_positions = []
        frame_joint_confidences = []

        for body_idx in range(num_bodies):
            # Skip body info line (tracking ID, etc.)
            line_idx += 1

            # Number of joints (should be 25 for NTU)
            num_joints = int(lines[line_idx].strip())
            line_idx += 1

            body_joints = []
            body_confidences = []

            for joint_idx in range(num_joints):
                joint_data = lines[line_idx].strip().split()
                line_idx += 1

                # Parse joint data: x, y, z positions + other data
                x, y, z = float(joint_data[0]), float(joint_data[1]), float(joint_data[2])
                # Confidence/tracking state is the last value
                confidence = float(joint_data[-1])

                body_joints.append([x, y, z])
                body_confidences.append(confidence)

            frame_joint_positions.append(body_joints)
            frame_joint_confidences.append(body_confidences)

        all_joint_positions.append(frame_joint_positions)
        all_joint_confidences.append(frame_joint_confidences)

    # Convert to consistent numpy arrays (pad with zeros for missing bodies)
    max_bodies = max(num_bodies_per_frame) if num_bodies_per_frame else 0

    padded_positions = np.zeros((num_frames, max_bodies, 25, 3), dtype=np.float32)
    padded_confidences = np.zeros((num_frames, max_bodies, 25), dtype=np.float32)

    for frame_idx in range(num_frames):
        num_bodies = num_bodies_per_frame[frame_idx]
        for body_idx in range(num_bodies):
            positions = np.array(all_joint_positions[frame_idx][body_idx])
            confidences = np.array(all_joint_confidences[frame_idx][body_idx])
            padded_positions[frame_idx, body_idx] = positions
            padded_confidences[frame_idx, body_idx] = confidences

    return {
        'num_frames': num_frames,
        'num_bodies_per_frame': num_bodies_per_frame,
        'joint_positions': padded_positions,
        'joint_confidences': padded_confidences,
        'max_bodies': max_bodies
    }

def find_matching_skeleton(video_path: str, skeleton_root: Path) -> Optional[Path]:
    """
    Find the corresponding skeleton file for a given NTU video file.

    Args:
        video_path: Path to the .avi video file
        skeleton_root: Root directory containing skeleton files

    Returns:
        Path to matching skeleton file, or None if not found
    """
    video_name = Path(video_path).stem
    # Remove the '_rgb' suffix to get the base name
    if video_name.endswith('_rgb'):
        base_name = video_name[:-4]
    else:
        base_name = video_name

    skeleton_name = f"{base_name}.skeleton"

    # Search in skeleton root and subdirectories
    skeleton_candidates = list(skeleton_root.rglob(skeleton_name))

    if skeleton_candidates:
        return skeleton_candidates[0]

    logger.warning(f"No matching skeleton found for {video_name}")
    return None

def convert_ntu_to_h36m_joints(ntu_joints: np.ndarray) -> np.ndarray:
    """
    Convert NTU 25-joint format to H36M 17-joint format for comparison.

    NTU joint indices (25 joints):
    0: base of spine, 1: middle of spine, 2: neck, 3: head
    4: left shoulder, 5: left elbow, 6: left wrist, 7: left hand
    8: right shoulder, 9: right elbow, 10: right wrist, 11: right hand
    12: left hip, 13: left knee, 14: left ankle, 15: left foot
    16: right hip, 17: right knee, 18: right ankle, 19: right foot
    20: spine shoulder, 21: left hand tip, 22: left thumb, 23: right hand tip, 24: right thumb

    H36M joint indices (17 joints):
    0: hip, 1: right hip, 2: right knee, 3: right ankle
    4: left hip, 5: left knee, 6: left ankle
    7: spine, 8: thorax, 9: neck/nose, 10: head
    11: left shoulder, 12: left elbow, 13: left wrist
    14: right shoulder, 15: right elbow, 16: right wrist
    """
    # Mapping from NTU to H36M joint indices
    ntu_to_h36m = {
        0: 0,   # hip (base of spine)
        1: 7,   # spine (middle of spine)
        2: 9,   # neck
        3: 10,  # head
        4: 11,  # left shoulder
        5: 12,  # left elbow
        6: 13,  # left wrist
        8: 14,  # right shoulder
        9: 15,  # right elbow
        10: 16, # right wrist
        12: 4,  # left hip
        13: 5,  # left knee
        14: 6,  # left ankle
        16: 1,  # right hip
        17: 2,  # right knee
        18: 3,  # right ankle
        20: 8,  # thorax (spine shoulder)
    }

    # Initialize H36M joints array
    h36m_shape = ntu_joints.shape[:-2] + (17, ntu_joints.shape[-1])
    h36m_joints = np.zeros(h36m_shape, dtype=ntu_joints.dtype)

    # Map available joints
    for ntu_idx, h36m_idx in ntu_to_h36m.items():
        if ntu_idx < ntu_joints.shape[-2]:  # Check if NTU joint exists
            h36m_joints[..., h36m_idx, :] = ntu_joints[..., ntu_idx, :]

    return h36m_joints