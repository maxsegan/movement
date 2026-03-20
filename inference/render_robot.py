"""
Render a humanoid robot from joint angles using MuJoCo.

Builds a MuJoCo scene with a realistic humanoid robot body composed of
capsules, ellipsoids, and boxes with metallic materials, positioned via
forward kinematics from our joint angle representation.

Supports two modes:
  --inference: Load the VLA model and render its predictions
  (default): Render directly from training data (ground truth)

Outputs: 3 individual PNGs (start, middle, end of trajectory) + a triptych.
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import sys
import numpy as np
from pathlib import Path
from PIL import Image
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from training.kinetics_dataset import (
    joint_angles_to_pose3d, pose3d_to_joint_angles,
    joint_angles_to_sincos, sincos_to_joint_angles,
    _parse_description,
)
from data_prep.constants import H36M_I, H36M_J


# H36M joint indices
HIP, R_HIP, R_KNEE, R_ANKLE = 0, 1, 2, 3
L_HIP, L_KNEE, L_ANKLE = 4, 5, 6
SPINE, THORAX, NECK, HEAD = 7, 8, 9, 10
L_SHOULDER, L_ELBOW, L_WRIST = 11, 12, 13
R_SHOULDER, R_ELBOW, R_WRIST = 14, 15, 16

# Body part groupings for materials
TORSO_BONES = [(HIP, SPINE), (SPINE, THORAX)]
HEAD_BONES = [(NECK, HEAD)]
NECK_BONES = [(THORAX, NECK)]
L_ARM_BONES = [(L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST)]
R_ARM_BONES = [(R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST)]
L_LEG_BONES = [(L_HIP, L_KNEE), (L_KNEE, L_ANKLE)]
R_LEG_BONES = [(R_HIP, R_KNEE), (R_KNEE, R_ANKLE)]
SHOULDER_BONES = [(THORAX, L_SHOULDER), (THORAX, R_SHOULDER)]
HIP_BONES = [(HIP, L_HIP), (HIP, R_HIP)]

# Material assignments: (bone_pairs, material_name, radius_scale)
# base_radius = 0.022, so effective radii are base * scale
BONE_MATERIALS = [
    (TORSO_BONES, "torso_mat", 2.8),
    (HEAD_BONES, "head_mat", 1.5),
    (NECK_BONES, "neck_mat", 1.2),
    ([(L_SHOULDER, L_ELBOW), (R_SHOULDER, R_ELBOW)], "upper_limb_mat", 1.6),
    ([(L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)], "limb_mat", 1.3),
    ([(L_HIP, L_KNEE), (R_HIP, R_KNEE)], "upper_limb_mat", 1.8),
    ([(L_KNEE, L_ANKLE), (R_KNEE, R_ANKLE)], "limb_mat", 1.5),
    (SHOULDER_BONES, "torso_mat", 2.2),
    (HIP_BONES, "torso_mat", 2.2),
]

# Joint material assignments: (joint_indices, material_name, radius)
JOINT_MATERIALS = [
    ([HIP], "joint_mat", 0.06),
    ([SPINE, THORAX], "torso_mat", 0.055),
    ([NECK], "joint_mat", 0.035),
    ([HEAD], "head_mat", 0.075),
    ([L_SHOULDER, R_SHOULDER], "joint_mat", 0.048),
    ([L_ELBOW, R_ELBOW], "joint_mat", 0.038),
    ([L_WRIST, R_WRIST], "joint_mat", 0.030),
    ([L_HIP, R_HIP], "joint_mat", 0.052),
    ([L_KNEE, R_KNEE], "joint_mat", 0.042),
    ([L_ANKLE, R_ANKLE], "joint_mat", 0.035),
]


def capsule_from_endpoints(p1, p2, radius):
    """Compute capsule position, quaternion, and half-length from two endpoints."""
    mid = (p1 + p2) / 2.0
    diff = p2 - p1
    length = np.linalg.norm(diff)
    half_len = length / 2.0

    # MuJoCo capsules are aligned along Z axis by default
    z_axis = np.array([0, 0, 1], dtype=np.float64)
    direction = diff / (length + 1e-10)

    # Quaternion to rotate z_axis to direction
    quat = _rotation_quat(z_axis, direction)

    return mid, quat, half_len


def _rotation_quat(from_vec, to_vec):
    """Compute quaternion rotating from_vec to to_vec (wxyz format)."""
    from_vec = from_vec / (np.linalg.norm(from_vec) + 1e-10)
    to_vec = to_vec / (np.linalg.norm(to_vec) + 1e-10)

    dot = np.clip(np.dot(from_vec, to_vec), -1.0, 1.0)

    if dot > 0.9999:
        return np.array([1, 0, 0, 0], dtype=np.float64)
    if dot < -0.9999:
        # 180-degree rotation around any perpendicular axis
        perp = np.cross(from_vec, np.array([1, 0, 0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(from_vec, np.array([0, 1, 0]))
        perp = perp / np.linalg.norm(perp)
        return np.array([0, perp[0], perp[1], perp[2]], dtype=np.float64)

    cross = np.cross(from_vec, to_vec)
    w = 1 + dot
    quat = np.array([w, cross[0], cross[1], cross[2]], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat


def build_mjcf_xml(pose3d, label=""):
    """Build a MuJoCo XML string for a humanoid robot at the given pose."""

    # Base capsule radius (scaled up for robotic look)
    base_radius = 0.022

    geom_lines = []

    # Add bone capsules
    for bone_pairs, mat_name, radius_scale in BONE_MATERIALS:
        radius = base_radius * radius_scale
        for i, j in bone_pairs:
            p1 = pose3d[i]
            p2 = pose3d[j]
            mid, quat, half_len = capsule_from_endpoints(p1, p2, radius)

            if half_len < 1e-4:
                continue

            geom_lines.append(
                f'    <geom type="capsule" '
                f'pos="{mid[0]:.6f} {mid[1]:.6f} {mid[2]:.6f}" '
                f'quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}" '
                f'size="{radius:.4f} {half_len:.6f}" '
                f'material="{mat_name}"/>'
            )

    # Add joint spheres
    for joint_indices, mat_name, radius in JOINT_MATERIALS:
        for idx in joint_indices:
            p = pose3d[idx]
            geom_lines.append(
                f'    <geom type="sphere" '
                f'pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" '
                f'size="{radius:.4f}" '
                f'material="{mat_name}"/>'
            )

    # Add a "chest plate" box geom between shoulders
    chest_center = (pose3d[THORAX] + pose3d[SPINE]) / 2
    chest_width = np.linalg.norm(pose3d[L_SHOULDER] - pose3d[R_SHOULDER]) * 0.38
    chest_height = np.linalg.norm(pose3d[THORAX] - pose3d[SPINE]) * 0.35
    chest_depth = base_radius * 2.5

    # Compute chest orientation from spine direction
    spine_dir = pose3d[THORAX] - pose3d[SPINE]
    spine_dir = spine_dir / (np.linalg.norm(spine_dir) + 1e-10)
    shoulder_dir = pose3d[R_SHOULDER] - pose3d[L_SHOULDER]
    shoulder_dir = shoulder_dir / (np.linalg.norm(shoulder_dir) + 1e-10)
    chest_normal = np.cross(shoulder_dir, spine_dir)
    chest_normal = chest_normal / (np.linalg.norm(chest_normal) + 1e-10)

    # Build rotation matrix -> quaternion for chest
    rot_mat = np.column_stack([shoulder_dir, spine_dir, chest_normal])
    chest_quat = _mat_to_quat(rot_mat)

    geom_lines.append(
        f'    <geom type="box" '
        f'pos="{chest_center[0]:.6f} {chest_center[1]:.6f} {chest_center[2]:.6f}" '
        f'quat="{chest_quat[0]:.6f} {chest_quat[1]:.6f} {chest_quat[2]:.6f} {chest_quat[3]:.6f}" '
        f'size="{chest_width:.4f} {chest_height:.4f} {chest_depth:.4f}" '
        f'material="chest_mat"/>'
    )

    # Add pelvis box
    pelvis_center = pose3d[HIP]
    pelvis_width = np.linalg.norm(pose3d[L_HIP] - pose3d[R_HIP]) * 0.5
    pelvis_height = base_radius * 3.0
    pelvis_depth = base_radius * 3.0

    hip_dir = pose3d[R_HIP] - pose3d[L_HIP]
    hip_dir = hip_dir / (np.linalg.norm(hip_dir) + 1e-10)
    pelvis_up = spine_dir
    pelvis_fwd = np.cross(hip_dir, pelvis_up)
    pelvis_fwd = pelvis_fwd / (np.linalg.norm(pelvis_fwd) + 1e-10)
    pelvis_rot = np.column_stack([hip_dir, pelvis_up, pelvis_fwd])
    pelvis_quat = _mat_to_quat(pelvis_rot)

    geom_lines.append(
        f'    <geom type="box" '
        f'pos="{pelvis_center[0]:.6f} {pelvis_center[1]:.6f} {pelvis_center[2]:.6f}" '
        f'quat="{pelvis_quat[0]:.6f} {pelvis_quat[1]:.6f} {pelvis_quat[2]:.6f} {pelvis_quat[3]:.6f}" '
        f'size="{pelvis_width:.4f} {pelvis_height:.4f} {pelvis_depth:.4f}" '
        f'material="torso_mat"/>'
    )

    # Add "feet" boxes at ankles
    for ankle_idx in [L_ANKLE, R_ANKLE]:
        foot_pos = pose3d[ankle_idx].copy()
        foot_pos[2] -= 0.01  # Slightly below ankle (Z-up)
        geom_lines.append(
            f'    <geom type="box" '
            f'pos="{foot_pos[0]:.6f} {foot_pos[1]:.6f} {foot_pos[2]:.6f}" '
            f'size="0.04 0.018 0.07" '
            f'material="limb_mat"/>'
        )

    # Add "hands" boxes at wrists
    for wrist_idx in [L_WRIST, R_WRIST]:
        hand_pos = pose3d[wrist_idx].copy()
        geom_lines.append(
            f'    <geom type="box" '
            f'pos="{hand_pos[0]:.6f} {hand_pos[1]:.6f} {hand_pos[2]:.6f}" '
            f'size="0.022 0.032 0.015" '
            f'material="limb_mat"/>'
        )

    # Add visor/eye slit on the head (a thin box in front of the head)
    head_pos = pose3d[HEAD]
    neck_to_head = pose3d[HEAD] - pose3d[NECK]
    neck_to_head_dir = neck_to_head / (np.linalg.norm(neck_to_head) + 1e-10)
    # Forward direction: perpendicular to spine in the shoulder plane
    fwd_dir = np.cross(shoulder_dir, neck_to_head_dir)
    fwd_dir = fwd_dir / (np.linalg.norm(fwd_dir) + 1e-10)
    visor_pos = head_pos + fwd_dir * 0.06 - neck_to_head_dir * 0.015
    visor_rot = np.column_stack([shoulder_dir, fwd_dir, neck_to_head_dir])
    visor_quat = _mat_to_quat(visor_rot)
    geom_lines.append(
        f'    <geom type="box" '
        f'pos="{visor_pos[0]:.6f} {visor_pos[1]:.6f} {visor_pos[2]:.6f}" '
        f'quat="{visor_quat[0]:.6f} {visor_quat[1]:.6f} {visor_quat[2]:.6f} {visor_quat[3]:.6f}" '
        f'size="0.045 0.012 0.015" '
        f'material="visor_mat"/>'
    )

    geom_str = "\n".join(geom_lines)

    xml = f"""<mujoco model="robot_pose">
  <visual>
    <global offwidth="1200" offheight="1600"/>
    <rgba fog="0.22 0.24 0.28 1"/>
    <headlight ambient="0.25 0.25 0.28" diffuse="0.15 0.15 0.18" specular="0.1 0.1 0.1"/>
  </visual>
  <option gravity="0 0 0"/>

  <asset>
    <!-- Dark charcoal torso -->
    <material name="torso_mat"
      rgba="0.18 0.20 0.22 1"
      specular="0.8" shininess="0.6" reflectance="0.15"/>
    <!-- Chest plate - slightly lighter with blue tint -->
    <material name="chest_mat"
      rgba="0.22 0.24 0.28 1"
      specular="0.9" shininess="0.7" reflectance="0.2"/>
    <!-- Silver metallic limbs -->
    <material name="limb_mat"
      rgba="0.55 0.58 0.62 1"
      specular="0.9" shininess="0.8" reflectance="0.25"/>
    <!-- Darker metallic upper limbs (thighs, upper arms) -->
    <material name="upper_limb_mat"
      rgba="0.35 0.38 0.42 1"
      specular="0.85" shininess="0.7" reflectance="0.2"/>
    <!-- Accent joints - cyan/teal highlights -->
    <material name="joint_mat"
      rgba="0.1 0.65 0.75 1"
      specular="0.95" shininess="0.9" reflectance="0.3"/>
    <!-- Neck - between torso and head -->
    <material name="neck_mat"
      rgba="0.35 0.38 0.42 1"
      specular="0.85" shininess="0.7" reflectance="0.2"/>
    <!-- Head - dark with slight metallic -->
    <material name="head_mat"
      rgba="0.15 0.17 0.20 1"
      specular="0.85" shininess="0.7" reflectance="0.2"/>
    <!-- Visor - glowing cyan -->
    <material name="visor_mat"
      rgba="0.0 0.85 0.95 1"
      specular="1.0" shininess="1.0" reflectance="0.5"
      emission="0.6"/>
    <!-- Ground -->
    <material name="ground_mat"
      rgba="0.28 0.30 0.34 1"
      specular="0.3" shininess="0.1" reflectance="0.3"/>
    <!-- Grid texture for ground -->
    <texture name="grid_tex" type="2d" builtin="checker"
      rgb1="0.25 0.27 0.30" rgb2="0.32 0.34 0.37"
      width="512" height="512"/>
    <material name="grid_mat" texture="grid_tex"
      texrepeat="8 8" specular="0.3" shininess="0.1" reflectance="0.3"/>
  </asset>

  <worldbody>
    <!-- Lights (Z-up convention) -->
    <!-- Key light: bright from upper-front-right -->
    <light pos="1.5 -2.0 3.0" dir="-0.4 0.6 -0.7" diffuse="1.0 1.0 1.05" specular="0.7 0.7 0.7" castshadow="true"/>
    <!-- Fill light: from upper-front-left -->
    <light pos="-1.2 -1.5 2.5" dir="0.4 0.5 -0.7" diffuse="0.6 0.62 0.7" specular="0.4 0.4 0.45" castshadow="false"/>
    <!-- Rim/back light: edge highlights -->
    <light pos="0.0 2.0 2.0" dir="0 -0.7 -0.3" diffuse="0.35 0.38 0.45" specular="0.25 0.25 0.3" castshadow="false"/>
    <!-- Ground bounce -->
    <light pos="0.0 0.0 -0.5" dir="0 0 1" diffuse="0.18 0.19 0.22" specular="0.08 0.08 0.08" castshadow="false"/>

    <!-- Ground plane (Z-up, so plane is at z=0) -->
    <geom type="plane" size="2 2 0.01" material="grid_mat"/>

    <!-- Robot body -->
{geom_str}
  </worldbody>
</mujoco>"""

    return xml


def _mat_to_quat(R):
    """Convert 3x3 rotation matrix to quaternion (wxyz)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat


def transform_pose_for_render(pose3d, scale=None):
    """Transform pose so Z is up (MuJoCo convention), centered, feet on ground.

    Raw poses are in camera coordinates (X-right, Y-down, Z-forward).
    MuJoCo uses Z-up convention, so we align hip-to-head with +Z.

    Args:
        scale: If provided, use this scale factor instead of auto-computing.
               Useful for consistent sizing across multiple frames.
    Returns:
        (pose, scale_used): Transformed pose and the scale that was applied.
    """
    pose = pose3d.copy().astype(np.float64)

    # Center on hip
    pose -= pose[HIP]

    # Find the "up" direction: average of hip->head and hip->thorax
    up_vec = pose[HEAD] - pose[HIP]
    up_vec2 = pose[THORAX] - pose[HIP]
    up_dir = (up_vec + up_vec2) / 2
    up_dir = up_dir / (np.linalg.norm(up_dir) + 1e-10)

    # Target: Z-up (MuJoCo convention)
    target_up = np.array([0, 0, 1], dtype=np.float64)

    # Rotate skeleton so "up" aligns with +Z
    R = _rotation_matrix_between(up_dir, target_up)
    pose = (R @ pose.T).T

    # Fix roll: ensure shoulders are roughly along X axis
    shoulder_dir = pose[R_SHOULDER] - pose[L_SHOULDER]
    shoulder_dir[2] = 0  # Project to XY plane
    if np.linalg.norm(shoulder_dir) > 1e-6:
        shoulder_dir = shoulder_dir / np.linalg.norm(shoulder_dir)
        # Rotation around Z axis to align shoulders with X
        angle = np.arctan2(shoulder_dir[1], shoulder_dir[0])
        c, s = np.cos(-angle), np.sin(-angle)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        pose = (Rz @ pose.T).T

    # Scale to humanoid height (~1.6m)
    if scale is None:
        height = pose[:, 2].max() - pose[:, 2].min()
        scale = 1.6 / height if height > 0.01 else 1.0
    pose *= scale

    # Shift so lowest point is at ground (z=0)
    min_z = pose[:, 2].min()
    pose[:, 2] -= min_z

    return pose, scale


def _rotation_matrix_between(from_vec, to_vec):
    """Compute rotation matrix that rotates from_vec to to_vec."""
    from_vec = from_vec / (np.linalg.norm(from_vec) + 1e-10)
    to_vec = to_vec / (np.linalg.norm(to_vec) + 1e-10)

    v = np.cross(from_vec, to_vec)
    c = np.dot(from_vec, to_vec)

    if c > 0.9999:
        return np.eye(3, dtype=np.float64)
    if c < -0.9999:
        # 180-degree rotation: find perpendicular axis
        perp = np.cross(from_vec, np.array([1, 0, 0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(from_vec, np.array([0, 1, 0]))
        perp = perp / np.linalg.norm(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=np.float64)

    R = np.eye(3) + vx + vx @ vx / (1 + c)
    return R


def render_pose(pose3d, width=1200, height=1600, scale=None, camera_center_z=None):
    """Render a single pose and return an image array."""
    pose, scale_used = transform_pose_for_render(pose3d, scale=scale)
    xml = build_mjcf_xml(pose)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height, width)

    # Configure camera
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE

    # Front-facing 3/4 view, slightly elevated (Z-up)
    if camera_center_z is None:
        camera_center_z = (pose[:, 2].max() + pose[:, 2].min()) / 2
    camera.lookat[:] = [0, 0, camera_center_z]
    camera.distance = 3.8
    camera.azimuth = -80  # Front-facing 3/4 view (robot faces -Y)
    camera.elevation = -15  # Slightly above

    # Scene options
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False

    renderer.update_scene(data, camera, scene_option)
    pixels = renderer.render()

    renderer.close()

    return pixels, scale_used, pose


def _pose_uprightness(pose):
    """Score how upright a pose is (head-ankle vertical separation after alignment)."""
    head = pose[HEAD]
    ankles = (pose[R_ANKLE] + pose[L_ANKLE]) / 2
    return np.linalg.norm(head - ankles)


def select_frames(pose_seq):
    """Select 3 frames: standing, deepest squat, standing again.

    First finds the tallest frame to establish a consistent scale,
    then picks the best standing->squat->standing progression.

    Returns:
        (frame_indices, ref_scale): List of 3 frame indices and the reference scale.
    """
    num_frames = len(pose_seq)

    # Step 1: Find the tallest frame to use as the reference scale.
    # Sample every 5th frame for speed.
    best_raw_height = 0
    best_scale = None
    for i in range(0, num_frames, 5):
        p, s = transform_pose_for_render(pose_seq[i])
        raw_height = p[:, 2].max() - p[:, 2].min()
        # The auto-scale targets 1.6m, so the raw height before scaling
        # determines the scale. We want the frame with the LARGEST raw height
        # (= smallest scale factor = most upright/tallest pose).
        if s is not None and (best_scale is None or s < best_scale):
            best_scale = s
            best_raw_height = raw_height

    # Step 2: Compute hip height for every frame using that scale
    hip_z = np.zeros(num_frames)
    for i in range(num_frames):
        p, _ = transform_pose_for_render(pose_seq[i], scale=best_scale)
        hip_z[i] = p[HIP, 2]

    # Step 3: Find the deepest squat frame (lowest hip) in the middle 60%
    margin = num_frames // 5
    search_start = margin
    search_end = num_frames - margin
    low_frame = search_start + int(np.argmin(hip_z[search_start:search_end]))

    # Find tallest standing frame BEFORE the squat
    start_frame = int(np.argmax(hip_z[:low_frame])) if low_frame > 0 else 0

    # Find tallest standing frame AFTER the squat
    end_frame = low_frame + int(np.argmax(hip_z[low_frame:])) if low_frame < num_frames else num_frames - 1

    print(f"  Frame selection: {start_frame}(hip={hip_z[start_frame]:.2f}), "
          f"{low_frame}(hip={hip_z[low_frame]:.2f}), {end_frame}(hip={hip_z[end_frame]:.2f})")

    return [start_frame, low_frame, end_frame], best_scale


def load_sample_trajectory(action_class=None, clip_id=None):
    """Load a sample trajectory from the dataset.

    If action_class and clip_id are given, loads that specific clip.
    Otherwise searches for a good trajectory automatically.
    """
    pose_dir = Path("/root/movement/data/kinetics_processed")

    # Load specific clip if requested
    if action_class and clip_id:
        f = pose_dir / action_class / f"{clip_id}.npz"
        d = np.load(f, allow_pickle=True)
        pose = d["pose3d"].astype(np.float32)
        print(f"Loaded: {action_class}/{clip_id} ({pose.shape[0]} frames)")
        return pose, action_class, clip_id

    # Auto-select: try action classes with clear upright motion
    candidates = [
        "squat", "doing aerobics", "lunges", "dancing ballet",
        "tai chi", "clean and jerk", "punching bag",
    ]

    for cls in candidates:
        cls_dir = pose_dir / cls
        if not cls_dir.exists():
            continue

        files = sorted(cls_dir.glob("*.npz"))
        best_score = 0
        best_file = None
        best_pose = None

        for f in files[:30]:
            try:
                d = np.load(f, allow_pickle=True)
                pose = d["pose3d"].astype(np.float32)
                if pose.shape[0] < 30:
                    continue
                # Score: uprightness consistency * motion
                uprights = [_pose_uprightness(pose[i]) for i in range(len(pose))]
                min_up = min(uprights)
                mean_up = np.mean(uprights)
                motion = np.abs(np.diff(pose, axis=0)).mean()
                # Prefer consistently upright with visible motion
                score = min_up * mean_up * motion
                if score > best_score:
                    best_score = score
                    best_file = f
                    best_pose = pose
            except Exception:
                continue

        if best_file is not None:
            print(f"Selected: {cls}/{best_file.stem} ({best_pose.shape[0]} frames, score={best_score:.4f})")
            return best_pose, cls, best_file.stem

    raise RuntimeError("No suitable trajectory found in kinetics_processed")


def load_vla_model(checkpoint_path=None, device="cuda:0"):
    """Load the VLA model from checkpoint."""
    import torch
    import yaml
    from training.vla_model import VLAModel, VLAConfig

    if checkpoint_path is None:
        checkpoint_path = PROJECT_ROOT / "checkpoints/kinetics_vla/best_model.pth"

    config_path = PROJECT_ROOT / "training/config_kinetics.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config["model_config"]
    vla_config = VLAConfig(
        qwen_model_name=mc.get("qwen_model_name", "Qwen/Qwen3-VL-4B-Instruct"),
        qwen_hidden_size=mc.get("qwen_hidden_size", 2560),
        use_intermediate_hidden=mc.get("use_intermediate_hidden", True),
        hidden_layer_index=mc.get("hidden_layer_index", 18),
        use_early_exit=mc.get("use_early_exit", True),
        use_deepstack_features=mc.get("use_deepstack_features", True),
        use_flash_attention=mc.get("use_flash_attention", True),
        projection_dim=mc.get("projection_dim", 1024),
        action_dim=mc.get("action_dim", 44),
        diffusion_hidden_dim=mc.get("diffusion_hidden_dim", 1536),
        num_diffusion_layers=mc.get("num_diffusion_layers", 24),
        num_diffusion_heads=mc.get("num_diffusion_heads", 24),
        num_future_tokens=mc.get("num_future_tokens", 4),
        action_horizon=mc.get("action_horizon", 16),
        num_frames=mc.get("num_frames", 4),
        use_lora=mc.get("use_lora", True),
        lora_rank=mc.get("lora_rank", 128),
        lora_alpha=mc.get("lora_alpha", 128),
        lora_dropout=mc.get("lora_dropout", 0.05),
        freeze_vision_encoder=mc.get("freeze_vision_encoder", True),
        freeze_qwen_layers=mc.get("freeze_qwen_layers", 0),
        use_thinking_mode=mc.get("use_thinking_mode", False),
        diffusion_steps=mc.get("diffusion_steps", 1),
        init_from_current_pose=mc.get("init_from_current_pose", False),
    )

    print(f"Loading model to {device}...")
    model = VLAModel(vla_config).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Remove 'module.' prefix if present (from DDP training)
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully.")
    return model


def prepare_inference_inputs(
    action_class, clip_id, start_frame, num_frames=4, image_size=224,
):
    """Prepare model inputs for a single sample, matching the dataset format.

    Returns:
        images: List of PIL Images (num_frames history frames)
        instruction: str
        robot_state_sincos: np.ndarray [44] (sin/cos at start_frame)
        ground_truth_sincos: np.ndarray [action_horizon, 44]
    """
    import cv2
    from training.kinetics_dataset import _draw_bbox_on_frame

    pose_dir = Path("/root/movement/data/kinetics_processed")
    desc_dir = Path("/root/movement/data/kinetics_full_output/descriptions")
    video_dir = Path("/root/movement/data/kinetics-dataset/k700-2020")

    # Load pose data
    npz_path = pose_dir / action_class / f"{clip_id}.npz"
    data = np.load(npz_path, allow_pickle=True)
    pose3d = data["pose3d"].astype(np.float32)
    bboxes = data["bboxes"].astype(np.float32)
    pose_indices = data["indices"].astype(np.int32)

    # Find video
    video_path = None
    for subdir in ["train", "val", "test"]:
        candidate = video_dir / subdir / action_class / f"{clip_id}.mp4"
        if candidate.exists():
            video_path = candidate
            break
    if video_path is None:
        raise FileNotFoundError(f"Video not found for {action_class}/{clip_id}")

    # Compute ground truth sin/cos for the action window
    action_horizon = 16
    end_frame = min(start_frame + action_horizon, len(pose3d))
    action_seq = pose3d[start_frame:end_frame]
    if len(action_seq) < action_horizon:
        pad = np.repeat(action_seq[-1:], action_horizon - len(action_seq), axis=0)
        action_seq = np.concatenate([action_seq, pad], axis=0)

    joint_angles = np.stack([pose3d_to_joint_angles(action_seq[t]) for t in range(action_horizon)])
    gt_sincos = joint_angles_to_sincos(joint_angles)
    robot_state = gt_sincos[0]  # Current state as proprioception

    # Load history images (matching dataset __getitem__ logic)
    history_start = max(0, start_frame - num_frames)
    history_end = max(0, start_frame - 1)
    video_start = pose_indices[history_start] if history_start < len(pose_indices) else 0
    video_end = pose_indices[history_end] if history_end < len(pose_indices) else pose_indices[0]
    video_frame_indices = np.linspace(video_start, video_end, num_frames, dtype=np.int32)

    cap = cv2.VideoCapture(str(video_path))
    images = []
    for frame_idx in video_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find matching bbox
        pose_idx = np.where(pose_indices == frame_idx)[0]
        if len(pose_idx) > 0 and pose_idx[0] < len(bboxes):
            bbox = bboxes[pose_idx[0]]
            frame = _draw_bbox_on_frame(frame, bbox)

        if image_size:
            frame = cv2.resize(frame, (image_size, image_size))

        images.append(Image.fromarray(frame))
    cap.release()

    # Build instruction
    desc_path = desc_dir / action_class / f"{clip_id}.txt"
    desc_body = _parse_description(desc_path)
    instruction = f"Task: {action_class}. Instruction: {desc_body}"

    return images, instruction, robot_state, gt_sincos


def run_inference_trajectory(model, action_class, clip_id, start_frame):
    """Run VLA model inference starting at start_frame and return the full predicted trajectory.

    The model predicts 16 future timesteps. We return all of them as 3D poses.

    Returns:
        predicted_poses: list of 16 (17, 3) arrays - the predicted trajectory
        gt_poses: list of 16 (17, 3) arrays - ground truth for comparison
    """
    import torch

    device = next(model.parameters()).device

    # Load raw pose data for reference bone lengths
    pose_dir = Path("/root/movement/data/kinetics_processed")
    npz_path = pose_dir / action_class / f"{clip_id}.npz"
    raw_pose3d = np.load(npz_path, allow_pickle=True)["pose3d"].astype(np.float32)

    print(f"  Running inference from frame {start_frame}...")
    images, instruction, robot_state, gt_sincos = prepare_inference_inputs(
        action_class, clip_id, start_frame=start_frame
    )

    robot_state_tensor = torch.from_numpy(robot_state).float().unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model.forward(
                [images],
                [instruction],
                robot_state=robot_state_tensor,
                compute_loss=False,
            )
        pred_sincos = output["actions"].squeeze(0).cpu().numpy()  # [16, 44]

    predicted_poses = []
    gt_poses = []

    for t in range(pred_sincos.shape[0]):
        ref_idx = min(start_frame + t, len(raw_pose3d) - 1)
        reference_pose = raw_pose3d[ref_idx]

        pred_angles = sincos_to_joint_angles(pred_sincos[t])
        pred_pose3d = joint_angles_to_pose3d(pred_angles, reference_pose=reference_pose)
        predicted_poses.append(pred_pose3d)

        gt_angles = sincos_to_joint_angles(gt_sincos[t])
        gt_pose3d = joint_angles_to_pose3d(gt_angles, reference_pose=reference_pose)
        gt_poses.append(gt_pose3d)

    # Report errors
    errors = [np.linalg.norm(p - g, axis=1).mean() for p, g in zip(predicted_poses, gt_poses)]
    print(f"    Per-timestep errors: {[f'{e:.4f}' for e in errors]}")
    print(f"    Mean error across trajectory: {np.mean(errors):.4f}")

    return predicted_poses, gt_poses


def run_inference_for_frames(model, action_class, clip_id, frame_indices):
    """Run VLA model inference for specific frames and return predicted 3D poses.

    For each target frame, runs a separate inference starting at that frame
    and extracts the prediction at t=0.

    Returns:
        predicted_poses: list of (17, 3) arrays - one per frame
        gt_poses: list of (17, 3) arrays - ground truth for comparison
    """
    import torch

    device = next(model.parameters()).device
    predicted_poses = []
    gt_poses = []

    # Load raw pose data for ground truth reference bone lengths
    pose_dir = Path("/root/movement/data/kinetics_processed")
    npz_path = pose_dir / action_class / f"{clip_id}.npz"
    raw_pose3d = np.load(npz_path, allow_pickle=True)["pose3d"].astype(np.float32)

    for fidx in frame_indices:
        print(f"  Inference for frame {fidx}...")

        images, instruction, robot_state, gt_sincos = prepare_inference_inputs(
            action_class, clip_id, start_frame=fidx
        )

        robot_state_tensor = torch.from_numpy(robot_state).float().unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model.forward(
                    [images],
                    [instruction],
                    robot_state=robot_state_tensor,
                    compute_loss=False,
                )
            pred_sincos = output["actions"].squeeze(0).cpu().numpy()  # [16, 44]

        # Extract prediction at t=0 (the pose at start_frame)
        pred_angles = sincos_to_joint_angles(pred_sincos[0])  # [22]
        gt_angles = sincos_to_joint_angles(gt_sincos[0])  # [22]

        # Convert to 3D poses via FK
        reference_pose = raw_pose3d[fidx]
        pred_pose3d = joint_angles_to_pose3d(pred_angles, reference_pose=reference_pose)
        gt_pose3d = joint_angles_to_pose3d(gt_angles, reference_pose=reference_pose)

        predicted_poses.append(pred_pose3d)
        gt_poses.append(gt_pose3d)

        error = np.linalg.norm(pred_pose3d - gt_pose3d, axis=1).mean()
        print(f"    Mean joint error: {error:.4f} (camera-space units)")

    return predicted_poses, gt_poses


def render_poses_triptych(poses, ref_scale, labels, suffix, out_dir=None):
    """Render 3 poses as individual images + triptych.

    Args:
        poses: list of 3 (17, 3) arrays
        ref_scale: consistent scale factor
        labels: list of 3 label strings
        suffix: filename suffix (e.g. "gt", "inference")
        out_dir: output directory (default: inference/)
    Returns:
        list of rendered image arrays
    """
    if out_dir is None:
        out_dir = Path(__file__).parent
    out_dir = Path(out_dir)

    # Camera center from first pose
    stand_pose, _ = transform_pose_for_render(poses[0], scale=ref_scale)
    max_z = stand_pose[:, 2].max()
    camera_center_z = max_z / 2

    rendered_images = []
    for i, (pose, label) in enumerate(zip(poses, labels)):
        print(f"  Rendering {label}...")
        pixels, _, _ = render_pose(pose, scale=ref_scale, camera_center_z=camera_center_z)
        rendered_images.append(pixels)

        out_path = out_dir / f"robot_render_{label}_{suffix}.png"
        Image.fromarray(pixels).save(out_path)
        print(f"  Saved: {out_path}")

    # Create triptych
    h, w = rendered_images[0].shape[:2]
    gap = 4
    triptych_w = w * 3 + gap * 2
    triptych = np.full((h, triptych_w, 3), 45, dtype=np.uint8)

    for i, img in enumerate(rendered_images):
        x_offset = i * (w + gap)
        triptych[:, x_offset:x_offset + w] = img

    triptych_path = out_dir / f"robot_render_triptych_{suffix}.png"
    Image.fromarray(triptych).save(triptych_path)
    print(f"Saved triptych: {triptych_path}")

    return rendered_images


def generate_comparison_figure(
    action_class, clip_id, frame_start, gt_pose3d, pred_pose3d, output_path=None,
):
    """Generate a GT vs prediction comparison figure matching figure_A_compare.png format.

    Layout: [start video frame | GT skeleton (green) + predicted skeleton (yellow) on black | end video frame]

    The middle panel overlays the ground truth and model-predicted 3D poses as
    2D-projected skeletons on a black background.

    Args:
        action_class: Action class name
        clip_id: Clip ID
        frame_start: Start frame index (input to model)
        gt_pose3d: Ground truth 3D pose at the target timestep (17, 3)
        pred_pose3d: Model-predicted 3D pose at the target timestep (17, 3)
        output_path: Output file path
    """
    import cv2
    from data_prep.constants import H36M_LR_EDGE_MASK

    pose_dir = Path("/root/movement/data/kinetics_processed")
    video_dir = Path("/root/movement/data/kinetics-dataset/k700-2020")

    # Load pose data for video frame extraction
    npz_path = pose_dir / action_class / f"{clip_id}.npz"
    data = np.load(npz_path, allow_pickle=True)
    bboxes = data["bboxes"].astype(np.float32)
    pose_indices = data["indices"].astype(np.int32)

    # Find video
    video_path = None
    for subdir in ["train", "val", "test"]:
        candidate = video_dir / subdir / action_class / f"{clip_id}.mp4"
        if candidate.exists():
            video_path = str(candidate)
            break
    if video_path is None:
        raise FileNotFoundError(f"Video not found for {action_class}/{clip_id}")

    # Extract video frames (start and end of action window)
    action_horizon = 16
    frame_end = min(frame_start + action_horizon - 1, len(pose_indices) - 1)

    cap = cv2.VideoCapture(video_path)

    def get_frame(frame_idx):
        vid_idx = int(pose_indices[frame_idx])
        cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read video frame {vid_idx}")
        return frame_bgr[:, :, ::-1]  # BGR -> RGB

    frame_start_rgb = get_frame(frame_start)
    frame_end_rgb = get_frame(frame_end)
    cap.release()

    # Crop around bounding box with padding
    def crop_around_bbox(frame, bbox):
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        pad = max(bw, bh) * 0.4
        cx1 = max(0, int(x1 - pad))
        cy1 = max(0, int(y1 - pad * 0.3))
        cx2 = min(w_frame, int(x2 + pad))
        cy2 = min(h_frame, int(y2 + pad * 0.6))
        return frame[cy1:cy2, cx1:cx2]

    start_crop = crop_around_bbox(frame_start_rgb, bboxes[frame_start])
    end_crop = crop_around_bbox(frame_end_rgb, bboxes[frame_end])

    # Project 3D poses to 2D (camera view: X=horizontal, -Y=vertical)
    def prepare_2d_projection(pose):
        pts = pose - pose[0:1]  # Center at hip
        proj = np.zeros((len(pts), 2))
        proj[:, 0] = pts[:, 0]       # X -> horizontal
        proj[:, 1] = -pts[:, 1]      # -Y -> vertical (head up)
        return proj

    proj_gt = prepare_2d_projection(gt_pose3d)
    proj_pred = prepare_2d_projection(pred_pose3d)

    # Normalize both projections to fit in image space, using consistent scale
    all_pts = np.concatenate([proj_gt, proj_pred], axis=0)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    span = (max_xy - min_xy).max()
    if span < 1e-6:
        span = 1.0

    # Target panel size
    panel_h = max(start_crop.shape[0], end_crop.shape[0])
    panel_w = max(start_crop.shape[1], end_crop.shape[1])
    margin = 0.15

    def to_pixel(proj, panel_w, panel_h):
        usable_w = panel_w * (1 - 2 * margin)
        usable_h = panel_h * (1 - 2 * margin)
        scale = min(usable_w, usable_h) / span
        center = (min_xy + max_xy) / 2
        pix = (proj - center) * scale
        pix[:, 0] += panel_w / 2
        pix[:, 1] = panel_h / 2 - pix[:, 1]  # Flip Y for image coords
        return pix.astype(int)

    pix_gt = to_pixel(proj_gt, panel_w, panel_h)
    pix_pred = to_pixel(proj_pred, panel_w, panel_h)

    # Draw skeletons on black background
    black_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    def draw_skeleton_on_panel(img, pix, bone_color_a, bone_color_b, joint_color,
                               bone_width=4, joint_radius=6):
        for i, (a, b) in enumerate(zip(H36M_I, H36M_J)):
            color = bone_color_a if H36M_LR_EDGE_MASK[i] else bone_color_b
            p1 = tuple(pix[a])
            p2 = tuple(pix[b])
            cv2.line(img, p1, p2, color, bone_width, cv2.LINE_AA)
        for j in range(len(pix)):
            cv2.circle(img, tuple(pix[j]), joint_radius, joint_color, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(pix[j]), joint_radius, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    # GT pose: green (RGB)
    draw_skeleton_on_panel(
        black_panel, pix_gt,
        bone_color_a=(0, 210, 0), bone_color_b=(0, 180, 0),
        joint_color=(200, 200, 200), bone_width=5, joint_radius=7,
    )
    # Predicted pose: yellow/gold (RGB)
    draw_skeleton_on_panel(
        black_panel, pix_pred,
        bone_color_a=(255, 220, 0), bone_color_b=(230, 200, 0),
        joint_color=(200, 200, 200), bone_width=5, joint_radius=7,
    )

    # Resize all panels to same height
    target_h = panel_h
    def resize_to_height(img, h):
        aspect = img.shape[1] / img.shape[0]
        new_w = int(h * aspect)
        return cv2.resize(img, (new_w, h))

    start_resized = resize_to_height(start_crop, target_h)
    end_resized = resize_to_height(end_crop, target_h)
    mid_resized = resize_to_height(black_panel, target_h)

    # Make middle panel wider (matching reference figure proportions)
    mid_target_w = int(max(start_resized.shape[1], end_resized.shape[1]) * 1.3)
    mid_canvas = np.zeros((target_h, mid_target_w, 3), dtype=np.uint8)
    x_off = (mid_target_w - mid_resized.shape[1]) // 2
    mid_canvas[:, x_off:x_off + mid_resized.shape[1]] = mid_resized

    sep_width = 2

    # Combine panels
    total_w = start_resized.shape[1] + sep_width + mid_canvas.shape[1] + sep_width + end_resized.shape[1]
    combined = np.zeros((target_h, total_w, 3), dtype=np.uint8)

    x = 0
    combined[:, x:x + start_resized.shape[1]] = start_resized
    x += start_resized.shape[1]
    combined[:, x:x + sep_width] = 180
    x += sep_width
    combined[:, x:x + mid_canvas.shape[1]] = mid_canvas
    x += mid_canvas.shape[1]
    combined[:, x:x + sep_width] = 180
    x += sep_width
    combined[:, x:x + end_resized.shape[1]] = end_resized

    # Save
    if output_path is None:
        output_path = Path(__file__).parent / "gt_vs_pred_comparison.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(output_path)
    print(f"Saved GT vs prediction comparison: {output_path}")

    return combined


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", action="store_true",
                        help="Use VLA model inference instead of training data")
    parser.add_argument("--trajectory", action="store_true",
                        help="Show full predicted trajectory (frames 0, 8, 15 of prediction)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--action-class", type=str, default="squat")
    parser.add_argument("--clip-id", type=str, default="DANpSOYfehc_000000_000010")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="Override start frame for trajectory mode")
    parser.add_argument("--compare", action="store_true",
                        help="Generate GT comparison figure (video start | skeletons | video end)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    action_class = args.action_class
    clip_id = args.clip_id

    print("Loading sample trajectory...")
    pose3d_seq, action_class, clip_id = load_sample_trajectory(
        action_class=action_class,
        clip_id=clip_id,
    )
    num_frames = pose3d_seq.shape[0]

    # Select frames showing standing -> squat -> standing
    frame_indices, ref_scale = select_frames(pose3d_seq)
    labels = ["standing", "squat", "standing"]

    if args.compare:
        # Generate GT vs prediction comparison figure (matching figure_A_compare.png)
        # Layout: [start video | GT(green) + pred(yellow) skeletons on black | end video]
        print("=== GT vs PREDICTION COMPARISON ===")
        import torch

        # Load model and run inference
        model = load_vla_model(checkpoint_path=args.checkpoint, device=args.device)
        start_frame = args.start_frame if args.start_frame is not None else frame_indices[0]

        predicted_poses, gt_poses = run_inference_trajectory(
            model, action_class, clip_id, start_frame=start_frame
        )

        # Use the last timestep (t=15) for maximum motion and to show prediction quality
        target_t = 15
        gt_raw = pose3d_seq[min(start_frame + target_t, len(pose3d_seq) - 1)]
        pred_3d = predicted_poses[target_t]

        print(f"Generating comparison figure (start={start_frame}, target=t+{target_t})...")
        generate_comparison_figure(
            action_class, clip_id,
            frame_start=start_frame,
            gt_pose3d=gt_raw,
            pred_pose3d=pred_3d,
        )

        # Also generate robot triptych from raw GT poses
        poses_to_render = [pose3d_seq[fi] for fi in frame_indices]
        print(f"Rendering robot triptych for frames {frame_indices}...")
        render_poses_triptych(poses_to_render, ref_scale, labels, "gt")

        del model
        torch.cuda.empty_cache()
        print("Done!")
        return

    if args.inference and args.trajectory:
        # Trajectory mode: predict 16 frames from a starting point
        print("=== TRAJECTORY INFERENCE MODE ===")
        model = load_vla_model(checkpoint_path=args.checkpoint, device=args.device)

        start = args.start_frame if args.start_frame is not None else frame_indices[0]
        predicted_poses, gt_poses = run_inference_trajectory(
            model, action_class, clip_id, start_frame=start
        )

        # Pick 3 evenly-spaced frames from the 16-step trajectory
        traj_indices = [0, 7, 15]
        poses_to_render = [predicted_poses[i] for i in traj_indices]
        gt_for_render = [gt_poses[i] for i in traj_indices]
        labels = [f"t{i}" for i in traj_indices]
        suffix = "traj_inference"

        # Also render GT trajectory for comparison
        print("Rendering predicted trajectory...")
        render_poses_triptych(poses_to_render, ref_scale, labels, suffix)
        print("Rendering GT trajectory...")
        render_poses_triptych(gt_for_render, ref_scale, labels, "traj_gt")

        import torch
        del model
        torch.cuda.empty_cache()

    elif args.inference:
        # Per-frame inference mode
        print("=== INFERENCE MODE ===")
        model = load_vla_model(checkpoint_path=args.checkpoint, device=args.device)
        predicted_poses, gt_poses = run_inference_for_frames(
            model, action_class, clip_id, frame_indices
        )
        poses_to_render = predicted_poses
        suffix = "inference"

        print(f"Rendering frames {frame_indices} from '{action_class}' ({suffix})...")
        render_poses_triptych(poses_to_render, ref_scale, labels, suffix)

        # Also render GT for comparison
        gt_through_fk = gt_poses
        print("Rendering GT (through FK) for comparison...")
        render_poses_triptych(gt_through_fk, ref_scale, labels, "inference_gt")

        import torch
        del model
        torch.cuda.empty_cache()
    else:
        # Ground truth mode (raw training data, no FK round-trip)
        poses_to_render = [pose3d_seq[fi] for fi in frame_indices]
        suffix = "gt"
        print(f"Rendering frames {frame_indices} from '{action_class}' ({suffix})...")
        render_poses_triptych(poses_to_render, ref_scale, labels, suffix)

    print("Done!")


if __name__ == "__main__":
    main()
