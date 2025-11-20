import bpy
import numpy as np
import os
import math

def clear_scene():
    """Clear all objects from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def setup_scene():
    """Set up the basic scene with lighting and background"""
    scene = bpy.context.scene

    # Use EEVEE for faster rendering (no denoising issues)
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.use_bloom = False
    scene.eevee.use_ssr = False

    # Set render resolution for video
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100

    # Set frame rate
    scene.render.fps = 25
    scene.render.fps_base = 1.0

    # Add background plane
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"

    # Create simple material for ground
    mat = bpy.data.materials.new(name="Ground_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.3, 0.4, 0.5, 1.0)
    ground.data.materials.append(mat)

    # Add sun light
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 2.0

    print("✓ Scene setup complete")

def create_animated_spheres(pose3d_data, start_frame=1):
    """Create animated spheres for each joint"""
    scene = bpy.context.scene

    # Joint colors for better visualization
    joint_colors = [
        (1, 0, 0, 1),  # Red for head joints
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
        (0, 1, 0, 1),  # Green for arm joints
        (0, 1, 0, 1),
        (0, 0.5, 0, 1),
        (0, 0.5, 0, 1),
        (0, 0.3, 0, 1),
        (0, 0.3, 0, 1),
        (0, 0, 1, 1),  # Blue for leg joints
        (0, 0, 1, 1),
        (0, 0, 0.5, 1),
        (0, 0, 0.5, 1),
        (0, 0, 0.3, 1),
        (0, 0, 0.3, 1),
    ]

    spheres = []

    # Create spheres for each joint
    num_joints = min(pose3d_data.shape[1], len(joint_colors))
    for i in range(num_joints):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.03, location=(0, 0, 0))
        sphere = bpy.context.active_object
        sphere.name = f"Joint_{i:02d}"

        # Create material
        mat = bpy.data.materials.new(name=f"Joint_Mat_{i:02d}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf and i < len(joint_colors):
            bsdf.inputs["Base Color"].default_value = joint_colors[i]
            bsdf.inputs["Metallic"].default_value = 0.5
            bsdf.inputs["Roughness"].default_value = 0.2

        sphere.data.materials.append(mat)
        spheres.append(sphere)

    # Add connections between joints (bones)
    connections = [
        # Head connections
        (0, 1), (0, 2), (1, 3), (2, 4),
        # Torso to arms
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
        # Torso to legs
        (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
    ]

    cylinders = []
    for conn in connections:
        if conn[0] < num_joints and conn[1] < num_joints:
            bpy.ops.mesh.primitive_cylinder_add(radius=0.01, depth=1, location=(0, 0, 0))
            cylinder = bpy.context.active_object
            cylinder.name = f"Bone_{conn[0]:02d}_{conn[1]:02d}"

            # Dark material for bones
            mat = bpy.data.materials.new(name=f"Bone_Mat_{conn[0]}_{conn[1]}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            bsdf = nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (0.2, 0.2, 0.2, 1)

            cylinder.data.materials.append(mat)
            cylinders.append((cylinder, conn))

    # Apply animation
    num_frames = len(pose3d_data)
    scene.frame_start = start_frame
    scene.frame_end = start_frame + num_frames - 1

    # Compute center offset to center the animation
    mean_pos = np.mean(pose3d_data, axis=(0, 1))

    for frame_idx, pose_frame in enumerate(pose3d_data):
        scene.frame_set(start_frame + frame_idx)

        # Animate spheres
        for joint_idx in range(num_joints):
            if joint_idx < len(spheres):
                sphere = spheres[joint_idx]
                # Center and scale the position
                pos = pose_frame[joint_idx] - mean_pos
                pos = pos * 2  # Scale up
                pos[1] = -pos[2]  # Swap Y and Z for better orientation
                pos[2] = pos[1] + 1.5  # Lift up
                sphere.location = pos
                sphere.keyframe_insert(data_path="location")

        # Update cylinder positions and orientations
        for cylinder, (start_joint, end_joint) in cylinders:
            if start_joint < num_joints and end_joint < num_joints:
                # Get joint positions
                start_pos = pose_frame[start_joint] - mean_pos
                end_pos = pose_frame[end_joint] - mean_pos

                # Scale and adjust positions
                start_pos = start_pos * 2
                end_pos = end_pos * 2
                start_pos[1] = -start_pos[2]
                start_pos[2] = start_pos[1] + 1.5
                end_pos[1] = -end_pos[2]
                end_pos[2] = end_pos[1] + 1.5

                # Position cylinder at midpoint
                mid_pos = (start_pos + end_pos) / 2
                cylinder.location = mid_pos
                cylinder.keyframe_insert(data_path="location")

                # Calculate rotation
                direction = end_pos - start_pos
                length = np.linalg.norm(direction)
                if length > 0.001:
                    cylinder.scale[2] = length / 2
                    cylinder.keyframe_insert(data_path="scale")

                    # Calculate rotation to align with direction
                    direction = direction / length
                    # Default cylinder points up (Z), we need to rotate it
                    up = np.array([0, 0, 1])
                    angle = np.arccos(np.clip(np.dot(up, direction), -1, 1))
                    axis = np.cross(up, direction)
                    if np.linalg.norm(axis) > 0.001:
                        axis = axis / np.linalg.norm(axis)
                        cylinder.rotation_mode = 'AXIS_ANGLE'
                        cylinder.rotation_axis_angle = (angle, *axis)
                        cylinder.keyframe_insert(data_path="rotation_axis_angle")

    print(f"✓ Created {len(spheres)} animated joint spheres with {len(cylinders)} connections")
    return spheres

def setup_camera():
    """Set up camera for the animation"""
    bpy.ops.object.camera_add(location=(6, -6, 3))
    camera = bpy.context.active_object
    camera.rotation_euler = (math.radians(70), 0, math.radians(45))

    # Set as active camera
    bpy.context.scene.camera = camera

    print("✓ Camera positioned")
    return camera

def render_video(output_path):
    """Render the animation as a video"""
    scene = bpy.context.scene

    # Set output format to video
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'

    # Set quality
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scene.render.ffmpeg.ffmpeg_preset = 'GOOD'

    # Set output path
    scene.render.filepath = output_path

    print(f"🎬 Rendering video to: {output_path}")
    print(f"  Frames: {scene.frame_start} to {scene.frame_end}")

    # Render animation
    bpy.ops.render.render(animation=True)

    print(f"✓ Video rendered successfully!")

def main():
    """Main function to create and render the animation"""
    print("\n=== Starting Simple Animation Replay ===\n")

    # Clear scene
    clear_scene()

    # Setup scene
    setup_scene()

    # Load keyframe data
    npz_file = "/root/movement/data/kinetics_processed/applauding/0BburtHMBts_000069_000079.npz"

    if not os.path.exists(npz_file):
        # Try another action if applauding doesn't exist
        alt_actions = ["air_drumming", "archery", "arm_wrestling"]
        for action in alt_actions:
            alt_path = f"/root/movement/data/kinetics_processed/{action}/"
            if os.path.exists(alt_path):
                files = os.listdir(alt_path)
                if files:
                    npz_file = os.path.join(alt_path, files[0])
                    print(f"Using alternative file: {npz_file}")
                    break

    if not os.path.exists(npz_file):
        print(f"❌ Error: No keyframe files found")
        return

    # Load the data
    data = np.load(npz_file, allow_pickle=True)
    pose3d_data = data['pose3d']

    print(f"✓ Loaded {len(pose3d_data)} frames from: {os.path.basename(npz_file)}")
    print(f"  Action: {npz_file.split('/')[-2]}")

    # Create animated spheres for joints
    create_animated_spheres(pose3d_data)

    # Setup camera
    setup_camera()

    # Save the scene
    blend_file = "/root/movement/inference/animated_skeleton.blend"
    bpy.ops.wm.save_as_mainfile(filepath=blend_file)
    print(f"✓ Scene saved to: {blend_file}")

    # Render video
    video_output = "/root/movement/inference/skeleton_animation.mp4"
    render_video(video_output)

    print(f"\n=== Animation Complete! ===")
    print(f"✅ Video saved to: {video_output}")
    print(f"   You can download this video file")

if __name__ == "__main__":
    main()