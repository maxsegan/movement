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

    # Set render engine to EEVEE for faster video rendering
    scene.render.engine = 'BLENDER_EEVEE'

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
        bsdf.inputs["Base Color"].default_value = (0.2, 0.3, 0.4, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.9
    ground.data.materials.append(mat)

    # Add backdrop
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 10, 10))
    backdrop = bpy.context.active_object
    backdrop.name = "Backdrop"
    backdrop.rotation_euler = (math.radians(-90), 0, 0)
    backdrop.data.materials.append(mat)

    # Add sun light
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 1.5
    sun.rotation_euler = (math.radians(-45), math.radians(-45), 0)

    # Add fill light
    bpy.ops.object.light_add(type='AREA', location=(-5, -5, 5))
    fill = bpy.context.active_object
    fill.name = "Fill Light"
    fill.data.energy = 50
    fill.data.size = 5

    print("✓ Scene setup complete")

def create_mblab_character():
    """Create an MB-Lab character"""
    scn = bpy.context.scene

    # Set character type
    scn.mblab_character_name = 'f_ca01'  # Female caucasian
    scn.mblab_use_ik = True  # Enable IK for better animation

    # Create the character
    bpy.ops.mbast.init_character()

    # Get the character object
    character = None
    for obj in bpy.context.scene.objects:
        if 'mblab' in obj.name.lower() or 'f_ca01' in obj.name.lower():
            character = obj
            break

    if character:
        character.location = (0, 0, 0)
        print(f"✓ MB-Lab character created: {character.name}")
    else:
        print("⚠ Warning: Could not find MB-Lab character")

    return character

def create_simple_armature():
    """Create a simple armature for animation if MB-Lab character not available"""
    bpy.ops.object.armature_add(location=(0, 0, 1))
    armature = bpy.context.active_object
    armature.name = "Simple_Armature"

    # Enter edit mode to add bones
    bpy.ops.object.mode_set(mode='EDIT')

    # Create a simple bone chain
    edit_bones = armature.data.edit_bones

    # Add basic skeleton bones (simplified 17-joint structure)
    bone_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    bones = {}
    for i, name in enumerate(bone_names):
        bone = edit_bones.new(name)
        bone.head = (0, 0, 1.5 - i * 0.1)
        bone.tail = (0, 0, 1.4 - i * 0.1)
        bones[name] = bone

    bpy.ops.object.mode_set(mode='OBJECT')

    print("✓ Simple armature created")
    return armature

def load_keyframe_data(npz_path):
    """Load keyframe data from npz file"""
    data = np.load(npz_path, allow_pickle=True)

    pose3d = data['pose3d']  # Shape: (num_frames, 17, 3)
    indices = data['indices'] if 'indices' in data else np.arange(len(pose3d))

    print(f"✓ Loaded {len(pose3d)} frames of animation data")
    print(f"  Pose shape: {pose3d.shape}")

    return pose3d, indices

def apply_animation_to_character(character, pose3d_data, start_frame=1):
    """Apply pose data to character as keyframes"""
    scene = bpy.context.scene

    # Set animation length
    num_frames = len(pose3d_data)
    scene.frame_start = start_frame
    scene.frame_end = start_frame + num_frames - 1

    # Try to find the armature
    armature = None
    if character and character.type == 'MESH':
        # Look for armature modifier
        for modifier in character.modifiers:
            if modifier.type == 'ARMATURE':
                armature = modifier.object
                break
    elif character and character.type == 'ARMATURE':
        armature = character

    if not armature:
        print("⚠ No armature found, creating animated empties instead")
        return create_animated_empties(pose3d_data, start_frame)

    # Apply animation to armature
    print(f"✓ Applying animation to armature: {armature.name}")

    # Select armature and enter pose mode
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    # Get pose bones
    pose_bones = armature.pose.bones

    # Simplified bone mapping (adjust based on your rig)
    bone_mapping = {
        0: 'head',      # nose
        5: 'shoulder_L', # left_shoulder
        6: 'shoulder_R', # right_shoulder
        7: 'upperarm_L', # left_elbow
        8: 'upperarm_R', # right_elbow
        9: 'forearm_L',  # left_wrist
        10: 'forearm_R', # right_wrist
        11: 'thigh_L',   # left_hip
        12: 'thigh_R',   # right_hip
        13: 'shin_L',    # left_knee
        14: 'shin_R',    # right_knee
        15: 'foot_L',    # left_ankle
        16: 'foot_R',    # right_ankle
    }

    # Apply keyframes
    for frame_idx, pose_frame in enumerate(pose3d_data):
        scene.frame_set(start_frame + frame_idx)

        for joint_idx, joint_pos in enumerate(pose_frame):
            if joint_idx in bone_mapping:
                bone_name = bone_mapping[joint_idx]
                if bone_name in pose_bones:
                    bone = pose_bones[bone_name]
                    # Apply position (scaled and adjusted)
                    bone.location = joint_pos * 0.1  # Scale down
                    bone.keyframe_insert(data_path="location")

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"✓ Animation applied: {num_frames} frames")

def create_animated_empties(pose3d_data, start_frame=1):
    """Create empty objects animated with pose data (fallback method)"""
    scene = bpy.context.scene

    # Joint names for visualization
    joint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    empties = []

    # Create empties for each joint
    for i, name in enumerate(joint_names[:pose3d_data.shape[1]]):
        bpy.ops.object.empty_add(type='SPHERE', location=(0, 0, 0))
        empty = bpy.context.active_object
        empty.name = f"Joint_{name}"
        empty.empty_display_size = 0.05
        empties.append(empty)

    # Apply animation
    num_frames = len(pose3d_data)
    scene.frame_start = start_frame
    scene.frame_end = start_frame + num_frames - 1

    for frame_idx, pose_frame in enumerate(pose3d_data):
        scene.frame_set(start_frame + frame_idx)

        for joint_idx, joint_pos in enumerate(pose_frame):
            if joint_idx < len(empties):
                empty = empties[joint_idx]
                # Center the animation around origin and scale
                empty.location = joint_pos + (0, -2, 1.5)  # Offset for visibility
                empty.keyframe_insert(data_path="location")

    print(f"✓ Created {len(empties)} animated empties")
    return empties

def setup_camera():
    """Set up camera for the animation"""
    bpy.ops.object.camera_add(location=(5, -8, 2))
    camera = bpy.context.active_object
    camera.rotation_euler = (math.radians(75), 0, math.radians(30))

    # Set as active camera
    bpy.context.scene.camera = camera

    # Set lens
    camera.data.lens = 35

    print("✓ Camera positioned")
    return camera

def render_video(output_path):
    """Render the animation as a video"""
    scene = bpy.context.scene

    # Set output format to video
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'HIGH'

    # Set output path
    scene.render.filepath = output_path

    print(f"🎬 Rendering video to: {output_path}")
    print(f"  Frames: {scene.frame_start} to {scene.frame_end}")

    # Render animation
    bpy.ops.render.render(animation=True)

    print(f"✓ Video rendered successfully!")

def main():
    """Main function to create and render the animation"""
    print("\n=== Starting Animation Replay ===\n")

    # Clear scene
    clear_scene()

    # Setup scene
    setup_scene()

    # Try to create MB-Lab character
    try:
        character = create_mblab_character()
    except Exception as e:
        print(f"⚠ Could not create MB-Lab character: {e}")
        print("  Creating simple armature instead...")
        character = create_simple_armature()

    # Load keyframe data
    npz_file = "/root/movement/data/kinetics_processed/applauding/0BburtHMBts_000069_000079.npz"

    if not os.path.exists(npz_file):
        print(f"❌ Error: Keyframe file not found: {npz_file}")
        return

    pose3d_data, indices = load_keyframe_data(npz_file)

    # Apply animation
    apply_animation_to_character(character, pose3d_data)

    # Setup camera
    setup_camera()

    # Save the scene
    blend_file = "/root/movement/inference/animated_scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=blend_file)
    print(f"✓ Scene saved to: {blend_file}")

    # Render video
    video_output = "/root/movement/inference/animation_replay.mp4"
    render_video(video_output)

    print(f"\n=== Animation Complete! ===")
    print(f"Video saved to: {video_output}")
    print(f"You can download the video from that location")

if __name__ == "__main__":
    main()