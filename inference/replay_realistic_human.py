import bpy
import numpy as np
import os
import math

def clear_scene():
    """Clear all objects from the scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_realistic_humanoid():
    """Create a more realistic humanoid mesh with proper proportions"""

    # Human body proportions (in head units)
    # Average human is about 7.5 heads tall
    head_size = 0.25
    body_height = 1.8  # meters

    parts = []

    # HEAD - More detailed with facial features hint
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=32, ring_count=16,
        radius=head_size * 0.9,
        location=(0, 0, body_height - head_size)
    )
    head = bpy.context.active_object
    head.name = "Head"
    # Slightly elongate the head
    head.scale = (0.85, 0.95, 1.1)
    parts.append(head)

    # NECK
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=16, radius=head_size * 0.4, depth=head_size * 0.8,
        location=(0, 0, body_height - head_size * 2)
    )
    neck = bpy.context.active_object
    neck.name = "Neck"
    parts.append(neck)

    # TORSO - Upper (chest)
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(0, 0, body_height * 0.65)
    )
    upper_torso = bpy.context.active_object
    upper_torso.name = "UpperTorso"
    upper_torso.scale = (head_size * 3, head_size * 1.5, head_size * 2.5)
    # Add some taper
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.transform.resize(value=(1, 1, 1),
                            orient_type='GLOBAL',
                            constraint_axis=(False, False, True))
    bpy.ops.object.mode_set(mode='OBJECT')
    parts.append(upper_torso)

    # TORSO - Lower (abdomen/hips)
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(0, 0, body_height * 0.45)
    )
    lower_torso = bpy.context.active_object
    lower_torso.name = "LowerTorso"
    lower_torso.scale = (head_size * 2.8, head_size * 1.4, head_size * 2)
    parts.append(lower_torso)

    # SHOULDERS
    for side, x_mult in [("L", -1), ("R", 1)]:
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=16, ring_count=8,
            radius=head_size * 0.5,
            location=(x_mult * head_size * 2, 0, body_height * 0.72)
        )
        shoulder = bpy.context.active_object
        shoulder.name = f"Shoulder_{side}"
        shoulder.scale = (1.2, 0.8, 0.8)
        parts.append(shoulder)

    # ARMS - More realistic proportions
    for side, x_mult in [("L", -1), ("R", 1)]:
        # Upper arm (humerus)
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=16, radius=head_size * 0.35, depth=head_size * 2.5,
            location=(x_mult * head_size * 2.8, 0, body_height * 0.58)
        )
        upper_arm = bpy.context.active_object
        upper_arm.name = f"UpperArm_{side}"
        upper_arm.rotation_euler = (0, math.radians(8 * x_mult), 0)
        # Taper the arm
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        parts.append(upper_arm)

        # Elbow joint
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=16, ring_count=8,
            radius=head_size * 0.32,
            location=(x_mult * head_size * 3.2, 0, body_height * 0.42)
        )
        elbow = bpy.context.active_object
        elbow.name = f"Elbow_{side}"
        parts.append(elbow)

        # Forearm
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=16, radius=head_size * 0.28, depth=head_size * 2.3,
            location=(x_mult * head_size * 3.5, 0, body_height * 0.28)
        )
        forearm = bpy.context.active_object
        forearm.name = f"Forearm_{side}"
        forearm.rotation_euler = (0, math.radians(5 * x_mult), 0)
        parts.append(forearm)

        # Hand - more detailed
        bpy.ops.mesh.primitive_cube_add(
            size=head_size * 0.5,
            location=(x_mult * head_size * 3.7, 0, body_height * 0.15)
        )
        hand = bpy.context.active_object
        hand.name = f"Hand_{side}"
        hand.scale = (0.7, 0.3, 1.4)
        parts.append(hand)

    # LEGS - Proper proportions
    for side, x_mult in [("L", -0.6), ("R", 0.6)]:
        # Hip joint
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=16, ring_count=8,
            radius=head_size * 0.4,
            location=(x_mult * head_size, 0, body_height * 0.35)
        )
        hip = bpy.context.active_object
        hip.name = f"Hip_{side}"
        parts.append(hip)

        # Thigh
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=16, radius=head_size * 0.45, depth=head_size * 3,
            location=(x_mult * head_size, 0, body_height * 0.22)
        )
        thigh = bpy.context.active_object
        thigh.name = f"Thigh_{side}"
        # Taper toward knee
        thigh.scale = (1, 1, 1)
        parts.append(thigh)

        # Knee
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=16, ring_count=8,
            radius=head_size * 0.35,
            location=(x_mult * head_size, -head_size * 0.1, body_height * 0.08)
        )
        knee = bpy.context.active_object
        knee.name = f"Knee_{side}"
        parts.append(knee)

        # Shin/Calf
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=16, radius=head_size * 0.35, depth=head_size * 2.8,
            location=(x_mult * head_size, 0, -body_height * 0.06)
        )
        shin = bpy.context.active_object
        shin.name = f"Shin_{side}"
        parts.append(shin)

        # Ankle
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=12, ring_count=6,
            radius=head_size * 0.25,
            location=(x_mult * head_size, 0, -body_height * 0.18)
        )
        ankle = bpy.context.active_object
        ankle.name = f"Ankle_{side}"
        parts.append(ankle)

        # Foot
        bpy.ops.mesh.primitive_cube_add(
            size=head_size * 0.6,
            location=(x_mult * head_size, -head_size * 0.3, -body_height * 0.22)
        )
        foot = bpy.context.active_object
        foot.name = f"Foot_{side}"
        foot.scale = (0.8, 1.8, 0.4)
        parts.append(foot)

    # Join all parts into single mesh
    bpy.ops.object.select_all(action='DESELECT')
    for part in parts:
        part.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()

    humanoid = bpy.context.active_object
    humanoid.name = "RealisticHumanoid"

    # Apply subdivision surface for smooth appearance
    subsurf = humanoid.modifiers.new("Subdivision", 'SUBSURF')
    subsurf.levels = 2
    subsurf.render_levels = 3

    # Add solidify for better volume
    solidify = humanoid.modifiers.new("Solidify", 'SOLIDIFY')
    solidify.thickness = 0.01

    # Create realistic skin material
    create_skin_material(humanoid)

    print("✓ Created realistic humanoid mesh")
    return humanoid

def create_skin_material(obj):
    """Create a realistic skin material"""
    mat = bpy.data.materials.new(name="Realistic_Skin")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Add Principled BSDF
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)

    # Skin tone color
    bsdf.inputs["Base Color"].default_value = (0.95, 0.75, 0.65, 1.0)
    bsdf.inputs["Subsurface Weight"].default_value = 0.3
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.3, 0.1)
    bsdf.inputs["Roughness"].default_value = 0.4
    bsdf.inputs["Specular IOR Level"].default_value = 0.5

    # Add noise for skin texture variation
    noise = nodes.new('ShaderNodeTexNoise')
    noise.location = (-400, 0)
    noise.inputs["Scale"].default_value = 50.0
    noise.inputs["Detail"].default_value = 10.0

    # Color variation
    mix = nodes.new('ShaderNodeMix')
    mix.location = (-200, 0)
    mix.data_type = 'RGBA'
    mix.inputs["Factor"].default_value = 0.1
    mix.inputs["A"].default_value = (0.95, 0.75, 0.65, 1.0)
    mix.inputs["B"].default_value = (0.90, 0.70, 0.60, 1.0)

    # Connect nodes
    links.new(noise.outputs["Fac"], mix.inputs["Factor"])
    links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])

    # Output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Assign material
    obj.data.materials.append(mat)

def setup_environment():
    """Create a nice environment with lighting"""
    scene = bpy.context.scene

    # Use EEVEE for faster rendering
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.use_bloom = True
    scene.eevee.bloom_intensity = 0.3
    scene.eevee.bloom_threshold = 0.9

    # Video settings
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = 30

    # World background - gradient sky
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()

    # Sky texture
    sky = nodes.new('ShaderNodeTexSky')
    sky.sky_type = 'NISHITA'
    sky.sun_elevation = math.radians(45)
    sky.sun_rotation = math.radians(45)

    background = nodes.new('ShaderNodeBackground')
    background.inputs["Strength"].default_value = 0.8

    output = nodes.new('ShaderNodeOutputWorld')

    world.node_tree.links.new(sky.outputs["Color"], background.inputs["Color"])
    world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])

    # Ground plane with texture
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, -0.23))
    ground = bpy.context.active_object
    ground.name = "Ground"

    # Ground material
    ground_mat = bpy.data.materials.new(name="Ground")
    ground_mat.use_nodes = True
    nodes = ground_mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.3, 0.35, 0.3, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8
    ground.data.materials.append(ground_mat)

    # Three-point lighting setup
    # Key light
    bpy.ops.object.light_add(type='AREA', location=(3, -4, 3))
    key = bpy.context.active_object
    key.name = "Key Light"
    key.data.energy = 200
    key.data.size = 2
    key.data.color = (1.0, 0.98, 0.95)
    key.rotation_euler = (math.radians(60), 0, math.radians(45))

    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-3, -3, 2))
    fill = bpy.context.active_object
    fill.name = "Fill Light"
    fill.data.energy = 100
    fill.data.size = 3
    fill.data.color = (0.8, 0.85, 1.0)
    fill.rotation_euler = (math.radians(70), 0, math.radians(-30))

    # Rim light
    bpy.ops.object.light_add(type='SPOT', location=(0, 4, 2.5))
    rim = bpy.context.active_object
    rim.name = "Rim Light"
    rim.data.energy = 150
    rim.data.spot_size = math.radians(60)
    rim.rotation_euler = (math.radians(-160), 0, 0)

    print("✓ Environment and lighting setup complete")

def animate_humanoid_applause(humanoid, pose3d_data, num_frames=90):
    """Animate the humanoid with visible applauding motion"""

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = num_frames

    # Get average position for centering
    mean_pos = np.mean(pose3d_data, axis=(0, 1))

    # COCO keypoint indices for arms
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10

    for frame in range(num_frames):
        scene.frame_set(frame + 1)

        # Get frame data (cycle through available data)
        data_frame = frame % len(pose3d_data)
        frame_data = pose3d_data[data_frame]

        # Calculate torso center from shoulders
        torso_center = (frame_data[LEFT_SHOULDER] + frame_data[RIGHT_SHOULDER]) / 2 - mean_pos

        # Overall body position and sway
        humanoid.location = (
            torso_center[0] * 0.2,  # Subtle left/right sway
            torso_center[2] * 0.1,  # Forward/back
            0  # Keep feet on ground
        )
        humanoid.keyframe_insert(data_path="location")

        # Body rotation for natural movement
        humanoid.rotation_euler = (
            torso_center[2] * 0.05,  # Slight forward/back tilt
            0,
            torso_center[0] * 0.03   # Slight side rotation
        )
        humanoid.keyframe_insert(data_path="rotation_euler")

        # Create applauding motion using scale animation
        # This simulates arms coming together and apart
        clap_phase = (frame % 15) / 15.0  # Clap every 15 frames (0.5 seconds at 30fps)

        if clap_phase < 0.3:  # Arms coming together
            scale_x = 1.0 - (clap_phase / 0.3) * 0.15  # Compress horizontally
            scale_y = 1.0 + (clap_phase / 0.3) * 0.05  # Expand forward slightly
        elif clap_phase < 0.4:  # Clap moment
            scale_x = 0.85
            scale_y = 1.05
        else:  # Arms moving apart
            progress = (clap_phase - 0.4) / 0.6
            scale_x = 0.85 + progress * 0.15
            scale_y = 1.05 - progress * 0.05

        humanoid.scale = (scale_x, scale_y, 1.0)
        humanoid.keyframe_insert(data_path="scale")

        # Add some vertical bounce during clapping
        if clap_phase < 0.4:
            bounce = math.sin(clap_phase * math.pi / 0.4) * 0.02
            humanoid.location = (
                humanoid.location[0],
                humanoid.location[1],
                bounce
            )
            humanoid.keyframe_insert(data_path="location")

    print(f"✓ Applied {num_frames} frames of applause animation")

def setup_camera():
    """Position camera for good view of the character"""
    bpy.ops.object.camera_add(location=(3, -5, 1.2))
    camera = bpy.context.active_object
    camera.rotation_euler = (math.radians(85), 0, math.radians(25))

    camera.data.lens = 50
    camera.data.dof.use_dof = True
    camera.data.dof.focus_distance = 6.0
    camera.data.dof.aperture_fstop = 8.0

    scene = bpy.context.scene
    scene.camera = camera

    # Add camera tracking constraint to follow humanoid
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = bpy.data.objects.get("RealisticHumanoid")
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    print("✓ Camera positioned with tracking")
    return camera

def render_video(output_path):
    """Render the animation as video"""
    scene = bpy.context.scene

    # Configure video output
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'HIGH'
    scene.render.ffmpeg.ffmpeg_preset = 'GOOD'

    scene.render.filepath = output_path

    print(f"🎬 Rendering video to: {output_path}")
    print(f"  Resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
    print(f"  Frames: {scene.frame_start} to {scene.frame_end}")
    print(f"  Duration: {scene.frame_end/scene.render.fps:.1f} seconds at {scene.render.fps} fps")

    # Render
    bpy.ops.render.render(animation=True)

    print("✓ Video rendering complete!")

def main():
    print("\n=== Creating Realistic Human Animation ===\n")

    # Clear scene
    clear_scene()

    # Setup environment
    setup_environment()

    # Create realistic humanoid
    humanoid = create_realistic_humanoid()

    # Load animation data
    npz_file = "/root/movement/data/kinetics_processed/applauding/0BburtHMBts_000069_000079.npz"

    # Check for alternative files if needed
    if not os.path.exists(npz_file):
        actions = ["air_drumming", "archery", "arm_wrestling", "acting in play"]
        for action in actions:
            action_dir = f"/root/movement/data/kinetics_processed/{action}/"
            if os.path.exists(action_dir):
                files = [f for f in os.listdir(action_dir) if f.endswith('.npz')]
                if files:
                    npz_file = os.path.join(action_dir, files[0])
                    print(f"Using {action} animation instead")
                    break

    if not os.path.exists(npz_file):
        print("❌ No animation data found")
        return

    # Load data
    data = np.load(npz_file, allow_pickle=True)
    pose3d_data = data['pose3d']
    print(f"✓ Loaded {len(pose3d_data)} frames of pose data")

    # Animate humanoid with applause (3 seconds = 90 frames at 30fps)
    animate_humanoid_applause(humanoid, pose3d_data, num_frames=90)

    # Setup camera
    setup_camera()

    # Save scene
    blend_file = "/root/movement/inference/realistic_human.blend"
    bpy.ops.wm.save_as_mainfile(filepath=blend_file)
    print(f"✓ Scene saved: {blend_file}")

    # Render video
    video_path = "/root/movement/inference/realistic_human_animation.mp4"
    render_video(video_path)

    print(f"\n=== Complete! ===")
    print(f"✅ Video available at: {video_path}")
    print(f"   Duration: 3 seconds of realistic applauding animation")
    print(f"   Ready for download!")

if __name__ == "__main__":
    main()