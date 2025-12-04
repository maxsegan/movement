#!/usr/bin/env python3
"""Render Kinetics/MotionAGFormer poses in a constructed Blender scene.

This script must be executed from Blender Python, e.g.:

    blender --background --python movement/inference/render_trace.py -- \\
        --trace movement/data/kinetics_processed/abseiling/035LtPeUFTE_000085_000095.npz \\
        --output-dir movement/inference/renders

By default an MB-Lab character is spawned and bound to an animated proxy mesh via
Surface Deform, giving a textured human avatar that follows the supplied pose trace.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

try:
    import bpy  # type: ignore
    import addon_utils  # type: ignore
    from mathutils import Vector, Quaternion
except ImportError as exc:  # pragma: no cover - guarded by Blender runtime
    raise SystemExit("This script has to run inside Blender. Use `blender --python ... --`.") from exc


JOINT_NAMES = [
    'Hip', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle',
    'Spine', 'Neck', 'Head', 'HeadTop',
    'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'
]
H36M_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16)
]
BONE_JOINT_MAP = {
    'pelvis': (0, 7),
    'spine01': (7, 8),
    'spine02': (8, 9),
    'spine03': (9, 10),
    'neck': (8, 9),
    'head': (9, 10),
    'thigh_L': (4, 5),
    'calf_L': (5, 6),
    'thigh_R': (1, 2),
    'calf_R': (2, 3),
    'clavicle_L': (8, 11),
    'upperarm_L': (11, 12),
    'lowerarm_L': (12, 13),
    'clavicle_R': (8, 14),
    'upperarm_R': (14, 15),
    'lowerarm_R': (15, 16),
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a kinetics_processed trace inside Blender")
    parser.add_argument("--trace", required=True, help="Path to the .npz trace file")
    parser.add_argument("--output-dir", default="movement/inference/renders",
                        help="Directory where the mp4 will be written")
    parser.add_argument("--scene-style", choices=["indoor", "outdoor", "studio"], default="indoor",
                        help="Lighting/look preset")
    parser.add_argument("--background-image", default=None,
                        help="Optional HDRI/texture to drive the world background and backplate")
    parser.add_argument("--engine", choices=["CYCLES", "BLENDER_EEVEE"], default="BLENDER_EEVEE",
                        help="Render engine")
    parser.add_argument("--samples", type=int, default=64, help="Samples per pixel (Cycles/Eevee TAA)")
    parser.add_argument("--resolution", default="1920x1080",
                        help="Resolution as WIDTHxHEIGHT (default: 1920x1080)")
    parser.add_argument("--frame-start", type=int, default=1, help="Timeline start frame")
    parser.add_argument("--limit", type=int, default=None, help="Optional frame limit for short previews")
    parser.add_argument("--scale", type=float, default=1.8, help="Global scale multiplier for the skeleton")
    parser.add_argument("--axis-mode", choices=[
        "x_negz_y",    # X = x,   Y = -z, Z = y   (default)
        "x_z_y",       # X = x,   Y =  z,  Z = y
        "x_z_negy",    # X = x,   Y =  z,  Z = -y
        "x_negy_z",    # X = x,   Y = -y, Z = z
        "negx_negz_y", # X = -x,  Y = -z, Z = y  (mirror LR)
        "x_negz_negy", # X = x,   Y = -z, Z = -y (flip up/down)
        "y_negz_x",    # X = y,   Y = -z, Z = x  (use x as up)
        "y_z_x",       # X = y,   Y =  z,  Z = x
        "y_negz_negx", # X = y,   Y = -z, Z = -x
        "y_z_negx",    # X = y,   Y =  z,  Z = -x
    ], default="x_negz_y",
        help="Mapping from MotionAGFormer (x,y,z) to Blender (X,Y,Z).")
    parser.add_argument("--avatar-style", choices=["mblab", "procedural"], default="mblab",
                        help="Choose MB-Lab skinned human or the procedural mannequin")
    parser.add_argument("--mblab-character", default="f_ca01",
                        help="Character identifier from MB-Lab (e.g., f_ca01, m_la01)")
    parser.add_argument("--clothing-style", choices=["none", "procedural"], default="procedural",
                        help="Add a simple cloth overlay to the MB-Lab mesh")
    parser.add_argument("--bbox-scale", type=float, default=6.0,
                        help="Horizontal/vertical meters corresponding to full image width/height when inferring translation")
    parser.add_argument("--exposure", type=float, default=0.0, help="Color management exposure adjustment")
    parser.add_argument("--preview", action="store_true",
                        help="Faster preview: Eevee, 720p, <=32 samples, and a short frame limit")
    parser.add_argument("--ground-to-floor", dest="ground_to_floor", action="store_true", default=True,
                        help="Shift the character so feet sit on the floor plane")
    parser.add_argument("--no-ground-to-floor", dest="ground_to_floor", action="store_false",
                        help="Disable foot-based grounding (keeps hip-centric coordinates)")
    parser.add_argument("--foot-ground-margin", type=float, default=0.04,
                        help="Meters to keep feet above the floor when grounding is enabled")
    parser.add_argument("--still", action="store_true", help="Render only the first frame (debug)")
    return parser.parse_args(argv)


def extract_cli_args() -> List[str]:
    if "--" not in sys.argv:
        return []
    idx = sys.argv.index("--")
    return sys.argv[idx + 1:]


def apply_preview_overrides(args: argparse.Namespace):
    """Clamp render settings for quick iteration when --preview is set."""
    if not args.preview:
        return
    args.engine = "BLENDER_EEVEE"
    args.samples = min(args.samples, 32)
    if args.resolution == "1920x1080":
        args.resolution = "1280x720"
    if args.limit is None:
        args.limit = 120


def load_trace(npz_path: Path, limit: int | None = None) -> Tuple[np.ndarray, np.ndarray | None,
                                                                  np.ndarray | None, float, float, float]:
    with np.load(npz_path, allow_pickle=True) as data:
        if 'pose3d' not in data:
            raise RuntimeError(f"{npz_path} is missing 'pose3d'")
        pose3d = np.nan_to_num(data['pose3d']).astype(np.float32)
        bboxes = data.get('bboxes')
        indices = data.get('indices')
        meta = data.get('meta')

    if limit:
        pose3d = pose3d[:limit]
        if bboxes is not None:
            bboxes = bboxes[:limit]
        if indices is not None:
            indices = indices[:limit]

    if meta is None:
        fps = 30.0
        width = 1280
        height = 720
    else:
        fps = float(meta[0]) if meta.size else 30.0
        width = float(meta[2]) if meta.size > 2 else 1280
        height = float(meta[3]) if meta.size > 3 else 720

    return pose3d, bboxes, indices, fps, width, height


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Remove unused data blocks to keep the .blend clean
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.images):
        for datablock in collection:
            if datablock.users == 0:
                collection.remove(datablock)


def configure_render_settings(engine: str, samples: int, resolution: str, exposure: float, fps: float) -> str:
    scene = bpy.context.scene
    available_engines = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items}
    if engine not in available_engines:
        engine = 'BLENDER_EEVEE'
    scene.render.engine = engine
    width, height = [int(v) for v in resolution.lower().split('x')]
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.fps = max(1, int(round(fps)))

    if engine == 'CYCLES':
        scene.cycles.samples = samples
        scene.cycles.preview_samples = min(samples, 16)
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.max_bounces = 12
        if hasattr(scene.cycles, "use_denoising"):
            scene.cycles.use_denoising = False
        if hasattr(scene.cycles, "denoiser"):
            try:
                scene.cycles.denoiser = 'NONE'
            except TypeError:
                pass
        for view_layer in scene.view_layers:
            if hasattr(view_layer, "cycles"):
                view_layer.cycles.use_denoising = False
    else:
        scene.eevee.taa_render_samples = samples
        scene.eevee.use_gtao = True
        scene.eevee.use_bloom = True

    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.exposure = exposure
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scene.render.ffmpeg.audio_codec = 'NONE'
    return engine


def create_material(name: str, base_color: Tuple[float, float, float], roughness: float = 0.5,
                     metallic: float = 0.0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    principled = mat.node_tree.nodes.get('Principled BSDF')
    if principled:
        base_socket = principled.inputs.get('Base Color')
        if base_socket:
            base_socket.default_value = (*base_color, 1.0)
        rough_socket = principled.inputs.get('Roughness')
        if rough_socket:
            rough_socket.default_value = roughness
        metal_socket = principled.inputs.get('Metallic')
        if metal_socket:
            metal_socket.default_value = metallic
    return mat


def create_textured_material(name: str, base_color: Tuple[float, float, float], roughness: float,
                             bump_strength: float = 0.3) -> bpy.types.Material:
    """Procedural material with slight noise for backgrounds/clothing."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    principled = nodes.get('Principled BSDF')
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    noise = nodes.new(type='ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 8.0
    noise.inputs['Detail'].default_value = 2.0
    bump = nodes.new(type='ShaderNodeBump')
    bump.inputs['Strength'].default_value = bump_strength
    color_ramp = nodes.new(type='ShaderNodeValToRGB')
    color_ramp.color_ramp.elements[0].position = 0.3
    color_ramp.color_ramp.elements[1].position = 0.7
    color_ramp.color_ramp.elements[0].color = (*base_color, 1.0)
    color_ramp.color_ramp.elements[1].color = tuple(min(1.0, c + 0.05) for c in (*base_color, 1.0))

    links.new(tex_coord.outputs['Object'], noise.inputs['Vector'])
    links.new(noise.outputs['Fac'], color_ramp.inputs['Fac'])
    links.new(noise.outputs['Fac'], bump.inputs['Height'])
    if principled:
        links.new(color_ramp.outputs['Color'], principled.inputs['Base Color'])
        links.new(bump.outputs['Normal'], principled.inputs['Normal'])
        principled.inputs['Roughness'].default_value = roughness
    return mat


def build_proxy_avatar() -> Tuple[bpy.types.Object, List[bpy.types.Object]]:
    mesh = bpy.data.meshes.new('AvatarMesh')
    verts = [(0.0, 0.0, 0.0) for _ in JOINT_NAMES]
    mesh.from_pydata(verts, H36M_EDGES, [])
    mesh.update()

    avatar = bpy.data.objects.new('Avatar', mesh)
    bpy.context.collection.objects.link(avatar)

    skin = avatar.modifiers.new('Skin', 'SKIN')
    subsurf = avatar.modifiers.new('Subdivision', 'SUBSURF')
    subsurf.levels = 2
    subsurf.render_levels = 3
    skin_layer = avatar.data.skin_vertices[0]

    base_radius = {
        'Hip': 0.12,
        'Spine': 0.1,
        'Neck': 0.08,
        'Head': 0.15,
        'HeadTop': 0.15,
    }
    for idx, joint in enumerate(JOINT_NAMES):
        data = skin_layer.data[idx]
        radius = base_radius.get(joint, 0.06)
        data.radius = (radius, radius)
    skin_layer.data[0].use_root = True

    mat = create_material('AvatarSkin', (0.59, 0.44, 0.34), roughness=0.45)
    if avatar.data.materials:
        avatar.data.materials[0] = mat
    else:
        avatar.data.materials.append(mat)

    bpy.context.view_layer.objects.active = avatar
    bpy.ops.object.shade_smooth()
    avatar.data.use_auto_smooth = True

    joints = []
    for idx, joint in enumerate(JOINT_NAMES):
        empty = bpy.data.objects.new(f"Joint_{joint}", None)
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = 0.04
        bpy.context.collection.objects.link(empty)
        joints.append(empty)

        hook = avatar.modifiers.new(f"Hook_{idx}", 'HOOK')
        hook.object = empty
        hook.vertex_indices_set([idx])
        hook.strength = 1.0
        hook.show_in_editmode = True
        hook.show_expanded = False

    bpy.context.view_layer.objects.active = avatar
    for modifier in [m for m in avatar.modifiers if m.type == 'HOOK']:
        while avatar.modifiers.find(modifier.name) > 0:
            bpy.ops.object.modifier_move_up(modifier=modifier.name)

    bpy.ops.object.modifier_apply(modifier=skin.name)
    bpy.ops.object.modifier_apply(modifier=subsurf.name)

    return avatar, joints


def setup_environment(style: str, background_image: str | None, floor_height: float = 0.0):
    scene = bpy.context.scene
    bg_image = None
    if background_image:
        candidate = Path(background_image).expanduser()
        if candidate.exists():
            bg_image = bpy.data.images.load(str(candidate))
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new('TraceWorld')
        scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    background = nodes['Background']

    if bg_image:
        env_tex = nodes.get('EnvTex') or nodes.new(type='ShaderNodeTexEnvironment')
        env_tex.image = bg_image
        env_tex.name = 'EnvTex'
        links.new(env_tex.outputs['Color'], background.inputs[0])
        background.inputs[1].default_value = 1.0
    elif style == 'outdoor':
        sky = nodes.new(type='ShaderNodeTexSky')
        sky.sun_elevation = math.radians(35)
        sky.turbidity = 2.0
        links.new(sky.outputs['Color'], background.inputs[0])
        background.inputs[1].default_value = 1.0
    else:
        background.inputs[0].default_value = (0.08, 0.09, 0.1, 1.0)
        background.inputs[1].default_value = 1.1

    # Floor
    bpy.ops.mesh.primitive_plane_add(size=14, location=(0, 0, floor_height))
    floor = bpy.context.active_object
    if style == 'outdoor':
        floor_mat = create_textured_material('FloorOutdoor', (0.15, 0.17, 0.12), roughness=0.7)
    else:
        floor_mat = create_textured_material('FloorIndoor', (0.32, 0.34, 0.36), roughness=0.6)
    floor.data.materials.append(floor_mat)

    if style != 'outdoor':
        bpy.ops.mesh.primitive_plane_add(size=14, location=(0, 7.5, floor_height + 2.5))
        wall = bpy.context.active_object
        wall.rotation_euler = (math.radians(90), 0, 0)
        wall_mat = create_textured_material('Wall', (0.28, 0.28, 0.3), roughness=0.8, bump_strength=0.12)
        wall.data.materials.append(wall_mat)
        if bg_image:
            # Add a backplate plane to catch the HDRI/texture for parallax cues
            bpy.ops.mesh.primitive_plane_add(size=18, location=(0, 9.0, floor_height + 2.5))
            plate = bpy.context.active_object
            plate.rotation_euler = (math.radians(90), 0, 0)
            mat = bpy.data.materials.new('Backplate')
            mat.use_nodes = True
            tex_node = mat.node_tree.nodes.new(type='ShaderNodeTexImage')
            tex_node.image = bg_image
            principled = mat.node_tree.nodes.get('Principled BSDF')
            if principled:
                mat.node_tree.links.new(tex_node.outputs['Color'], principled.inputs['Base Color'])
                principled.inputs['Roughness'].default_value = 0.8
            plate.data.materials.append(mat)

    add_lights(style)


def add_lights(style: str):
    def area(name: str, energy: float, size: float, location: Tuple[float, float, float],
             rotation: Tuple[float, float, float]):
        light_data = bpy.data.lights.new(name=name, type='AREA')
        light_data.energy = energy
        light_data.size = size
        light_obj = bpy.data.objects.new(name, light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = location
        light_obj.rotation_euler = tuple(math.radians(v) for v in rotation)
        return light_obj

    if style == 'outdoor':
        sun = bpy.data.lights.new('Sun', type='SUN')
        sun.energy = 5.0
        sun_obj = bpy.data.objects.new('Sun', sun)
        bpy.context.collection.objects.link(sun_obj)
        sun_obj.rotation_euler = (math.radians(50), math.radians(-30), 0)
    else:
        area('KeyLight', 3000, 4.0, (3.0, -4.0, 3.0), (65, 0, -35))
        area('FillLight', 2200, 3.0, (-2.5, -4.2, 2.4), (70, 0, 25))
        area('RimLight', 1600, 2.5, (0, 4.5, 2.8), (110, 0, 180))


def convert_pose_sequence(pose_seq: np.ndarray, scale: float, axis_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    centered_frames = []
    roots = []
    for pose in pose_seq:
        coords = np.zeros_like(pose, dtype=np.float32)
        if axis_mode == "x_negz_y":
            # X = x, Y = -z (forward), Z = y (up)
            coords[:, 0] = pose[:, 0] * scale
            coords[:, 1] = -pose[:, 2] * scale
            coords[:, 2] = pose[:, 1] * scale
        elif axis_mode == "x_z_y":
            # X = x, Y = z (forward), Z = y (up)
            coords[:, 0] = pose[:, 0] * scale
            coords[:, 1] = pose[:, 2] * scale
            coords[:, 2] = pose[:, 1] * scale
        elif axis_mode == "x_z_negy":
            # X = x, Y = z (forward), Z = -y (down)
            coords[:, 0] = pose[:, 0] * scale
            coords[:, 1] = pose[:, 2] * scale
            coords[:, 2] = -pose[:, 1] * scale
        elif axis_mode == "x_negy_z":
            # X = x, Y = -y (forward), Z = z (up)
            coords[:, 0] = pose[:, 0] * scale
            coords[:, 1] = -pose[:, 1] * scale
            coords[:, 2] = pose[:, 2] * scale
        elif axis_mode == "negx_negz_y":
            # X = -x (mirror), Y = -z, Z = y
            coords[:, 0] = -pose[:, 0] * scale
            coords[:, 1] = -pose[:, 2] * scale
            coords[:, 2] = pose[:, 1] * scale
        elif axis_mode == "x_negz_negy":
            # X = x, Y = -z, Z = -y
            coords[:, 0] = pose[:, 0] * scale
            coords[:, 1] = -pose[:, 2] * scale
            coords[:, 2] = -pose[:, 1] * scale
        elif axis_mode == "y_negz_x":
            # X = y (side), Y = -z (forward), Z = x (up)
            coords[:, 0] = pose[:, 1] * scale
            coords[:, 1] = -pose[:, 2] * scale
            coords[:, 2] = pose[:, 0] * scale
        elif axis_mode == "y_z_x":
            # X = y, Y = z, Z = x
            coords[:, 0] = pose[:, 1] * scale
            coords[:, 1] = pose[:, 2] * scale
            coords[:, 2] = pose[:, 0] * scale
        elif axis_mode == "y_negz_negx":
            # X = y, Y = -z, Z = -x
            coords[:, 0] = pose[:, 1] * scale
            coords[:, 1] = -pose[:, 2] * scale
            coords[:, 2] = -pose[:, 0] * scale
        elif axis_mode == "y_z_negx":
            # X = y, Y = z, Z = -x
            coords[:, 0] = pose[:, 1] * scale
            coords[:, 1] = pose[:, 2] * scale
            coords[:, 2] = -pose[:, 0] * scale
        else:
            coords[:, 0] = pose[:, 0] * scale
            coords[:, 1] = -pose[:, 2] * scale
            coords[:, 2] = pose[:, 1] * scale
        root = coords[0].copy()
        centered_frames.append(coords - root)
        roots.append(root)
    return np.stack(centered_frames, axis=0), np.stack(roots, axis=0)


def estimate_root_translation(bboxes: np.ndarray | None, width: float, height: float,
                              meters_per_image: float) -> np.ndarray | None:
    if bboxes is None or bboxes.size == 0 or width <= 0 or height <= 0:
        return None
    centers = 0.5 * (bboxes[:, :2] + bboxes[:, 2:])
    norm_x = (centers[:, 0] / width) - 0.5
    norm_y = 0.5 - (centers[:, 1] / height)
    trans = np.zeros((centers.shape[0], 3), dtype=np.float32)
    trans[:, 0] = (norm_x - norm_x[0]) * meters_per_image
    trans[:, 1] = (norm_y - norm_y[0]) * meters_per_image
    return trans


def ground_roots_to_floor(centered: np.ndarray, roots: np.ndarray, margin: float) -> Tuple[np.ndarray, float]:
    """Shift roots so the ankles sit near the floor plane instead of the hip origin."""
    absolute = centered + roots[:, None, :]
    if absolute.shape[1] <= 6:
        return roots, 0.0
    foot_indices = [3, 6]  # R_Ankle, L_Ankle
    foot_heights = absolute[:, foot_indices, 2]
    per_frame_min = np.min(foot_heights, axis=1)
    baseline = np.percentile(per_frame_min, 5)
    offset = baseline - margin
    adjusted_roots = roots.copy()
    adjusted_roots[:, 2] -= offset
    return adjusted_roots, 0.0


def apply_pose_to_joints(joints: Sequence[bpy.types.Object], pose: np.ndarray, frame: int):
    bpy.context.scene.frame_set(frame)
    for empty, location in zip(joints, pose):
        empty.location = Vector(location)


def animate_joints(joints: Sequence[bpy.types.Object], poses: np.ndarray, start_frame: int):
    for frame_idx, pose in enumerate(poses):
        bpy.context.scene.frame_set(start_frame + frame_idx)
        for empty, location in zip(joints, pose):
            empty.location = Vector(location)
            empty.keyframe_insert(data_path='location', frame=start_frame + frame_idx)


def setup_camera(target: bpy.types.Object) -> bpy.types.Object:
    rig = bpy.data.objects.new('CameraRig', None)
    rig.empty_display_type = 'PLAIN_AXES'
    rig.empty_display_size = 0.2
    bpy.context.collection.objects.link(rig)
    rig.parent = target
    rig.location = (0.0, -6.0, 3.0)

    cam_data = bpy.data.cameras.new('TraceCamera')
    cam_data.lens = 35
    cam_data.dof.use_dof = True
    cam_data.dof.focus_distance = 4.0
    cam_obj = bpy.data.objects.new('TraceCamera', cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.parent = rig
    cam_obj.location = (0.0, 0.0, 0.0)

    track = cam_obj.constraints.new('TRACK_TO')
    track.target = target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    track.influence = 1.0
    bpy.context.scene.camera = cam_obj
    return cam_obj


def configure_output(output_dir: Path, trace_path: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output_dir / f"{trace_path.stem}.mp4")


def ensure_mblab_addon():
    is_enabled, is_loaded = addon_utils.check("MB-Lab")
    if not (is_enabled and is_loaded):
        addon_utils.modules_refresh()
        try:
            addon_utils.enable("MB-Lab", default_set=True, persistent=True)
        except Exception as exc:
            raise RuntimeError("Failed to enable MB-Lab addon. Install it under Blender's addons folder.") from exc


def available_mblab_characters(scene: bpy.types.Scene) -> List[str]:
    module = sys.modules.get('MB-Lab')
    if module is None:
        try:
            addon_utils.enable('MB-Lab', default_set=True)
            module = sys.modules.get('MB-Lab')
        except Exception:
            module = None
    if module is not None:
        humanoid = getattr(module, 'mblab_humanoid', None)
        if humanoid and getattr(humanoid, 'humanoid_types', None):
            return [item[0] for item in humanoid.humanoid_types]

    prop = scene.bl_rna.properties.get('mblab_character_name')
    if prop:
        return [item.identifier for item in prop.enum_items]
    return []


def create_mblab_character(character_id: str) -> Tuple[List[bpy.types.Object], bpy.types.Object | None]:
    ensure_mblab_addon()
    scene = bpy.context.scene
    chars = available_mblab_characters(scene)
    if not chars:
        raise RuntimeError("MB-Lab characters database not found. Verify MB-Lab/data is installed.")
    chosen = character_id if character_id in chars else chars[0]
    scene.mblab_character_name = chosen
    scene.mblab_use_ik = False
    scene.mblab_use_lamps = False
    bpy.ops.mbast.init_character()
    meshes = [obj for obj in bpy.data.objects
              if obj.type == 'MESH' and obj.parent and obj.parent.type == 'ARMATURE']
    armature = meshes[0].parent if meshes else None
    return meshes, armature


def compute_objects_height(objs: Sequence[bpy.types.Object]) -> float:
    min_z = float('inf')
    max_z = float('-inf')
    for obj in objs:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_z = min(min_z, world_corner.z)
            max_z = max(max_z, world_corner.z)
    height = max_z - min_z
    return height if height > 1e-4 else 1.0


def scale_and_center_mblab(armature: bpy.types.Object, mesh_objects: Sequence[bpy.types.Object],
                           target_height: float):
    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)
    armature.location = (0.0, 0.0, 0.0)
    armature.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    current_height = compute_objects_height(mesh_objects)
    if current_height <= 0:
        current_height = 1.0
    if target_height <= 0:
        target_height = 1.75
    scale_factor = target_height / current_height
    armature.scale = (scale_factor, scale_factor, scale_factor)
    bpy.context.view_layer.update()
    pelvis = armature.data.bones.get("pelvis")
    if pelvis:
        pelvis_world = armature.matrix_world @ pelvis.head_local
        armature.location -= pelvis_world
    bpy.context.view_layer.update()


def create_clothing_materials() -> Tuple[bpy.types.Material, bpy.types.Material]:
    shirt = create_textured_material('Cloth_Shirt', (0.22, 0.35, 0.55), roughness=0.7, bump_strength=0.25)
    pants = create_textured_material('Cloth_Pants', (0.08, 0.1, 0.12), roughness=0.65, bump_strength=0.2)
    return shirt, pants


def assign_clothing_to_mesh(mesh_obj: bpy.types.Object, shirt: bpy.types.Material, pants: bpy.types.Material):
    mesh = mesh_obj.data
    if mesh.materials.find(shirt.name) == -1:
        mesh.materials.append(shirt)
    if mesh.materials.find(pants.name) == -1:
        mesh.materials.append(pants)
    shirt_idx = mesh.materials.find(shirt.name)
    pants_idx = mesh.materials.find(pants.name)

    xs = [v.co.x for v in mesh.vertices]
    zs = [v.co.z for v in mesh.vertices]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    width = max(0.01, max_x - min_x)
    height = max(0.01, max_z - min_z)
    center_x = 0.5 * (min_x + max_x)
    waist_z = min_z + 0.4 * height
    shoulder_z = min_z + 0.75 * height

    for poly in mesh.polygons:
        avg = Vector((0.0, 0.0, 0.0))
        for v_idx in poly.vertices:
            avg += mesh.vertices[v_idx].co
        avg /= len(poly.vertices)
        torso_band = waist_z <= avg.z <= shoulder_z and abs(avg.x - center_x) <= 0.45 * width
        leg_band = avg.z < waist_z and abs(avg.x - center_x) <= 0.55 * width
        if torso_band:
            poly.material_index = shirt_idx
        elif leg_band:
            poly.material_index = pants_idx


def apply_procedural_clothing(mesh_objects: Sequence[bpy.types.Object]):
    if not mesh_objects:
        return
    shirt, pants = create_clothing_materials()
    for mesh_obj in mesh_objects:
        assign_clothing_to_mesh(mesh_obj, shirt, pants)


def compute_rest_vectors(armature: bpy.types.Object) -> dict[str, Vector]:
    rest = {}
    for bone_name in BONE_JOINT_MAP:
        bone = armature.data.bones.get(bone_name)
        if bone:
            vec = bone.tail_local - bone.head_local
            if vec.length > 1e-6:
                rest[bone_name] = vec.normalized()
    return rest


def animate_mblab_armature(armature: bpy.types.Object, joint_frames: np.ndarray,
                           roots: np.ndarray, start_frame: int):
    rest_vectors = compute_rest_vectors(armature)
    pose_bones = armature.pose.bones
    for name in BONE_JOINT_MAP:
        if name in pose_bones:
            pose_bones[name].rotation_mode = 'QUATERNION'

    for frame_idx, (frame, root_vec) in enumerate(zip(joint_frames, roots)):
        current_frame = start_frame + frame_idx
        vectors = [Vector(j.tolist()) for j in frame]
        armature.location = Vector(root_vec.tolist())
        armature.keyframe_insert(data_path='location', frame=current_frame)

        for bone_name, (idx_a, idx_b) in BONE_JOINT_MAP.items():
            if bone_name not in pose_bones or bone_name not in rest_vectors:
                continue
            target_vec = vectors[idx_b] - vectors[idx_a]
            if target_vec.length < 1e-6:
                continue
            rest_vec = rest_vectors[bone_name]
            rotation = rest_vec.rotation_difference(target_vec.normalized())
            pose_bone = pose_bones[bone_name]
            pose_bone.rotation_quaternion = rotation
            pose_bone.keyframe_insert(data_path='rotation_quaternion', frame=current_frame)


def animate_focus_target(target: bpy.types.Object, roots: np.ndarray, start_frame: int):
    for frame_idx, root in enumerate(roots):
        current_frame = start_frame + frame_idx
        target.location = Vector(root.tolist())
        target.keyframe_insert(data_path='location', frame=current_frame)


def main():
    args = parse_args(extract_cli_args())
    apply_preview_overrides(args)
    trace_path = Path(args.trace).expanduser().resolve()
    poses3d, bboxes, _, fps, width, height = load_trace(trace_path, args.limit)
    if poses3d.size == 0:
        raise RuntimeError(f"No pose data found in {trace_path}")

    centered, base_roots = convert_pose_sequence(poses3d, args.scale, args.axis_mode)
    bbox_roots = estimate_root_translation(bboxes, width, height, args.bbox_scale)
    if bbox_roots is None:
        roots = base_roots
    else:
        roots = base_roots + bbox_roots
    floor_height = 0.0
    if args.ground_to_floor:
        roots, floor_height = ground_roots_to_floor(centered, roots, args.foot_ground_margin)
    absolute = centered + roots[:, None, :]
    target_height = float(np.max(absolute[..., 2]) - np.min(absolute[..., 2]))

    clear_scene()
    resolved_engine = configure_render_settings(args.engine, args.samples, args.resolution, args.exposure, fps)
    setup_environment(args.scene_style, args.background_image, floor_height=floor_height)
    scene = bpy.context.scene
    scene.frame_start = args.frame_start
    total_frames = centered.shape[0] if not args.still else 1
    frames_to_use_absolute = absolute[:total_frames]
    roots_used = roots[:total_frames]
    scene.frame_end = args.frame_start + frames_to_use_absolute.shape[0] - 1

    if args.avatar_style == "mblab":
        meshes, armature = create_mblab_character(args.mblab_character)
        if not meshes or armature is None:
            raise RuntimeError("MB-Lab failed to create a character mesh.")
        scale_and_center_mblab(armature, meshes, target_height)
        if args.clothing_style == "procedural":
            apply_procedural_clothing(meshes)
        focus_target = bpy.data.objects.new('MotionFocus', None)
        focus_target.empty_display_size = 0.3
        focus_target.empty_display_type = 'PLAIN_AXES'
        bpy.context.collection.objects.link(focus_target)
        camera = setup_camera(focus_target)
        animate_mblab_armature(armature, frames_to_use_absolute, roots_used, args.frame_start)
        animate_focus_target(focus_target, roots_used, args.frame_start)
    else:
        driver_mesh, joints = build_proxy_avatar()
        apply_pose_to_joints(joints, frames_to_use_absolute[0], args.frame_start)
        bpy.context.view_layer.update()
        camera = setup_camera(joints[0])
        animate_joints(joints, frames_to_use_absolute, args.frame_start)

    bpy.context.scene.render.engine = resolved_engine
    configure_output(Path(args.output_dir), trace_path)

    if args.still:
        scene.frame_set(args.frame_start)
        bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
