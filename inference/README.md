# Inference Rendering Toolkit

This folder now contains a single entry point, `render_trace.py`, for turning pose traces exported into `movement/data/kinetics_processed` (or `pose_processed`) into photorealistic videos straight from Blender. By default the script instantiates an MB-Lab character and procedurally retargets MotionAGFormer joint sequences onto the MB-Lab armature (every major bone gets a quaternion per frame) while inferring hip translation from the 2D bounding boxes, so you get a textured human avatar that mirrors each trace without touching the MB-Lab UI. Lights, camera, and materials are configured automatically so you can focus on supplying trace files.

## Requirements
- Blender 3.6+ with access to `bpy` (run headless via `blender --background` for batch rendering).
- The MB-Lab addon (1.8.x) installed in Blender's addons path. Run `blender --python enable_mblab.py` once if needed so Blender registers it.
- MotionAGFormer `.npz` outputs that include `pose3d`, `bboxes`, and `meta` arrays (see `movement/data/kinetics_processed`).
- Enough disk space under `movement/inference/renders` for the resulting `H.264` mp4 files.

## Usage
From the repository root (defaults to MB-Lab avatar + Cycles):

```bash
blender --background --python movement/inference/render_trace.py -- \
    --trace movement/data/kinetics_processed/abseiling/035LtPeUFTE_000085_000095.npz \
    --output-dir movement/inference/renders \
    --scene-style indoor \
    --engine CYCLES \
    --samples 64
```

Key options:
- `--avatar-style mblab|procedural` toggles between the MB-Lab human (default) and the original proxy mannequin.
- `--mblab-character f_ca01` picks the MB-Lab preset (check `MB-Lab/data/characters_config.json` for identifiers).
- `--scene-style` chooses the preset (`indoor` gives softbox lights, `outdoor` swaps in sun/sky, `studio` mirrors the indoor rig).
- `--background-image` wires a supplied HDRI/JPEG/PNG into the world and a backplate plane for a more realistic backdrop.
- `--clothing-style procedural|none` adds a simple shirt + pants material to MB-Lab meshes without touching the head/hands.
- `--preview` flips to Eevee, 720p, <=32 samples, and caps frames at 120 for fast iteration.
- `--ground-to-floor/--no-ground-to-floor` toggles foot anchoring; grounding keeps ankles a few centimeters above the floor plane.
- `--resolution 1920x1080` sets the render size (any `WIDTHxHEIGHT` string works).
- `--limit 120` renders only the specified number of frames for previews.
- `--still` renders a single frame for quick look dev.
- `--scale` stretches the skeleton if a trace looks too small/large; defaults to roughly 1.8 m tall.
- `--axis-mode` controls coordinate remapping from MotionAGFormer → Blender. Defaults to `x_negz_y` (X=x, Y=-z, Z=y). Additional options include `y_negz_x`/`y_z_x` to treat the source x-axis as vertical if your poses appear lying sideways, plus mirrored/flip variants.

Outputs land in the directory supplied to `--output-dir` (defaults to `movement/inference/renders`) with filenames derived from the source `.npz`. Keep the folder empty or `.gitignore`d because renders can be large.

## Workflow Tips
1. Inspect the `.npz` via `python3 - <<'PY'` scripts if you need to confirm `meta` contains valid FPS before rendering.
2. For higher realism, bump `--samples` to 128 or more and change `--engine` to `BLENDER_EEVEE` only when you need near-real-time iteration.
3. If MB-Lab retargeting runs slowly, switch to `--avatar-style procedural` while iterating, then flip back to `mblab` for final renders.
4. The script estimates root translation from the 2D boxes inside each `.npz`; if a dataset lacks bounding boxes, expect the avatar to stay centered.
5. Further customize the MB-Lab result by tweaking the helper block inside `render_trace.py` (`create_mblab_character`, `scale_and_center_mblab`, `animate_mblab_armature`) before rendering.

## Round-trip sanity check
`movement/inference/roundtrip_validate.py` renders a trace, then re-extracts 2D keypoints with ViTPose and lifts them with MotionAGFormer to measure drift against the source trace. This is useful to debug axis/bone mapping issues.

```bash
CUDA_VISIBLE_DEVICES=0 python3 movement/inference/roundtrip_validate.py \
  --trace movement/data/kinetics_processed/push_up_temp_centered.npz \
  --limit 120 \
  --output-dir movement/inference/renders/roundtrip
```

Outputs:
- Rendered mp4 (same name as the trace) in the chosen output dir.
- `{trace}_roundtrip.npz` containing re-extracted 2D/3D poses and metrics.
- `{trace}_roundtrip_metrics.json` with MPJPE stats after root-centering.

Note: ViTPose and MotionAGFormer checkpoints must be available locally; the script will use `usyd-community/vitpose-plus-large` and `models/motionagformer-b-h36m.pth.tr`.

## Future-frame overlay (no Blender)
`movement/inference/overlay_future_skeletons.py` overlays the next three pose frames onto each video frame (t+1/t+2/t+3 with fading alpha). It re-runs ViTPose to align scale/rotation but pins the pelvis to the current frame to sidestep global orientation issues. Example:

```bash
CUDA_VISIBLE_DEVICES=0 python movement/inference/overlay_future_skeletons.py \
  --video movement/data/kinetics_full_output/bbox_videos/'push up'/04p4MWfhpAI_000067_000077_bbox.mp4 \
  --pose movement/data/kinetics_processed/push_up_temp_centered.npz \
  --output movement/inference/renders/push_up_future_overlay.mp4 \
  --limit 180 \
  --device cuda:0
```

The output mp4 is a quick visual sanity check without invoking Blender.
