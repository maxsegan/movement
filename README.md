# MIMIC: Motion Imitation from Massive Internet Clips

A fully open pipeline for bootstrapping humanoid foundation models from internet-scale video. MIMIC extracts 3D human pose from Kinetics-700 videos, converts it to 22-DoF humanoid joint angles, and trains a 4.0B-parameter vision-language-action model (VLAM) for full-body humanoid control -- all on consumer GPUs.

**Paper:** *Bootstrapping Humanoid Foundation Models from Internet-Scale Video*

## Dataset: MoveNet-332

332K processed video clips from Kinetics-700 covering 704 action classes. Each clip contains 2D/3D pose sequences, 22-DoF joint angles, bounding boxes, and VLM-generated action descriptions.

**Stats:** 4.7M training samples, 96K validation samples, 10Hz frame rate, ~24s per clip.

**Download:** [maxsegan/movenet-332 on HuggingFace](https://huggingface.co/datasets/maxsegan/movenet-332)

## Model

Dual-system architecture following GR00T N1:

- **System 2 (vision-language encoder):** Qwen3-VL-4B with early exit at layer 18 and LoRA fine-tuning (rank 128)
- **System 1 (action head):** 24-layer Diffusion Transformer (1.28B params) with AdaLN timestep conditioning and cross-attention to Qwen hidden states
- **Output:** 22 joint angles (sin/cos encoded) over 16 future timesteps at 10Hz

**Weights:** [maxsegan/mimic-vlam on HuggingFace](https://huggingface.co/maxsegan/mimic-vlam)

## Repository Structure

```
data_prep/                        # Video processing pipeline
  pipeline/pipeline.py            #   Main orchestrator
  vitpose.py                      #   2D pose estimation (ViTPose-Large)
  pose3d.py                       #   3D lifting (MotionAGFormer, 243-frame clips)
  clip_filtering.py               #   Quality filtering (coverage, motion, tracking)
  fast_video_loader.py            #   FFmpeg-based frame extraction
  process_videos.py               #   Multi-GPU batch processing
  geometry.py                     #   Camera projection & normalization
  keypoints.py                    #   COCO <-> H36M joint format conversion
  temporal.py                     #   Overlapping clip strategy
  constants.py                    #   Shared constants (clip length, hop size)

training/                         # Model training
  train_vla.py                    #   Training loop (DDP, mixed precision)
  vla_model.py                    #   VLA architecture (VLAConfig, VLAModel)
  kinetics_dataset.py             #   Dataset class + joint angle conversion
  config_kinetics.yaml            #   Training hyperparameters

inference/                        # Evaluation & visualization
  eval_rolling.py                 #   Rolling inference vs baselines
  eval_ntu_pipeline.py            #   Pipeline validation against NTU MoCap
  paper_figure.py                 #   Paper figure generation
  render_robot.py                 #   MuJoCo humanoid rendering
  compare_gt_pred.py              #   GT vs predicted skeleton overlay
  benchmark_steps.py              #   Diffusion step count benchmarking
  generate_pipeline_figure.py     #   Pipeline diagram generation

scripts/
  prepare_hf_dataset.py           #   Export to HuggingFace Parquet format

tests/
  test_data_prep.py               #   Unit tests for geometry, temporal, keypoints
```

## Setup

```bash
# Clone
git clone https://github.com/maxsegan/movement.git
cd movement

# Create environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

External models are downloaded automatically on first use:
- [ViTPose-Large](https://huggingface.co/usyd-community/vitpose-plus-large) (2D pose)
- [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer) (3D lifting) -- checkpoint at `models/motionagformer-b-h36m.pth.tr`
- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (VLA backbone)

## Usage

### Data Preparation

Process Kinetics-700 videos into pose NPZ files:

```bash
python data_prep/process_videos.py \
  --data_root /data/kinetics-dataset/k700-2020 \
  --out_dir /data/kinetics_processed \
  --target_fps 10 \
  --gpus 0,1,2,3
```

The pipeline runs: video loading -> hard cut detection -> ViTPose 2D estimation -> MotionAGFormer 3D lifting -> clip filtering -> NPZ export.

Export to HuggingFace dataset format:

```bash
python scripts/prepare_hf_dataset.py --workers 16
```

### Training

```bash
python training/train_vla.py --config training/config_kinetics.yaml
```

Key training settings (see `config_kinetics.yaml`):
- 4 GPUs, batch size 8/GPU, gradient accumulation 16 (effective batch 512)
- Learning rate 1e-5, cosine annealing
- Flow matching loss with Beta(1.5, 1.0) timestep sampling
- LoRA on Qwen attention/MLP, frozen vision encoder
- Checkpoints every 750 steps

### Inference

```bash
# Rolling evaluation against baselines
python inference/eval_rolling.py

# Pipeline validation against NTU MoCap ground truth
python inference/eval_ntu_pipeline.py

# Generate paper figures
python inference/paper_figure.py

# Render MuJoCo humanoid from predicted joint angles
python inference/render_robot.py
```

### Tests

```bash
pytest tests/
```

## Pipeline Details

### Data Flow

1. **Video** -> FFmpeg frame extraction at target FPS
2. **Frames** -> Hard cut detection (histogram correlation)
3. **Frames** -> ViTPose-Large 2D keypoint estimation (17 H36M joints)
4. **2D poses** -> MotionAGFormer 3D lifting (243-frame overlapping clips, flip ensemble)
5. **3D poses** -> Quality filtering (body coverage >40%, motion significance, tracking consistency)
6. **3D poses** -> Inverse kinematics to 22-DoF joint angles
7. **Angles + frames** -> VLM captioning (Qwen3-VL-32B) for instruction conditioning

### Joint Representation

22 degrees of freedom covering the humanoid skeleton:

| Group | Joints | DoF |
|---|---|---|
| Spine | abdomen_y, abdomen_z, abdomen_x | 3 |
| Right leg | hip_x, hip_z, hip_y, knee | 4 |
| Left leg | hip_x, hip_z, hip_y, knee | 4 |
| Right arm | shoulder_x, shoulder_y, shoulder_z, elbow | 4 |
| Left arm | shoulder_x, shoulder_y, shoulder_z, elbow | 4 |
| Neck | neck_x, neck_y, neck_z | 3 |

Angles are encoded as sin/cos pairs (44 dimensions) to avoid discontinuities.

## Acknowledgments

This project builds on:
- [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer) for 3D pose lifting
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) for 2D pose estimation
- [Qwen3-VL](https://github.com/QwenLM/Qwen2.5-VL) as the vision-language backbone
- [Kinetics-700](https://www.deepmind.com/open-source/kinetics) video dataset
- [NTU RGB+D 120](https://rose1.ntu.edu.sg/dataset/actionRecognition/) for pipeline validation

Architecture inspired by [GR00T N1](https://arxiv.org/abs/2503.14734) and [Helix](https://arxiv.org/abs/2502.07092).

## License

MIT
