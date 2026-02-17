#!/usr/bin/env python3
"""
Test different numbers of diffusion steps and compare quality.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.kinetics_dataset import (
    KineticsPoseDataset, pose3d_to_joint_angles, JOINT_ANGLES_DIM
)
from training.vla_model import VLAModel, VLAConfig


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, config: dict, device: str) -> VLAModel:
    mc = config['model_config']
    vla_config = VLAConfig(
        qwen_model_name=mc.get('qwen_model_name', 'Qwen/Qwen3-VL-4B-Instruct'),
        qwen_hidden_size=mc.get('qwen_hidden_size', 2560),
        use_intermediate_hidden=mc.get('use_intermediate_hidden', True),
        hidden_layer_index=mc.get('hidden_layer_index', 18),
        use_early_exit=mc.get('use_early_exit', True),
        use_deepstack_features=mc.get('use_deepstack_features', True),
        use_flash_attention=mc.get('use_flash_attention', False),
        projection_dim=mc.get('projection_dim', 512),
        action_dim=mc.get('action_dim', 51),
        diffusion_hidden_dim=mc.get('diffusion_hidden_dim', 512),
        num_diffusion_layers=mc.get('num_diffusion_layers', 4),
        num_diffusion_heads=mc.get('num_diffusion_heads', 8),
        num_future_tokens=mc.get('num_future_tokens', 4),
        action_horizon=mc.get('action_horizon', 16),
        num_frames=mc.get('num_frames', 4),
        use_lora=mc.get('use_lora', True),
        lora_rank=mc.get('lora_rank', 128),
        lora_alpha=mc.get('lora_alpha', 128),
        lora_dropout=mc.get('lora_dropout', 0.05),
        freeze_vision_encoder=mc.get('freeze_vision_encoder', True),
        freeze_qwen_layers=mc.get('freeze_qwen_layers', 0),
        use_thinking_mode=mc.get('use_thinking_mode', False),
        diffusion_steps=mc.get('diffusion_steps', 8),
    )

    model = VLAModel(vla_config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def evaluate_sample(model, sample, config, num_steps, seed=42, solver="euler"):
    """Run inference with specific number of steps and return RMSE."""
    # Set seed for deterministic sampling (same initial noise for all step counts)
    torch.manual_seed(seed)
    np.random.seed(seed)

    images = sample['images']
    instruction = sample['instruction']
    gt_actions = sample['actions']
    robot_state = sample['robot_state']

    action_horizon = config['model_config'].get('action_horizon', 16)
    action_dim = config['model_config'].get('action_dim', 51)
    use_joint_angles = action_dim == JOINT_ANGLES_DIM

    if use_joint_angles:
        robot_state_3d = robot_state.reshape(17, 3)
        robot_state = pose3d_to_joint_angles(robot_state_3d)

    # Set diffusion steps and solver
    model.set_inference_steps(num_steps)
    model.set_inference_solver(solver)

    img_arrays = [np.array(img) for img in images]
    with torch.no_grad():
        pred_actions = model.get_action(
            images=img_arrays,
            instruction=instruction,
            robot_state=robot_state,
        )

    gt_poses = gt_actions.reshape(action_horizon, 17, 3)

    if use_joint_angles:
        pred_angles = pred_actions.reshape(action_horizon, JOINT_ANGLES_DIM)
        gt_angles = np.array([pose3d_to_joint_angles(gt_poses[t]) for t in range(action_horizon)])
        mse = np.mean((gt_angles - pred_angles) ** 2)
        rmse_deg = np.degrees(np.sqrt(mse))
    else:
        mse = np.mean((gt_poses - pred_actions.reshape(action_horizon, 17, 3)) ** 2)
        rmse_deg = np.sqrt(mse) * 100

    return rmse_deg


def main():
    config_path = "training/config_kinetics.yaml"
    checkpoint_path = "checkpoints/kinetics_vla/best_model.pth"
    device = "cuda:0"

    config = load_config(config_path)

    print("Loading model...")
    model = load_model(checkpoint_path, config, device)
    print("Model loaded!\n")

    # Load dataset
    action_horizon = config['model_config'].get('action_horizon', 16)
    num_frames = config['model_config'].get('num_frames', 4)
    dc = config['dataset']

    val_dataset = KineticsPoseDataset(
        pose_dir=dc['pose_dir'],
        desc_dir=dc['desc_dir'],
        video_dir=dc['video_dir'],
        split='val',
        val_split=dc.get('val_split', 0.02),
        action_horizon=action_horizon,
        num_frames=num_frames,
        sample_stride=dc.get('sample_stride', 3),
        resize=dc.get('image_size', 224),
        normalize_pose=dc.get('normalize_pose', True),
        use_joint_angles=False,
        seed=dc.get('seed', 42),
    )

    # Test on a few samples
    np.random.seed(42)
    test_indices = np.random.choice(len(val_dataset), size=10, replace=False)

    # Test configurations: (steps, solver)
    configs = [
        (8, "euler"),
        (8, "rk4"),
        (20, "euler"),
        (20, "rk4"),
        (50, "rk4"),
    ]

    print(f"Testing {len(configs)} configurations on {len(test_indices)} samples\n")
    print("=" * 70)

    results = {cfg: [] for cfg in configs}

    for i, idx in enumerate(test_indices):
        sample = val_dataset[idx]
        clip_idx, _ = val_dataset.samples[idx]
        clip = val_dataset.clips[clip_idx]

        print(f"\nSample {i+1}: {clip.action_class}")

        # Use sample index as seed so same sample gets same noise across configs
        sample_seed = int(idx)
        for steps, solver in configs:
            rmse = evaluate_sample(model, sample, config, steps, seed=sample_seed, solver=solver)
            results[(steps, solver)].append(rmse)
            print(f"  {steps:3d} steps {solver:5s}: {rmse:.2f}° RMSE")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for cfg in configs:
        steps, solver = cfg
        rmses = results[cfg]
        print(f"\n{steps} steps ({solver}):")
        print(f"  Mean RMSE: {np.mean(rmses):.2f}°")
        print(f"  Std RMSE:  {np.std(rmses):.2f}°")

    # Pairwise comparison vs baseline (8 steps euler)
    print("\n" + "=" * 70)
    print("COMPARISON vs 8 steps Euler (baseline)")
    print("=" * 70)

    baseline = np.array(results[(8, "euler")])
    for cfg in configs[1:]:
        steps, solver = cfg
        current = np.array(results[cfg])
        diff = baseline - current  # Positive = improvement
        pct = np.mean(diff) / np.mean(baseline) * 100
        improved = np.sum(diff > 0)
        print(f"\n{steps} steps {solver}:")
        print(f"  Mean improvement: {np.mean(diff):+.2f}° ({pct:+.1f}%)")
        print(f"  Samples improved: {improved}/{len(diff)}")


if __name__ == "__main__":
    main()
