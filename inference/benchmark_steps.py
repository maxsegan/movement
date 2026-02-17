#!/usr/bin/env python3
"""
Benchmark different numbers of diffusion steps to find the optimal value.
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


def evaluate_sample(model, sample, config, num_steps, seed):
    """Run inference with specific number of steps and return RMSE."""
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

    model.set_inference_steps(num_steps)

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

    # Sample indices
    np.random.seed(42)
    num_samples = 30
    test_indices = np.random.choice(len(val_dataset), size=num_samples, replace=False)

    # Step counts to test
    step_counts = list(range(1, 21))  # 1 to 20

    print(f"Benchmarking {len(step_counts)} step counts on {num_samples} samples")
    print("=" * 70)

    results = {steps: [] for steps in step_counts}

    for i, idx in enumerate(test_indices):
        sample = val_dataset[idx]
        sample_seed = int(idx)

        if (i + 1) % 10 == 0:
            print(f"  Processing sample {i+1}/{num_samples}...")

        for steps in step_counts:
            rmse = evaluate_sample(model, sample, config, steps, seed=sample_seed)
            results[steps].append(rmse)

    # Compute statistics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Steps':>6} | {'Mean RMSE':>10} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print("-" * 50)

    best_steps = None
    best_mean = float('inf')

    for steps in step_counts:
        rmses = results[steps]
        mean_rmse = np.mean(rmses)
        std_rmse = np.std(rmses)
        min_rmse = np.min(rmses)
        max_rmse = np.max(rmses)

        if mean_rmse < best_mean:
            best_mean = mean_rmse
            best_steps = steps

        print(f"{steps:>6} | {mean_rmse:>10.2f}° | {std_rmse:>8.2f}° | {min_rmse:>8.2f}° | {max_rmse:>8.2f}°")

    print("-" * 50)
    print(f"\nBest: {best_steps} steps with {best_mean:.2f}° mean RMSE")

    # Show ranking
    print("\n" + "=" * 70)
    print("RANKING (best to worst)")
    print("=" * 70)

    ranked = sorted(step_counts, key=lambda s: np.mean(results[s]))
    for i, steps in enumerate(ranked[:10]):
        mean_rmse = np.mean(results[steps])
        print(f"  {i+1}. {steps:2d} steps: {mean_rmse:.2f}°")


if __name__ == "__main__":
    main()
