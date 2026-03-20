#!/usr/bin/env python3
"""
Upload model checkpoint to Hugging Face Hub.

Usage:
  # Upload latest step checkpoint:
  python scripts/upload_model.py

  # Upload specific checkpoint:
  python scripts/upload_model.py --checkpoint checkpoints/kinetics_vla/best_model.pth

  # Upload with a specific repo name:
  python scripts/upload_model.py --repo maxsegan/mimic-vlam
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo


def find_latest_checkpoint(checkpoint_dir="checkpoints/kinetics_vla"):
    """Find the latest step checkpoint by step number."""
    ckpt_dir = Path(checkpoint_dir)
    step_ckpts = sorted(
        ckpt_dir.glob("checkpoint_step_*.pth"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not step_ckpts:
        raise FileNotFoundError(f"No step checkpoints in {ckpt_dir}")
    return step_ckpts[-1]


def extract_model_card(checkpoint_path, config):
    """Generate a model card (README.md) from checkpoint metadata."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    step = ckpt.get("global_step", "unknown")
    best_val = ckpt.get("best_val_loss", "unknown")

    model_cfg = config.get("model_config", {}) if isinstance(config, dict) else {}
    dit_layers = model_cfg.get("dit_layers", 24)
    dit_dim = model_cfg.get("dit_hidden_dim", 1536)
    action_horizon = model_cfg.get("action_horizon", 16)

    card = f"""---
license: apache-2.0
tags:
  - robotics
  - humanoid
  - vision-language-action
  - vlam
  - diffusion-transformer
  - pose-estimation
datasets:
  - maxsegan/movenet-332
language:
  - en
---

# MIMIC: Motion Imitation from Massive Internet Clips

A 4.0B-parameter vision-language-action model for full-body humanoid control,
trained entirely from internet-scale human video.

## Model Details

- **Architecture**: Qwen3-VL-4B (early exit at layer 18) + {dit_layers}L/{dit_dim}D DiT action head
- **Parameters**: ~4.0B total (2.2B truncated LLM + 415M vision encoder + 1.28B DiT + 132M LoRA)
- **Action space**: 22-DoF joint angles at 10Hz
- **Action horizon**: {action_horizon} steps (1.6s)
- **Training data**: [MoveNet-332](https://huggingface.co/datasets/maxsegan/movenet-332) (~332K clips, ~4.7M samples from Kinetics-700)
- **Training compute**: 4x RTX Pro Blackwell GPUs (~576 GPU-hours)
- **Checkpoint step**: {step}
- **Best validation loss**: {best_val}

## Usage

```python
from training.vla_model import QwenVLAModel
import torch, yaml

config = yaml.safe_load(open("training/config_kinetics.yaml"))
model = QwenVLAModel(**config["model_config"])

ckpt = torch.load("checkpoint.pth", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval().cuda()
```

See the [GitHub repo](https://github.com/maxsegan/movement) for full inference and training code.

## Training

Trained with flow matching loss on the MoveNet-332 dataset. The vision encoder
(SigLIP) is frozen throughout; the LLM backbone uses LoRA (rank 128). The DiT
action head is trained from scratch.

## Citation

Paper forthcoming.

## License

Apache 2.0
"""
    return card, ckpt


def main():
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint (default: latest step checkpoint)",
    )
    parser.add_argument(
        "--repo", type=str, default="maxsegan/mimic-vlam",
        help="HuggingFace repo ID",
    )
    parser.add_argument(
        "--include-optimizer", action="store_true",
        help="Include optimizer state (large, only needed for resuming training)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be uploaded without uploading",
    )
    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_latest_checkpoint()
    print(f"Checkpoint: {ckpt_path} ({ckpt_path.stat().st_size / 1e9:.1f} GB)")

    # Load config from checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    step = ckpt.get("global_step", 0)

    # Generate model card
    model_card, _ = extract_model_card(ckpt_path, config)

    if args.dry_run:
        print(f"\nWould upload to: {args.repo}")
        print(f"Step: {step}")
        print(f"Include optimizer: {args.include_optimizer}")
        print(f"\nModel card:\n{model_card[:500]}...")
        return

    # Strip optimizer state if not needed (saves ~50% file size)
    if not args.include_optimizer:
        print("Stripping optimizer/scheduler state for smaller upload...")
        upload_ckpt = {
            "model_state_dict": ckpt["model_state_dict"],
            "config": ckpt.get("config"),
            "global_step": ckpt.get("global_step"),
            "current_epoch": ckpt.get("current_epoch"),
            "best_val_loss": ckpt.get("best_val_loss"),
        }
        stripped_path = ckpt_path.parent / f"upload_step_{step}.pth"
        torch.save(upload_ckpt, stripped_path)
        upload_file = stripped_path
        print(f"Stripped checkpoint: {stripped_path} ({stripped_path.stat().st_size / 1e9:.1f} GB)")
    else:
        upload_file = ckpt_path

    # Create repo if needed
    api = HfApi()
    create_repo(args.repo, repo_type="model", exist_ok=True)

    # Upload model card
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
    )
    print("Uploaded README.md")

    # Upload config alongside checkpoint
    config_path = upload_file.parent / "upload_config.json"
    if config:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # Use upload_large_folder for reliable large file upload
    # Stage files in a temp directory with the desired repo layout
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)
        # Symlink checkpoint to avoid copying 11GB
        (staging / "checkpoint.pth").symlink_to(upload_file.resolve())
        # Write model card
        (staging / "README.md").write_text(model_card)
        # Write config
        if config:
            shutil.copy(config_path, staging / "config.json")

        print(f"Uploading checkpoint ({upload_file.stat().st_size / 1e9:.1f} GB) via upload_large_folder...")
        api.upload_large_folder(
            repo_id=args.repo,
            repo_type="model",
            folder_path=str(staging),
        )

    print(f"Uploaded to https://huggingface.co/{args.repo}")

    # Clean up
    if not args.include_optimizer and stripped_path.exists():
        stripped_path.unlink()
        print(f"Cleaned up {stripped_path}")
    if config_path.exists():
        config_path.unlink()

    print(f"\nDone! Model available at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
