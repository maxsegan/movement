"""
Optimized training script for Qwen-based VLA model
Includes staged training, gradual unfreezing, and memory optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import wandb
from tqdm import tqdm
import argparse
import os
from datetime import datetime
import yaml
import gc
from PIL import Image


from vla_model import VLAModel, VLAConfig


class VLADataset(Dataset):
    """
    Optimized dataset for Qwen-based VLA training
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        config: Dict[str, Any] = None,
        transform=None
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.config = config or {}
        self.transform = transform
        
        # Load sequences
        self.sequences = self._load_sequences()
        print(f"Loaded {len(self.sequences)} sequences for {split}")
        
        # Calculate frame sampling indices
        self.frame_indices = self._calculate_frame_indices()
        
    def _load_sequences(self) -> List[Path]:
        """Load sequence directories"""
        sequences = []
        split_file = self.data_path / f"{self.split}.txt"
        
        if split_file.exists():
            with open(split_file, 'r') as f:
                seq_names = f.read().strip().split('\n')
            sequences = [self.data_path / 'sequences' / name for name in seq_names]
        else:
            seq_dir = self.data_path / 'sequences'
            sequences = sorted([d for d in seq_dir.iterdir() if d.is_dir()])
            
            # Train/val split
            if self.split == 'train':
                sequences = sequences[:int(0.9 * len(sequences))]
            else:
                sequences = sequences[int(0.9 * len(sequences)):]
        
        return sequences
    
    def _calculate_frame_indices(self) -> List[int]:
        """Calculate which frames to sample for 4Hz from 30Hz data"""
        sequence_length = self.config.get('sequence_length', 10)
        sampling_rate = self.config.get('sampling_rate', 30)
        image_sampling_rate = self.config.get('image_sampling_rate', 4)
        
        total_frames = sequence_length * sampling_rate
        image_frames = sequence_length * image_sampling_rate
        
        # Evenly spaced frames
        indices = np.linspace(0, total_frames - 1, image_frames, dtype=int)
        return indices.tolist()
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_path = self.sequences[idx]
        
        # Load instruction
        with open(seq_path / 'instruction.txt', 'r') as f:
            instruction = f.read().strip()
        
        # Load actions and proprioception
        actions = np.load(seq_path / 'actions.npy')
        proprioception = np.load(seq_path / 'proprioception.npy')
        
        # Load images (already as PIL for Qwen)
        images = []
        image_dir = seq_path / 'images'
        image_files = sorted(image_dir.glob('*.png'))
        
        num_frames = self.config.get('num_frames', 4)
        for idx in self.frame_indices[:num_frames]:
            if idx < len(image_files):
                # Load as PIL directly for Qwen processor
                img = Image.open(image_files[idx]).convert('RGB')
                
                # Qwen3-VL handles any resolution - no resize needed
                
                # Apply augmentation if in training
                if self.split == 'train' and self.transform:
                    img = self.transform(img)
                
                images.append(img)
        
        # Sample action sequence
        action_horizon = self.config.get('action_horizon', 8)
        max_start = len(actions) - action_horizon
        if max_start > 0:
            start_idx = np.random.randint(0, max_start) if self.split == 'train' else 0
        else:
            start_idx = 0
        
        action_sequence = actions[start_idx:start_idx + action_horizon]
        robot_state = proprioception[start_idx]
        
        # Pad if necessary
        if len(action_sequence) < action_horizon:
            pad_length = action_horizon - len(action_sequence)
            action_sequence = np.pad(action_sequence, ((0, pad_length), (0, 0)), 'edge')
        
        return {
            'images': images,  # List of PIL images
            'instruction': instruction,
            'actions': action_sequence.astype(np.float32),
            'robot_state': robot_state.astype(np.float32),
            'seq_name': seq_path.name
        }


def collate_fn_qwen(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for Qwen
    Images stay as PIL for the processor
    """
    # Keep images as list of lists (batch of frame lists)
    images = [b['images'] for b in batch]
    
    # Text instructions
    instructions = [b['instruction'] for b in batch]
    
    # Numerical data
    actions = torch.stack([torch.from_numpy(b['actions']) for b in batch])
    robot_states = torch.stack([torch.from_numpy(b['robot_state']) for b in batch])
    
    seq_names = [b['seq_name'] for b in batch]
    
    return {
        'images': images,
        'instructions': instructions,
        'actions': actions,
        'robot_states': robot_states,
        'seq_names': seq_names
    }


class StagedTrainingScheduler:
    """
    Manages staged unfreezing and curriculum learning
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.current_stage = 0
        
        self.stages = [
            # Stage 1: Only train diffusion head
            {"epoch": 0, "unfreeze_layers": 0, "lr_scale": 1.0},
            # Stage 2: Unfreeze last 4 Qwen layers
            {"epoch": 10, "unfreeze_layers": 4, "lr_scale": 0.5},
            # Stage 3: Unfreeze last 8 layers
            {"epoch": 20, "unfreeze_layers": 8, "lr_scale": 0.3},
            # Stage 4: Unfreeze all with LoRA
            {"epoch": 30, "unfreeze_layers": -1, "lr_scale": 0.1},
        ]
    
    def update(self, epoch: int) -> float:
        """
        Update training stage based on epoch
        
        Returns:
            Learning rate scale for this stage
        """
        for stage in self.stages:
            if epoch >= stage["epoch"] and stage["epoch"] > self.current_stage:
                self.current_stage = stage["epoch"]
                self.model.unfreeze_backbone(stage["unfreeze_layers"])
                print(f"Entering stage at epoch {epoch}: "
                      f"unfreezing {stage['unfreeze_layers']} layers, "
                      f"LR scale {stage['lr_scale']}")
                return stage["lr_scale"]
        
        # Return current stage's LR scale
        for stage in reversed(self.stages):
            if epoch >= stage["epoch"]:
                return stage["lr_scale"]
        
        return 1.0


class VLATrainer:
    """
    Optimized trainer for Qwen-based VLA
    """
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_config = VLAConfig(**self.config['model_config'])
        self.model = VLAModel(model_config)
        self.model_config = model_config
        
        if self.model:
            # Move to device
            self.model = self.model.to(self.device)
            
            # Enable gradient checkpointing if specified
            if self.config.get('gradient_checkpointing', False):
                self.model.qwen_model.gradient_checkpointing_enable()
        
        # Setup optimization
        self.setup_optimization()
        
        # Setup data loaders
        self.setup_data()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize staged training scheduler
        # self.stage_scheduler = StagedTrainingScheduler(self.model, model_config)
        
        # Initialize metrics
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.current_epoch = 0
    
    def setup_optimization(self):
        """Setup optimizer with different LR groups"""
        if not self.model:
            return
            
        # Separate parameter groups
        param_groups = []
        
        # Diffusion head - highest learning rate
        diffusion_params = list(self.model.action_head.parameters())
        param_groups.append({
            'params': diffusion_params,
            'lr': self.config['learning_rate'],
            'name': 'diffusion_head'
        })
        
        # Qwen LoRA parameters (if using LoRA)
        if self.config['model_config'].get('use_lora', False):
            lora_params = [p for n, p in self.model.qwen_model.named_parameters() 
                          if 'lora' in n.lower() and p.requires_grad]
            if lora_params:
                param_groups.append({
                    'params': lora_params,
                    'lr': self.config['learning_rate'] * 0.1,  # Lower LR for LoRA
                    'name': 'lora'
                })
        
        # Other trainable parameters
        other_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and not any(
                param is p for group in param_groups for p in group['params']
            ):
                other_params.append(param)
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.config['learning_rate'] * 0.5,
                'name': 'other'
            })
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Initial restart period
            T_mult=2,  # Increase period after each restart
            eta_min=1e-6
        )
        
        # Mixed precision scaler
        if self.config.get('use_amp', True):
            self.scaler = GradScaler()
        
        print("Optimizer groups:")
        for group in param_groups:
            print(f"  {group['name']}: {len(group['params'])} params, LR={group['lr']}")
    
    def setup_data(self):
        """Setup data loaders"""
        # Create datasets
        train_dataset = QwenVLADataset(
            self.config['data_path'],
            split='train',
            config=self.config.get('model_config', {})
        )
        
        val_dataset = QwenVLADataset(
            self.config['data_path'],
            split='val',
            config=self.config.get('model_config', {})
        )
        
        # Data loaders with memory pinning
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,  # Fewer workers to save memory
            pin_memory=True,
            collate_fn=collate_fn_qwen,
            persistent_workers=True  # Keep workers alive
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn_qwen,
            persistent_workers=True
        )
    
    def setup_logging(self):
        """Setup logging and checkpointing"""
        # Create directories
        os.makedirs(self.config['log_dir'], exist_ok=True)
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.config['log_dir'])
        
        # Weights & Biases
        if os.environ.get('WANDB_API_KEY'):
            wandb.init(
                project="qwen-vla-humanoid",
                config=self.config,
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def train_epoch(self):
        """Train for one epoch with memory optimization"""
        if not self.model:
            print("Model not initialized")
            return 0
            
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Update training stage
        # lr_scale = self.stage_scheduler.update(self.current_epoch)
        
        # Adjust learning rates based on stage
        # for group in self.optimizer.param_groups:
        #     group['lr'] = group['lr'] * lr_scale
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            actions = batch['actions'].to(self.device)
            robot_states = batch['robot_states'].to(self.device)
            
            # Forward pass with mixed precision
            if self.config.get('use_amp', True):
                with autocast('cuda'):
                    output = self.model(
                        batch['images'],
                        batch['instructions'],
                        actions=actions,
                        robot_state=robot_states
                    )
                    loss = output['loss']
            else:
                output = self.model(
                    batch['images'],
                    batch['instructions'],
                    actions=actions,
                    robot_state=robot_states
                )
                loss = output['loss']
            
            # Gradient accumulation
            loss = loss / self.config['gradient_accumulation_steps']
            
            # Backward pass
            if self.config.get('use_amp', True):
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient step
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                if self.config.get('use_amp', True):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clip']
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config['log_every_n_steps'] == 0:
                    self.log_metrics({
                        'train/loss': loss.item() * self.config['gradient_accumulation_steps'],
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'train/epoch': self.current_epoch
                    })
                
                # Checkpointing
                if self.global_step % self.config['save_every_n_steps'] == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pth")
            
            total_loss += loss.item() * self.config['gradient_accumulation_steps']
            num_batches += 1
            
            pbar.set_postfix({
                'loss': total_loss / num_batches,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Clear cache periodically to prevent memory issues
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate(self):
        """Validation with memory optimization"""
        if not self.model:
            return float('inf')
            
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                actions = batch['actions'].to(self.device)
                robot_states = batch['robot_states'].to(self.device)
                
                # Forward pass
                with autocast('cuda', enabled=self.config.get('use_amp', True)):
                    output = self.model(
                        batch['images'],
                        batch['instructions'],
                        actions=actions,
                        robot_state=robot_states
                    )
                    loss = output['loss']
                
                total_loss += loss.item()
                num_batches += 1
                
                # Clear cache
                if num_batches % 50 == 0:
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint("best_model.pth")
            print(f"New best validation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to TensorBoard and WandB"""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.global_step)
        
        if wandb.run:
            wandb.log(metrics, step=self.global_step)
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        if not self.model:
            return
            
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
        }
        
        if hasattr(self, 'scaler'):
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = Path(self.config['checkpoint_dir']) / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        if not self.model:
            return
            
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and hasattr(self, 'scaler'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint from {path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self):
        """Main training loop"""
        print("Starting Qwen VLA training...")
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            if epoch % self.config.get('evaluate_every_n_epochs', 2) == 0:
                val_loss = self.validate()
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                self.log_metrics({
                    'epoch/train_loss': train_loss,
                    'epoch/val_loss': val_loss,
                    'epoch/epoch': epoch
                })
            else:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
                self.log_metrics({
                    'epoch/train_loss': train_loss,
                    'epoch/epoch': epoch
                })
            
            # Step scheduler
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
            
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
        
        print("Training completed!")
        self.writer.close()
        if wandb.run:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train Qwen-based VLA')
    parser.add_argument('--config', type=str, default='config_qwen3.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = QwenVLATrainer(args.config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()