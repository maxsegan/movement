"""
Optimized training script for Qwen-based VLA model
Includes staged training, gradual unfreezing, and memory optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, DistributedSampler
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
import itertools

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vla_model import VLAModel, VLAConfig
from kinetics_dataset import KineticsPoseDataset


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
    Manages staged unfreezing and LR decay based on % progress through training.

    Stage 1 (0-10%): Train diffusion head only, vision+LLM frozen
    Stage 2 (10-100%): Unfreeze vision + LLM (with LoRA), cosine LR decay
    """

    def __init__(self, model, config, total_steps: int, unfreeze_pct: float = 0.10):
        self.model = model
        self.config = config
        self.total_steps = total_steps
        self.unfreeze_pct = unfreeze_pct
        self.unfreeze_step = int(total_steps * unfreeze_pct)
        self.unfrozen = False
        self.base_lrs = None  # Will store initial LRs

        print(f"StagedTrainingScheduler: unfreeze at step {self.unfreeze_step} ({unfreeze_pct*100:.0f}% of {total_steps})")

    def update(self, global_step: int, optimizer) -> Dict[str, Any]:
        """
        Update training stage based on % through training.

        Args:
            global_step: Current global step
            optimizer: The optimizer to adjust LR for

        Returns:
            Dict with stage info: transitioned, lr_scale, progress
        """
        # Store base LRs on first call
        if self.base_lrs is None:
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        progress = global_step / self.total_steps
        transitioned = False

        # Check if we need to unfreeze
        if not self.unfrozen and global_step >= self.unfreeze_step:
            self.unfrozen = True
            transitioned = True

            # Unfreeze vision encoder and LLM
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            newly_unfrozen = []

            if hasattr(model_ref, 'qwen_model'):
                qwen = model_ref.qwen_model
                # Handle PEFT wrapped model (if LoRA is used)
                if hasattr(qwen, 'base_model'):
                    base_model = qwen.base_model.model
                else:
                    base_model = qwen

                # Unfreeze vision encoder
                if hasattr(base_model, 'visual'):
                    for param in base_model.visual.parameters():
                        param.requires_grad = True
                        newly_unfrozen.append(param)
                    print(f"Step {global_step} ({progress*100:.1f}%): Unfroze vision encoder")
                elif hasattr(base_model, 'model') and hasattr(base_model.model, 'visual'):
                    for param in base_model.model.visual.parameters():
                        param.requires_grad = True
                        newly_unfrozen.append(param)
                    print(f"Step {global_step} ({progress*100:.1f}%): Unfroze vision encoder")

                # Unfreeze LLM layers
                if hasattr(base_model, 'layers'):
                    for param in base_model.layers.parameters():
                        param.requires_grad = True
                        newly_unfrozen.append(param)
                    print(f"Step {global_step} ({progress*100:.1f}%): Unfroze LLM layers")
                elif hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
                    for param in base_model.model.layers.parameters():
                        param.requires_grad = True
                        newly_unfrozen.append(param)
                    print(f"Step {global_step} ({progress*100:.1f}%): Unfroze LLM layers")

            # Store newly unfrozen params for optimizer to add
            self.newly_unfrozen_params = newly_unfrozen
            print(f"Total newly unfrozen params: {len(newly_unfrozen)}")

        # Cosine LR decay from unfreeze point onwards
        if self.unfrozen:
            # Progress within the unfrozen phase (10% to 100%)
            unfrozen_progress = (global_step - self.unfreeze_step) / (self.total_steps - self.unfreeze_step)
            unfrozen_progress = min(1.0, max(0.0, unfrozen_progress))
            # Cosine decay from 0.5x to 0.1x base LR
            lr_scale = 0.1 + 0.4 * (1 + np.cos(np.pi * unfrozen_progress)) / 2
        else:
            lr_scale = 1.0

        # Update learning rates (handle case where new param groups were added)
        # If optimizer has more param groups than we tracked, update base_lrs
        while len(self.base_lrs) < len(optimizer.param_groups):
            # New param group was added - use its current LR as base
            new_idx = len(self.base_lrs)
            self.base_lrs.append(optimizer.param_groups[new_idx]['lr'])

        for i, group in enumerate(optimizer.param_groups):
            group['lr'] = self.base_lrs[i] * lr_scale

        return {
            'transitioned': transitioned,
            'lr_scale': lr_scale,
            'progress': progress,
            'unfrozen': self.unfrozen,
        }


class VLATrainer:
    """
    Optimized trainer for Qwen-based VLA
    """
    
    def __init__(self, config_path: str):
        self.distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self._init_distributed()

        self.device = torch.device(
            'cuda', self.local_rank
        ) if torch.cuda.is_available() else torch.device('cpu')

        model_config = VLAConfig(**self.config['model_config'])

        # In distributed mode, have rank 0 load first to download/cache the model
        # Then other ranks load from cache
        if self.distributed:
            if self.rank == 0:
                self.model = VLAModel(model_config)
            dist.barrier()  # Wait for rank 0 to finish
            if self.rank != 0:
                self.model = VLAModel(model_config)
            dist.barrier()  # Sync all ranks before continuing
        else:
            self.model = VLAModel(model_config)
        self.model_config = model_config
        
        if self.model:
            # Move to device
            self.model = self.model.to(self.device)
            
            # Enable gradient checkpointing if specified (non-reentrant for DDP+LoRA compatibility)
            if self.config.get('gradient_checkpointing', False):
                self.model.qwen_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )

            # Wrap with DDP if distributed
            if self.distributed:
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,  # All params used, no need for extra graph traversal
                    static_graph=False  # Must be False - graph changes due to use_cache toggling
                )
        
        # Setup optimization
        self.setup_optimization()
        
        # Setup data loaders
        self.setup_data()
        
        # Setup logging
        self.setup_logging()

        # Calculate total steps for scheduler
        steps_per_epoch = len(self.train_loader) // self.config['gradient_accumulation_steps']
        total_steps = steps_per_epoch * self.config['num_epochs']

        # Initialize staged training scheduler (%-based unfreezing)
        unfreeze_pct = self.config.get('unfreeze_pct', 0.10)  # Default 10%
        self.stage_scheduler = StagedTrainingScheduler(
            self.model, model_config, total_steps=total_steps, unfreeze_pct=unfreeze_pct
        )

        # Initialize metrics
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.current_epoch = 0
        self.resumed_from_checkpoint = False  # Track if we resumed, to skip batches only once

    def _init_distributed(self):
        """Initialize distributed training if launched with torchrun."""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            dist.init_process_group(backend='nccl')
            self.distributed = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(self.local_rank)
            print(f"Distributed training enabled: rank {self.rank}/{self.world_size} on local GPU {self.local_rank}")

    @property
    def is_main_process(self) -> bool:
        return (not self.distributed) or self.rank == 0
    
    def setup_optimization(self):
        """Setup optimizer with different LR groups"""
        if not self.model:
            return
        model_ref = self.model.module if isinstance(self.model, DDP) else self.model
            
        # Separate parameter groups
        param_groups = []
        
        # Diffusion head - highest learning rate
        diffusion_params = list(model_ref.action_head.parameters())
        param_groups.append({
            'params': diffusion_params,
            'lr': self.config['learning_rate'],
            'name': 'diffusion_head'
        })
        
        # Qwen LoRA parameters (if using LoRA)
        if self.config['model_config'].get('use_lora', False):
            lora_params = [p for n, p in model_ref.qwen_model.named_parameters() 
                          if 'lora' in n.lower() and p.requires_grad]
            if lora_params:
                param_groups.append({
                    'params': lora_params,
                    'lr': self.config['learning_rate'] * 0.1,  # Lower LR for LoRA
                    'name': 'lora'
                })
        
        # Other trainable parameters
        other_params = []
        for name, param in model_ref.named_parameters():
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
        
        # Mixed precision scaler - disabled for BFloat16 models (Qwen uses BFloat16)
        # BFloat16 doesn't need loss scaling due to its larger dynamic range
        self.scaler = GradScaler(enabled=False)
        
        print("Optimizer groups:")
        for group in param_groups:
            print(f"  {group['name']}: {len(group['params'])} params, LR={group['lr']}")
    
    def setup_data(self):
        """Setup data loaders"""
        dataset_cfg = self.config.get('dataset', {})
        action_horizon = self.config['model_config'].get('action_horizon', 16)
        num_frames = self.config['model_config'].get('num_frames', 4)
        resize = dataset_cfg.get('image_size', 224)
        num_workers = dataset_cfg.get('num_workers', 2)

        sample_stride = dataset_cfg.get('sample_stride', 3)

        # Create datasets
        train_dataset = KineticsPoseDataset(
            pose_dir=dataset_cfg['pose_dir'],
            desc_dir=dataset_cfg['desc_dir'],
            video_dir=dataset_cfg['video_dir'],
            split='train',
            val_split=dataset_cfg.get('val_split', 0.02),
            action_horizon=action_horizon,
            num_frames=num_frames,
            sample_stride=sample_stride,
            resize=resize,
            max_samples_per_class=dataset_cfg.get('max_samples_per_class'),
            normalize_pose=dataset_cfg.get('normalize_pose', True),
            use_joint_angles=dataset_cfg.get('use_joint_angles', True),
            seed=dataset_cfg.get('seed', 42),
        )

        val_dataset = KineticsPoseDataset(
            pose_dir=dataset_cfg['pose_dir'],
            desc_dir=dataset_cfg['desc_dir'],
            video_dir=dataset_cfg['video_dir'],
            split='val',
            val_split=dataset_cfg.get('val_split', 0.02),
            action_horizon=action_horizon,
            num_frames=num_frames,
            sample_stride=sample_stride,
            resize=resize,
            max_samples_per_class=dataset_cfg.get('max_samples_per_class'),
            normalize_pose=dataset_cfg.get('normalize_pose', True),
            use_joint_angles=dataset_cfg.get('use_joint_angles', True),
            seed=dataset_cfg.get('seed', 42),
        )

        train_sampler = None
        val_sampler = None
        if self.distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
            val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        # Note: Weighted sampling disabled - dataset too large (>2^24) for torch.multinomial
        # With 24M+ samples across 700 classes, distribution is reasonably balanced

        # Data loaders with memory pinning
        # Use 'fork' multiprocessing - safe here since CUDA is initialized after fork
        # (workers don't use CUDA, only the main process does)
        mp_context = 'fork' if num_workers > 0 else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn_qwen,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            multiprocessing_context=mp_context
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            sampler=val_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn_qwen,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            multiprocessing_context=mp_context
        )
    
    def setup_logging(self):
        """Setup logging and checkpointing"""
        # Create directories
        os.makedirs(self.config['log_dir'], exist_ok=True)
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.config['log_dir']) if self.is_main_process else None
        
        # Weights & Biases - works with wandb login or WANDB_API_KEY
        if self.is_main_process:
            try:
                wandb.init(
                    project="qwen-vla-humanoid",
                    config=self.config,
                    name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    resume="allow"
                )
                print("WandB initialized successfully")
            except wandb.errors.UsageError as e:
                print(f"WandB initialization failed (not logged in?): {e}")
                print("Continuing without WandB logging")
    
    def train_epoch(self):
        """Train for one epoch with memory optimization"""
        if not self.model:
            print("Model not initialized")
            return 0
            
        self.model.train()
        total_loss = 0
        num_batches = 0
        max_steps = self.config.get('max_train_steps')
        if self.distributed and hasattr(self.train_loader, 'sampler') and isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(self.current_epoch)

        # When resuming, we continue from where we left off but don't try to skip batches
        # The shuffled order is different each run anyway, so we just continue training
        # with the restored model/optimizer state
        if self.resumed_from_checkpoint:
            print(f"Resuming training from step {self.global_step} (model & optimizer restored)")
            self.resumed_from_checkpoint = False

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):

            if max_steps and self.global_step >= max_steps:
                break

            # Move data to device
            actions = batch['actions'].to(self.device)
            robot_states = batch['robot_states'].to(self.device)
            
            # Forward pass with mixed precision (bf16 matches model dtype)
            if self.config.get('use_amp', True):
                with autocast('cuda', dtype=torch.bfloat16):
                    output = self.model(
                        batch['images'],
                        batch['instructions'],
                        actions=actions,
                        robot_state=robot_states,
                        compute_loss=True
                    )
                    loss = output['loss']
            else:
                output = self.model(
                    batch['images'],
                    batch['instructions'],
                    actions=actions,
                    robot_state=robot_states,
                    compute_loss=True
                )
                loss = output['loss']

            # Check for NaN loss and skip bad batches
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss at step {self.global_step}, batch {batch_idx}, skipping...")
                self.optimizer.zero_grad()
                continue

            # Gradient accumulation
            loss = loss / self.config['gradient_accumulation_steps']

            # Backward pass (no scaling needed for BFloat16)
            loss.backward()

            # Gradient step
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                # Check for NaN gradients before step
                has_nan_grad = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        has_nan_grad = True
                        print(f"Warning: NaN/Inf gradient in {name} at step {self.global_step}")
                        break

                if has_nan_grad:
                    print(f"Skipping optimizer step due to NaN gradients")
                    self.optimizer.zero_grad()
                    continue

                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
                self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1

                # Update training stage (handles unfreezing and LR decay)
                stage_info = self.stage_scheduler.update(self.global_step, self.optimizer)
                if stage_info.get('transitioned'):
                    # Add newly unfrozen params to optimizer
                    if hasattr(self.stage_scheduler, 'newly_unfrozen_params') and self.stage_scheduler.newly_unfrozen_params:
                        new_lr = self.config['learning_rate'] * stage_info['lr_scale'] * 0.1  # Lower LR for VLM
                        self.optimizer.add_param_group({
                            'params': self.stage_scheduler.newly_unfrozen_params,
                            'lr': new_lr,
                            'name': 'vlm_unfrozen'
                        })
                        print(f"Added {len(self.stage_scheduler.newly_unfrozen_params)} params to optimizer with LR={new_lr}")
                    self.log_metrics({
                        'stage/step': self.global_step,
                        'stage/lr_scale': stage_info['lr_scale'],
                        'stage/unfrozen': 1.0,
                    })

                # Logging
                if self.global_step % self.config['log_every_n_steps'] == 0:
                    self.log_metrics({
                        'train/loss': loss.item() * self.config['gradient_accumulation_steps'],
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'train/epoch': self.current_epoch,
                        'train/progress': stage_info.get('progress', 0),
                    })
                
                # Checkpointing with validation
                if self.global_step % self.config['save_every_n_steps'] == 0:
                    # Run validation
                    val_loss = self.validate()
                    self.model.train()  # Back to training mode

                    self.save_checkpoint(
                        f"checkpoint_step_{self.global_step}.pth",
                        metrics={
                            'train_loss': loss.item() * self.config['gradient_accumulation_steps'],
                            'val_loss': val_loss
                        }
                    )
                    self.log_metrics({
                        'val/loss': val_loss,
                        'val/step': self.global_step
                    })
            
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
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        if self.distributed:
            tensor = torch.tensor([avg_loss, 1.0], device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            avg_loss = tensor[0].item() / tensor[1].item()
        return avg_loss
    
    def validate(self, max_batches: Optional[int] = None):
        """Validation with memory optimization

        Args:
            max_batches: Limit number of batches for quick validation during training.
                        None = full validation set.
        """
        if not self.model:
            return float('inf')

        self.model.eval()
        total_loss = 0
        num_batches = 0

        # Use config default if not specified
        if max_batches is None:
            max_batches = self.config.get('val_max_batches')

        if self.distributed and hasattr(self.val_loader, 'sampler') and isinstance(self.val_loader.sampler, DistributedSampler):
            self.val_loader.sampler.set_epoch(self.current_epoch)

        with torch.no_grad():
            val_total = max_batches if max_batches else len(self.val_loader)
            for batch in tqdm(self.val_loader, desc="Validation", leave=False, total=val_total):
                actions = batch['actions'].to(self.device)
                robot_states = batch['robot_states'].to(self.device)

                # Forward pass (bf16 matches model dtype)
                with autocast('cuda', dtype=torch.bfloat16, enabled=self.config.get('use_amp', True)):
                    output = self.model(
                        batch['images'],
                        batch['instructions'],
                        actions=actions,
                        robot_state=robot_states,
                        compute_loss=True
                    )
                    loss = output['loss']

                total_loss += loss.item()
                num_batches += 1

                # Early exit if max_batches specified
                if max_batches and num_batches >= max_batches:
                    break

                # Clear cache
                if num_batches % 50 == 0:
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        if self.distributed:
            tensor = torch.tensor([avg_loss, 1.0], device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            avg_loss = tensor[0].item() / tensor[1].item()
        
        # Save best model (main process only)
        if self.is_main_process and avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint("best_model.pth", metrics={'val_loss': avg_loss})
            print(f"New best validation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to TensorBoard and WandB"""
        if self.is_main_process and self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.global_step)
        
        if self.is_main_process and wandb.run:
            wandb.log(metrics, step=self.global_step)
    
    def save_checkpoint(self, filename: str, metrics: Optional[Dict[str, float]] = None):
        """Save model checkpoint and log to wandb"""
        if not self.model or not self.is_main_process:
            return

        checkpoint = {
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict(),
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

        # Rotate checkpoints - keep only last N step checkpoints (not best_model or epoch checkpoints)
        if filename.startswith('checkpoint_step_'):
            self._rotate_checkpoints()

        # Log checkpoint event to wandb (metrics only, no artifact upload to save bandwidth)
        if wandb.run:
            log_data = {
                'checkpoint/global_step': self.global_step,
                'checkpoint/epoch': self.current_epoch,
            }
            # Only log best_val_loss if validation has run
            if self.best_val_loss != float('inf'):
                log_data['checkpoint/best_val_loss'] = self.best_val_loss
            if metrics:
                log_data.update({f'checkpoint/{k}': v for k, v in metrics.items()})
            wandb.log(log_data, step=self.global_step)
    
    def _rotate_checkpoints(self):
        """Keep only the last N step checkpoints to save disk space"""
        max_checkpoints = self.config.get('max_checkpoints', 5)
        checkpoint_dir = Path(self.config['checkpoint_dir'])

        # Find all step checkpoints (not best_model or epoch checkpoints)
        step_checkpoints = sorted(
            checkpoint_dir.glob('checkpoint_step_*.pth'),
            key=lambda p: int(p.stem.split('_')[-1])
        )

        # Delete oldest checkpoints if we have too many
        while len(step_checkpoints) > max_checkpoints:
            oldest = step_checkpoints.pop(0)
            oldest.unlink()
            print(f"Deleted old checkpoint: {oldest.name}")

    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        if not self.model:
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Get checkpoint step to determine if we need to unfreeze before loading optimizer
        checkpoint_step = checkpoint.get('global_step', 0)

        # Load model state first
        model_state = checkpoint['model_state_dict']
        target_model = self.model.module if isinstance(self.model, DDP) else self.model
        target_model.load_state_dict(model_state)

        # If checkpoint is past unfreeze point, we need to trigger unfreeze BEFORE loading optimizer
        # This ensures optimizer has matching param groups for the saved state
        if checkpoint_step >= self.stage_scheduler.unfreeze_step:
            print(f"Checkpoint at step {checkpoint_step} is past unfreeze point ({self.stage_scheduler.unfreeze_step})")
            print("Triggering unfreeze before loading optimizer state...")

            # Manually trigger the unfreeze logic (similar to stage_scheduler.update)
            stage_info = self.stage_scheduler.update(checkpoint_step, self.optimizer)

            # Add newly unfrozen params to optimizer
            if hasattr(self.stage_scheduler, 'newly_unfrozen_params') and self.stage_scheduler.newly_unfrozen_params:
                new_lr = self.config['learning_rate'] * stage_info['lr_scale'] * 0.1
                self.optimizer.add_param_group({
                    'params': self.stage_scheduler.newly_unfrozen_params,
                    'lr': new_lr,
                    'name': 'vlm_unfrozen'
                })
                print(f"Added {len(self.stage_scheduler.newly_unfrozen_params)} params to optimizer")

        # Now load optimizer state (param groups should match)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'scaler_state_dict' in checkpoint and hasattr(self, 'scaler'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.global_step = checkpoint_step
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.resumed_from_checkpoint = True  # Mark that we resumed to enable batch skipping

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
            val_loss = None
            if epoch % self.config.get('evaluate_every_n_epochs', 2) == 0:
                val_loss = self.validate()
                if self.is_main_process:
                    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                    self.log_metrics({
                        'epoch/train_loss': train_loss,
                        'epoch/val_loss': val_loss,
                        'epoch/epoch': epoch
                    })
            else:
                if self.is_main_process:
                    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
                    self.log_metrics({
                        'epoch/train_loss': train_loss,
                        'epoch/epoch': epoch
                    })

            # Step scheduler
            self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                epoch_metrics = {'train_loss': train_loss}
                if val_loss is not None:
                    epoch_metrics['val_loss'] = val_loss
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth", metrics=epoch_metrics)
            
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
        
        print("Training completed!")
        self.writer.close()
        if wandb.run:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train Qwen-based VLA')
    parser.add_argument('--config', type=str, default='training/config_kinetics.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = VLATrainer(args.config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
