"""
Vision-Language-Action Model using Qwen3-VL-2B as backbone
Leverages the latest Qwen3-VL improvements for embodied AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from dataclasses import dataclass
import math
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
from peft import LoraConfig, get_peft_model, TaskType
import warnings


@dataclass
class VLAConfig:
    """Configuration for Qwen3-VL based VLA model"""
    # Qwen3-VL backbone - using the latest version
    qwen_model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    use_intermediate_hidden: bool = True  # Extract intermediate layers
    hidden_layer_index: int = -4  # Which layer to extract (-4 = 4th from last)
    
    # Qwen3-VL specific settings
    qwen_hidden_size: int = 1536  # Qwen3-VL-2B hidden size
    use_deepstack_features: bool = True  # Use multi-level ViT features
    use_flash_attention: bool = True  # Use flash_attention_2 for efficiency
    
    # Projection from Qwen hidden states to action space
    projection_dim: int = 1024
    
    # Enhanced diffusion head for Qwen3 features
    action_dim: int = 22  # Adjust based on your humanoid DOF
    diffusion_hidden_dim: int = 768  # Larger for richer Qwen3 features
    num_diffusion_layers: int = 8
    num_diffusion_heads: int = 12
    diffusion_dropout: float = 0.1
    diffusion_steps: int = 10  # For 30Hz inference
    
    # Action chunking and horizon
    action_horizon: int = 8  # Predict 8 future actions at 30Hz
    action_chunking: bool = True
    
    # Frame processing (Qwen3-VL supports dynamic resolution)
    num_frames: int = 4  # Process 4 frames at 4Hz
    image_size: Optional[int] = None  # Qwen3-VL handles any resolution
    use_dynamic_resolution: bool = True  # Qwen3-VL's dynamic resolution
    
    # LoRA configuration for efficient fine-tuning
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None  # Will set defaults in init
    
    # Training stages
    freeze_vision_encoder: bool = True  # Initially freeze
    freeze_qwen_layers: int = 20  # Freeze first N layers initially
    
    # Qwen3-VL Thinking mode (for reasoning-enhanced version)
    use_thinking_mode: bool = False  # Set True for Qwen3-VL-2B-Thinking
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            # Target attention and MLP layers in Qwen3-VL
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        # Switch to Thinking model if requested
        if self.use_thinking_mode:
            self.qwen_model_name = "Qwen/Qwen3-VL-2B-Thinking"


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FlowMatchingDiffusion(nn.Module):
    """
    Enhanced Flow Matching Diffusion Head optimized for Qwen3-VL features
    Takes advantage of Qwen3's improved spatial and temporal understanding
    """
    
    def __init__(self, config: VLAConfig):
        super().__init__()
        self.config = config
        
        # Time embedding with learnable components
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(config.diffusion_hidden_dim),
            nn.Linear(config.diffusion_hidden_dim, config.diffusion_hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(config.diffusion_hidden_dim * 2, config.diffusion_hidden_dim),
        )
        
        # Project from Qwen3 hidden states (with DeepStack features)
        self.qwen_projection = nn.Sequential(
            nn.Linear(config.qwen_hidden_size, config.projection_dim),
            nn.LayerNorm(config.projection_dim),
            nn.SiLU(),
            nn.Linear(config.projection_dim, config.diffusion_hidden_dim)
        )
        
        # Optional: Project multi-level features if using DeepStack
        if config.use_deepstack_features:
            self.deepstack_projection = nn.Linear(
                config.diffusion_hidden_dim * 2,  # Combine multiple levels
                config.diffusion_hidden_dim
            )
        
        # Action encoder/decoder with residual connections
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_dim * config.action_horizon, config.diffusion_hidden_dim),
            nn.LayerNorm(config.diffusion_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.diffusion_dropout)
        )
        
        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(config.action_dim, config.diffusion_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.diffusion_hidden_dim // 2, config.diffusion_hidden_dim),
            nn.LayerNorm(config.diffusion_hidden_dim)
        )
        
        # Spatial-aware cross-attention (leveraging Qwen3's spatial understanding)
        self.spatial_cross_attention = nn.MultiheadAttention(
            embed_dim=config.diffusion_hidden_dim,
            num_heads=config.num_diffusion_heads,
            dropout=config.diffusion_dropout,
            batch_first=True
        )
        
        # Self-attention transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.diffusion_hidden_dim,
            nhead=config.num_diffusion_heads,
            dim_feedforward=config.diffusion_hidden_dim * 4,
            dropout=config.diffusion_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_diffusion_layers
        )
        
        # Final action prediction with skip connection
        self.action_decoder = nn.Sequential(
            nn.Linear(config.diffusion_hidden_dim * 2, config.diffusion_hidden_dim * 2),
            nn.LayerNorm(config.diffusion_hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(config.diffusion_dropout),
            nn.Linear(config.diffusion_hidden_dim * 2, config.action_dim * config.action_horizon),
        )
        
        # Learnable query tokens for action generation
        self.action_queries = nn.Parameter(
            torch.randn(1, config.action_horizon, config.diffusion_hidden_dim)
        )
    
    def forward(
        self,
        qwen_features: torch.Tensor,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        robot_state: Optional[torch.Tensor] = None,
        deepstack_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for flow matching
        
        Args:
            qwen_features: Hidden states from Qwen3-VL [B, L, D]
            noisy_actions: Noisy action sequence [B, action_dim * action_horizon]
            timesteps: Diffusion timesteps [B]
            robot_state: Current robot proprioception [B, state_dim]
            deepstack_features: Optional multi-level features from Qwen3's DeepStack
        
        Returns:
            Predicted velocity field [B, action_dim * action_horizon]
        """
        B = qwen_features.shape[0]
        device = qwen_features.device
        
        # Time embedding
        t_emb = self.time_embed(timesteps)  # [B, D]
        
        # Project Qwen3 features
        vl_features = self.qwen_projection(qwen_features)  # [B, L, D]
        
        # Incorporate DeepStack features if available
        if deepstack_features is not None and self.config.use_deepstack_features:
            deep_features = self.deepstack_projection(deepstack_features)
            vl_features = vl_features + deep_features.unsqueeze(1)
        
        # Encode noisy actions
        action_emb = self.action_encoder(noisy_actions)  # [B, D]
        
        # Add proprioception if available
        if robot_state is not None:
            proprio_emb = self.proprio_encoder(robot_state)  # [B, D]
            action_emb = action_emb + proprio_emb
        
        # Expand action queries
        queries = self.action_queries.expand(B, -1, -1)  # [B, H, D]
        
        # Add time conditioning to queries
        queries = queries + t_emb.unsqueeze(1)  # [B, H, D]
        
        # Add action information to queries
        queries[:, 0, :] = queries[:, 0, :] + action_emb  # First query gets action info
        
        # Spatial-aware cross-attention (leveraging Qwen3's improved spatial understanding)
        attended_features, _ = self.spatial_cross_attention(
            queries, vl_features, vl_features
        )  # [B, H, D]
        
        # Self-attention refinement
        refined_features = self.transformer(attended_features)  # [B, H, D]
        
        # Combine with original queries (residual)
        combined = torch.cat([
            refined_features.flatten(1, 2), 
            action_emb.unsqueeze(1).expand(-1, self.config.action_horizon, -1).flatten(1, 2)
        ], dim=-1)
        
        # Decode to velocity field
        velocity = self.action_decoder(combined)  # [B, action_dim * action_horizon]
        
        return velocity
    
    @torch.no_grad()
    def sample(
        self,
        qwen_features: torch.Tensor,
        robot_state: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
        deepstack_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample actions using flow matching ODE solver
        
        Args:
            qwen_features: Hidden states from Qwen3-VL [B, L, D]
            robot_state: Current robot state [B, state_dim]
            num_steps: Number of diffusion steps
            temperature: Sampling temperature for diversity
            deepstack_features: Optional multi-level features
        
        Returns:
            Sampled action sequence [B, action_horizon, action_dim]
        """
        if num_steps is None:
            num_steps = self.config.diffusion_steps
            
        B = qwen_features.shape[0]
        device = qwen_features.device
        
        # Initialize from noise
        x = torch.randn(
            B, 
            self.config.action_dim * self.config.action_horizon,
            device=device
        ) * temperature
        
        # ODE integration with adaptive step size
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            
            # Predict velocity field
            with torch.cuda.amp.autocast(enabled=True):
                v = self.forward(qwen_features, x, t, robot_state, deepstack_features)
            
            # Euler integration
            x = x + v * dt
        
        # Reshape to [B, action_horizon, action_dim]
        actions = x.reshape(B, self.config.action_horizon, self.config.action_dim)
        
        return actions


class VLAModel(nn.Module):
    """
    Complete VLA model using Qwen3-VL backbone with diffusion action head
    Leverages Qwen3-VL's improvements for embodied AI
    """
    
    def __init__(self, config: VLAConfig):
        super().__init__()
        self.config = config
        
        # Load Qwen3-VL model
        print(f"Loading Qwen3-VL from {config.qwen_model_name}")
        
        # Load with appropriate settings
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": None  # We'll handle device placement
        }
        
        # Add flash attention if supported
        if config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.qwen_model = VLForConditionalGeneration.from_pretrained(
            config.qwen_model_name,
            **model_kwargs
        )
        
        # Load processor for preprocessing
        self.processor = AutoProcessor.from_pretrained(config.qwen_model_name)
        
        # Apply LoRA if specified
        if config.use_lora:
            self._apply_lora()
        
        # Freeze layers as specified
        self._freeze_layers()
        
        # Diffusion action head
        self.action_head = FlowMatchingDiffusion(config)
        
        # Register hooks for intermediate features and DeepStack
        self.intermediate_features = None
        self.deepstack_features = None
        if config.use_intermediate_hidden:
            self._register_hooks()
    
    def _apply_lora(self):
        """Apply LoRA to Qwen3 model"""
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
        )
        
        # Apply LoRA
        self.qwen_model = get_peft_model(self.qwen_model, lora_config)
        print(f"Applied LoRA with rank {self.config.lora_rank}")
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.qwen_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.qwen_model.parameters())
        print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    def _freeze_layers(self):
        """Freeze specified layers for staged training"""
        # Freeze vision encoder if specified
        if self.config.freeze_vision_encoder:
            if hasattr(self.qwen_model, 'visual'):
                for param in self.qwen_model.visual.parameters():
                    param.requires_grad = False
                print("Froze vision encoder")
        
        # Freeze first N transformer layers
        if self.config.freeze_qwen_layers > 0:
            if hasattr(self.qwen_model, 'model') and hasattr(self.qwen_model.model, 'layers'):
                layers_to_freeze = list(self.qwen_model.model.layers)[:self.config.freeze_qwen_layers]
                for layer in layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
                print(f"Froze first {self.config.freeze_qwen_layers} Qwen layers")
    
    def _register_hooks(self):
        """Register forward hooks to extract intermediate and DeepStack features"""
        def hook_fn(module, input, output):
            # Store intermediate hidden states
            if hasattr(output, 'hidden_states') and output.hidden_states is not None:
                self.intermediate_features = output.hidden_states[self.config.hidden_layer_index]
                
                # Extract multi-level features for DeepStack
                if self.config.use_deepstack_features:
                    # Get features from multiple layers
                    mid_layer = len(output.hidden_states) // 2
                    early_features = output.hidden_states[mid_layer - 2]
                    late_features = output.hidden_states[self.config.hidden_layer_index]
                    
                    # Combine multi-level features
                    self.deepstack_features = torch.cat([early_features, late_features], dim=-1)
            elif isinstance(output, tuple) and len(output) > 0:
                self.intermediate_features = output[0]
            else:
                self.intermediate_features = output
        
        # Register hook on the model layers
        if hasattr(self.qwen_model, 'model') and hasattr(self.qwen_model.model, 'layers'):
            num_layers = len(self.qwen_model.model.layers)
            layer_idx = self.config.hidden_layer_index if self.config.hidden_layer_index >= 0 else num_layers + self.config.hidden_layer_index
            layer_idx = max(0, min(layer_idx, num_layers - 1))
            
            self.qwen_model.model.layers[layer_idx].register_forward_hook(hook_fn)
            print(f"Registered hook on layer {layer_idx}")
    
    def encode_inputs(
        self,
        images: List[torch.Tensor],
        instructions: List[str],
        device: torch.device
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode images and instructions using Qwen3-VL
        
        Args:
            images: List of image tensors [B, 3, H, W] or PIL images
            instructions: List of text instructions
            device: Device to run on
            
        Returns:
            Tuple of (hidden features, optional deepstack features)
        """
        B = len(instructions)
        
        # Prepare messages for Qwen3-VL format
        messages_batch = []
        for i in range(B):
            content = []
            
            # Add images - Qwen3-VL supports dynamic resolution
            for j, img in enumerate(images):
                if isinstance(img[i], torch.Tensor):
                    # Convert tensor to PIL
                    img_np = img[i].cpu().numpy().transpose(1, 2, 0)
                    img_np = (img_np * 255).astype(np.uint8)
                    from PIL import Image
                    pil_img = Image.fromarray(img_np)
                else:
                    pil_img = img[i]
                
                content.append({"type": "image", "image": pil_img})
            
            # Add instruction
            content.append({"type": "text", "text": instructions[i]})
            
            messages = [{"role": "user", "content": content}]
            messages_batch.append(messages)
        
        # Process with Qwen3 processor
        text = self.processor.apply_chat_template(
            messages_batch[0],  # Process one at a time for now
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process inputs (Qwen3-VL handles any resolution)
        inputs = self.processor(
            text=[text] * B,
            images=[img[0] for img in images],  # Just use first image for now
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Forward pass through Qwen3
        with torch.cuda.amp.autocast(enabled=True):
            outputs = self.qwen_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Extract features
        if self.config.use_intermediate_hidden and self.intermediate_features is not None:
            features = self.intermediate_features
            deepstack = self.deepstack_features if self.config.use_deepstack_features else None
        else:
            # Use last hidden state
            features = outputs.hidden_states[-1] if outputs.hidden_states else outputs.logits
            deepstack = None
        
        return features, deepstack
    
    def forward(
        self,
        images: List[torch.Tensor],
        instructions: List[str],
        actions: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        robot_state: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference
        
        Args:
            images: List of image frames [B, 3, H, W]
            instructions: List of text instructions
            actions: Ground truth actions for training [B, action_horizon, action_dim]
            timesteps: Diffusion timesteps for training
            robot_state: Current robot proprioception
        
        Returns:
            Dictionary with 'actions' for inference or 'loss' for training
        """
        device = images[0].device if images else robot_state.device
        
        # Encode vision and language with Qwen3-VL
        qwen_features, deepstack_features = self.encode_inputs(images, instructions, device)
        
        if self.training and actions is not None:
            # Training: compute diffusion loss
            B = actions.shape[0]
            
            # Flatten actions
            actions_flat = actions.reshape(B, -1)
            
            # Sample random timesteps
            if timesteps is None:
                timesteps = torch.rand(B, device=device)
            
            # Sample noise
            noise = torch.randn_like(actions_flat)
            
            # Flow matching interpolation
            noisy_actions = timesteps.unsqueeze(-1) * actions_flat + \
                          (1 - timesteps.unsqueeze(-1)) * noise
            
            # Predict velocity field
            velocity_pred = self.action_head(
                qwen_features, noisy_actions, timesteps, 
                robot_state, deepstack_features
            )
            
            # Target velocity
            velocity_target = actions_flat - noise
            
            # Compute loss
            loss = F.mse_loss(velocity_pred, velocity_target)
            
            return {'loss': loss}
        else:
            # Inference: sample actions
            with torch.no_grad():
                actions = self.action_head.sample(
                    qwen_features, robot_state, 
                    deepstack_features=deepstack_features
                )
            
            return {'actions': actions}
    
    def get_action(
        self,
        images: List[np.ndarray],
        instruction: str,
        robot_state: Optional[np.ndarray] = None,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Convenience method for inference
        
        Args:
            images: List of numpy image arrays [H, W, 3]
            instruction: Natural language instruction
            robot_state: Current robot state
            temperature: Sampling temperature
        
        Returns:
            Predicted actions [action_horizon, action_dim]
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Convert images to tensors
        img_tensors = []
        for img in images:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            img_tensors.append(img_tensor)
        
        # Convert robot state
        if robot_state is not None:
            robot_state_tensor = torch.from_numpy(robot_state).float().unsqueeze(0).to(device)
        else:
            robot_state_tensor = None
        
        # Get actions
        with torch.no_grad():
            output = self.forward(
                img_tensors, [instruction],
                robot_state=robot_state_tensor
            )
            actions = output['actions']
        
        return actions.squeeze(0).cpu().numpy()
    
    def unfreeze_backbone(self, num_layers: int = -1):
        """
        Gradually unfreeze Qwen3 backbone layers
        
        Args:
            num_layers: Number of layers to unfreeze from the end (-1 for all)
        """
        if num_layers == -1:
            # Unfreeze everything
            for param in self.qwen_model.parameters():
                param.requires_grad = True
            print("Unfroze all Qwen3 layers")
        else:
            # Unfreeze last N layers
            if hasattr(self.qwen_model, 'model') and hasattr(self.qwen_model.model, 'layers'):
                layers = list(self.qwen_model.model.layers)
                layers_to_unfreeze = layers[-num_layers:] if num_layers > 0 else []
                
                for layer in layers_to_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True
                
                print(f"Unfroze last {num_layers} Qwen3 layers")