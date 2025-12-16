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
    hidden_layer_index: int = 18  # Which layer to extract (18 = middle of 36 layers, like GR00T)
    use_early_exit: bool = True  # Stop forward pass after hidden_layer_index for 2x speedup
    
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
    num_future_tokens: int = 4  # Learnable future tokens for planning context (like GR00T)
    
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


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN) for timestep conditioning.
    Modulates layer norm scale and shift based on conditioning input.
    Used in DiT (Diffusion Transformer) architectures.
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # Project conditioning to scale and shift
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2)
        )
        # Initialize to identity transform
        nn.init.zeros_(self.cond_proj[-1].weight)
        nn.init.zeros_(self.cond_proj[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D] or [B, D]
            cond: Conditioning tensor [B, cond_dim]
        Returns:
            Normalized and modulated tensor
        """
        # Get scale and shift from conditioning
        scale_shift = self.cond_proj(cond)  # [B, 2*D]
        scale, shift = scale_shift.chunk(2, dim=-1)  # [B, D] each

        # Normalize
        x = self.norm(x)

        # Apply modulation (broadcast over sequence dim if needed)
        if x.dim() == 3:
            scale = scale.unsqueeze(1)  # [B, 1, D]
            shift = shift.unsqueeze(1)  # [B, 1, D]

        return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with AdaLN conditioning.
    Interleaves self-attention over [state, actions] with cross-attention to VL features.
    Following GR00T N1 architecture.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # AdaLN for self-attention
        self.norm1 = AdaLN(hidden_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # AdaLN for cross-attention
        self.norm2 = AdaLN(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # AdaLN for FFN
        self.norm3 = AdaLN(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        vl_features: torch.Tensor,
        t_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Sequence of [state_token, action_tokens] [B, 1+H, D]
            vl_features: Vision-language features [B, L, D]
            t_emb: Timestep embedding [B, D]
        Returns:
            Updated sequence [B, 1+H, D]
        """
        # Self-attention over state+action sequence with AdaLN
        residual = x
        x = self.norm1(x, t_emb)
        x, _ = self.self_attn(x, x, x)
        x = residual + x

        # Cross-attention to VL features with AdaLN
        residual = x
        x = self.norm2(x, t_emb)
        x, _ = self.cross_attn(x, vl_features, vl_features)
        x = residual + x

        # FFN with AdaLN
        residual = x
        x = self.norm3(x, t_emb)
        x = self.ffn(x)
        x = residual + x

        return x


class FlowMatchingDiffusion(nn.Module):
    """
    Flow Matching Diffusion Head with DiT architecture.

    Key improvements over previous version:
    1. Per-timestep action encoding with positional embeddings
    2. State token as separate token in the sequence
    3. AdaLN for timestep conditioning throughout
    4. DiT-style interleaved self-attention and cross-attention

    Architecture follows GR00T N1: https://github.com/NVIDIA/Isaac-GR00T
    """

    def __init__(self, config: VLAConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.diffusion_hidden_dim

        # Time embedding with MLP
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Project from Qwen3 hidden states
        self.qwen_projection = nn.Sequential(
            nn.Linear(config.qwen_hidden_size, config.projection_dim),
            nn.LayerNorm(config.projection_dim),
            nn.SiLU(),
            nn.Linear(config.projection_dim, hidden_dim)
        )

        # DeepStack features projection
        if config.use_deepstack_features:
            self.deepstack_projection = nn.Linear(
                config.qwen_hidden_size * 2,
                hidden_dim
            )

        # Per-timestep action encoder (NOT flattened!)
        # Each action timestep is encoded separately
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Learnable positional embeddings for action horizon
        self.action_pos_embed = nn.Parameter(
            torch.randn(1, config.action_horizon, hidden_dim) * 0.02
        )

        # Proprioception/state encoder -> produces state TOKEN
        self.state_encoder = nn.Sequential(
            nn.Linear(config.action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable future tokens for planning context (like GR00T)
        # These provide additional learnable context between state and actions
        self.num_future_tokens = config.num_future_tokens
        if self.num_future_tokens > 0:
            self.future_tokens = nn.Parameter(
                torch.randn(1, config.num_future_tokens, hidden_dim) * 0.02
            )

        # DiT blocks with AdaLN
        self.dit_blocks = nn.ModuleList([
            DiTBlock(hidden_dim, config.num_diffusion_heads, config.diffusion_dropout)
            for _ in range(config.num_diffusion_layers)
        ])

        # Final layer norm with AdaLN
        self.final_norm = AdaLN(hidden_dim, hidden_dim)

        # Per-timestep action decoder (NOT flattened!)
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, config.action_dim),
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
        Forward pass for flow matching.

        Args:
            qwen_features: Hidden states from Qwen3-VL [B, L, D_qwen]
            noisy_actions: Noisy action sequence [B, action_horizon, action_dim]
            timesteps: Diffusion timesteps [B]
            robot_state: Current robot proprioception [B, action_dim]
            deepstack_features: Optional multi-level features [B, L, 2*D_qwen]

        Returns:
            Predicted velocity field [B, action_horizon, action_dim]
        """
        B = qwen_features.shape[0]
        device = qwen_features.device
        H = self.config.action_horizon

        # Time embedding
        t_emb = self.time_embed(timesteps)  # [B, D]

        # Project VL features
        vl_features = self.qwen_projection(qwen_features)  # [B, L, D]

        # Add DeepStack features if available
        if deepstack_features is not None and self.config.use_deepstack_features:
            pooled_deep = deepstack_features.mean(dim=1)  # [B, 2*D_qwen]
            deep_features = self.deepstack_projection(pooled_deep)  # [B, D]
            vl_features = vl_features + deep_features.unsqueeze(1)

        # Encode noisy actions PER TIMESTEP
        # noisy_actions: [B, H, action_dim] -> action_tokens: [B, H, D]
        action_tokens = self.action_encoder(noisy_actions)  # [B, H, D]

        # Add positional embeddings to action tokens
        action_tokens = action_tokens + self.action_pos_embed  # [B, H, D]

        # Add timestep embedding to action tokens
        action_tokens = action_tokens + t_emb.unsqueeze(1)  # [B, H, D]

        # Encode robot state as a separate STATE TOKEN
        if robot_state is not None:
            state_token = self.state_encoder(robot_state)  # [B, D]
            state_token = state_token.unsqueeze(1)  # [B, 1, D]
        else:
            # If no state provided, use zeros
            state_token = torch.zeros(B, 1, self.config.diffusion_hidden_dim, device=device)

        # Build sequence: [state_token, future_tokens, action_tokens] (like GR00T)
        # Future tokens provide learnable planning context
        if self.num_future_tokens > 0:
            # Expand future tokens for batch
            future_tokens = self.future_tokens.expand(B, -1, -1)  # [B, num_future, D]
            sequence = torch.cat([state_token, future_tokens, action_tokens], dim=1)  # [B, 1+F+H, D]
        else:
            sequence = torch.cat([state_token, action_tokens], dim=1)  # [B, 1+H, D]

        # Apply DiT blocks with interleaved self-attention and cross-attention
        for block in self.dit_blocks:
            sequence = block(sequence, vl_features, t_emb)

        # Final normalization
        sequence = self.final_norm(sequence, t_emb)

        # Extract action tokens (skip state token and future tokens)
        # Sequence is [state(1), future(F), actions(H)] -> extract actions
        skip_tokens = 1 + self.num_future_tokens  # Skip state + future tokens
        action_output = sequence[:, skip_tokens:, :]  # [B, H, D]

        # Decode to velocity per timestep
        velocity = self.action_decoder(action_output)  # [B, H, action_dim]

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
        Sample actions using flow matching ODE solver.

        Args:
            qwen_features: Hidden states from Qwen3-VL [B, L, D]
            robot_state: Current robot state [B, action_dim]
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
        dtype = qwen_features.dtype

        # Initialize from noise [B, H, action_dim]
        x = torch.randn(
            B,
            self.config.action_horizon,
            self.config.action_dim,
            device=device,
            dtype=dtype
        ) * temperature

        # ODE integration
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)

            # Predict velocity field
            with torch.cuda.amp.autocast(enabled=True):
                v = self.forward(qwen_features, x, t, robot_state, deepstack_features)

            # Euler integration
            x = x + v * dt

        return x


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
            "device_map": None,  # We'll handle device placement
            "local_files_only": True,  # Use cached files only to avoid DDP race conditions
        }
        
        # Add flash attention if supported
        if config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        try:
            self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.qwen_model_name,
                **model_kwargs
            )
        except ImportError as e:
            if config.use_flash_attention:
                print("FlashAttention2 not available; retrying without it.")
                model_kwargs.pop("attn_implementation", None)
                self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    config.qwen_model_name,
                    **model_kwargs
                )
            else:
                raise e
        
        # Load processor for preprocessing
        self.processor = AutoProcessor.from_pretrained(config.qwen_model_name, local_files_only=True)
        
        # Apply LoRA if specified
        if config.use_lora:
            self._apply_lora()
        
        # Freeze layers as specified
        self._freeze_layers()
        
        # Diffusion action head
        self.action_head = FlowMatchingDiffusion(config)
        
        # Truncate layers if using early exit (must be before hooks)
        if config.use_early_exit:
            self._truncate_layers()

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
    
    def _get_language_model(self):
        """Get reference to language model layers, handling PEFT wrapping"""
        if hasattr(self.qwen_model, 'model') and hasattr(self.qwen_model.model, 'language_model'):
            return self.qwen_model.model.language_model
        elif hasattr(self.qwen_model, 'base_model') and hasattr(self.qwen_model.base_model, 'model'):
            # Handle PEFT wrapped model
            return self.qwen_model.base_model.model.model.language_model
        else:
            return None

    def _truncate_layers(self):
        """Remove layers after hidden_layer_index for early exit (2x speedup)"""
        lm = self._get_language_model()
        if lm is None:
            print("Warning: Could not find language model layers for truncation")
            return

        num_layers = len(lm.layers)
        target_idx = self.config.hidden_layer_index
        if target_idx < 0:
            target_idx = num_layers + target_idx

        # Keep layers 0 to target_idx (inclusive), remove the rest
        keep_layers = target_idx + 1
        if keep_layers < num_layers:
            # Truncate the ModuleList
            lm.layers = lm.layers[:keep_layers]
            removed = num_layers - keep_layers
            print(f"Truncated LLM: kept {keep_layers} layers, removed {removed} layers ({removed/num_layers*100:.0f}% reduction)")

    def _register_hooks(self):
        """Register forward hooks to extract intermediate features"""
        lm = self._get_language_model()
        if lm is None:
            print("Warning: Could not find language model layers for hooks")
            return

        num_layers = len(lm.layers)
        # After truncation, the last layer is our target
        target_idx = num_layers - 1

        # For deepstack: also hook an earlier layer
        early_layer_idx = target_idx // 2

        self.early_features = None

        def early_hook_fn(module, input, output):
            """Capture features from early layer for deepstack"""
            if isinstance(output, tuple):
                self.early_features = output[0]
            else:
                self.early_features = output

        def target_hook_fn(module, input, output):
            """Capture features from final (target) layer"""
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            self.intermediate_features = hidden_states

            # Build deepstack features if enabled
            if self.config.use_deepstack_features and self.early_features is not None:
                self.deepstack_features = torch.cat([self.early_features, hidden_states], dim=-1)
            else:
                self.deepstack_features = None

        # Register hooks
        if self.config.use_deepstack_features:
            lm.layers[early_layer_idx].register_forward_hook(early_hook_fn)
            print(f"Registered deepstack hook on layer {early_layer_idx}")

        lm.layers[target_idx].register_forward_hook(target_hook_fn)
        print(f"Registered output hook on layer {target_idx}")
    
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
        
        # Prepare chat messages and images per sample
        messages_batch = []
        image_inputs = []
        for frames, instruction in zip(images, instructions):
            content = [{"type": "image", "image": frame} for frame in frames]
            content.append({"type": "text", "text": instruction})
            messages_batch.append([{"role": "user", "content": content}])
            image_inputs.append(frames)
        
        # Apply chat template to each sample
        texts = self.processor.apply_chat_template(
            messages_batch,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process inputs (Qwen3-VL handles any resolution)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Forward pass through Qwen3 (layers already truncated if use_early_exit=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.qwen_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # Extract features from hooks
        if self.config.use_intermediate_hidden and self.intermediate_features is not None:
            features = self.intermediate_features
            deepstack = self.deepstack_features
        else:
            features = outputs.hidden_states[-1] if outputs.hidden_states else outputs.logits
            deepstack = None

        return features, deepstack
    
    def forward(
        self,
        images: List[torch.Tensor],
        instructions: List[str],
        actions: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        robot_state: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
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
        device = next(self.parameters()).device
        
        # Encode vision and language with Qwen3-VL
        qwen_features, deepstack_features = self.encode_inputs(images, instructions, device)
        
        if compute_loss and actions is not None:
            # Training: compute diffusion loss
            B = actions.shape[0]

            # actions shape: [B, action_horizon, action_dim] - keep this shape!

            # Sample random timesteps
            if timesteps is None:
                timesteps = torch.rand(B, device=device)

            # Sample noise with same shape as actions
            noise = torch.randn_like(actions)  # [B, H, action_dim]

            # Flow matching interpolation (broadcast timesteps over H and action_dim)
            t_expanded = timesteps.view(B, 1, 1)  # [B, 1, 1]
            noisy_actions = t_expanded * actions + (1 - t_expanded) * noise

            # Predict velocity field [B, H, action_dim]
            velocity_pred = self.action_head(
                qwen_features, noisy_actions, timesteps,
                robot_state, deepstack_features
            )

            # Target velocity [B, H, action_dim]
            velocity_target = actions - noise

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
