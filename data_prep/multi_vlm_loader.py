#!/usr/bin/env python3
"""
Multi-VLM loader for Qwen3-VL, CogVLM2, and other video understanding models.
"""

import torch
import logging
from typing import Dict, Tuple, Optional, Any
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig
)


def load_vlm_model(
    config: Dict,
    device: str = "cuda:0",
    cache_dir: str = "/root/movement/models/vlm_cache"
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Load a VLM model with 4-bit quantization support.

    Args:
        config: Model configuration dict with 'name', 'model_id', 'use_4bit'
        device: Device to load model on
        cache_dir: Cache directory for models

    Returns:
        Tuple of (model, processor) or (None, None) if loading fails
    """
    logger = logging.getLogger(__name__)

    try:
        # Handle different model types
        model_id = config["model_id"]

        # Check if model is pre-quantized (FP8, A3B, etc.)
        is_prequantized = "fp8" in model_id.lower() or "a3b" in model_id.lower()

        # Quantization config - skip for pre-quantized models
        quantization_config = None
        if config.get("use_4bit", False) and not is_prequantized:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        # Qwen VL models (both Qwen2.5-VL and Qwen3-VL)
        if "qwen" in model_id.lower() and "vl" in model_id.lower():
            logger.info(f"Loading Qwen VL model: {model_id}")

            # Determine dtype - auto for pre-quantized or 4-bit, float16 otherwise
            if is_prequantized or config.get("use_4bit"):
                torch_dtype = "auto"  # Let model use its native dtype
            else:
                torch_dtype = torch.float16  # Use float16 for consistency

            # Try to import Qwen3-VL specific class first
            try:
                if "qwen3-vl" in model_id.lower():
                    from transformers import Qwen3VLForConditionalGeneration
                    model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_id,
                        quantization_config=quantization_config,
                        torch_dtype=torch_dtype,
                        device_map=device if isinstance(device, str) else device,
                        trust_remote_code=True,
                        cache_dir=cache_dir
                    )
                else:
                    # Qwen2.5-VL uses AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=quantization_config,
                        torch_dtype=torch_dtype,
                        device_map=device if isinstance(device, str) else device,
                        trust_remote_code=True,
                        cache_dir=cache_dir
                    )
            except ImportError:
                # Fallback to generic loading
                logger.warning(f"Qwen3VLForConditionalGeneration not available, using AutoModelForCausalLM")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    torch_dtype=torch_dtype,
                    device_map=device if isinstance(device, str) else device,
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )

            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            model.eval()
            return model, processor

        # CogVLM2-Video
        elif "cogvlm" in model_id.lower():
            logger.info(f"Loading CogVLM2 model: {model_id}")

            # Determine dtype - CogVLM2 works best with float16
            if config.get("use_4bit"):
                torch_dtype = "auto"
            else:
                torch_dtype = torch.float16

            # CogVLM2 uses AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map=device if isinstance(device, str) else device,
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            model.eval()
            return model, tokenizer

        # InternVL2
        elif "internvl" in model_id.lower():
            logger.info(f"Loading InternVL2 model: {model_id}")

            from transformers import AutoModel

            # InternVL2 works with float16
            torch_dtype = "auto" if config.get("use_4bit") else torch.float16

            model = AutoModel.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map=device if isinstance(device, str) else device,
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            model.eval()
            return model, tokenizer

        # Video-LLaVA
        elif "video-llava" in model_id.lower():
            logger.info(f"Loading Video-LLaVA model: {model_id}")

            try:
                # Try importing Video-LLaVA specific classes
                from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

                # Video-LLaVA needs float16, not bfloat16
                if config.get("use_4bit"):
                    torch_dtype = "auto"
                else:
                    torch_dtype = torch.float16  # Use float16 for Video-LLaVA

                model = VideoLlavaForConditionalGeneration.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    cache_dir=cache_dir
                )

                processor = VideoLlavaProcessor.from_pretrained(
                    model_id,
                    cache_dir=cache_dir
                )

                model.eval()
                return model, processor
            except ImportError:
                # Fallback if Video-LLaVA classes not available
                logger.warning("Video-LLaVA specific classes not found, trying generic loading")

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    torch_dtype="auto" if (is_prequantized or config.get("use_4bit")) else torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )

                processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )

                model.eval()
                return model, processor

        # Default fallback
        else:
            logger.info(f"Loading generic model: {model_id}")

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16 if not config.get("use_4bit") else "auto",
                device_map="auto",
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            model.eval()
            return model, processor

    except Exception as e:
        logger.error(f"Failed to load {config['name']}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def generate_video_description_qwen_vl(
    model,
    processor,
    frames,
    prompt: str = "Describe the human action in this video."
) -> str:
    """
    Generate description using Qwen VL models (Qwen2.5-VL or Qwen3-VL).
    """
    try:
        # Prepare messages in Qwen VL format (same for both 2.5 and 3)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[{"type": "image", "image": frame} for frame in frames]
                ]
            }
        ]

        # Process inputs
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=frames,
            videos=None,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        # Decode
        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Remove the prompt from output
        if prompt in output_text:
            output_text = output_text.split(prompt)[-1].strip()

        return output_text

    except Exception as e:
        logging.error(f"Error generating with Qwen VL: {e}")
        return f"Error: {str(e)}"


def generate_video_description_cogvlm2(
    model,
    tokenizer,
    frames,
    prompt: str = "Describe the human action in this video."
) -> str:
    """
    Generate description using CogVLM2-Video model.
    """
    try:
        from PIL import Image

        # CogVLM2 expects a specific format
        # Combine frames into a video-like input
        query = f"<Video>{prompt}</Video>"

        # Create inputs
        inputs = tokenizer(
            query,
            return_tensors='pt',
            padding=True
        ).to(model.device)

        # Add image tokens for video frames
        # CogVLM2-Video processes frames as a sequence

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                images=frames,  # Pass frames directly
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean response
        if prompt in response:
            response = response.split(prompt)[-1].strip()

        return response

    except Exception as e:
        logging.error(f"Error generating with CogVLM2: {e}")
        return f"Error: {str(e)}"


def generate_video_description_internvl2(
    model,
    tokenizer,
    frames,
    prompt: str = "Describe the human action in this video."
) -> str:
    """
    Generate description using InternVL2 model.
    """
    try:
        # InternVL2 format
        # Combine multiple frames
        num_frames = len(frames)

        # Create placeholders for images
        image_placeholders = "".join([f"<image>" for _ in range(num_frames)])
        query = f"{image_placeholders}\n{prompt}"

        # Process
        inputs = tokenizer(
            query,
            return_tensors='pt'
        ).to(model.device)

        # Generate with frames
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                images=frames,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up
        if prompt in response:
            response = response.split(prompt)[-1].strip()

        return response

    except Exception as e:
        logging.error(f"Error generating with InternVL2: {e}")
        return f"Error: {str(e)}"


def generate_video_description_video_llava(
    model,
    processor,
    frames,
    prompt: str = "Describe the human action in this video."
) -> str:
    """
    Generate description using Video-LLaVA model.
    """
    try:
        # Video-LLaVA expects USER: <video>\n{prompt} ASSISTANT: format
        # The processor handles the video token insertion

        # Create the conversation format Video-LLaVA expects
        text_prompt = f"USER: <video>\n{prompt} ASSISTANT:"

        # Process inputs - Video-LLaVA processor handles video frames
        inputs = processor(
            text=text_prompt,
            videos=frames,  # Pass as videos, not images
            return_tensors="pt",
            padding=True
        )

        # Move inputs to device and ensure correct dtype
        # Convert to float16 to match model dtype
        for key in inputs:
            if hasattr(inputs[key], 'to'):
                if inputs[key].dtype in [torch.bfloat16, torch.float32]:
                    inputs[key] = inputs[key].to(model.device).to(torch.float16)
                else:
                    inputs[key] = inputs[key].to(model.device)

        # Generate
        with torch.no_grad():
            # Set model to use float16 for generation
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                generate_ids = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )

        # Decode response
        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Extract only the assistant's response
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        elif prompt in response:
            response = response.split(prompt)[-1].strip()

        return response

    except Exception as e:
        logging.error(f"Error generating with Video-LLaVA: {e}")
        return f"Error: {str(e)}"


def generate_description_with_model(
    model_name: str,
    model,
    processor_or_tokenizer,
    frames,
    prompt: str = "Describe the human action or activity shown in this video clip."
) -> str:
    """
    Route to appropriate generation function based on model type.
    """
    if "qwen" in model_name.lower() and "vl" in model_name.lower():
        return generate_video_description_qwen_vl(model, processor_or_tokenizer, frames, prompt)
    elif "cogvlm" in model_name.lower():
        return generate_video_description_cogvlm2(model, processor_or_tokenizer, frames, prompt)
    elif "internvl" in model_name.lower():
        return generate_video_description_internvl2(model, processor_or_tokenizer, frames, prompt)
    elif "video-llava" in model_name.lower():
        return generate_video_description_video_llava(model, processor_or_tokenizer, frames, prompt)
    else:
        # Generic generation fallback
        try:
            logging.warning(f"Using generic generation for {model_name}")

            if hasattr(processor_or_tokenizer, 'apply_chat_template'):
                # Has processor
                inputs = processor_or_tokenizer(
                    text=prompt,
                    images=frames,
                    return_tensors="pt"
                ).to(model.device)
            else:
                # Has tokenizer
                inputs = processor_or_tokenizer(
                    prompt,
                    return_tensors="pt"
                ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True
                )

            if hasattr(processor_or_tokenizer, 'batch_decode'):
                response = processor_or_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            else:
                response = processor_or_tokenizer.decode(outputs[0], skip_special_tokens=True)

            if prompt in response:
                response = response.split(prompt)[-1].strip()

            return response

        except Exception as e:
            logging.error(f"Error in generic generation: {e}")
            return f"Error: {str(e)}"
