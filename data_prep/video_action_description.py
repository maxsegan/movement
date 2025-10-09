#!/usr/bin/env python3
"""
Optimized Video action description module for high-throughput processing.
Key optimizations:
- 8-bit quantization for 75% memory reduction
- Lower resolution processing (224x224 max)
- Aggressive frame sampling (max 4 frames)
- Batch processing across videos
- Smaller, faster model options
"""

import torch
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig
)
from PIL import Image
import logging
from dataclasses import dataclass
import json


@dataclass
class ActionDescription:
    """Container for action descriptions with metadata."""
    frame_indices: List[int]
    description: str
    confidence: float
    person_bbox: Optional[np.ndarray] = None


class FastVideoActionDescriber:
    """
    Optimized VLM for high-throughput video processing.
    """

    # Optimization configurations
    MAX_RESOLUTION = 480  # Higher resolution for better VLM quality
    MAX_FRAMES_VLM = 8  # More frames for better temporal understanding
    BATCH_SIZE = 4  # Process multiple videos at once

    def __init__(
        self,
        device: str = "cuda:0",
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        use_quantization: bool = False,
        use_flash_attention: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize optimized video describer.

        Args:
            device: GPU device
            model_name: VLM model to use
            use_quantization: Enable 8-bit quantization
            use_flash_attention: Use flash attention if available
            cache_dir: Model cache directory
        """
        self.device = device
        self.model_name = model_name
        self.cache_dir = cache_dir or "/root/movement/models/vlm_cache"
        self.logger = logging.getLogger(__name__)

        # Load optimized model
        self._load_optimized_model(use_quantization, use_flash_attention)

    def _load_optimized_model(self, use_quantization: bool, use_flash_attention: bool):
        """Load model with all optimizations."""
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )

            # Quantization config for 8-bit
            quantization_config = None
            if use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.bfloat16,
                    bnb_8bit_use_double_quant=True,
                    bnb_8bit_quant_type="nf4"
                )

            # Load model with optimizations
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16 if not use_quantization else None,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                device_map={"": self.device},  # Explicit device mapping
                attn_implementation="flash_attention_2" if use_flash_attention else "eager"
            )

            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

            self.model.eval()

            # Log model size
            param_count = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Loaded model with {param_count/1e9:.1f}B parameters")

            if use_quantization:
                self.logger.info("Model loaded with 8-bit quantization")

        except Exception as e:
            self.logger.error(f"Failed to load optimized VLM: {e}")
            raise

    def preprocess_frames(
        self,
        frames: List[np.ndarray],
        bboxes: Optional[List[np.ndarray]] = None,
        target_size: int = 480
    ) -> List[Image.Image]:
        """
        Preprocess frames with bounding box visualization.

        Args:
            frames: List of video frames
            bboxes: Optional person bounding boxes for drawing
            target_size: Target resolution

        Returns:
            List of preprocessed PIL images
        """
        processed = []

        for i, frame in enumerate(frames):
            # Draw bounding box on frame if available (don't crop, just annotate)
            if bboxes and i < len(bboxes):
                bbox = bboxes[i]
                if not np.any(np.isnan(bbox)):
                    x1, y1, x2, y2 = bbox.astype(int)
                    frame = frame.copy()
                    # Draw thick red box to clearly mark the person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=4)  # Red in RGB

            # Resize to target resolution
            if frame.shape[0] > target_size or frame.shape[1] > target_size:
                scale = target_size / max(frame.shape[:2])
                new_w = int(frame.shape[1] * scale)
                new_h = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Convert to PIL
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            processed.append(Image.fromarray(frame))

        return processed

    def sample_frames_fast(
        self,
        total_frames: int,
        max_frames: int = 4
    ) -> List[int]:
        """
        Fast frame sampling strategy.

        Args:
            total_frames: Total number of frames
            max_frames: Maximum frames to sample

        Returns:
            List of frame indices to sample
        """
        if total_frames <= max_frames:
            return list(range(total_frames))

        # Sample evenly with preference for beginning, middle, end
        if max_frames >= 3:
            # Always include first, middle, last
            indices = [0, total_frames // 2, total_frames - 1]

            # Add more frames if needed
            if max_frames > 3:
                step = total_frames // (max_frames - 1)
                for i in range(1, max_frames - 2):
                    indices.append(i * step)

            return sorted(list(set(indices)))
        else:
            # Uniform sampling
            step = total_frames // max_frames
            return list(range(0, total_frames, step))[:max_frames]

    def generate_description_batch(
        self,
        video_paths: List[str],
        all_keypoints: List[np.ndarray],
        all_bboxes: List[np.ndarray],
        all_indices: List[np.ndarray]
    ) -> List[ActionDescription]:
        """
        Process multiple videos in a batch for efficiency.

        Args:
            video_paths: List of video paths
            all_keypoints: List of keypoints arrays
            all_bboxes: List of bbox arrays
            all_indices: List of frame indices

        Returns:
            List of action descriptions
        """
        descriptions = []

        # Process in batches
        for batch_start in range(0, len(video_paths), self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, len(video_paths))
            batch_paths = video_paths[batch_start:batch_end]

            batch_images = []
            batch_prompts = []

            for i in range(len(batch_paths)):
                # Load and sample frames
                cap = cv2.VideoCapture(batch_paths[i])
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Fast sampling
                sample_indices = self.sample_frames_fast(
                    min(total_frames, len(all_indices[batch_start + i])),
                    self.MAX_FRAMES_VLM
                )

                frames = []
                bboxes = []

                for idx in sample_indices:
                    if idx < len(all_indices[batch_start + i]):
                        frame_no = all_indices[batch_start + i][idx]
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
                        ret, frame = cap.read()
                        if ret:
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            if idx < len(all_bboxes[batch_start + i]):
                                bboxes.append(all_bboxes[batch_start + i][idx])

                cap.release()

                # Preprocess frames
                if frames:
                    processed = self.preprocess_frames(
                        frames, bboxes, self.MAX_RESOLUTION
                    )
                    batch_images.append(processed)
                    batch_prompts.append(self.get_fast_prompt())

            # Batch inference
            if batch_images:
                try:
                    results = self._batch_inference(batch_images, batch_prompts)
                    for result in results:
                        descriptions.append(ActionDescription(
                            frame_indices=sample_indices,
                            description=result,
                            confidence=0.8  # Fixed confidence for speed
                        ))
                except Exception as e:
                    self.logger.error(f"Batch inference failed: {e}")
                    # Fallback to empty descriptions
                    for _ in range(len(batch_images)):
                        descriptions.append(ActionDescription(
                            frame_indices=[],
                            description="",
                            confidence=0.0
                        ))

        return descriptions

    def _batch_inference(
        self,
        batch_images: List[List[Image.Image]],
        batch_prompts: List[str]
    ) -> List[str]:
        """
        Run batch inference on multiple videos.

        Args:
            batch_images: List of image lists for each video
            batch_prompts: List of prompts

        Returns:
            List of generated descriptions
        """
        results = []

        with torch.no_grad():
            for images, prompt in zip(batch_images, batch_prompts):
                try:
                    # Create input with images for Qwen2.5-VL
                    content = []
                    for img in images:
                        content.append({"type": "image", "image": img})
                    content.append({"type": "text", "text": prompt})

                    messages = [
                        {"role": "user", "content": content}
                    ]

                    # Process with model
                    text = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    inputs = self.processor(
                        text=text,
                        images=images,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)

                    # Generate with constraints
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,  # Allow detailed descriptions, stops early on EOS
                        do_sample=False,  # Greedy for speed
                        num_beams=1  # No beam search for speed
                    )

                    # Decode
                    response = self.processor.decode(
                        outputs[0],
                        skip_special_tokens=True
                    )

                    # Extract description
                    if "assistant" in response:
                        description = response.split("assistant")[-1].strip()
                    else:
                        description = response.strip()

                    self.logger.info(f"VLM generated description ({len(description)} chars): {description[:100]}...")
                    results.append(description)  # Return full description

                except Exception as e:
                    self.logger.error(f"Single inference failed: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    results.append("")

        return results

    def get_fast_prompt(self) -> str:
        """Get optimized prompt for fast inference with directive instructions."""
        return """Describe the actions of the person featured in the box prescriptively in at most a few clear and concise sentences such that somebody could recreate their motion closely based on the description. For example, "Carefully stack the four plates, lifting them one by one starting with the plate on the right" or "Powerfully swing the hammer down on the large tire repeatedly with a wide athletic stance"."""

    def process_video_fast(
        self,
        video_path: str,
        keypoints: np.ndarray,
        bboxes: np.ndarray,
        indices: np.ndarray
    ) -> List[ActionDescription]:
        """
        Fast single video processing.

        Args:
            video_path: Path to video
            keypoints: 2D keypoints
            bboxes: Bounding boxes
            indices: Frame indices

        Returns:
            List with single action description
        """
        return self.generate_description_batch(
            [video_path], [keypoints], [bboxes], [indices]
        )

    def process_video_with_tracking(
        self,
        video_path: str,
        keypoints: np.ndarray,
        bboxes: np.ndarray,
        indices: np.ndarray,
        segment_duration: float = 5.0,
        fps: float = 10.0
    ) -> List[ActionDescription]:
        """
        Process video with tracking data (compatible interface).
        Ignores segment_duration and fps parameters - processes entire video.

        Args:
            video_path: Path to video
            keypoints: 2D keypoints
            bboxes: Bounding boxes
            indices: Frame indices
            segment_duration: Ignored (for compatibility)
            fps: Ignored (for compatibility)

        Returns:
            List with single action description
        """
        return self.generate_description_batch(
            [video_path], [keypoints], [bboxes], [indices]
        )


def process_videos_parallel(
    video_list: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    gpu_ids: List[int] = [0, 1, 2, 3],
    batch_size: int = 4
) -> List[List[ActionDescription]]:
    """
    Process videos in parallel across multiple GPUs.

    Args:
        video_list: List of (video_path, keypoints, bboxes, indices) tuples
        gpu_ids: GPU IDs to use
        batch_size: Batch size per GPU

    Returns:
        List of descriptions for each video
    """
    from multiprocessing import Pool, Queue
    import multiprocessing as mp

    # Split videos across GPUs
    videos_per_gpu = len(video_list) // len(gpu_ids)
    gpu_batches = []

    for i, gpu_id in enumerate(gpu_ids):
        start = i * videos_per_gpu
        end = start + videos_per_gpu if i < len(gpu_ids) - 1 else len(video_list)
        gpu_batches.append((gpu_id, video_list[start:end]))

    # Process on each GPU
    def process_on_gpu(args):
        gpu_id, videos = args
        device = f"cuda:{gpu_id}"

        # Create describer for this GPU
        describer = FastVideoActionDescriber(
            device=device,
            use_quantization=True,
            use_flash_attention=False  # Disable if not available
        )

        # Process videos
        all_descriptions = []
        paths = [v[0] for v in videos]
        keypoints = [v[1] for v in videos]
        bboxes = [v[2] for v in videos]
        indices = [v[3] for v in videos]

        # Process in batches
        for i in range(0, len(videos), batch_size):
            batch_end = min(i + batch_size, len(videos))
            batch_desc = describer.generate_description_batch(
                paths[i:batch_end],
                keypoints[i:batch_end],
                bboxes[i:batch_end],
                indices[i:batch_end]
            )
            all_descriptions.extend(batch_desc)

        return all_descriptions

    # Run in parallel
    with Pool(len(gpu_ids)) as pool:
        results = pool.map(process_on_gpu, gpu_batches)

    # Flatten results
    all_results = []
    for gpu_results in results:
        all_results.extend(gpu_results)

    return all_results


if __name__ == "__main__":
    # Test the optimized VLM
    import time

    print("Testing optimized VLM...")

    # Create test data
    test_video = "/root/movement/data/kinetics-dataset/k700-2020/train/acting in play/0bdVrgImymc_000020_000030.mp4"
    test_keypoints = np.random.randn(50, 17, 2).astype(np.float32)
    test_bboxes = np.array([[100, 100, 200, 200]] * 50, dtype=np.float32)
    test_indices = np.arange(50)

    # Single GPU test
    describer = FastVideoActionDescriber(
        device="cuda:0",
        use_quantization=True
    )

    start = time.time()
    descriptions = describer.process_video_fast(
        test_video,
        test_keypoints,
        test_bboxes,
        test_indices
    )
    elapsed = time.time() - start

    print(f"Processed 1 video in {elapsed:.2f} seconds")
    if descriptions:
        print(f"Description: {descriptions[0].description}")

    # Estimate throughput
    videos_per_minute = 60 / elapsed
    print(f"Estimated throughput: {videos_per_minute:.1f} videos/minute per GPU")
    print(f"With 4 GPUs: {videos_per_minute * 4:.1f} videos/minute")
    print(f"Time for 800k videos: {800000 / (videos_per_minute * 4 * 60):.1f} hours")
