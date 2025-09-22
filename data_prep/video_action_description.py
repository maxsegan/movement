#!/usr/bin/env python3
"""
Video action description module using Video-Language Models (VLMs).
Generates directive-style descriptions of tracked person's actions.
"""

import torch
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import av
from PIL import Image
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import queue
import threading


@dataclass
class ActionDescription:
    """Container for action descriptions with metadata."""
    frame_indices: List[int]
    description: str
    confidence: float
    person_bbox: Optional[np.ndarray] = None


class VideoActionDescriber:
    """
    Generates action descriptions for tracked persons using VLMs.
    Optimized for multi-GPU processing without redundant model loading.
    """

    # Class-level model pool for GPU distribution
    _model_pool = {}
    _model_lock = threading.Lock()

    def __init__(
        self,
        device: str = "cuda:0",
        model_name: str = "microsoft/Phi-3.5-vision-instruct",
        batch_size: int = 1,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the video action describer.

        Args:
            device: Device to run the model on (cuda:0, cuda:1, etc.)
            model_name: Name of the VLM to use
            batch_size: Batch size for processing
            cache_dir: Optional cache directory for models
        """
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir or "/root/movement/models/vlm_cache"
        self.logger = logging.getLogger(__name__)

        # Ensure model is loaded for this device
        self._load_model_for_instance()

    @classmethod
    def _ensure_model_loaded(cls, device: str, model_name: str, cache_dir: str):
        """Ensure model is loaded on the specified device (singleton per GPU)."""
        with cls._model_lock:
            if device not in cls._model_pool:
                cls._model_pool[device] = cls._load_model(device, model_name, cache_dir)
        return cls._model_pool[device]

    @staticmethod
    def _load_model(device: str, model_name: str, cache_dir: str):
        """Load the VLM model and processor."""
        try:
            from transformers import AutoConfig

            # Use Phi-3.5-vision as it's good at video understanding and instruction following
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            )

            # Load config and set attention implementation to avoid FlashAttention2 requirement
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            config._attn_implementation = 'eager'  # Use eager attention instead of flash_attn

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float16 if "cuda" in device else torch.float32,
                trust_remote_code=True,
                cache_dir=cache_dir,
                device_map=device
            )

            model.eval()

            return {
                'processor': processor,
                'model': model,
                'tokenizer': processor.tokenizer if hasattr(processor, 'tokenizer') else None
            }
        except Exception as e:
            logging.error(f"Failed to load VLM model: {e}")
            raise

    def _load_model_for_instance(self):
        """Ensure model is loaded for this instance's device."""
        model_info = self.__class__._ensure_model_loaded(
            self.device, self.model_name, self.cache_dir
        )
        self.processor = model_info['processor']
        self.model = model_info['model']
        self.tokenizer = model_info['tokenizer']

    def extract_person_region(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        padding: float = 0.2
    ) -> np.ndarray:
        """
        Extract and crop the region around the tracked person.

        Args:
            frame: Full video frame (H, W, 3)
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Padding ratio around the bbox

        Returns:
            Cropped image focused on the person
        """
        if bbox is None or np.any(np.isnan(bbox)):
            return frame

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)

        # Add padding
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        return frame[y1:y2, x1:x2]

    def sample_frames_for_description(
        self,
        frames: np.ndarray,
        bboxes: np.ndarray,
        indices: np.ndarray,
        max_frames: int = 8
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Sample representative frames for action description.

        Args:
            frames: Video frames (T, H, W, 3)
            bboxes: Person bounding boxes (T, 4)
            indices: Original frame indices
            max_frames: Maximum frames to sample

        Returns:
            Sampled frames and their indices
        """
        num_frames = len(frames)

        if num_frames <= max_frames:
            frame_indices = list(range(num_frames))
        else:
            # Uniform sampling with preference for frames with valid bboxes
            valid_bbox_frames = np.where(~np.any(np.isnan(bboxes), axis=1))[0]

            if len(valid_bbox_frames) >= max_frames:
                # Sample from frames with valid bboxes
                step = len(valid_bbox_frames) // max_frames
                frame_indices = valid_bbox_frames[::step][:max_frames]
            else:
                # Uniform sampling across all frames
                step = num_frames // max_frames
                frame_indices = list(range(0, num_frames, step))[:max_frames]

        sampled_frames = []
        for idx in frame_indices:
            if idx < len(frames):
                frame = frames[idx]
                bbox = bboxes[idx] if idx < len(bboxes) else None

                # Crop to person region if bbox is available
                if bbox is not None and not np.any(np.isnan(bbox)):
                    frame = self.extract_person_region(frame, bbox)

                sampled_frames.append(frame)

        return sampled_frames, [indices[i] for i in frame_indices if i < len(indices)]

    def generate_action_prompt(self, has_person: bool = True) -> str:
        """
        Generate the prompt for action description.

        Args:
            has_person: Whether a tracked person is visible

        Returns:
            Formatted prompt for the VLM
        """
        if has_person:
            prompt = """You are a movie director providing detailed action instructions to an actor.

Watch this video clip and describe ONLY what the main tracked person (highlighted/centered) is doing.
Ignore other people in the scene.

Provide a clear, directive instruction as if you were directing them to perform this exact action.
Use imperative mood (command form) and be specific about the movement details.

Examples:
- "Dice the onion with quick, precise chopping motions while keeping your fingers curled"
- "Serve the tennis ball with a high toss and powerful overhead swing"
- "Walk forward with confident strides while swinging your arms naturally"

Now describe the action in the video as a directive to the tracked person:"""
        else:
            prompt = "Describe what action should be performed in this scene:"

        return prompt

    def describe_action(
        self,
        frames: np.ndarray,
        bboxes: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None
    ) -> ActionDescription:
        """
        Generate action description for video frames.

        Args:
            frames: Video frames (T, H, W, 3)
            bboxes: Person bounding boxes (T, 4) or None
            indices: Original frame indices

        Returns:
            ActionDescription object with the generated description
        """
        if frames is None or len(frames) == 0:
            return ActionDescription(
                frame_indices=[],
                description="No frames available",
                confidence=0.0
            )

        # Default values if not provided
        if indices is None:
            indices = np.arange(len(frames))
        if bboxes is None:
            bboxes = np.full((len(frames), 4), np.nan)

        try:
            # Sample frames for description
            sampled_frames, sampled_indices = self.sample_frames_for_description(
                frames, bboxes, indices
            )

            # Convert frames to PIL Images
            pil_images = []
            for frame in sampled_frames:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(frame))

            # Check if we have a tracked person
            has_person = np.any(~np.isnan(bboxes))

            # Generate prompt
            prompt = self.generate_action_prompt(has_person)

            # Prepare input for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ] + [
                        {"type": "image", "image": img} for img in pil_images
                    ]
                }
            ]

            # Process with model
            # For Phi-3.5-vision, format the prompt with image placeholders
            num_images = len(pil_images)
            image_tags = "".join([f"<|image_{i+1}|>" for i in range(num_images)])
            text_prompt = messages[0]['content'][0]['text']
            prompt = f"<|user|>\n{image_tags}\n{text_prompt}\n<|end|>\n<|assistant|>\n"

            inputs = self.processor(
                text=prompt,
                images=pil_images,
                return_tensors="pt"
            ).to(self.device)

            # Generate description
            with torch.no_grad():
                # Disable cache to avoid DynamicCache errors
                generation_args = {
                    **inputs,
                    'max_new_tokens': 150,
                    'do_sample': False,  # Disable sampling to avoid cache issues
                    'use_cache': False,  # Explicitly disable cache
                }

                outputs = self.model.generate(**generation_args)

            # Decode output
            generated_text = self.processor.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Calculate confidence based on model's output (simplified)
            confidence = 0.85 if has_person else 0.5

            return ActionDescription(
                frame_indices=sampled_indices.tolist() if isinstance(sampled_indices, np.ndarray) else sampled_indices,
                description=generated_text,
                confidence=float(confidence),
                person_bbox=bboxes[len(bboxes)//2].tolist() if has_person and isinstance(bboxes[len(bboxes)//2], np.ndarray) else (bboxes[len(bboxes)//2] if has_person else None)
            )

        except Exception as e:
            self.logger.error(f"Error generating action description: {e}")
            return ActionDescription(
                frame_indices=indices.tolist() if isinstance(indices, np.ndarray) else indices,
                description=f"Error: {str(e)}",
                confidence=0.0
            )

    def process_video_with_tracking(
        self,
        video_path: str,
        keypoints: np.ndarray,
        bboxes: np.ndarray,
        indices: np.ndarray,
        segment_duration: float = 3.0,
        fps: float = 20.0
    ) -> List[ActionDescription]:
        """
        Process entire video with tracking data to generate descriptions.

        Args:
            video_path: Path to video file
            keypoints: 2D keypoints (T, 17, 2)
            bboxes: Bounding boxes (T, 4)
            indices: Frame indices used
            segment_duration: Duration of each segment in seconds
            fps: Video FPS

        Returns:
            List of action descriptions for video segments
        """
        segment_frames = int(segment_duration * fps)
        num_segments = max(1, len(indices) // segment_frames)

        descriptions = []

        # Open video
        container = av.open(video_path)
        stream = container.streams.video[0]

        # Process each segment
        for seg_idx in range(num_segments):
            start_idx = seg_idx * segment_frames
            end_idx = min((seg_idx + 1) * segment_frames, len(indices))

            seg_indices = indices[start_idx:end_idx]
            seg_bboxes = bboxes[start_idx:end_idx]

            # Read frames for this segment
            frames = []
            container.seek(0)
            frame_idx = 0

            for frame in container.decode(stream):
                if frame_idx in seg_indices:
                    img = frame.to_ndarray(format='rgb24')
                    frames.append(img)
                frame_idx += 1

                if frame_idx > seg_indices[-1]:
                    break

            if frames:
                frames_array = np.array(frames)

                # Generate description for segment
                description = self.describe_action(
                    frames_array,
                    seg_bboxes,
                    seg_indices
                )

                descriptions.append(description)

        container.close()

        return descriptions


class MultiGPUVideoDescriber:
    """
    Manages video description across multiple GPUs.
    """

    def __init__(
        self,
        num_gpus: int = 4,
        model_name: str = "microsoft/Phi-3.5-vision-instruct"
    ):
        """
        Initialize multi-GPU describer.

        Args:
            num_gpus: Number of GPUs to use
            model_name: VLM model to use
        """
        self.num_gpus = num_gpus
        self.model_name = model_name
        self.describers = {}
        self.gpu_queue = queue.Queue()

        # Initialize queue with GPU IDs
        for i in range(num_gpus):
            self.gpu_queue.put(i)

    def get_describer(self) -> Tuple[VideoActionDescriber, int]:
        """
        Get an available describer and its GPU ID.

        Returns:
            Describer instance and GPU ID
        """
        gpu_id = self.gpu_queue.get()

        if gpu_id not in self.describers:
            self.describers[gpu_id] = VideoActionDescriber(
                device=f"cuda:{gpu_id}",
                model_name=self.model_name
            )

        return self.describers[gpu_id], gpu_id

    def release_describer(self, gpu_id: int):
        """Release a describer back to the pool."""
        self.gpu_queue.put(gpu_id)

    def process_batch(
        self,
        video_data: List[Dict],
        max_workers: int = 4
    ) -> List[List[ActionDescription]]:
        """
        Process multiple videos in parallel across GPUs.

        Args:
            video_data: List of dicts with video_path, keypoints, bboxes, indices
            max_workers: Max parallel workers

        Returns:
            List of description lists for each video
        """
        results = [None] * len(video_data)

        def process_single(idx, data):
            describer, gpu_id = self.get_describer()
            try:
                result = describer.process_video_with_tracking(**data)
                results[idx] = result
            finally:
                self.release_describer(gpu_id)

        with ThreadPoolExecutor(max_workers=min(max_workers, self.num_gpus)) as executor:
            futures = []
            for i, data in enumerate(video_data):
                future = executor.submit(process_single, i, data)
                futures.append(future)

            # Wait for all to complete
            for future in futures:
                future.result()

        return results