"""
YOLOv8 batch detector with proper batch processing support.
Falls back to PyTorch model for dynamic batching capability.
"""

import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Any


class YOLOBatchDetector:
    """
    Wrapper for YOLOv8 with true batch processing support.
    Uses PyTorch model for dynamic batching.
    """

    def __init__(self, model_path: str = 'yolov8x.pt', device: str = 'cuda:0', verbose: bool = False):
        """
        Initialize YOLO detector with batch support.

        Args:
            model_path: Path to YOLO model (using .pt for batch support)
            device: Device to run on
            verbose: Whether to print loading messages
        """
        # Use PyTorch model for dynamic batching
        if Path('yolov8x.pt').exists():
            self.model_path = 'yolov8x.pt'
            if verbose:
                print("Using yolov8x.pt with dynamic batching support")
        else:
            # Download if needed
            if verbose:
                print("Downloading YOLOv8x model...")
            self.model_path = 'yolov8x.pt'

        self.device = device
        self.model = YOLO(self.model_path)

        # Move model to device
        if hasattr(self.model.model, 'to'):
            self.model.model = self.model.model.to(device)

    def __call__(self, images: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """
        Run detection on a batch of images with TRUE BATCH PROCESSING.

        Args:
            images: List of tensors in format [C, H, W] with values in [0, 1]

        Returns:
            List of dictionaries with 'boxes', 'labels', and 'scores' for each image
        """
        if not images:
            return []

        # Convert all images to numpy in batch
        batch_size = len(images)

        # Stack tensors and convert to numpy
        # Convert from [C, H, W] to [H, W, C] and from [0,1] to [0,255]
        img_batch = []
        for img_tensor in images:
            img_np = img_tensor.cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
            img_np = (img_np * 255).astype(np.uint8)
            img_batch.append(img_np)

        # Run YOLO detection on ENTIRE BATCH at once (person class only)
        # This is the key difference - process all images in one call!
        results = self.model(img_batch, classes=[0], verbose=False, batch=batch_size)

        # Process results
        outputs = []
        for result in results:
            output = {
                'boxes': torch.empty((0, 4), device=self.device),
                'labels': torch.empty((0,), dtype=torch.int64, device=self.device),
                'scores': torch.empty((0,), device=self.device)
            }

            if result.boxes is not None and len(result.boxes) > 0:
                # Get boxes in xyxy format
                boxes = result.boxes.xyxy  # Already a tensor

                # All detections are persons (class 0 in YOLO = class 1 in COCO)
                labels = torch.ones(len(boxes), dtype=torch.int64) * 1

                # Get confidence scores
                scores = result.boxes.conf

                # Move to correct device
                output['boxes'] = boxes.to(self.device)
                output['labels'] = labels.to(self.device)
                output['scores'] = scores.to(self.device)

            outputs.append(output)

        return outputs

    def eval(self):
        """Set model to evaluation mode (compatibility with PyTorch models)."""
        return self

    def to(self, device):
        """Move model to device (compatibility with PyTorch models)."""
        self.device = str(device)
        if hasattr(self.model.model, 'to'):
            self.model.model = self.model.model.to(device)
        return self


def load_yolo_batch_detector(detector_type: str = 'yolov8x', device: str = 'cuda:0', verbose: bool = False):
    """
    Load YOLO detector with batch processing support.

    Args:
        detector_type: Type of YOLO model (yolov8x, yolov8l, etc)
        device: Device to run on
        verbose: Whether to print loading messages

    Returns:
        YOLOBatchDetector instance
    """
    # Map detector types to model paths
    model_map = {
        'yolov8x': 'yolov8x.pt',
        'yolov8l': 'yolov8l.pt',
        'yolov8m': 'yolov8m.pt',
    }

    model_path = model_map.get(detector_type, detector_type)
    return YOLOBatchDetector(model_path, device, verbose)