"""
YOLOv8 TensorRT detector wrapper for the movement pipeline.
Provides a compatible interface with Faster R-CNN.
"""

import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Any


class YOLOTensorRTDetector:
    """
    Wrapper for YOLOv8 with TensorRT optimization.
    Provides a compatible interface with torchvision Faster R-CNN.
    """

    def __init__(self, model_path: str = 'yolov8x.engine', device: str = 'cuda:0'):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model (can be .pt, .engine, etc)
            device: Device to run on
        """
        # Check if TensorRT engine exists, otherwise fall back
        if Path(model_path).exists():
            self.model_path = model_path
            print(f"Loading YOLO model from {model_path}")
        elif Path('yolov8x.engine').exists():
            self.model_path = 'yolov8x.engine'
            print("Using existing yolov8x.engine")
        elif Path('yolov8x.pt').exists():
            self.model_path = 'yolov8x.pt'
            print("Using yolov8x.pt (consider exporting to TensorRT for better performance)")
        else:
            # Download if needed
            print("Downloading YOLOv8x model...")
            self.model_path = 'yolov8x.pt'

        self.device = device
        self.model = YOLO(self.model_path)

        # Move model to device (for PyTorch models)
        if hasattr(self.model.model, 'to'):
            self.model.model = self.model.model.to(device)

    def __call__(self, images: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """
        Run detection on a batch of images.

        Args:
            images: List of tensors in format [C, H, W] with values in [0, 1]

        Returns:
            List of dictionaries with 'boxes', 'labels', and 'scores' for each image
        """
        outputs = []

        # Process images one by one (TensorRT engine has batch size 1)
        for img_tensor in images:
            # Convert from [C, H, W] to [H, W, C] and from [0,1] to [0,255]
            img_np = img_tensor.cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
            img_np = (img_np * 255).astype(np.uint8)

            # Run YOLO detection on single image (person class only)
            results = self.model(img_np, classes=[0], verbose=False)

            # Initialize output for this image
            output = {
                'boxes': torch.empty((0, 4), device=self.device),
                'labels': torch.empty((0,), dtype=torch.int64, device=self.device),
                'scores': torch.empty((0,), device=self.device)
            }

            # Process result (should be single result for single image)
            if results and len(results) > 0:
                result = results[0]
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


def load_yolo_detector(detector_type: str = 'yolov8x', device: str = 'cuda:0'):
    """
    Load YOLO detector for the pipeline.

    Args:
        detector_type: Type of YOLO model (yolov8x, yolov8l, etc)
        device: Device to run on

    Returns:
        YOLOTensorRTDetector instance
    """
    # Map detector types to model paths
    model_map = {
        'yolov8x': 'yolov8x.engine' if Path('yolov8x.engine').exists() else 'yolov8x.pt',
        'yolov8l': 'yolov8l.engine' if Path('yolov8l.engine').exists() else 'yolov8l.pt',
        'yolov8m': 'yolov8m.engine' if Path('yolov8m.engine').exists() else 'yolov8m.pt',
    }

    model_path = model_map.get(detector_type, detector_type)
    return YOLOTensorRTDetector(model_path, device)