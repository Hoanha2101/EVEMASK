"""
YOLOv8 Model Extractor
Extracts raw PyTorch models from Ultralytics YOLOv8 format for custom deployment.

This script provides:
- Conversion from Ultralytics YOLOv8 format to raw PyTorch
- Model extraction for custom inference pipelines
- Support for all YOLOv8 model variants (detection, segmentation, classification)

Usage Example:
    python getpytorch.py --weights weights/yolo/seg_v1.0.0.pt --output weights/pytorch/seg_v1.0.0.pth

Author: EVEMASK Team
"""

import torch
from ultralytics import YOLO
import argparse

def save_model(weights_path, output_path):
    """
    Extract and save raw PyTorch model from Ultralytics YOLOv8 format.
    
    This function:
    1. Loads the YOLOv8 model using Ultralytics wrapper
    2. Extracts the underlying PyTorch model
    3. Saves the complete model (architecture + weights) to file
    
    Args:
        weights_path (str): Path to YOLOv8 .pt weights file
        output_path (str): Output path to save the PyTorch model
    """
    # Load the YOLOv8 model from file using Ultralytics wrapper
    # This handles model loading, validation, and initialization
    model_ultra = YOLO(weights_path)
    
    # Access the raw PyTorch model inside the Ultralytics wrapper
    # This gives us the pure PyTorch nn.Module without Ultralytics overhead
    pure_model = model_ultra.model
    
    # Save the entire model (including architecture and weights)
    # This preserves the complete model structure for custom inference
    torch.save(pure_model, output_path)
    print(f"Model saved to: {output_path}")

if __name__ == "__main__":
    # Command line argument parser
    parser = argparse.ArgumentParser(description="Save Ultralytics YOLOv8 model in raw PyTorch format")
    
    # Required arguments
    parser.add_argument("--weights", type=str, required=True, 
                       help="Path to YOLOv8 .pt weights file")
    parser.add_argument("--output", type=str, required=True, 
                       help="Output path to save the PyTorch model")

    # Parse command line arguments
    args = parser.parse_args()
    
    # Extract and save the model
    save_model(args.weights, args.output)