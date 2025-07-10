########################
### Sample CLI: python getpytorch.py --weights weights/yolo/yolov8_seg_aug_best_l.pt --output weights/pytorch/yolov8_seg_aug_best_l.pth
#######################

import torch
from ultralytics import YOLO
import argparse

def save_model(weights_path, output_path):
    # Load the YOLOv8 model from file
    model_ultra = YOLO(weights_path)
    # Access the raw PyTorch model inside the Ultralytics wrapper
    pure_model = model_ultra.model
    # Save the entire model (including architecture and weights)
    torch.save(pure_model, output_path)
    print(f"Model saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save Ultralytics YOLOv8 model in raw PyTorch format")
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLOv8 .pt weights file")
    parser.add_argument("--output", type=str, required=True, help="Output path to save the PyTorch model")

    args = parser.parse_args()
    save_model(args.weights, args.output)