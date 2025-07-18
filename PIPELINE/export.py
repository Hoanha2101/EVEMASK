"""
PyTorch to ONNX Model Exporter
Converts PyTorch models to ONNX format for deployment and optimization.

This script provides:
- PyTorch model loading with smart format detection
- ONNX export with dynamic batch support
- Support for different precision modes (FP32/FP16)
- Model trimming for segmentation models
- Feature extraction model definition

Usage Examples:
    # YOLO segmentation model export
    python export.py --pth weights/pytorch/yolov8_seg_aug_best_l.pth --output weights/onnx/yolov8_seg_aug_best_l.onnx --input-shape 1 3 640 640 --input-name input --output-names pred0 pred1_0_0 pred1_0_1 pred1_0_2 pred1_1 pred1_2 --mode float32bit --device cuda --opset 19 --typeModel seg
    
    # Feature extraction model export
    python export.py --pth weights/pytorch/SupConLoss_BBVGG16.pth --output weights/onnx/SupConLoss_BBVGG16.onnx --input-shape 1 3 224 224 --input-name input --output-names output --mode float16bit --device cuda --opset 12 --typeModel fe

Author: EVEMASK Team
"""

########################
### Sample CLI: python export.py --pth weights/pytorch/yolov8_seg_aug_best_l.pth --output weights/onnx/yolov8_seg_aug_best_l.onnx --input-shape 1 3 640 640 --input-name input --output-names pred0 pred1_0_0 pred1_0_1 pred1_0_2 pred1_1 pred1_2 --mode float32bit --device cuda --opset 19 --typeModel seg
### Sample CLI: python export.py --pth weights/pytorch/SupConLoss_BBVGG16.pth --output weights/onnx/SupConLoss_BBVGG16.onnx --input-shape 1 3 224 224 --input-name input --output-names output --mode float16bit --device cuda --opset 12 --typeModel fe
#######################

import torch
import torch.nn as nn
import argparse
from torchvision import models
import os

def smart_load_model(pth_path, model_type, emb_dim=256):
    """
    Intelligently load PyTorch model from various formats.
    
    This function can handle different PyTorch model formats:
    - Checkpoint dictionaries with 'model_state_dict'
    - Raw state dictionaries
    - Full nn.Module objects
    
    Args:
        pth_path (str): Path to PyTorch model file
        model_type (str): Type of model ('seg' for segmentation, 'fe' for feature extraction)
        emb_dim (int): Embedding dimension for feature extraction models
        
    Returns:
        torch.nn.Module: Loaded PyTorch model
        
    Raises:
        ValueError: If model format doesn't match expected type
        RuntimeError: If model format is unsupported
    """
    # Load checkpoint from file
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)

    # Handle checkpoint with 'model_state_dict' (training checkpoint)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("Detected checkpoint with 'model_state_dict'")
        if model_type == "fe":
            # Create feature extraction model and load weights
            model = Network(emb_dim=emb_dim)
            model.load_state_dict(checkpoint["model_state_dict"])
            return model
        else:
            raise ValueError("Segmentation/classification model should not contain 'model_state_dict'. Please check --typeModel or retrain properly.")
    
    # Handle raw state dictionary
    elif isinstance(checkpoint, dict):
        print("Detected raw state_dict")
        if model_type == "fe":
            # Create feature extraction model and load weights
            model = Network(emb_dim=emb_dim)
            model.load_state_dict(checkpoint)
            return model
        else:
            raise ValueError("Segmentation/classification model should not be a raw state_dict. Please check --typeModel.")
    
    # Handle full nn.Module object
    elif isinstance(checkpoint, torch.nn.Module):
        print("Detected full nn.Module object")
        return checkpoint
    
    else:
        raise RuntimeError("Unsupported model format.")

# Feature extractor model (ResNet50 → embedding)
class Network(nn.Module):
    """
    Feature extraction network based on ResNet50 architecture.
    This network extracts feature embeddings from input images:
    1. Uses ResNet50 backbone for feature extraction
    2. Flattens features and passes through fully connected layers
    3. Outputs embedding vectors for similarity matching
    Attributes:
        backbone: ResNet50 feature extraction layers
        fc: Fully connected layers for embedding generation
    """
    def __init__(self, emb_dim=128):
        """
        Initialize feature extraction network.
        Args:
            emb_dim (int): Dimension of output embedding vector
        """
        super(Network, self).__init__()
        base_model = models.resnet50(pretrained=True)
        # Remove the last fully connected layer (fc)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # Output: [B, 2048, 1, 1]
        # Fully connected layers for embedding generation
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.PReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        x = self.backbone(x)           # [B, 2048, 1, 1]
        x = torch.flatten(x, 1)        # [B, 2048]
        x = self.fc(x)                 # [B, emb_dim]
        return x

def convert_pytorch_model_to_onnx(
    model,
    path_onnx,
    input_shape=(1, 3, 224, 224),
    input_name="input",
    output_names=["output"],
    mode='float32bit',
    device='cuda',
    opset_version=12
):
    """
    Convert PyTorch model to ONNX format.
    
    This function performs the complete ONNX export process:
    1. Creates dummy input with specified shape
    2. Converts model to specified precision (FP32/FP16)
    3. Sets up dynamic axes for batch dimension
    4. Exports to ONNX format
    
    Args:
        model (torch.nn.Module): PyTorch model to export
        path_onnx (str): Output path for ONNX file
        input_shape (tuple): Input tensor shape (batch, channels, height, width)
        input_name (str): Name of input tensor in ONNX
        output_names (list): Names of output tensors in ONNX
        mode (str): Precision mode ('float32bit' or 'float16bit')
        device (str): Device to use for export ('cuda' or 'cpu')
        opset_version (int): ONNX opset version
        
    Returns:
        str: Path to exported ONNX file
    """
    # Create dummy input tensor for export
    dummy_input = torch.randn(*input_shape)

    # Convert model and input to specified precision
    if mode == 'float16bit':
        print("Converting model and inputs to float16")
        model = model.half()
        dummy_input = dummy_input.half()
    else:
        print("Converting model and inputs to float32")
        model = model.float()
        dummy_input = dummy_input.float()

    # Set device for export
    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    print(f"Exporting ONNX model on device: {device}")
    model.to(device).eval()  # Set to evaluation mode
    dummy_input = dummy_input.to(device)

    # Set up dynamic axes for batch dimension
    dynamic_axes = {input_name: {0: 'batch_size'}}
    for name in output_names:
        dynamic_axes[name] = {0: 'batch_size'}

    # Export to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        path_onnx,
        input_names=[input_name],
        output_names=output_names,
        dynamic_axes=dynamic_axes,  # Enable dynamic batch size
        do_constant_folding=True,  # Optimize constant operations
        opset_version=opset_version,
        verbose=False
    )

    print(f"ONNX model saved to {path_onnx}")
    return path_onnx

if __name__ == "__main__":
    # Command line argument parser
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")

    # Required arguments
    parser.add_argument("--pth", type=str, required=True, 
                       help="Path to .pth PyTorch model file")
    parser.add_argument("--output", type=str, required=True, 
                       help="Output ONNX file path")
    parser.add_argument("--typeModel", type=str, choices=["seg", "fe"], required=True, 
                       help="Model type: seg (segmentation) or fe (feature extraction)")
    
    # Model configuration arguments
    parser.add_argument("--input-shape", type=int, nargs=4, default=(1, 3, 224, 224), 
                       help="Input shape e.g. 1 3 224 224")
    parser.add_argument("--input-name", type=str, default="input", 
                       help="Name of ONNX input tensor")
    parser.add_argument("--output-names", type=str, nargs='+', default=["output"], 
                       help="List of output names")
    
    # Export configuration arguments
    parser.add_argument("--mode", type=str, choices=["float32bit", "float16bit"], default="float32bit", 
                       help="Precision mode")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", 
                       help="Device to use")
    parser.add_argument("--opset", type=int, default=12, 
                       help="ONNX opset version")

    # Parse command line arguments
    args = parser.parse_args()

    # Load model based on type
    if args.typeModel == "seg":
        model = smart_load_model(args.pth, model_type="seg")
    elif args.typeModel == "fe":
        model = smart_load_model(args.pth, model_type="fe", emb_dim=224)

    # Export to ONNX format
    onnx_path = convert_pytorch_model_to_onnx(
        model=model,
        path_onnx=args.output,
        input_shape=tuple(args.input_shape),
        input_name=args.input_name,
        output_names=args.output_names,
        mode=args.mode,
        device=args.device,
        opset_version=args.opset
    )

    # Trim ONNX model for segmentation models
    # This removes unused outputs to reduce model size and improve performance
    if args.typeModel == "seg":
        print("Trimming unused outputs from ONNX model...")
        import onnx_graphsurgeon as gs
        import onnx

        # Load ONNX graph for modification
        graph = gs.import_onnx(onnx.load(onnx_path))
        
        # Keep only essential outputs for segmentation
        # pred0: detection outputs, pred1_2: mask prototypes
        graph.outputs = [o for o in graph.outputs if o.name in ["pred0", "pred1_2"]]
        
        # Clean up and optimize graph
        graph.cleanup().toposort()

        # Save trimmed model
        trimmed_path = onnx_path.replace(".onnx", "_trimmed.onnx")
        onnx.save(gs.export_onnx(graph), trimmed_path)
        print(f"Exported trimmed ONNX model to {trimmed_path}")

