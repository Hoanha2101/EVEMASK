"""
TensorRT Engine Builder
Converts ONNX models to optimized TensorRT engines for high-performance inference.

This script provides:
- ONNX to TensorRT engine conversion
- Support for FP16 precision optimization
- Dynamic shape configuration for flexible batch sizes
- Memory optimization and workspace management

Usage Examples:
    # YOLO segmentation model with dynamic shapes
    python build.py --onnx weights/onnx/yolov8_seg_aug_best_l.onnx --engine weights/trtPlans/yolov8_seg_aug_best_l.trt --fp16 --dynamic --dynamic-shapes "{\"input\": ((1, 3, 640, 640), (2, 3, 640, 640), (3, 3, 640, 640))}"
    
    # Feature extraction model with dynamic shapes
    python build.py --onnx weights/onnx/SupConLoss_BBVGG16.onnx --engine weights/trtPlans/SupConLoss_BBVGG16.trt --fp16 --dynamic --dynamic-shapes "{\"input\": ((1,3,224,224), (8,3,224,224), (32,3,224,224))}"

Author: EVEMASK Team
"""

import tensorrt as trt
import argparse
import os
import ast

# Initialize TensorRT logger with warning level to reduce verbose output
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_trt_engine(
    onnx_file_path,
    engine_file_path,
    use_fp16=False,
    dynamic=False,
    dynamic_shapes=None,
    fixed_batch_size=1
):
    """
    Build TensorRT engine from ONNX model.
    
    This function performs the complete conversion process from ONNX to TensorRT:
    1. Creates TensorRT builder and configuration
    2. Parses ONNX model
    3. Configures optimization settings (FP16, dynamic shapes)
    4. Builds and serializes the engine
    
    Args:
        onnx_file_path (str): Path to input ONNX model file
        engine_file_path (str): Path where TensorRT engine will be saved
        use_fp16 (bool): Enable FP16 precision for faster inference
        dynamic (bool): Enable dynamic shape support
        dynamic_shapes (dict): Dictionary defining dynamic shape ranges
        fixed_batch_size (int): Fixed batch size when not using dynamic shapes
    """
    # Create TensorRT builder
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    
    # Configure optimization settings
    config.set_tactic_sources(trt.TacticSource.CUBLAS_LT)  # Use cuBLAS for optimization
    config.max_workspace_size = 1 << 32  # 4 GB workspace for optimization
    
    # Enable FP16 if supported and requested
    if builder.platform_has_fast_fp16 and use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled")

    # Create network with explicit batch dimension
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    print(f"Parsing ONNX file: {onnx_file_path}")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed to parse the ONNX file.")
            # Print detailed error messages
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    print("Successfully parsed ONNX file")

    # Configure dynamic shapes if enabled
    if dynamic:
        print(f"===> Using dynamic shapes: {dynamic_shapes}")
        profile = builder.create_optimization_profile()
        
        # Set shape ranges for each input
        for name, (min_shape, opt_shape, max_shape) in dynamic_shapes.items():
            profile.set_shape(name, min_shape, opt_shape, max_shape)
            print(f"   Input '{name}': min={min_shape}, opt={opt_shape}, max={max_shape}")
            
        config.add_optimization_profile(profile)
    else:
        # Set fixed batch size for all inputs
        for i in range(network.num_inputs):
            shape = list(network.get_input(i).shape)
            shape[0] = fixed_batch_size  # Set batch dimension
            network.get_input(i).shape = shape
        print(f"===> Using fixed batch size: {fixed_batch_size}")

    # Remove existing engine file if it exists
    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
            print(f"Removed existing engine file: {engine_file_path}")
        except Exception as e:
            print(f"Cannot remove existing file: {engine_file_path}. Error: {e}")

    # Build and serialize the TensorRT engine
    print("Creating TensorRT Engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine:
        # Save the engine to file
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
        print(f"Engine saved at: {engine_file_path}")
    else:
        print("Failed to build engine")


if __name__ == "__main__":
    # Command line argument parser
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT engine")

    # Required arguments
    parser.add_argument("--onnx", type=str, required=True, 
                       help="Path to the ONNX file")
    parser.add_argument("--engine", type=str, required=True, 
                       help="Path to save the TensorRT engine")
    
    # Optional optimization arguments
    parser.add_argument("--fp16", action="store_true", 
                       help="Enable FP16 mode for faster inference")
    parser.add_argument("--dynamic", action="store_true", 
                       help="Enable dynamic shape mode for flexible batch sizes")
    parser.add_argument("--dynamic-shapes", type=str, default="", 
                       help="Dynamic shape dict, e.g. '{\"input\": [(1,3,224,224),(8,3,224,224),(32,3,224,224)]}'")
    parser.add_argument("--fixed-batch-size", type=int, default=1, 
                       help="Fixed batch size if not using dynamic shapes")

    # Parse command line arguments
    args = parser.parse_args()

    # Convert dynamic shapes string to dictionary if provided
    dynamic_shapes = ast.literal_eval(args.dynamic_shapes) if args.dynamic_shapes else None

    # Build the TensorRT engine
    build_trt_engine(
        onnx_file_path=args.onnx,
        engine_file_path=args.engine,
        use_fp16=args.fp16,
        dynamic=args.dynamic,
        dynamic_shapes=dynamic_shapes,
        fixed_batch_size=args.fixed_batch_size
    )
