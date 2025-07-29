#!/usr/bin/env python3
"""
ONNX and TensorRT Model Comparison Tool

This script analyzes and compares ONNX and TensorRT models to show:
- Layer counts and types
- File sizes
- Performance optimizations
- Model structure differences

example:
python benchmarks/fusion.py --onnx weights/onnx/seg_v1.0.0_trimmed.onnx --trt weights/trtPlans/seg_v1.0.0_trimmed.trt
python benchmarks/fusion.py --onnx weights/onnx/supconloss_bbresnet50_50e.onnx --trt weights/trtPlans/supconloss_bbresnet50_50e.trt

Author: EVEMASK Team
Version: 1.0.0
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

# Import required libraries
try:
    import onnx
    import tensorrt as trt
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install onnx tensorrt")
    sys.exit(1)


def analyze_onnx_model(onnx_path):
    """
    Analyze ONNX model structure and layers
    
    Args:
        onnx_path (str): Path to ONNX model file
        
    Returns:
        tuple: (layer_counts_dict, total_layers_count)
    """
    print(f"Loading ONNX model: {onnx_path}")
    
    # Check if file exists
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    # Load and parse ONNX model
    model = onnx.load(onnx_path)
    graph = model.graph
    
    # Count different layer types
    layer_counts = defaultdict(int)
    for node in graph.node:
        layer_counts[node.op_type] += 1
    
    # Calculate file size
    file_size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
    
    # Print analysis results
    print(f"\n=== ONNX Model Analysis ===")
    print(f"File: {Path(onnx_path).name}")
    print(f"Total layers: {len(graph.node)}")
    print(f"Inputs: {len(graph.input)}")
    print(f"Outputs: {len(graph.output)}")
    print(f"File size: {file_size_mb:.2f} MB")
    
    print("\nLayer breakdown:")
    for layer_type in sorted(layer_counts.keys()):
        count = layer_counts[layer_type]
        print(f"  {layer_type}: {count}")
    
    return dict(layer_counts), len(graph.node)


def analyze_tensorrt_engine(engine_path):
    """
    Analyze TensorRT engine structure
    
    Args:
        engine_path (str): Path to TensorRT engine file
        
    Returns:
        tuple: (layer_info_dict, estimated_layer_count)
    """
    print(f"Loading TensorRT engine: {engine_path}")
    
    # Check if file exists
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
    
    # Initialize TensorRT runtime
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    # Load engine from file
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
    
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine")
    
    # Calculate file size
    file_size_mb = Path(engine_path).stat().st_size / (1024 * 1024)
    
    # Print basic engine info
    print(f"\n=== TensorRT Engine Analysis ===")
    print(f"File: {Path(engine_path).name}")
    print(f"TensorRT version: {trt.__version__}")
    print(f"Number of bindings: {engine.num_bindings}")
    print(f"Max batch size: {engine.max_batch_size}")
    print(f"File size: {file_size_mb:.2f} MB")
    
    # Analyze input and output bindings
    inputs = []
    outputs = []
    
    for i in range(engine.num_bindings):
        binding_name = engine.get_binding_name(i)
        binding_shape = engine.get_binding_shape(i)
        binding_dtype = engine.get_binding_dtype(i)
        
        binding_info = {
            'name': binding_name,
            'shape': list(binding_shape) if binding_shape else [],
            'dtype': str(binding_dtype)
        }
        
        if engine.binding_is_input(i):
            inputs.append(binding_info)
        else:
            outputs.append(binding_info)
    
    # Print binding information
    print(f"Inputs: {len(inputs)}")
    for inp in inputs:
        print(f"  {inp['name']}: {inp['shape']} ({inp['dtype']})")
    
    print(f"Outputs: {len(outputs)}")
    for out in outputs:
        print(f"  {out['name']}: {out['shape']} ({out['dtype']})")
    
    # Estimate layer count
    estimated_layers = 0
    layer_info = {}
    
    # Try to get actual layer count if available
    if hasattr(engine, 'num_layers'):
        try:
            estimated_layers = engine.num_layers
            print(f"Reported layers: {estimated_layers}")
            layer_info['Total_Layers'] = estimated_layers
        except:
            pass
    
    # Fallback estimation based on file size and complexity
    if estimated_layers == 0:
        estimated_layers = len(inputs) + len(outputs) + max(1, int(file_size_mb / 10))
        print(f"Estimated layers: {estimated_layers}")
        layer_info['Estimated_Layers'] = estimated_layers
    
    return layer_info, estimated_layers


def compare_models(onnx_path, trt_path):
    """
    Compare ONNX and TensorRT models side by side
    
    Args:
        onnx_path (str): Path to ONNX model
        trt_path (str): Path to TensorRT engine
        
    Returns:
        dict: Comparison results
    """
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Analyze both models
    onnx_layers, onnx_total = analyze_onnx_model(onnx_path)
    trt_layers, trt_total = analyze_tensorrt_engine(trt_path)
    
    # Calculate file sizes
    onnx_size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
    trt_size_mb = Path(trt_path).stat().st_size / (1024 * 1024)
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    print("Metric".ljust(25) + "ONNX".ljust(15) + "TensorRT".ljust(15) + "Difference")
    print("-" * 60)
    
    print("Total Layers".ljust(25) + str(onnx_total).ljust(15) + str(trt_total).ljust(15) + str(onnx_total - trt_total))
    print("File Size (MB)".ljust(25) + f"{onnx_size_mb:.2f}".ljust(15) + f"{trt_size_mb:.2f}".ljust(15) + f"{onnx_size_mb - trt_size_mb:.2f}")
    
    # Calculate optimization percentages
    if onnx_total > 0:
        layer_reduction_percent = ((onnx_total - trt_total) / onnx_total) * 100
        print(f"\nLayer reduction: {onnx_total - trt_total} layers ({layer_reduction_percent:.1f}%)")
    
    if onnx_size_mb > 0:
        size_reduction_percent = ((onnx_size_mb - trt_size_mb) / onnx_size_mb) * 100
        print(f"Size reduction: {onnx_size_mb - trt_size_mb:.2f} MB ({size_reduction_percent:.1f}%)")
    
    return {
        'onnx_layers': onnx_total,
        'trt_layers': trt_total,
        'onnx_size_mb': onnx_size_mb,
        'trt_size_mb': trt_size_mb,
        'layer_reduction': onnx_total - trt_total,
        'size_reduction_mb': onnx_size_mb - trt_size_mb
    }


def main():
    """Main function to handle command line arguments and execute analysis"""
    parser = argparse.ArgumentParser(
        description="Compare ONNX and TensorRT models for optimization analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fusion.py --onnx model.onnx --trt model.trt
  python fusion.py --onnx model.onnx --onnx-only
  python fusion.py --trt model.trt --trt-only
        """
    )
    
    # Command line arguments
    parser.add_argument('--onnx', type=str, help='Path to ONNX model file')
    parser.add_argument('--trt', type=str, help='Path to TensorRT engine file')
    parser.add_argument('--onnx-only', action='store_true', help='Only analyze ONNX model')
    parser.add_argument('--trt-only', action='store_true', help='Only analyze TensorRT engine')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.onnx and not args.trt:
        print("Error: Please specify at least one model file")
        parser.print_help()
        return
    
    try:
        # Execute based on arguments
        if args.onnx_only and args.onnx:
            analyze_onnx_model(args.onnx)
        elif args.trt_only and args.trt:
            analyze_tensorrt_engine(args.trt)
        elif args.onnx and args.trt:
            result = compare_models(args.onnx, args.trt)
            print(f"\nComparison completed successfully!")
        else:
            print("Error: Invalid argument combination")
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    # Use default paths if no arguments provided
    if len(sys.argv) == 1:
        onnx_path = "weights/onnx/seg_v1.0.0_trimmed.onnx"
        trt_path = "weights/trtPlans/seg_v1.0.0_trimmed.trt"
        
        if os.path.exists(onnx_path) and os.path.exists(trt_path):
            print("Using default paths...")
            compare_models(onnx_path, trt_path)
        else:
            print("Default files not found. Use --help for usage information.")
    else:
        main()