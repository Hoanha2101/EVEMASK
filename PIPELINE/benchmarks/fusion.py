"""
ONNX and TensorRT Model Comparison Tool

This script analyzes and compares ONNX models to show:
- Layer counts and types
- File sizes
- Performance optimizations
- Model structure differences
- Comprehensive model analysis metrics
- Visualization of model comparisons

example:
python benchmarks/fusion.py --onnx weights/onnx/seg_v1.0.0_trimmed.onnx --trt weights/trtPlans/seg_v1.0.0_trimmed.trt
python benchmarks/fusion.py --onnx weights/onnx/supconloss_bbresnet50_50e.onnx --trt weights/trtPlans/supconloss_bbresnet50_50e.trt

Author: EVEMASK Team
Version: 1.0.0
"""

import argparse
import os
import sys
import csv
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Import required libraries
try:
    import onnx
    import tensorrt as trt
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install onnx tensorrt matplotlib numpy")
    sys.exit(1)


def analyze_onnx_model(onnx_path):
    """
    Analyze ONNX model structure and layers
    
    Args:
        onnx_path (str): Path to ONNX model file
        
    Returns:
        tuple: (layer_counts_dict, total_layers_count, detailed_info_dict)
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
    
    # Collect detailed information
    detailed_info = {
        'model_path': onnx_path,
        'model_name': Path(onnx_path).name,
        'total_layers': len(graph.node),
        'inputs_count': len(graph.input),
        'outputs_count': len(graph.output),
        'file_size_mb': file_size_mb,
        'ir_version': model.ir_version,
        'producer_name': model.producer_name,
        'model_version': model.model_version,
        'opset_version': model.opset_import[0].version if model.opset_import else 0,
        'layer_breakdown': dict(layer_counts)
    }
    
    # Print analysis results
    print(f"\n=== ONNX Model Analysis ===")
    print(f"File: {Path(onnx_path).name}")
    print(f"Total layers: {len(graph.node)}")
    print(f"Inputs: {len(graph.input)}")
    print(f"Outputs: {len(graph.output)}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name}")
    print(f"Model Version: {model.model_version}")
    print(f"Opset Version: {model.opset_import[0].version if model.opset_import else 'Unknown'}")
    
    print("\nLayer breakdown:")
    for layer_type in sorted(layer_counts.keys()):
        count = layer_counts[layer_type]
        print(f"  {layer_type}: {count}")
    
    return dict(layer_counts), len(graph.node), detailed_info


def analyze_tensorrt_engine(engine_path):
    """
    Analyze TensorRT engine structure
    
    Args:
        engine_path (str): Path to TensorRT engine file
        
    Returns:
        tuple: (layer_info_dict, estimated_layer_count, detailed_info_dict)
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
    
    # Collect detailed information
    detailed_info = {
        'engine_path': engine_path,
        'engine_name': Path(engine_path).name,
        'tensorrt_version': trt.__version__,
        'num_bindings': engine.num_bindings,
        'max_batch_size': engine.max_batch_size,
        'file_size_mb': file_size_mb,
        'estimated_layers': estimated_layers,
        'inputs': inputs,
        'outputs': outputs,
        'layer_info': layer_info
    }
    
    return layer_info, estimated_layers, detailed_info


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
    onnx_layers, onnx_total, onnx_details = analyze_onnx_model(onnx_path)
    trt_layers, trt_total, trt_details = analyze_tensorrt_engine(trt_path)
    
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
    
    comparison_results = {
        'onnx_layers': onnx_total,
        'trt_layers': trt_total,
        'onnx_size_mb': onnx_size_mb,
        'trt_size_mb': trt_size_mb,
        'layer_reduction': onnx_total - trt_total,
        'size_reduction_mb': onnx_size_mb - trt_size_mb,
        'layer_reduction_percent': layer_reduction_percent if onnx_total > 0 else 0,
        'size_reduction_percent': size_reduction_percent if onnx_size_mb > 0 else 0,
        'onnx_details': onnx_details,
        'trt_details': trt_details
    }
    
    return comparison_results


def save_analysis_results_to_csv(onnx_details, trt_details, comparison_results, output_dir):
    """
    Save comprehensive analysis results to CSV files
    
    Args:
        onnx_details (dict): Detailed ONNX model information
        trt_details (dict): Detailed TensorRT engine information
        comparison_results (dict): Model comparison results
        output_dir (str): Directory to save CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save ONNX model analysis
    onnx_csv_path = os.path.join(output_dir, 'onnx_model_analysis.csv')
    with open(onnx_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Model Path', onnx_details['model_path']])
        writer.writerow(['Model Name', onnx_details['model_name']])
        writer.writerow(['Total Layers', onnx_details['total_layers']])
        writer.writerow(['Inputs Count', onnx_details['inputs_count']])
        writer.writerow(['Outputs Count', onnx_details['outputs_count']])
        writer.writerow(['File Size (MB)', f"{onnx_details['file_size_mb']:.2f}"])
        writer.writerow(['IR Version', onnx_details['ir_version']])
        writer.writerow(['Producer Name', onnx_details['producer_name']])
        writer.writerow(['Model Version', onnx_details['model_version']])
        writer.writerow(['Opset Version', onnx_details['opset_version']])
        
        # Layer breakdown
        writer.writerow([])
        writer.writerow(['Layer Type', 'Count'])
        for layer_type, count in onnx_details['layer_breakdown'].items():
            writer.writerow([layer_type, count])
    
    # Save TensorRT engine analysis
    trt_csv_path = os.path.join(output_dir, 'tensorrt_engine_analysis.csv')
    with open(trt_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Engine Path', trt_details['engine_path']])
        writer.writerow(['Engine Name', trt_details['engine_name']])
        writer.writerow(['TensorRT Version', trt_details['tensorrt_version']])
        writer.writerow(['Number of Bindings', trt_details['num_bindings']])
        writer.writerow(['Max Batch Size', trt_details['max_batch_size']])
        writer.writerow(['File Size (MB)', f"{trt_details['file_size_mb']:.2f}"])
        writer.writerow(['Estimated Layers', trt_details['estimated_layers']])
        
        # Input bindings
        writer.writerow([])
        writer.writerow(['Input Binding', 'Shape', 'Data Type'])
        for inp in trt_details['inputs']:
            writer.writerow([inp['name'], str(inp['shape']), inp['dtype']])
        
        # Output bindings
        writer.writerow([])
        writer.writerow(['Output Binding', 'Shape', 'Data Type'])
        for out in trt_details['outputs']:
            writer.writerow([out['name'], str(out['shape']), out['dtype']])
    
    # Save comparison results
    comparison_csv_path = os.path.join(output_dir, 'model_comparison_summary.csv')
    with open(comparison_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Comparison Metric', 'ONNX', 'TensorRT', 'Difference', 'Improvement (%)'])
        writer.writerow(['Total Layers', comparison_results['onnx_layers'], comparison_results['trt_layers'], 
                        comparison_results['layer_reduction'], f"{comparison_results['layer_reduction_percent']:.1f}%"])
        writer.writerow(['File Size (MB)', f"{comparison_results['onnx_size_mb']:.2f}", 
                        f"{comparison_results['trt_size_mb']:.2f}", 
                        f"{comparison_results['size_reduction_mb']:.2f}", 
                        f"{comparison_results['size_reduction_percent']:.1f}%"])
    
    # Save detailed comparison
    detailed_csv_path = os.path.join(output_dir, 'detailed_comparison.csv')
    with open(detailed_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Analysis Type', 'ONNX Value', 'TensorRT Value', 'Notes'])
        writer.writerow(['Model Format', 'ONNX', 'TensorRT Engine', 'Optimized format'])
        writer.writerow(['File Size (MB)', f"{onnx_details['file_size_mb']:.2f}", 
                        f"{trt_details['file_size_mb']:.2f}", 'Compression achieved'])
        writer.writerow(['Layer Count', onnx_details['total_layers'], 
                        trt_details['estimated_layers'], 'Layer fusion applied'])
        writer.writerow(['Input Bindings', onnx_details['inputs_count'], 
                        len(trt_details['inputs']), 'Binding optimization'])
        writer.writerow(['Output Bindings', onnx_details['outputs_count'], 
                        len(trt_details['outputs']), 'Output optimization'])

    return {
        'onnx_csv': onnx_csv_path,
        'trt_csv': trt_csv_path,
        'comparison_csv': comparison_csv_path,
        'detailed_csv': detailed_csv_path
    }


def create_visualization_charts(onnx_details, trt_details, comparison_results, output_dir):
    """
    Create visualization charts for model comparison
    
    Args:
        onnx_details (dict): Detailed ONNX model information
        trt_details (dict): Detailed TensorRT engine information
        comparison_results (dict): Model comparison results
        output_dir (str): Directory to save charts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comprehensive comparison chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Chart 1: File Size Comparison
    models = ['ONNX', 'TensorRT']
    sizes = [onnx_details['file_size_mb'], trt_details['file_size_mb']]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars1 = ax1.bar(models, sizes, color=colors, alpha=0.7)
    ax1.set_title('File Size Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('File Size (MB)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, size in zip(bars1, sizes):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{size:.2f} MB', ha='center', va='bottom', fontweight='bold')
    
    # Chart 2: Layer Count Comparison
    layer_counts = [onnx_details['total_layers'], trt_details['estimated_layers']]
    
    bars2 = ax2.bar(models, layer_counts, color=colors, alpha=0.7)
    ax2.set_title('Layer Count Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Layers', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars2, layer_counts):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Chart 3: ONNX Layer Breakdown
    layer_types = list(onnx_details['layer_breakdown'].keys())
    layer_counts_onnx = list(onnx_details['layer_breakdown'].values())
    
    ax3.pie(layer_counts_onnx, labels=layer_types, autopct='%1.1f%%', startangle=90)
    ax3.set_title('ONNX Model Layer Distribution', fontsize=14, fontweight='bold')
    
    # Chart 4: Optimization Benefits
    optimization_metrics = ['Layer Reduction', 'Size Reduction']
    reduction_values = [comparison_results['layer_reduction_percent'], 
                       comparison_results['size_reduction_percent']]
    
    bars4 = ax4.bar(optimization_metrics, reduction_values, color=['#FFD93D', '#6BCF7F'], alpha=0.7)
    ax4.set_title('Optimization Benefits', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Reduction (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars4, reduction_values):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save comprehensive chart
    chart_path = os.path.join(output_dir, 'model_comparison_analysis.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Create individual charts
    # Individual chart 1: File size comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    plt.title('Model File Size Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('File Size (MB)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, size in zip(bars, sizes):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{size:.2f} MB', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    size_chart_path = os.path.join(output_dir, 'file_size_comparison.png')
    plt.savefig(size_chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Individual chart 2: Layer count comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, layer_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    plt.title('Model Layer Count Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Layers', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, layer_counts):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    layer_chart_path = os.path.join(output_dir, 'layer_count_comparison.png')
    plt.savefig(layer_chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {
        'comprehensive_chart': chart_path,
        'size_chart': size_chart_path,
        'layer_chart': layer_chart_path
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
    parser.add_argument('--output-dir', type=str, default='benchmarks/results/fusion_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.onnx and not args.trt:
        print("Error: Please specify at least one model file")
        parser.print_help()
        return
    
    try:
        # Execute based on arguments
        if args.onnx_only and args.onnx:
            onnx_layers, onnx_total, onnx_details = analyze_onnx_model(args.onnx)
            # Save ONNX-only analysis
            os.makedirs(args.output_dir, exist_ok=True)
            save_analysis_results_to_csv(onnx_details, {}, {}, args.output_dir)
            
        elif args.trt_only and args.trt:
            trt_layers, trt_total, trt_details = analyze_tensorrt_engine(args.trt)
            # Save TensorRT-only analysis
            os.makedirs(args.output_dir, exist_ok=True)
            save_analysis_results_to_csv({}, trt_details, {}, args.output_dir)
            
        elif args.onnx and args.trt:
            # Full comparison analysis
            comparison_results = compare_models(args.onnx, args.trt)
            
            # Extract details for saving
            onnx_layers, onnx_total, onnx_details = analyze_onnx_model(args.onnx)
            trt_layers, trt_total, trt_details = analyze_tensorrt_engine(args.trt)
            
            # Save all results to CSV
            csv_paths = save_analysis_results_to_csv(onnx_details, trt_details, comparison_results, args.output_dir)
            
            # Create and save visualization charts
            chart_paths = create_visualization_charts(onnx_details, trt_details, comparison_results, args.output_dir)
            
            # Save metadata
            metadata = {
                'analysis_timestamp': datetime.now().isoformat(),
                'onnx_model': args.onnx,
                'tensorrt_engine': args.trt,
                'csv_files': csv_paths,
                'chart_files': chart_paths,
                'comparison_summary': {
                    'layer_reduction': comparison_results['layer_reduction'],
                    'size_reduction_mb': comparison_results['size_reduction_mb'],
                    'layer_reduction_percent': comparison_results['layer_reduction_percent'],
                    'size_reduction_percent': comparison_results['size_reduction_percent']
                }
            }
            
            metadata_path = os.path.join(args.output_dir, 'analysis_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"Comparison completed successfully!")
            
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
            comparison_results = compare_models(onnx_path, trt_path)
            
            # Extract details for saving
            onnx_layers, onnx_total, onnx_details = analyze_onnx_model(onnx_path)
            trt_layers, trt_total, trt_details = analyze_tensorrt_engine(trt_path)
            
            # Save all results to CSV
            output_dir = 'benchmarks/results/fusion_analysis'
            csv_paths = save_analysis_results_to_csv(onnx_details, trt_details, comparison_results, output_dir)
            
            # Create and save visualization charts
            chart_paths = create_visualization_charts(onnx_details, trt_details, comparison_results, output_dir)
            
            # Save metadata
            metadata = {
                'analysis_timestamp': datetime.now().isoformat(),
                'onnx_model': onnx_path,
                'tensorrt_engine': trt_path,
                'csv_files': csv_paths,
                'chart_files': chart_paths,
                'comparison_summary': {
                    'layer_reduction': comparison_results['layer_reduction'],
                    'size_reduction_mb': comparison_results['size_reduction_mb'],
                    'layer_reduction_percent': comparison_results['layer_reduction_percent'],
                    'size_reduction_percent': comparison_results['size_reduction_percent']
                }
            }
            
            metadata_path = os.path.join(output_dir, 'analysis_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"Default analysis completed successfully!")

    else:
        main()