"""
Benchmark entry point for the EVEMASK Pipeline system.
This script compares inference performance and memory usage between original and trimmed ONNX models used in the EVEMASK system.

Author: EVEMASK Team
Version: 1.0.0
"""

import argparse
import time
import numpy as np
import cv2
import onnxruntime as ort
import psutil
import gc
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class ModelBenchmark:
    def __init__(self, model_path: str, providers: List[str] = None):
        """
        Initialize the model benchmark class.
        
        Args:
            model_path: Path to the ONNX model file
            providers: ONNX Runtime providers (default: ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"Loaded model: {model_path}")
        print(f"Input shape: {self.input_shape}")
        print(f"Output names: {self.output_names}")
        print(f"Provider: {self.session.get_providers()}")
        
    def generate_dummy_images(self, num_images: int, input_shape: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Generate dummy images for benchmarking purposes.
        
        Args:
            num_images: Number of images to generate
            input_shape: Model input shape (B, C, H, W)
        
        Returns:
            Array of dummy images
        """
        batch_size, channels, height, width = input_shape
        
        # Generate random images with realistic values (0-255 normalized to 0-1)
        images = np.random.rand(num_images, channels, height, width).astype(np.float32)
        return images
    
    def warmup(self, num_warmup: int = 10):
        """
        Run warmup iterations with dummy data to stabilize model performance.
        
        Args:
            num_warmup: Number of warmup iterations
        """
        print(f"Warming up model with {num_warmup} iterations...")
        
        # Determine input shape (handle dynamic batch size)
        input_shape = list(self.input_shape)
        if input_shape[0] == -1 or input_shape[0] is None:
            input_shape[0] = 1  # Set batch size to 1 for warmup
            
        warmup_images = self.generate_dummy_images(num_warmup, input_shape)
        
        warmup_times = []
        for i in range(num_warmup):
            # Single image inference
            input_data = warmup_images[i:i+1]
            
            start_time = time.perf_counter()
            _ = self.session.run(self.output_names, {self.input_name: input_data})
            end_time = time.perf_counter()
            
            warmup_time = (end_time - start_time) * 1000  # Convert to ms
            warmup_times.append(warmup_time)
            
            if (i + 1) % 5 == 0:
                print(f"  Warmup {i+1}/{num_warmup}: {warmup_time:.2f}ms")
        
        avg_warmup_time = np.mean(warmup_times)
        print(f"Warmup completed. Average time: {avg_warmup_time:.2f}ms")
        
        # Force garbage collection to free memory
        gc.collect()
        
    def benchmark(self, num_iterations: int = 100) -> Dict:
        """
        Benchmark the model's inference performance.
        
        Args:
            num_iterations: Number of benchmark iterations
        
        Returns:
            Dictionary containing benchmark results
        """
        print(f"Starting benchmark with {num_iterations} iterations...")
        
        # Determine input shape
        input_shape = list(self.input_shape)
        if input_shape[0] == -1 or input_shape[0] is None:
            input_shape[0] = 1
            
        # Generate benchmark images
        benchmark_images = self.generate_dummy_images(num_iterations, input_shape)
        
        # Initialize metrics
        inference_times = []
        memory_usage_before = []
        memory_usage_after = []
        
        # Get initial memory usage in MB
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        print("Running benchmark...")
        for i in range(num_iterations):
            # Memory before inference
            mem_before = process.memory_info().rss / 1024 / 1024
            memory_usage_before.append(mem_before)
            
            # Single image inference
            input_data = benchmark_images[i:i+1]
            
            # Inference timing
            start_time = time.perf_counter()
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
            end_time = time.perf_counter()
            
            # Memory after inference
            mem_after = process.memory_info().rss / 1024 / 1024
            memory_usage_after.append(mem_after)
            
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)
            
            # Progress update every 20 iterations
            if (i + 1) % 20 == 0:
                current_avg = np.mean(inference_times[-20:])
                print(f"  Progress: {i+1}/{num_iterations} - Recent avg: {current_avg:.2f}ms")
        
        # Calculate statistics for results
        results = {
            'model_path': self.model_path,
            'num_iterations': num_iterations,
            'input_shape': input_shape,
            'output_names': self.output_names,
            'num_outputs': len(self.output_names),
            
            # Timing statistics
            'inference_times': inference_times,
            'mean_time': np.mean(inference_times),
            'std_time': np.std(inference_times),
            'min_time': np.min(inference_times),
            'max_time': np.max(inference_times),
            'median_time': np.median(inference_times),
            'p95_time': np.percentile(inference_times, 95),
            'p99_time': np.percentile(inference_times, 99),
            
            # Memory statistics
            'initial_memory_mb': initial_memory,
            'mean_memory_before_mb': np.mean(memory_usage_before),
            'mean_memory_after_mb': np.mean(memory_usage_after),
            'memory_increase_mb': np.mean(memory_usage_after) - np.mean(memory_usage_before),
            
            # Throughput
            'fps': 1000 / np.mean(inference_times),  # Images per second
        }
        
        print(f"Benchmark completed!")
        print(f"   Mean inference time: {results['mean_time']:.2f} ± {results['std_time']:.2f}ms")
        print(f"   Median inference time: {results['median_time']:.2f}ms")
        print(f"   FPS: {results['fps']:.1f}")
        print(f"   Memory usage: {results['mean_memory_after_mb']:.1f}MB")
        
        return results

def compare_models(original_results: Dict, trimmed_results: Dict) -> Dict:
    """
    Compare benchmark results between the original and trimmed models.
    
    Args:
        original_results: Benchmark results from the original model
        trimmed_results: Benchmark results from the trimmed model
    
    Returns:
        Dictionary containing comparison metrics
    """
    comparison = {
        'speed_improvement_percent': (
            (original_results['mean_time'] - trimmed_results['mean_time']) / 
            original_results['mean_time'] * 100
        ),
        'memory_reduction_percent': (
            (original_results['mean_memory_after_mb'] - trimmed_results['mean_memory_after_mb']) / 
            original_results['mean_memory_after_mb'] * 100
        ),
        'fps_improvement_percent': (
            (trimmed_results['fps'] - original_results['fps']) / 
            original_results['fps'] * 100
        ),
        'outputs_reduction_percent': (
            (original_results['num_outputs'] - trimmed_results['num_outputs']) / 
            original_results['num_outputs'] * 100
        ),
        
        # Absolute differences
        'time_difference_ms': original_results['mean_time'] - trimmed_results['mean_time'],
        'memory_difference_mb': original_results['mean_memory_after_mb'] - trimmed_results['mean_memory_after_mb'],
        'fps_difference': trimmed_results['fps'] - original_results['fps'],
    }
    
    return comparison

def plot_results(original_results: Dict, trimmed_results: Dict, output_dir: str = "benchmark_results_onnx"):
    """
    Create visualization plots for benchmark results.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Inference time comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Box plot of inference times
    ax1.boxplot([original_results['inference_times'], trimmed_results['inference_times']], 
                labels=['Original', 'Trimmed'])
    ax1.set_ylabel('Inference Time (ms)')
    ax1.set_title('Inference Time Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Bar chart of mean metrics
    metrics = ['Mean Time (ms)', 'FPS', 'Memory (MB)']
    original_vals = [original_results['mean_time'], original_results['fps'], 
                     original_results['mean_memory_after_mb']]
    trimmed_vals = [trimmed_results['mean_time'], trimmed_results['fps'], 
                    trimmed_results['mean_memory_after_mb']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x - width/2, original_vals, width, label='Original', alpha=0.8)
    ax2.bar(x + width/2, trimmed_vals, width, label='Trimmed', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_title('Performance Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Time series plot
    iterations = range(len(original_results['inference_times']))
    ax3.plot(iterations, original_results['inference_times'], label='Original', alpha=0.7)
    ax3.plot(iterations, trimmed_results['inference_times'], label='Trimmed', alpha=0.7)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Inference Time (ms)')
    ax3.set_title('Inference Time Over Iterations')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Improvement percentages
    comparison = compare_models(original_results, trimmed_results)
    improvements = ['Speed\nImprovement', 'Memory\nReduction', 'FPS\nImprovement', 'Outputs\nReduction']
    values = [comparison['speed_improvement_percent'], comparison['memory_reduction_percent'],
              comparison['fps_improvement_percent'], comparison['outputs_reduction_percent']]
    
    colors = ['green' if v > 0 else 'red' for v in values]
    bars = ax4.bar(improvements, values, color=colors, alpha=0.7)
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('Performance Improvements')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/benchmark_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_results(original_results: Dict, trimmed_results: Dict, comparison: Dict, 
                output_dir: str = "benchmark_results_onnx"):
    """
    Save benchmark results to JSON files.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save individual results
    with open(f"{output_dir}/original_model_results.json", 'w') as f:
        json.dump(original_results, f, indent=2, default=str)
    
    with open(f"{output_dir}/trimmed_model_results.json", 'w') as f:
        json.dump(trimmed_results, f, indent=2, default=str)
    
    # Save comparison
    summary = {
        'original_model': {
            'path': original_results['model_path'],
            'mean_time_ms': original_results['mean_time'],
            'fps': original_results['fps'],
            'memory_mb': original_results['mean_memory_after_mb'],
            'num_outputs': original_results['num_outputs']
        },
        'trimmed_model': {
            'path': trimmed_results['model_path'],
            'mean_time_ms': trimmed_results['mean_time'],
            'fps': trimmed_results['fps'],
            'memory_mb': trimmed_results['mean_memory_after_mb'],
            'num_outputs': trimmed_results['num_outputs']
        },
        'improvements': comparison
    }
    
    with open(f"{output_dir}/benchmark_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Results saved to {output_dir}/")

def print_summary(original_results: Dict, trimmed_results: Dict, comparison: Dict):
    """
    Print a formatted summary of benchmark results.
    """
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    print(f"\nORIGINAL MODEL: {Path(original_results['model_path']).name}")
    print(f"   + Mean inference time: {original_results['mean_time']:.2f} ± {original_results['std_time']:.2f}ms")
    print(f"   + FPS: {original_results['fps']:.1f}")
    print(f"   + Memory usage: {original_results['mean_memory_after_mb']:.1f}MB")
    print(f"   + Number of outputs: {original_results['num_outputs']}")
    
    print(f"\nTRIMMED MODEL: {Path(trimmed_results['model_path']).name}")
    print(f"   + Mean inference time: {trimmed_results['mean_time']:.2f} ± {trimmed_results['std_time']:.2f}ms")
    print(f"   + FPS: {trimmed_results['fps']:.1f}")
    print(f"   + Memory usage: {trimmed_results['mean_memory_after_mb']:.1f}MB")
    print(f"   + Number of outputs: {trimmed_results['num_outputs']}")
    
    print(f"\nIMPROVEMENTS:")
    print(f"   + Speed improvement: {comparison['speed_improvement_percent']:+.1f}% ({comparison['time_difference_ms']:+.2f}ms)")
    print(f"   + Memory reduction: {comparison['memory_reduction_percent']:+.1f}% ({comparison['memory_difference_mb']:+.1f}MB)")
    print(f"   + FPS improvement: {comparison['fps_improvement_percent']:+.1f}% ({comparison['fps_difference']:+.1f} FPS)")
    print(f"   + Outputs reduction: {comparison['outputs_reduction_percent']:+.1f}%")
    
    print("\nCONCLUSION:")
    if comparison['speed_improvement_percent'] > 0:
        print(f"   Trimmed model is {comparison['speed_improvement_percent']:.1f}% FASTER")
    else:
        print(f"   Trimmed model is {abs(comparison['speed_improvement_percent']):.1f}% SLOWER")
    
    if comparison['memory_reduction_percent'] > 0:
        print(f"   Memory usage reduced by {comparison['memory_reduction_percent']:.1f}%")
    else:
        print(f"   Memory usage increased by {abs(comparison['memory_reduction_percent']):.1f}%")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8 Models - Trimmed vs Non-trimmed")
    parser.add_argument("--original", type=str, required=True, help="Path to original ONNX model")
    parser.add_argument("--trimmed", type=str, required=True, help="Path to trimmed ONNX model")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--output-dir", type=str, default="benchmark_results_onnx", help="Output directory for results")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    
    args = parser.parse_args()
    
    print("YOLOv8 Model Benchmark - Trimmed vs Non-trimmed")
    print("="*60)
    
    # Initialize benchmarks
    original_benchmark = ModelBenchmark(args.original)
    trimmed_benchmark = ModelBenchmark(args.trimmed)
    
    # Warmup phase
    print(f"\nWARMUP PHASE ({args.warmup} iterations)")
    print("-" * 40)
    original_benchmark.warmup(args.warmup)
    trimmed_benchmark.warmup(args.warmup)
    
    # Benchmark phase
    print(f"\nBENCHMARK PHASE ({args.iterations} iterations)")
    print("-" * 40)
    
    print("\n1. Benchmarking ORIGINAL model...")
    original_results = original_benchmark.benchmark(args.iterations)
    
    print("\n2. Benchmarking TRIMMED model...")
    trimmed_results = trimmed_benchmark.benchmark(args.iterations)
    
    # Compare results
    comparison = compare_models(original_results, trimmed_results)
    
    # Print summary
    print_summary(original_results, trimmed_results, comparison)
    
    # Save results
    save_results(original_results, trimmed_results, comparison, args.output_dir)
    
    # Generate plots
    if not args.no_plots:
        print(f"\nGenerating visualization plots...")
        plot_results(original_results, trimmed_results, args.output_dir)
    
    print(f"\nBenchmark completed! Results saved to '{args.output_dir}/'")

if __name__ == "__main__":
    main()