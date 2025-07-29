"""
RoI Operations Benchmark for Full HD Input Processing
This module benchmarks RoIPool and RoIAlign operations on feature maps derived from Full HD images.
The implementation simulates realistic conditions for object detection pipelines where input images
are 1920x1080 pixels and processed through CNN backbones with typical downsampling ratios.

Key Components:
- Feature map simulation with realistic dimensions
- ROI generation for Full HD coordinate space
- Performance benchmarking with GPU synchronization
- Memory usage monitoring
- Spatial scale calculations for real-world scenarios

Author: EVEMASK Team
Version: 1.0.0
"""

import torch
import torchvision
import time
import random

def setup_device():
    """
    Initialize and configure the compute device.
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using compute device: {device}")
    if device.type == 'cuda':
        print(f"[DEVICE] GPU: {torch.cuda.get_device_name()}")
        print(f"[DEVICE] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    return device

def create_feature_map(device, downsample_factor=32):
    """
    Create feature map tensor simulating CNN backbone output for Full HD input.
    
    Args:
        device (torch.device): Target device for tensor allocation
        downsample_factor (int): Downsampling ratio from original image (default: 32x)
        
    Returns:
        torch.Tensor: Feature map tensor with shape [batch, channels, height, width]
    """
    # Full HD dimensions
    original_height, original_width = 1080, 1920
    
    # Calculate feature map dimensions after downsampling
    feature_height = original_height // downsample_factor  # 1080/32 = 33.75 -> 34
    feature_width = original_width // downsample_factor    # 1920/32 = 60
    feature_channels = 256  # Typical channel count from ResNet/FPN backbone
    
    input_tensor = torch.randn(1, feature_channels, feature_height, feature_width).to(device)
    
    print(f"[FEATURE_MAP] Original image size: {original_width}x{original_height}")
    print(f"[FEATURE_MAP] Downsample factor: {downsample_factor}x")
    print(f"[FEATURE_MAP] Feature map size: {feature_width}x{feature_height}")
    print(f"[FEATURE_MAP] Feature channels: {feature_channels}")
    print(f"[FEATURE_MAP] Tensor shape: {input_tensor.shape}")
    
    return input_tensor, (feature_height, feature_width), (original_height, original_width)

def generate_rois(num_rois, feature_size, device, coordinate_space='feature'):
    """
    Generate random regions of interest (ROIs) for benchmarking.
    
    Args:
        num_rois (int): Number of ROIs to generate
        feature_size (tuple): Feature map dimensions (height, width)
        device (torch.device): Target device for tensor allocation
        coordinate_space (str): 'feature' for feature map coords, 'original' for image coords
        
    Returns:
        torch.Tensor: ROI tensor with shape [num_rois, 5] format: [batch_idx, x1, y1, x2, y2]
    """
    feature_height, feature_width = feature_size
    
    if coordinate_space == 'feature':
        # Generate ROIs in feature map coordinate space
        rois_list = []
        for i in range(num_rois):
            x1 = random.uniform(0, feature_width - 2)
            y1 = random.uniform(0, feature_height - 2)
            x2 = random.uniform(x1 + 1, feature_width)
            y2 = random.uniform(y1 + 1, feature_height)
            rois_list.append([0, x1, y1, x2, y2])  # batch_index=0, x1, y1, x2, y2
            
        print(f"[ROI_GEN] Generated {num_rois} ROIs in feature map coordinates")
        print(f"[ROI_GEN] Coordinate range: X[0, {feature_width}], Y[0, {feature_height}]")
        
    else:  # original coordinate space
        # Generate ROIs in original image coordinate space
        original_height, original_width = 1080, 1920
        rois_list = []
        for i in range(num_rois):
            x1 = random.uniform(0, original_width - 100)
            y1 = random.uniform(0, original_height - 100)
            x2 = random.uniform(x1 + 50, min(x1 + 300, original_width))
            y2 = random.uniform(y1 + 50, min(y1 + 300, original_height))
            rois_list.append([0, x1, y1, x2, y2])
            
        print(f"[ROI_GEN] Generated {num_rois} ROIs in original image coordinates")
        print(f"[ROI_GEN] Coordinate range: X[0, {original_width}], Y[0, {original_height}]")
    
    rois = torch.tensor(rois_list, dtype=torch.float32).to(device)
    print(f"[ROI_GEN] ROI tensor shape: {rois.shape}")
    print(f"[ROI_GEN] Sample ROI: [{rois[0, 0]:.0f}, {rois[0, 1]:.1f}, {rois[0, 2]:.1f}, {rois[0, 3]:.1f}, {rois[0, 4]:.1f}]")
    
    return rois

def initialize_roi_operations(output_size, spatial_scale, device):
    """
    Initialize RoIPool and RoIAlign operations with specified parameters.
    
    Args:
        output_size (tuple): Output dimensions for pooled features (height, width)
        spatial_scale (float): Scale factor from image coordinates to feature map coordinates
        device (torch.device): Target device for operations
        
    Returns:
        tuple: (roi_pool, roi_align) initialized operations
    """
    roi_pool = torchvision.ops.RoIPool(
        output_size=output_size, 
        spatial_scale=spatial_scale
    ).to(device)
    
    roi_align = torchvision.ops.RoIAlign(
        output_size=output_size, 
        spatial_scale=spatial_scale, 
        sampling_ratio=2,
        aligned=True  # Use aligned=True for higher accuracy
    ).to(device)
    
    print(f"[ROI_OPS] Initialized RoIPool and RoIAlign")
    print(f"[ROI_OPS] Output size: {output_size}")
    print(f"[ROI_OPS] Spatial scale: {spatial_scale:.6f}")
    print(f"[ROI_OPS] RoIAlign sampling ratio: 2")
    print(f"[ROI_OPS] RoIAlign aligned: True")
    
    return roi_pool, roi_align

def gpu_warmup(roi_pool, roi_align, input_tensor, rois, warmup_iterations=5):
    """
    Perform GPU warmup to stabilize performance measurements.
    
    Args:
        roi_pool: RoIPool operation
        roi_align: RoIAlign operation
        input_tensor (torch.Tensor): Input feature map
        rois (torch.Tensor): Regions of interest
        warmup_iterations (int): Number of warmup iterations
    """
    print(f"[WARMUP] Starting GPU warmup with {warmup_iterations} iterations...")
    
    for i in range(warmup_iterations):
        _ = roi_pool(input_tensor, rois)
        _ = roi_align(input_tensor, rois)
        
    torch.cuda.synchronize()
    print(f"[WARMUP] GPU warmup completed")

def benchmark_roi_operations(roi_pool, roi_align, input_tensor, rois, num_iterations=100):
    """
    Benchmark RoIPool and RoIAlign operations with precise timing.
    
    Args:
        roi_pool: RoIPool operation
        roi_align: RoIAlign operation
        input_tensor (torch.Tensor): Input feature map
        rois (torch.Tensor): Regions of interest
        num_iterations (int): Number of benchmark iterations
        
    Returns:
        tuple: (roi_pool_time, roi_align_time, pool_output, align_output)
    """
    print(f"[BENCHMARK] Starting performance benchmark with {num_iterations} iterations...")
    
    # Benchmark RoIPool
    print(f"[BENCHMARK] Testing RoIPool...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for iteration in range(num_iterations):
        pool_output = roi_pool(input_tensor, rois)
        
    torch.cuda.synchronize()
    roi_pool_time = time.time() - start_time
    
    print(f"[BENCHMARK] RoIPool completed: {roi_pool_time:.4f}s total")
    print(f"[BENCHMARK] RoIPool average: {roi_pool_time/num_iterations*1000:.2f}ms per iteration")
    print(f"[BENCHMARK] RoIPool output shape: {pool_output.shape}")
    
    # Benchmark RoIAlign
    print(f"[BENCHMARK] Testing RoIAlign...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for iteration in range(num_iterations):
        align_output = roi_align(input_tensor, rois)
        
    torch.cuda.synchronize()
    roi_align_time = time.time() - start_time
    
    print(f"[BENCHMARK] RoIAlign completed: {roi_align_time:.4f}s total")
    print(f"[BENCHMARK] RoIAlign average: {roi_align_time/num_iterations*1000:.2f}ms per iteration")
    print(f"[BENCHMARK] RoIAlign output shape: {align_output.shape}")
    
    return roi_pool_time, roi_align_time, pool_output, align_output

def analyze_performance(roi_pool_time, roi_align_time):
    """
    Analyze and compare performance between RoIPool and RoIAlign.
    
    Args:
        roi_pool_time (float): Total time for RoIPool operations
        roi_align_time (float): Total time for RoIAlign operations
    """
    print(f"[ANALYSIS] Performance comparison:")
    
    speedup_ratio = roi_align_time / roi_pool_time
    if speedup_ratio > 1:
        faster_op = "RoIPool"
        speedup = speedup_ratio
    else:
        faster_op = "RoIAlign"
        speedup = 1 / speedup_ratio
        
    print(f"[ANALYSIS] {faster_op} is {speedup:.2f}x faster")
    print(f"[ANALYSIS] RoIPool: {roi_pool_time:.4f}s")
    print(f"[ANALYSIS] RoIAlign: {roi_align_time:.4f}s")
    print(f"[ANALYSIS] Time difference: {abs(roi_align_time - roi_pool_time):.4f}s")

def monitor_gpu_memory():
    """
    Monitor and report GPU memory usage.
    """
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        
        print(f"[MEMORY] GPU memory allocated: {allocated_gb:.2f} GB")
        print(f"[MEMORY] GPU memory reserved: {reserved_gb:.2f} GB")
        print(f"[MEMORY] Memory utilization: {allocated_gb/reserved_gb*100:.1f}%")
    else:
        print(f"[MEMORY] Running on CPU - no GPU memory to monitor")

def demonstrate_real_world_usage(input_tensor, feature_size, original_size, device):
    """
    Demonstrate real-world usage with proper spatial scaling.
    
    Args:
        input_tensor (torch.Tensor): Feature map tensor
        feature_size (tuple): Feature map dimensions
        original_size (tuple): Original image dimensions
        device (torch.device): Compute device
    """
    print(f"[DEMO] Real-world usage demonstration:")
    
    original_height, original_width = original_size
    feature_height, feature_width = feature_size
    
    # Calculate spatial scale
    spatial_scale = feature_height / original_height
    
    print(f"[DEMO] Original image: {original_width}x{original_height}")
    print(f"[DEMO] Feature map: {feature_width}x{feature_height}")
    print(f"[DEMO] Spatial scale: {spatial_scale:.6f}")
    
    # Generate ROIs in original image coordinates
    demo_rois = generate_rois(100, feature_size, device, coordinate_space='original')
    
    # Initialize RoIAlign with real spatial scale
    roi_align_real = torchvision.ops.RoIAlign(
        output_size=(7, 7), 
        spatial_scale=spatial_scale, 
        sampling_ratio=2,
        aligned=True
    ).to(device)
    
    # Process ROIs
    demo_output = roi_align_real(input_tensor, demo_rois)
    
    print(f"[DEMO] Processed {demo_rois.shape[0]} ROIs")
    print(f"[DEMO] Output tensor shape: {demo_output.shape}")
    print(f"[DEMO] Output features per ROI: {demo_output.shape[1]} channels, {demo_output.shape[2]}x{demo_output.shape[3]} spatial")

def main():
    """
    Main execution function for RoI operations benchmark.
    """
    print("="*80)
    print("RoI OPERATIONS BENCHMARK FOR FULL HD PROCESSING")
    print("="*80)
    
    # Setup and initialization
    device = setup_device()
    input_tensor, feature_size, original_size = create_feature_map(device)
    
    # ROI generation
    num_rois = 1000
    rois = generate_rois(num_rois, feature_size, device, coordinate_space='feature')
    
    # Initialize operations
    output_size = (7, 7)  # Standard output size for object detection
    spatial_scale = 1.0   # Using feature map coordinates
    roi_pool, roi_align = initialize_roi_operations(output_size, spatial_scale, device)
    
    # GPU warmup
    gpu_warmup(roi_pool, roi_align, input_tensor, rois)
    
    # Memory monitoring before benchmark
    print(f"[STATUS] Pre-benchmark memory status:")
    monitor_gpu_memory()
    
    # Performance benchmark
    roi_pool_time, roi_align_time, pool_output, align_output = benchmark_roi_operations(
        roi_pool, roi_align, input_tensor, rois, num_iterations=100
    )
    
    # Performance analysis
    analyze_performance(roi_pool_time, roi_align_time)
    
    # Memory monitoring after benchmark
    print(f"[STATUS] Post-benchmark memory status:")
    monitor_gpu_memory()
    
    # Real-world demonstration
    demonstrate_real_world_usage(input_tensor, feature_size, original_size, device)
    
    print("="*80)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("="*80)

if __name__ == "__main__":
    main()