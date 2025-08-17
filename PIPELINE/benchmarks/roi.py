"""
Simplified RoI Operations Benchmark for Full HD Input Processing
This module compares RoIPool vs RoIAlign inference time on Full HD images.
Simplified version with focus on performance comparison only.

Author: EVEMASK Team
Version: 1.0.0
"""

import torch
import torchvision
import time
import random

def main():
    """
    Main benchmark function comparing RoIPool vs RoIAlign performance.
    """
    print("="*60)
    print("ROI BENCHMARK: FULL HD - 100 ROIs - 224x224 OUTPUT")
    print("="*60)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using: {device}")
    
    # Full HD input simulation
    # Original: 1920x1080
    feature_height, feature_width = 1080, 1920
    feature_channels = 256
    
    input_tensor = torch.randn(1, feature_channels, feature_height, feature_width).to(device)
    print(f"[INPUT] Feature map: {feature_width}x{feature_height}, Channels: {feature_channels}")
    
    # Generate 100 ROIs in feature map coordinates
    num_rois = 10
    rois_list = []
    for i in range(num_rois):
        x1 = random.uniform(0, feature_width - 2)
        y1 = random.uniform(0, feature_height - 2)
        x2 = random.uniform(x1 + 1, feature_width)
        y2 = random.uniform(y1 + 1, feature_height)
        rois_list.append([0, x1, y1, x2, y2])  # batch_idx, x1, y1, x2, y2
    
    rois = torch.tensor(rois_list, dtype=torch.float32).to(device)
    print(f"[ROI] Generated {num_rois} ROIs")
    
    # Initialize RoI operations with 224x224 output
    output_size = (224, 224)
    spatial_scale = 1.0
    
    roi_pool = torchvision.ops.RoIPool(
        output_size=output_size, 
        spatial_scale=spatial_scale
    ).to(device)
    
    roi_align = torchvision.ops.RoIAlign(
        output_size=output_size, 
        spatial_scale=spatial_scale, 
        sampling_ratio=2,
        aligned=True
    ).to(device)
    
    print(f"[SETUP] Output size: {output_size}")
    
    # GPU warmup
    print(f"[WARMUP] Warming up GPU...")
    for _ in range(5):
        _ = roi_pool(input_tensor, rois)
        _ = roi_align(input_tensor, rois)
    torch.cuda.synchronize()
    
    # Benchmark iterations
    num_iterations = 10
    print(f"[BENCHMARK] Running {num_iterations} iterations...")
    
    # Benchmark RoIPool
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        pool_output = roi_pool(input_tensor, rois)
    torch.cuda.synchronize()
    roi_pool_time = time.time() - start_time
    
    # Benchmark RoIAlign
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        align_output = roi_align(input_tensor, rois)
    torch.cuda.synchronize()
    roi_align_time = time.time() - start_time
    
    # Results
    print("="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"RoIPool Total Time:  {roi_pool_time:.4f}s")
    print(f"RoIPool Per Iter:    {roi_pool_time/num_iterations*1000:.2f}ms")
    print(f"RoIPool Output:      {pool_output.shape}")
    print("-"*40)
    print(f"RoIAlign Total Time: {roi_align_time:.4f}s")
    print(f"RoIAlign Per Iter:   {roi_align_time/num_iterations*1000:.2f}ms")
    print(f"RoIAlign Output:     {align_output.shape}")
    print("-"*40)
    
    # Performance comparison
    if roi_pool_time < roi_align_time:
        speedup = roi_align_time / roi_pool_time
        print(f"WINNER: RoIPool is {speedup:.2f}x FASTER")
    else:
        speedup = roi_pool_time / roi_align_time
        print(f"WINNER: RoIAlign is {speedup:.2f}x FASTER")
    
    time_diff = abs(roi_align_time - roi_pool_time)
    print(f"Time Difference: {time_diff:.4f}s ({time_diff/num_iterations*1000:.2f}ms per iter)")
    
    print("="*60)

if __name__ == "__main__":
    main()