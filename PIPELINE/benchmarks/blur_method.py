import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import json

# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

def create_mask_from_polygon(image_shape: Tuple[int, int], polygon: List[Tuple[int, int]]) -> np.ndarray:
    """
    Tạo mask từ polygon.
    Args:
        image_shape: (height, width)
        polygon: List of (x, y) coordinates
    Returns:
        mask: numpy array (H, W) with values 0-255
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    if len(polygon) > 2:
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    return mask

def resize_mask_to_image(mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Resize mask to match target dimensions.
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    resized_mask = F.interpolate(mask, size=(target_h, target_w), mode='bilinear', align_corners=False)
    resized_mask = resized_mask.squeeze().clamp(0, 1)
    return resized_mask

# ========================================================================
# BLUR METHODS
# ========================================================================

def censored_options(image_tensor: torch.Tensor, downscale_factor: int = 20) -> torch.Tensor:
    """
    Your custom blur method (from original code).
    """
    device = image_tensor.device
    _, h, w = image_tensor.shape
    
    # Downscale
    small_h, small_w = max(1, h // downscale_factor), max(1, w // downscale_factor)
    downscaled = F.interpolate(image_tensor.unsqueeze(0), size=(small_h, small_w), 
                              mode='bilinear', align_corners=False)
    
    # Upscale back
    upscaled = F.interpolate(downscaled, size=(h, w), mode='bilinear', align_corners=False)
    return upscaled.squeeze(0)

class BlurMethod:
    """Base class for blur methods"""
    def __init__(self, name: str):
        self.name = name
    
    def apply(self, image: np.ndarray, polygon: List[Tuple[int, int]], **kwargs) -> Tuple[np.ndarray, float]:
        """
        Apply blur method and return result with inference time.
        Returns:
            (blurred_image, inference_time_ms)
        """
        raise NotImplementedError

class CustomBlurMethod(BlurMethod):
    """Your custom blur method"""
    def __init__(self, downscale_factor: int = 20):
        super().__init__(f"Custom Blur")
        self.downscale_factor = downscale_factor
    
    def apply(self, image: np.ndarray, polygon: List[Tuple[int, int]], **kwargs) -> Tuple[np.ndarray, float]:
        # Convert to tensor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        start_time = time.perf_counter()
        
        # Create mask
        mask = create_mask_from_polygon(image.shape[:2], polygon)
        
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(device) / 255.0
        
        # Apply custom blur
        blur_img = censored_options(image_tensor, self.downscale_factor)
        
        # Apply mask
        mask_tensor = torch.from_numpy(mask).float().to(device) / 255.0
        h, w = image_tensor.shape[1], image_tensor.shape[2]
        
        if mask_tensor.shape != (h, w):
            mask_tensor = resize_mask_to_image(mask_tensor, h, w)
        
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        
        # Blend
        processed_image = (1 - mask_tensor) * image_tensor + mask_tensor * blur_img
        processed_image = processed_image.permute(1, 2, 0).cpu().numpy()
        processed_image = (processed_image * 255).astype(np.uint8)
        
        # GPU synchronization for accurate timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000  # Convert to ms
        
        return processed_image, inference_time

class GaussianBlurMethod(BlurMethod):
    """Gaussian blur method"""
    def __init__(self, kernel_size: int = 51, sigma: float = 0):
        super().__init__(f"Gaussian Blur")
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def apply(self, image: np.ndarray, polygon: List[Tuple[int, int]], **kwargs) -> Tuple[np.ndarray, float]:
        start_time = time.perf_counter()
        
        # Create mask
        mask = create_mask_from_polygon(image.shape[:2], polygon)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)
        
        # Apply mask
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
        result = image * (1 - mask_3ch) + blurred * mask_3ch
        result = result.astype(np.uint8)
        
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000
        
        return result, inference_time

class MedianBlurMethod(BlurMethod):
    """Median blur method"""
    def __init__(self, kernel_size: int = 51):
        super().__init__(f"Median Blur")
        self.kernel_size = kernel_size
    
    def apply(self, image: np.ndarray, polygon: List[Tuple[int, int]], **kwargs) -> Tuple[np.ndarray, float]:
        start_time = time.perf_counter()
        
        # Create mask
        mask = create_mask_from_polygon(image.shape[:2], polygon)
        
        # Apply median blur
        blurred = cv2.medianBlur(image, self.kernel_size)
        
        # Apply mask
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
        result = image * (1 - mask_3ch) + blurred * mask_3ch
        result = result.astype(np.uint8)
        
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000
        
        return result, inference_time

class MotionBlurMethod(BlurMethod):
    """Motion blur method"""
    def __init__(self, kernel_size: int = 15, angle: float = 0):
        super().__init__(f"Motion Blur")
        self.kernel_size = kernel_size
        self.angle = angle
    
    def apply(self, image: np.ndarray, polygon: List[Tuple[int, int]], **kwargs) -> Tuple[np.ndarray, float]:
        start_time = time.perf_counter()
        
        # Create mask
        mask = create_mask_from_polygon(image.shape[:2], polygon)
        
        # Create motion blur kernel
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        kernel[int((self.kernel_size-1)/2), :] = np.ones(self.kernel_size)
        kernel = kernel / self.kernel_size
        
        # Rotate kernel if angle is specified
        if self.angle != 0:
            center = (self.kernel_size//2, self.kernel_size//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, self.angle, 1.0)
            kernel = cv2.warpAffine(kernel, rotation_matrix, (self.kernel_size, self.kernel_size))
        
        # Apply motion blur
        blurred = cv2.filter2D(image, -1, kernel)
        
        # Apply mask
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
        result = image * (1 - mask_3ch) + blurred * mask_3ch
        result = result.astype(np.uint8)
        
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000
        
        return result, inference_time

class FastBoxBlurMethod(BlurMethod):
    """Fast box blur method"""
    def __init__(self, kernel_size: int = 51):
        super().__init__(f"Box Blur")
        self.kernel_size = kernel_size
    
    def apply(self, image: np.ndarray, polygon: List[Tuple[int, int]], **kwargs) -> Tuple[np.ndarray, float]:
        start_time = time.perf_counter()
        
        # Create mask
        mask = create_mask_from_polygon(image.shape[:2], polygon)
        
        # Apply box blur (fast)
        blurred = cv2.blur(image, (self.kernel_size, self.kernel_size))
        
        # Apply mask
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
        result = image * (1 - mask_3ch) + blurred * mask_3ch
        result = result.astype(np.uint8)
        
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000
        
        return result, inference_time

# ========================================================================
# PERFORMANCE COMPARISON FRAMEWORK
# ========================================================================

class BlurPerformanceComparison:
    """Framework to compare inference time of different blur methods"""
    
    def __init__(self):
        self.methods = []
        self.results = []
    
    def add_method(self, method: BlurMethod):
        """Add a blur method to comparison"""
        self.methods.append(method)
    
    def compare(self, image: np.ndarray, polygon: List[Tuple[int, int]], 
                num_runs: int = 10, warmup_runs: int = 3) -> Dict[str, Any]:
        """
        Compare inference time of all blur methods.
        Args:
            image: Input image (H, W, C)
            polygon: List of (x, y) coordinates
            num_runs: Number of runs for timing average
            warmup_runs: Number of warmup runs (not counted in timing)
        Returns:
            Performance comparison results dictionary
        """
        results = {}
        
        for method in self.methods:
            print(f"Testing {method.name}...")
            
            # Warmup runs (especially important for GPU methods)
            print(f"  Warming up ({warmup_runs} runs)...")
            for _ in range(warmup_runs):
                try:
                    method.apply(image, polygon)
                except Exception as e:
                    print(f"Warmup error in {method.name}: {e}")
                    break
            
            # Actual timing runs
            print(f"  Measuring performance ({num_runs} runs)...")
            times = []
            sample_result = None
            
            for _ in range(num_runs):
                try:
                    result_img, inference_time = method.apply(image, polygon)
                    times.append(inference_time)
                    if sample_result is None:
                        sample_result = result_img
                except Exception as e:
                    print(f"Error in {method.name}: {e}")
                    continue
            
            if not times or sample_result is None:
                print(f"Failed to benchmark {method.name}")
                continue
            
            # Calculate timing statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            results[method.name] = {
                'avg_inference_time_ms': avg_time,
                'std_inference_time_ms': std_time,
                'min_inference_time_ms': min_time,
                'max_inference_time_ms': max_time,
                'all_times': times,
                'sample_result': sample_result
            }
            
            print(f"  ✓ {method.name}: {avg_time:.2f}±{std_time:.2f}ms (range: {min_time:.2f}-{max_time:.2f}ms)")
        
        return results
    
    def print_performance_table(self, results: Dict[str, Any]):
        """Print performance comparison results in table format"""
        print("\n" + "="*80)
        print("BLUR METHODS PERFORMANCE COMPARISON")
        print("="*80)
        
        # Header
        print(f"{'Method Name':<20} {'Avg Time (ms)':<15} {'Min Time (ms)':<15} {'Max Time (ms)':<15}")
        print("-" * 80)
        
        # Sort by average inference time
        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_inference_time_ms'])
        
        for method_name, data in sorted_results:
            avg_str = f"{data['avg_inference_time_ms']:.2f}±{data['std_inference_time_ms']:.2f}"
            min_str = f"{data['min_inference_time_ms']:.2f}"
            max_str = f"{data['max_inference_time_ms']:.2f}"
            
            print(f"{method_name:<20} {avg_str:<15} {min_str:<15} {max_str:<15}")
        
        print("-" * 80)
        
        # Performance ranking
        print("\nPERFORMANCE RANKING (by average inference time):")
        for i, (method_name, data) in enumerate(sorted_results, 1):
            speedup = sorted_results[-1][1]['avg_inference_time_ms'] / data['avg_inference_time_ms']
            print(f"{i}. {method_name}: {data['avg_inference_time_ms']:.2f}ms (speedup: {speedup:.1f}x)")
    
    def visualize_performance(self, results: Dict[str, Any], save_path: str = None):
        """Visualize performance comparison"""
        method_names = list(results.keys())
        avg_times = [results[name]['avg_inference_time_ms'] for name in method_names]
        std_times = [results[name]['std_inference_time_ms'] for name in method_names]
        
        # Sort by average time
        sorted_indices = np.argsort(avg_times)
        method_names = [method_names[i] for i in sorted_indices]
        avg_times = [avg_times[i] for i in sorted_indices]
        std_times = [std_times[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, 6))
        
        # Bar chart with error bars
        bars = plt.bar(method_names, avg_times, yerr=std_times, capsize=5, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
        
        plt.title('Blur Methods Inference Time Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Inference Time (ms)', fontsize=12)
        plt.xlabel('Blur Method', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, avg_time, std_time in zip(bars, avg_times, std_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_time + 0.5,
                    f'{avg_time:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def visualize_sample_results(self, image: np.ndarray, polygon, 
                               results: Dict[str, Any], save_path: str = None):
        """Visualize sample results from each method"""
        n_methods = len(results) + 1  # +1 for original
        cols = min(3, n_methods)
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Show original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Draw polygon on original
        if polygon is not None and len(polygon) > 0:
            if isinstance(polygon, list):
                poly_array = np.array(polygon)
            else:
                poly_array = polygon
            
            if len(poly_array) > 0:
                axes[0, 0].plot(poly_array[:, 0], poly_array[:, 1], 'r-', linewidth=2)
                axes[0, 0].plot([poly_array[-1, 0], poly_array[0, 0]], 
                               [poly_array[-1, 1], poly_array[0, 1]], 'r-', linewidth=2)
        
        # Show results sorted by performance
        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_inference_time_ms'])
        
        idx = 1
        for method_name, data in sorted_results:
            row = idx // cols
            col = idx % cols
            
            axes[row, col].imshow(data['sample_result'])
            
            # Create title with timing info
            title = f"{method_name}\n"
            title += f"Avg: {data['avg_inference_time_ms']:.1f}ms\n"
            title += f"Range: {data['min_inference_time_ms']:.1f}-{data['max_inference_time_ms']:.1f}ms"
            
            axes[row, col].set_title(title, fontsize=10, fontweight='bold')
            axes[row, col].axis('off')
            idx += 1
        
        # Hide unused subplots
        for i in range(idx, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()

# ========================================================================
# EXAMPLE USAGE
# ========================================================================

def create_performance_comparison():
    """Create a performance comparison with various blur methods"""
    
    # Initialize comparison framework
    comparison = BlurPerformanceComparison()
    
    # Add different blur methods
    comparison.add_method(CustomBlurMethod(downscale_factor=20))
    comparison.add_method(GaussianBlurMethod(kernel_size=51))
    comparison.add_method(MedianBlurMethod(kernel_size=31))
    comparison.add_method(MotionBlurMethod(kernel_size=21))
    comparison.add_method(FastBoxBlurMethod(kernel_size=31))
    
    return comparison

def run_performance_benchmark():
    """Example of how to use the performance comparison framework"""
    
    # Try to use real image file, fallback to sample image
    try:
        image = cv2.imread("benchmarks/sample/blur/draftkings.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with open("benchmarks/sample/blur/draftkings.json", "r") as f:
            data = json.load(f)
        polygon = data["shapes"][0]["points"]
        print("Using real image file")
    except:
        # Fallback to sample image if file not found
        print("File not found, using sample image")
        image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        polygon = [(100, 100), (300, 120), (280, 250), (120, 230)]
    
    # Create comparison
    comparison = create_performance_comparison()
    
    # Run performance benchmark
    print("Running blur methods performance benchmark...")
    print(f"Image size: {image.shape}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    results = comparison.compare(image, polygon, num_runs=10, warmup_runs=3)
    
    # Print results
    comparison.print_performance_table(results)
    
    # Visualize performance
    comparison.visualize_performance(results)
    
    # Visualize sample results
    comparison.visualize_sample_results(image, polygon, results)
    
    return results

# Run benchmark
if __name__ == "__main__":
    results = run_performance_benchmark()