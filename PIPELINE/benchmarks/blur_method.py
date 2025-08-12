"""
Blur Methods Performance Comparison Framework.
This system provides comprehensive benchmarking capabilities for comparing different image blurring
techniques in terms of inference speed, visual quality, and processing efficiency.

The framework includes:
- Custom downscale/upscale blur method
- Traditional OpenCV blur methods (Gaussian, Median, Motion, Box)
- Performance benchmarking with statistical analysis
- Visualization of results and performance comparisons
- GPU acceleration support for custom methods

Author: EVEMASK Team
Version: 1.0.0
"""

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
    Create a binary mask from polygon coordinates.
    
    This function generates a mask where the polygon area is filled with white (255)
    and the background is black (0), enabling precise region selection for blurring.
    
    Args:
        image_shape (Tuple[int, int]): Target image dimensions (height, width)
        polygon (List[Tuple[int, int]]): List of (x, y) coordinates defining the polygon
        
    Returns:
        np.ndarray: Binary mask with shape (H, W) and values 0-255
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    if len(polygon) > 2:
        # Convert polygon points to OpenCV format and fill the polygon
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    return mask

def resize_mask_to_image(mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Resize mask tensor to match target image dimensions.
    
    This function ensures the mask has the correct dimensions and format for
    blending operations with the target image.
    
    Args:
        mask (torch.Tensor): Input mask tensor
        target_h (int): Target height
        target_w (int): Target width
        
    Returns:
        torch.Tensor: Resized mask tensor with values clamped to [0, 1]
    """
    # Add batch and channel dimensions if needed
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)  # Add batch dimension
    
    # Resize using bilinear interpolation
    resized_mask = F.interpolate(mask, size=(target_h, target_w), mode='bilinear', align_corners=False)
    resized_mask = resized_mask.squeeze().clamp(0, 1)  # Remove extra dims and clamp values
    return resized_mask

# ========================================================================
# BLUR METHODS
# ========================================================================

def censored_options(image_tensor: torch.Tensor, downscale_factor: int = 20) -> torch.Tensor:
    """
    Custom blur method using downscaling and upscaling technique.
    
    This function creates a blur effect by downscaling the image to a smaller size
    and then upscaling it back to the original size, which naturally creates
    a smooth blur effect due to information loss during the process.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor in format (C, H, W)
        downscale_factor (int): Factor by which to downscale (higher = more blur)
        
    Returns:
        torch.Tensor: Blurred image tensor in format (C, H, W)
    """
    device = image_tensor.device
    _, h, w = image_tensor.shape
    
    # Downscale image to create blur effect
    small_h, small_w = max(1, h // downscale_factor), max(1, w // downscale_factor)
    downscaled = F.interpolate(image_tensor.unsqueeze(0), size=(small_h, small_w), 
                              mode='bilinear', align_corners=False)
    
    # Upscale back to original size (creates blur effect)
    upscaled = F.interpolate(downscaled, size=(h, w), mode='bilinear', align_corners=False)
    return upscaled.squeeze(0)

class BlurMethod:
    """
    Base class for all blur methods.
    
    This abstract class defines the interface that all blur methods must implement.
    Each blur method should provide a name and an apply method that returns
    the blurred image along with inference time measurements.
    """
    def __init__(self, name: str):
        """
        Initialize blur method with a descriptive name.
        
        Args:
            name (str): Human-readable name for the blur method
        """
        self.name = name
    
    def apply(self, image: np.ndarray, polygon: List[Tuple[int, int]], **kwargs) -> Tuple[np.ndarray, float]:
        """
        Apply blur method to the specified region and return result with inference time.
        
        Args:
            image (np.ndarray): Input image in RGB format
            polygon (List[Tuple[int, int]]): Polygon coordinates defining the blur region
            **kwargs: Additional method-specific parameters
            
        Returns:
            Tuple[np.ndarray, float]: (blurred_image, inference_time_ms)
        """
        raise NotImplementedError

class CustomBlurMethod(BlurMethod):
    """
    Custom blur method using downscaling and upscaling technique.
    
    This method creates a blur effect by downscaling the image to a smaller size
    and then upscaling it back to the original size. The blur intensity is
    controlled by the downscale factor - higher values create stronger blur effects.
    """
    def __init__(self, downscale_factor: int = 30):
        """
        Initialize custom blur method.
        
        Args:
            downscale_factor (int): Factor by which to downscale (higher = more blur)
        """
        super().__init__(f"Custom Blur")
        self.downscale_factor = downscale_factor
    
    def apply(self, image: np.ndarray, polygon: List[Tuple[int, int]], **kwargs) -> Tuple[np.ndarray, float]:
        """
        Apply custom blur to the specified polygon region.
        
        Args:
            image (np.ndarray): Input image in RGB format
            polygon (List[Tuple[int, int]]): Polygon coordinates defining the blur region
            **kwargs: Additional parameters (not used)
            
        Returns:
            Tuple[np.ndarray, float]: (blurred_image, inference_time_ms)
        """
        # Convert to tensor and move to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        start_time = time.perf_counter()
        
        # Create binary mask from polygon
        mask = create_mask_from_polygon(image.shape[:2], polygon)
        
        # Convert image to tensor and normalize
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(device) / 255.0
        
        # Apply custom blur using downscale/upscale technique
        blur_img = censored_options(image_tensor, self.downscale_factor)
        
        # Prepare mask tensor for blending
        mask_tensor = torch.from_numpy(mask).float().to(device) / 255.0
        h, w = image_tensor.shape[1], image_tensor.shape[2]
        
        # Ensure mask has correct dimensions
        if mask_tensor.shape != (h, w):
            mask_tensor = resize_mask_to_image(mask_tensor, h, w)
        
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        
        # Blend original image with blurred image using mask
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
    """
    Gaussian blur method using OpenCV's GaussianBlur function.
    
    This method applies a Gaussian blur filter which creates a smooth blur effect
    by convolving the image with a Gaussian kernel. The blur intensity is controlled
    by the kernel size and sigma parameters.
    """
    def __init__(self, kernel_size: int = 51, sigma: float = 0):
        """
        Initialize Gaussian blur method.
        
        Args:
            kernel_size (int): Size of the Gaussian kernel (must be odd)
            sigma (float): Standard deviation of the Gaussian kernel
        """
        super().__init__(f"Gaussian Blur")
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def apply(self, image: np.ndarray, polygon: List[Tuple[int, int]], **kwargs) -> Tuple[np.ndarray, float]:
        """
        Apply Gaussian blur to the specified polygon region.
        
        Args:
            image (np.ndarray): Input image in RGB format
            polygon (List[Tuple[int, int]]): Polygon coordinates defining the blur region
            **kwargs: Additional parameters (not used)
            
        Returns:
            Tuple[np.ndarray, float]: (blurred_image, inference_time_ms)
        """
        start_time = time.perf_counter()
        
        # Create binary mask from polygon
        mask = create_mask_from_polygon(image.shape[:2], polygon)
        
        # Apply Gaussian blur to entire image
        blurred = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma)
        
        # Blend original image with blurred image using mask
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
        result = image * (1 - mask_3ch) + blurred * mask_3ch
        result = result.astype(np.uint8)
        
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000
        
        return result, inference_time

class MedianBlurMethod(BlurMethod):
    """
    Median blur method using OpenCV's medianBlur function.
    
    This method applies a median filter which replaces each pixel with the median
    value of its neighborhood. Median blur is effective at removing noise while
    preserving edges, making it useful for certain types of image processing.
    """
    def __init__(self, kernel_size: int = 51):
        """
        Initialize median blur method.
        
        Args:
            kernel_size (int): Size of the median filter kernel (must be odd)
        """
        super().__init__(f"Median Blur")
        self.kernel_size = kernel_size
    
    def apply(self, image: np.ndarray, polygon: List[Tuple[int, int]], **kwargs) -> Tuple[np.ndarray, float]:
        """
        Apply median blur to the specified polygon region.
        
        Args:
            image (np.ndarray): Input image in RGB format
            polygon (List[Tuple[int, int]]): Polygon coordinates defining the blur region
            **kwargs: Additional parameters (not used)
            
        Returns:
            Tuple[np.ndarray, float]: (blurred_image, inference_time_ms)
        """
        start_time = time.perf_counter()
        
        # Create binary mask from polygon
        mask = create_mask_from_polygon(image.shape[:2], polygon)
        
        # Apply median blur to entire image
        blurred = cv2.medianBlur(image, self.kernel_size)
        
        # Blend original image with blurred image using mask
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
        result = image * (1 - mask_3ch) + blurred * mask_3ch
        result = result.astype(np.uint8)
        
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000
        
        return result, inference_time

class MotionBlurMethod(BlurMethod):
    """
    Motion blur method using custom directional kernel.
    
    This method simulates motion blur by applying a directional filter kernel
    that creates a streaking effect in the specified direction. The blur intensity
    is controlled by the kernel size and the direction by the angle parameter.
    """
    def __init__(self, kernel_size: int = 15, angle: float = 0):
        """
        Initialize motion blur method.
        
        Args:
            kernel_size (int): Size of the motion blur kernel
            angle (float): Direction of motion blur in degrees
        """
        super().__init__(f"Motion Blur")
        self.kernel_size = kernel_size
        self.angle = angle
    
    def apply(self, image: np.ndarray, polygon: List[Tuple[int, int]], **kwargs) -> Tuple[np.ndarray, float]:
        """
        Apply motion blur to the specified polygon region.
        
        Args:
            image (np.ndarray): Input image in RGB format
            polygon (List[Tuple[int, int]]): Polygon coordinates defining the blur region
            **kwargs: Additional parameters (not used)
            
        Returns:
            Tuple[np.ndarray, float]: (blurred_image, inference_time_ms)
        """
        start_time = time.perf_counter()
        
        # Create binary mask from polygon
        mask = create_mask_from_polygon(image.shape[:2], polygon)
        
        # Create motion blur kernel (horizontal line)
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        kernel[int((self.kernel_size-1)/2), :] = np.ones(self.kernel_size)
        kernel = kernel / self.kernel_size
        
        # Rotate kernel if angle is specified
        if self.angle != 0:
            center = (self.kernel_size//2, self.kernel_size//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, self.angle, 1.0)
            kernel = cv2.warpAffine(kernel, rotation_matrix, (self.kernel_size, self.kernel_size))
        
        # Apply motion blur using custom kernel
        blurred = cv2.filter2D(image, -1, kernel)
        
        # Blend original image with blurred image using mask
        mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
        result = image * (1 - mask_3ch) + blurred * mask_3ch
        result = result.astype(np.uint8)
        
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000
        
        return result, inference_time

class FastBoxBlurMethod(BlurMethod):
    """
    Fast box blur method using OpenCV's blur function.
    
    This method applies a simple box filter (averaging filter) which is computationally
    efficient and provides a uniform blur effect. Box blur is faster than Gaussian blur
    but may produce less smooth results.
    """
    def __init__(self, kernel_size: int = 51):
        """
        Initialize fast box blur method.
        
        Args:
            kernel_size (int): Size of the box filter kernel
        """
        super().__init__(f"Box Blur")
        self.kernel_size = kernel_size
    
    def apply(self, image: np.ndarray, polygon: List[Tuple[int, int]], **kwargs) -> Tuple[np.ndarray, float]:
        """
        Apply fast box blur to the specified polygon region.
        
        Args:
            image (np.ndarray): Input image in RGB format
            polygon (List[Tuple[int, int]]): Polygon coordinates defining the blur region
            **kwargs: Additional parameters (not used)
            
        Returns:
            Tuple[np.ndarray, float]: (blurred_image, inference_time_ms)
        """
        start_time = time.perf_counter()
        
        # Create binary mask from polygon
        mask = create_mask_from_polygon(image.shape[:2], polygon)
        
        # Apply fast box blur to entire image
        blurred = cv2.blur(image, (self.kernel_size, self.kernel_size))
        
        # Blend original image with blurred image using mask
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
    """
    Framework to compare inference time and performance of different blur methods.
    
    This class provides a comprehensive benchmarking system that can test multiple
    blur methods on the same input image and polygon, measuring inference times
    with statistical analysis and providing detailed performance comparisons.
    """
    
    def __init__(self):
        """Initialize the performance comparison framework."""
        self.methods = []
        self.results = []
    
    def add_method(self, method: BlurMethod):
        """
        Add a blur method to the comparison framework.
        
        Args:
            method (BlurMethod): Blur method instance to add for comparison
        """
        self.methods.append(method)
    
    def compare(self, image: np.ndarray, polygon: List[Tuple[int, int]], 
                num_runs: int = 10, warmup_runs: int = 3) -> Dict[str, Any]:
        """
        Compare inference time of all blur methods.
        
        This method runs each blur method multiple times to get accurate timing
        statistics, including warmup runs to ensure consistent performance.
        
        Args:
            image (np.ndarray): Input image (H, W, C)
            polygon (List[Tuple[int, int]]): List of (x, y) coordinates defining blur region
            num_runs (int): Number of runs for timing average
            warmup_runs (int): Number of warmup runs (not counted in timing)
            
        Returns:
            Dict[str, Any]: Performance comparison results dictionary
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
            
            print(f"  {method.name}: {avg_time:.2f}±{std_time:.2f}ms (range: {min_time:.2f}-{max_time:.2f}ms)")
        
        return results
    
    def print_performance_table(self, results: Dict[str, Any]):
        """
        Print performance comparison results in a formatted table.
        
        This method displays the benchmark results in an easy-to-read table format,
        showing average, minimum, and maximum inference times for each method,
        along with a performance ranking based on speed.
        
        Args:
            results (Dict[str, Any]): Performance comparison results from compare() method
        """
        print("\n" + "="*80)
        print("BLUR METHODS PERFORMANCE COMPARISON")
        print("="*80)
        
        # Table header
        print(f"{'Method Name':<20} {'Avg Time (ms)':<15} {'Min Time (ms)':<15} {'Max Time (ms)':<15}")
        print("-" * 80)
        
        # Sort results by average inference time (fastest first)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_inference_time_ms'])
        
        for method_name, data in sorted_results:
            avg_str = f"{data['avg_inference_time_ms']:.2f}±{data['std_inference_time_ms']:.2f}"
            min_str = f"{data['min_inference_time_ms']:.2f}"
            max_str = f"{data['max_inference_time_ms']:.2f}"
            
            print(f"{method_name:<20} {avg_str:<15} {min_str:<15} {max_str:<15}")
        
        print("-" * 80)
        
        # Performance ranking with speedup calculations
        print("\nPERFORMANCE RANKING (by average inference time):")
        for i, (method_name, data) in enumerate(sorted_results, 1):
            speedup = sorted_results[-1][1]['avg_inference_time_ms'] / data['avg_inference_time_ms']
            print(f"{i}. {method_name}: {data['avg_inference_time_ms']:.2f}ms (speedup: {speedup:.1f}x)")
    
    def visualize_performance(self, results: Dict[str, Any], save_path: str = None):
        """
        Visualize performance comparison with bar chart.
        
        This method creates a bar chart showing the average inference times for each
        blur method, with error bars representing standard deviation. Methods are
        sorted by performance (fastest to slowest).
        
        Args:
            results (Dict[str, Any]): Performance comparison results from compare() method
            save_path (str, optional): Path to save the visualization image
        """
        method_names = list(results.keys())
        avg_times = [results[name]['avg_inference_time_ms'] for name in method_names]
        std_times = [results[name]['std_inference_time_ms'] for name in method_names]
        
        # Sort by average time (fastest first)
        sorted_indices = np.argsort(avg_times)
        method_names = [method_names[i] for i in sorted_indices]
        avg_times = [avg_times[i] for i in sorted_indices]
        std_times = [std_times[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, 6))
        
        # Create bar chart with error bars
        bars = plt.bar(method_names, avg_times, yerr=std_times, capsize=5, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
        
        plt.title('Blur Methods Inference Time Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Inference Time (ms)', fontsize=12)
        plt.xlabel('Blur Method', fontsize=12)
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
        """
        Visualize sample results from each blur method.
        
        This method creates a grid of images showing the original image with the
        polygon region marked, followed by the results from each blur method
        sorted by performance. Each result image includes timing information.
        
        Args:
            image (np.ndarray): Original input image
            polygon: Polygon coordinates defining the blur region
            results (Dict[str, Any]): Performance comparison results from compare() method
            save_path (str, optional): Path to save the visualization image
        """
        n_methods = len(results) + 1  # +1 for original image
        cols = min(3, n_methods)
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Show original image with polygon overlay
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Draw polygon boundary on original image
        if polygon is not None and len(polygon) > 0:
            if isinstance(polygon, list):
                poly_array = np.array(polygon)
            else:
                poly_array = polygon
            
            if len(poly_array) > 0:
                axes[0, 0].plot(poly_array[:, 0], poly_array[:, 1], 'r-', linewidth=2)
                axes[0, 0].plot([poly_array[-1, 0], poly_array[0, 0]], 
                               [poly_array[-1, 1], poly_array[0, 1]], 'r-', linewidth=2)
        
        # Show results sorted by performance (fastest first)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_inference_time_ms'])
        
        idx = 1
        for method_name, data in sorted_results:
            row = idx // cols
            col = idx % cols
            
            axes[row, col].imshow(data['sample_result'])
            
            # Create title with timing information
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
    """
    Create a performance comparison framework with various blur methods.
    
    This function initializes the BlurPerformanceComparison framework and adds
    different types of blur methods for benchmarking, including custom methods
    and traditional OpenCV-based methods.
    
    Returns:
        BlurPerformanceComparison: Configured comparison framework
    """
    
    # Initialize comparison framework
    comparison = BlurPerformanceComparison()
    
    # Add different blur methods with optimized parameters
    comparison.add_method(CustomBlurMethod(downscale_factor=40))  # Custom downscale/upscale method
    comparison.add_method(GaussianBlurMethod(kernel_size=51))     # Traditional Gaussian blur
    comparison.add_method(MedianBlurMethod(kernel_size=31))       # Median filter blur
    comparison.add_method(MotionBlurMethod(kernel_size=21))       # Motion blur simulation
    comparison.add_method(FastBoxBlurMethod(kernel_size=31))      # Fast box blur
    
    return comparison

def run_performance_benchmark():
    """
    Example of how to use the performance comparison framework.
    
    This function demonstrates the complete workflow for benchmarking blur methods:
    1. Load or create test image and polygon data
    2. Initialize comparison framework with various blur methods
    3. Run performance benchmarks with statistical analysis
    4. Display results in table format
    5. Generate visualizations of performance and sample results
    
    Returns:
        Dict[str, Any]: Performance comparison results
    """
    
    # Try to use real image file, fallback to sample image
    try:
        image = cv2.imread("benchmarks/sample/blur/draftkings2.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with open("benchmarks/sample/blur/draftkings2.json", "r") as f:
            data = json.load(f)
        polygon = data["shapes"][0]["points"]
        print("Using real image file")
    except:
        # Fallback to sample image if file not found
        print("File not found, using sample image")
        image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        polygon = [(100, 100), (300, 120), (280, 250), (120, 230)]
    
    # Create comparison framework with various blur methods
    comparison = create_performance_comparison()
    
    # Run performance benchmark with statistical analysis
    print("Running blur methods performance benchmark...")
    print(f"Image size: {image.shape}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    results = comparison.compare(image, polygon, num_runs=10, warmup_runs=3)
    
    # Display results in formatted table
    comparison.print_performance_table(results)
    
    # Generate performance visualization
    comparison.visualize_performance(results)
    
    # Generate sample results visualization
    comparison.visualize_sample_results(image, polygon, results)
    
    return results

# Main execution - run the blur methods performance benchmark
if __name__ == "__main__":
    results = run_performance_benchmark()