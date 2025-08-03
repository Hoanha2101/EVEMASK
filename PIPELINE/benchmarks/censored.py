"""
Custom Blur Method for Image Censoring System.
This system provides advanced image blurring capabilities using downscaling and upscaling techniques
to create natural-looking censoring effects while preserving image quality outside the censored regions.

The system features:
- Polygon-based mask creation for precise censoring areas
- Multi-step blur processing with downscaling and upscaling
- Real-time visualization of processing steps
- Configurable blur intensity through downscale factors
- GPU acceleration support for faster processing

Author: EVEMASK Team
Version: 1.0.0
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Tuple
import json

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

def custom_blur_with_steps(image: np.ndarray, polygon: List[Tuple[int, int]], downscale_factor: int = 20):
    """
    Custom blur method with visualization of processing steps.
    
    This function implements a multi-step blurring process:
    1. Downscale the image to create blur effect
    2. Upscale back to original size
    3. Apply mask to blend original and blurred regions
    4. Return all intermediate steps for visualization
    
    Args:
        image (np.ndarray): Input image in RGB format
        polygon (List[Tuple[int, int]]): Polygon coordinates defining the blur region
        downscale_factor (int): Factor by which to downscale (higher = more blur)
        
    Returns:
        dict: Dictionary containing all processing steps:
            - original: Original input image
            - downscaled: Image after downscaling
            - upscaled: Image after upscaling (blurred effect)
            - mask: Binary mask of the polygon region
            - final_result: Final blended result
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Step 1: Convert original image to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(device) / 255.0
    _, h, w = image_tensor.shape
    
    # Step 2: Downscale image to create blur effect
    small_h, small_w = max(1, h // downscale_factor), max(1, w // downscale_factor)
    downscaled = F.interpolate(image_tensor.unsqueeze(0), size=(small_h, small_w), 
                              mode='bilinear', align_corners=False)
    
    # Step 3: Upscale back to original size (creates blur effect)
    upscaled = F.interpolate(downscaled, size=(h, w), mode='bilinear', align_corners=False)
    upscaled = upscaled.squeeze(0)
    
    # Step 4: Create mask and blend original with blurred regions
    mask = create_mask_from_polygon(image.shape[:2], polygon)
    mask_tensor = torch.from_numpy(mask).float().to(device) / 255.0
    
    # Ensure mask has correct dimensions
    if mask_tensor.shape != (h, w):
        mask_tensor = resize_mask_to_image(mask_tensor, h, w)
    
    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    
    # Final blend: original image outside mask, blurred image inside mask
    final_result = (1 - mask_tensor) * image_tensor + mask_tensor * upscaled
    
    # Convert all results back to numpy for visualization
    def tensor_to_numpy(tensor):
        return (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    results = {
        'original': image,
        'downscaled': tensor_to_numpy(downscaled.squeeze(0)),
        'upscaled': tensor_to_numpy(upscaled),
        'mask': mask,
        'final_result': tensor_to_numpy(final_result)
    }
    
    return results

def visualize_blur_steps(image: np.ndarray, polygon: List[Tuple[int, int]], 
                        downscale_factor: int = 20, save_path: str = None):
    """
    Visualize the 3 main processing steps: downscaled, upscaled, and blended result.
    
    This function creates a side-by-side comparison of the blur processing pipeline,
    showing how the image changes at each step of the custom blur method.
    
    Args:
        image (np.ndarray): Input image to process
        polygon (List[Tuple[int, int]]): Polygon coordinates for blur region
        downscale_factor (int): Factor used for downscaling (blur intensity)
        save_path (str, optional): Path to save the visualization image
    """
    # Get all processing steps
    results = custom_blur_with_steps(image, polygon, downscale_factor)
    
    # Create visualization with 3 images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Step 1: Downscaled image
    axes[0].imshow(results['downscaled'])
    axes[0].set_title(f'1. Downscaled\n(Factor: {downscale_factor}x)\nSize: {results["downscaled"].shape[:2]}', 
                     fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Step 2: Upscaled image (blurred effect)
    axes[1].imshow(results['upscaled'])
    axes[1].set_title('2. Upscaled\n(Blur Effect)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Step 3: Final blended result
    axes[2].imshow(results['final_result'])
    axes[2].set_title('3. Final Blended Result', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Add main title
    fig.suptitle(f'Custom Blur Method - Key Steps', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    # Print processing information
    print(f"Original: {results['original'].shape} → Downscaled: {results['downscaled'].shape} → Final: {results['final_result'].shape}")
    print(f"Downscale factor: {downscale_factor}x | Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

def run_custom_blur_demo():
    """
    Demo function to run custom blur visualization with different parameters.
    
    This function demonstrates the custom blur method by:
    1. Loading a real image and polygon data (or creating sample data)
    2. Testing different downscale factors to show blur intensity variations
    3. Generating visualizations for each parameter setting
    
    The demo shows how different downscale factors affect the blur intensity
    and processing pipeline visualization.
    """
    # Try to load real image, fallback to sample data
    try:
        image = cv2.imread("benchmarks/sample/blur/draftkings.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        with open("benchmarks/sample/blur/draftkings.json", "r") as f:
            data = json.load(f)
        polygon = data["shapes"][0]["points"]
        print("Loaded real image and polygon data")
    except:
        # Fallback to sample data
        print("File not found, creating sample image and polygon")
        image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        # Add some patterns to make blur effect more visible
        image[100:300, 200:400] = [255, 100, 100]  # Red rectangle
        image[50:150, 50:200] = [100, 255, 100]    # Green rectangle
        
        polygon = [(150, 120), (450, 140), (430, 280), (170, 260)]
    
    print("\n" + "="*60)
    print("CUSTOM BLUR METHOD - STEP BY STEP VISUALIZATION")
    print("="*60)
    
    # Test different downscale factors to demonstrate blur intensity variations
    downscale_factors = [10, 20, 50]
    
    for factor in downscale_factors:
        print(f"\nTesting with downscale factor: {factor}x")
        visualize_blur_steps(image, polygon, downscale_factor=factor)

# Main execution - run the custom blur demonstration
if __name__ == "__main__":
    run_custom_blur_demo()