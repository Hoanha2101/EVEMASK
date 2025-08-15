"""
Enhanced Custom Blur Method for Image Censoring System with Original + Polygon Mask Visualization.
This system provides advanced image blurring capabilities using downscaling and upscaling techniques
to create natural-looking censoring effects while preserving image quality outside the censored regions.

The system features:
- Polygon-based mask creation for precise censoring areas
- Multi-step blur processing with downscaling and upscaling
- Real-time visualization of processing steps
- Enhanced visualization showing original image + polygon mask
- Configurable blur intensity through downscale factors
- GPU acceleration support for faster processing

Author: EVEMASK Team
Version: 1.1.0 - Enhanced with Original + Polygon Mask Display
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Tuple
import json
import os
import csv
import time

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

def create_colored_polygon_outline(image_shape: Tuple[int, int], polygon: List[Tuple[int, int]], 
                                  outline_color: Tuple[int, int, int] = (255, 0, 0),
                                  background_color: Tuple[int, int, int] = (255, 255, 255),
                                  thickness: int = 3) -> np.ndarray:
    """
    Create a colored visualization showing polygon outline (viền) on white background.
    
    Args:
        image_shape (Tuple[int, int]): Target image dimensions (height, width)
        polygon (List[Tuple[int, int]]): List of (x, y) coordinates defining the polygon
        outline_color (Tuple[int, int, int]): RGB color for polygon outline (default: red)
        background_color (Tuple[int, int, int]): RGB color for background (default: white)
        thickness (int): Thickness of the polygon outline (default: 3)
        
    Returns:
        np.ndarray: Image with shape (H, W, 3) showing polygon outline only
    """
    # Create white background
    outline_image = np.full((image_shape[0], image_shape[1], 3), background_color, dtype=np.uint8)
    
    if len(polygon) > 2:
        # Convert polygon points to OpenCV format and draw outline only
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(outline_image, [pts], isClosed=True, color=outline_color, thickness=thickness)
    
    return outline_image

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
            - colored_mask: Colored visualization of polygon (red on white)
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
    
    # Step 4: Create masks
    mask = create_mask_from_polygon(image.shape[:2], polygon)
    polygon_outline = create_colored_polygon_outline(image.shape[:2], polygon)
    
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
        'polygon_outline': polygon_outline,
        'final_result': tensor_to_numpy(final_result)
    }
    
    return results

def visualize_original_and_polygon(image: np.ndarray, polygon: List[Tuple[int, int]], 
                                  save_path: str = None):
    """
    Visualize original image on the left and polygon outline on the right.
    
    This function creates a side-by-side comparison showing:
    - Left: Original input image
    - Right: Polygon outline visualization (red outline on white background)
    
    Args:
        image (np.ndarray): Input image to display
        polygon (List[Tuple[int, int]]): Polygon coordinates for outline visualization
        save_path (str, optional): Path to save the visualization image
    """
    # Create polygon outline visualization
    polygon_outline = create_colored_polygon_outline(image.shape[:2], polygon)
    
    # Create visualization with 2 images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Right: Polygon outline
    axes[1].imshow(polygon_outline)
    axes[1].set_title('Polygon Outline', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Add main title
    fig.suptitle('Original Image And Polygon Outline Visualization', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    

def visualize_complete_pipeline(image: np.ndarray, polygon: List[Tuple[int, int]], 
                               downscale_factor: int = 20, save_path: str = None):
    """
    Complete visualization showing original + mask, then the 3-step blur process.
    
    This function creates a comprehensive visualization with:
    - Top row: Original image + Polygon mask
    - Bottom row: Downscaled + Upscaled + Final result
    
    Args:
        image (np.ndarray): Input image to process
        polygon (List[Tuple[int, int]]): Polygon coordinates for blur region
        downscale_factor (int): Factor used for downscaling (blur intensity)
        save_path (str, optional): Path to save the visualization image
    """
    # Get all processing steps
    results = custom_blur_with_steps(image, polygon, downscale_factor)
    
    # Create comprehensive visualization with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top row: Original and polygon mask
    axes[0, 0].imshow(results['original'])
    axes[0, 0].set_title('1. Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(results['polygon_outline'])
    axes[0, 1].set_title('2. Polygon Outline', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Empty space in top right
    axes[0, 2].axis('off')
    axes[0, 2].text(0.5, 0.5, f'Downscale Factor:\n{downscale_factor}x\n\nDevice: {"CUDA" if torch.cuda.is_available() else "CPU"}', 
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # Bottom row: Processing steps
    axes[1, 0].imshow(results['downscaled'])
    axes[1, 0].set_title(f'3. Downscaled\n(Factor: {downscale_factor}x)\nSize: {results["downscaled"].shape[:2]}', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(results['upscaled'])
    axes[1, 1].set_title('4. Upscaled\n(Blur Effect)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(results['final_result'])
    axes[1, 2].set_title('5. Final Blended Result', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Add main title
    fig.suptitle('Complete Custom Blur Pipeline - From Original to Final Result', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def run_enhanced_blur_demo():
    """
    Enhanced demo function showcasing original + polygon mask visualization.
    
    This function demonstrates:
    1. Original image + polygon mask side-by-side view
    2. Complete pipeline visualization
    3. Different downscale factors for blur intensity comparison
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
        # Fallback to sample data with more visible patterns
        print("File not found, creating sample image and polygon")
        image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        
        # Add some colorful patterns to make blur effect more visible
        image[50:150, 100:250] = [255, 100, 100]    # Red rectangle
        image[200:300, 300:450] = [100, 255, 100]   # Green rectangle
        image[100:200, 400:550] = [100, 100, 255]   # Blue rectangle
        
        # Add some text-like patterns
        cv2.rectangle(image, (150, 180), (400, 220), (0, 0, 0), -1)
        cv2.putText(image, "SAMPLE TEXT", (160, 205), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        polygon = [(150, 120), (450, 140), (430, 280), (170, 260)]
    
    print("\n" + "="*70)
    print("ENHANCED CUSTOM BLUR METHOD - COMPLETE VISUALIZATION")
    print("="*70)
    
    # Prepare output directory for saving figures and results
    save_dir = "benchmarks/results/censored_demo/"
    os.makedirs(save_dir, exist_ok=True)

    # 1. Show and save original + polygon mask visualization
    print("\n1. Displaying Original Image + Polygon Mask:")
    orig_poly_fig_path = os.path.join(save_dir, "original_polygon.png")
    visualize_original_and_polygon(image, polygon, save_path=orig_poly_fig_path)
    
    # 2. Show and save complete pipeline visualization
    print("\n2. Complete Processing Pipeline:")
    full_pipeline_fig_path = os.path.join(save_dir, "complete_pipeline.png")
    visualize_complete_pipeline(image, polygon, downscale_factor=20, save_path=full_pipeline_fig_path)
    
    # 3. Test different blur intensities and save results + CSV
    print("\n3. Testing Different Blur Intensities:")
    downscale_factors = [10, 30, 50]
    csv_rows = []
    device_label = "CUDA" if torch.cuda.is_available() else "CPU"
    img_h, img_w = image.shape[0], image.shape[1]

    for factor in downscale_factors:
        print(f"\nTesting blur intensity with downscale factor: {factor}x")
        start_time = time.perf_counter()
        results = custom_blur_with_steps(image, polygon, downscale_factor=factor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Save per-factor comparison figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(results['original'])
        axes[0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(results['final_result'])
        axes[1].set_title(f'Blurred (Factor: {factor}x)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        fig.suptitle(f'Blur Intensity Comparison - Factor: {factor}x', fontsize=14, fontweight='bold')
        plt.tight_layout()
        factor_fig_path = os.path.join(save_dir, f"comparison_factor_{factor}.png")
        plt.savefig(factor_fig_path, dpi=150, bbox_inches='tight')
        plt.show()

        # Save per-factor final result image as PNG
        final_rgb = results['final_result']
        final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
        final_img_path = os.path.join(save_dir, f"final_result_factor_{factor}.png")
        cv2.imwrite(final_img_path, final_bgr)

        # Collect CSV row
        csv_rows.append({
            'downscale_factor': factor,
            'device': device_label,
            'image_height': img_h,
            'image_width': img_w,
            'total_time_ms': total_time_ms
        })

    # Write CSV with timing per factor
    csv_path = os.path.join(save_dir, "censored_demo_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['downscale_factor', 'device', 'image_height', 'image_width', 'total_time_ms'])
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"\nFigures saved at: {orig_poly_fig_path}, {full_pipeline_fig_path}")
    print(f"Per-factor comparison figures and final images saved in: {save_dir}")
    print(f"CSV timing results saved at: {csv_path}")

# Main execution - run the enhanced blur demonstration
if __name__ == "__main__":
    run_enhanced_blur_demo()