"""
Image generation and augmentation utilities for the EVEMASK Pipeline system.
Provides a variety of image transformation functions for data augmentation, including rotation, flipping, scaling, brightness/contrast adjustment, noise addition, blurring, cropping, and color shifting.
Handles batch image generation for dataset expansion.

Author: EVEMASK Team
Version: 1.0.0
"""

# ========================================================================
# IMPORTS
# ========================================================================
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from typing import List, Callable, Tuple
import albumentations as A

class GenImage:
    """
    The GenImage class provides methods for generating augmented images from a dataset.
    Supports multiple image transformations for data augmentation and dataset expansion.
    """
    def __init__(self, base_path: str = "recognizeData"):
        """
        Initialize the GenImage class.

        Args:
            base_path (str): Path to the root directory containing subfolders of images.
        """
        self.base_path = base_path
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        
        self._check_and_setup_directory()
        
    def _check_and_setup_directory(self):
        """
        Check the recognizeData directory:
        - If it does not exist: create it and exit the program
        - If it exists but is empty: exit the program
        - If it exists and has data: continue
        """
        if not os.path.exists(self.base_path):
            print(f"Directory '{self.base_path}' does not exist!")
            print(f"Creating directory '{self.base_path}'...")
            os.makedirs(self.base_path, exist_ok=True)
            print(f"Successfully created directory '{self.base_path}'!")
            print("Please add folders containing images to this directory and rerun the program.")
            print(f"   Desired structure:")
            print(f"   {self.base_path}/")
            print(f"   ├── Logo1/")
            print(f"   │   ├── 1.png")
            print(f"   │   ├── 2.png")
            print(f"   │   └── ...")
            print(f"   ├── Logo2/")
            print(f"   │   └── ...")
            print(f"   └── ...")
            exit(0)
        
        # Check if the directory is empty
        subfolders = self.get_subfolders()
        if not subfolders:
            print(f"Directory '{self.base_path}' exists but has no subfolders!")
            print("Please add folders containing images to this directory and rerun the program.")
            print(f"   Desired structure:")
            print(f"   {self.base_path}/")
            print(f"   ├── Logo1/")
            print(f"   │   ├── 1.png")
            print(f"   │   ├── 2.png")
            print(f"   │   └── ...")
            print(f"   ├── Logo2/")
            print(f"   │   └── ...")
            print(f"   └── ...")
            exit(0)
        
        # Check if subfolders contain images
        has_images = False
        for folder in subfolders:
            images = self.get_images_in_folder(folder)
            if images:
                has_images = True
                break
        
        if not has_images:
            print(f"Directory '{self.base_path}' has subfolders but no images!")
            print("Please add images to the subfolders and rerun the program.")
            print(f"   Supported image formats: {', '.join(self.supported_formats)}")
            print(f"   Current subfolders: {', '.join(subfolders)}")
            exit(0)
        
        print(f"Directory '{self.base_path}' is ready!")
        print(f"Found {len(subfolders)} folders: {', '.join(subfolders)}")
        
    def get_subfolders(self) -> List[str]:
        """
        Get a list of all subfolders in the base directory.

        Returns:
            List[str]: List of subfolder names.
        """
        subfolders = []
        for item in os.listdir(self.base_path):
            item_path = os.path.join(self.base_path, item)
            if os.path.isdir(item_path):
                subfolders.append(item)
        return subfolders
    
    def get_images_in_folder(self, folder_name: str) -> List[str]:
        """
        Get a list of image file paths in a given subfolder.

        Args:
            folder_name (str): Name of the subfolder.
        Returns:
            List[str]: List of image file paths.
        """
        folder_path = os.path.join(self.base_path, folder_name)
        images = []
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_name} does not exist!")
            return images
            
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in self.supported_formats):
                images.append(os.path.join(folder_path, file))
        return images
    
    # ========================================================================
    # IMAGE TRANSFORMATION FUNCTIONS
    # ========================================================================
    def rotate_image(self, image_path: str, output_dir: str, prefix: str = "rotated") -> List[str]:
        """
        Rotate an image by multiple angles and save the results.

        Args:
            image_path (str): Path to the input image.
            output_dir (str): Directory to save the rotated images.
            prefix (str): Prefix for output filenames.
        Returns:
            List[str]: List of file paths to the generated images.
        """
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        generated_images = []
        angles = [15, 30, 45, 90, 180, 270, -15, -30, -45]
        
        for i, angle in enumerate(angles):
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, matrix, (w, h), borderValue=(255, 255, 255))
            
            output_path = os.path.join(output_dir, f"{prefix}_{angle}deg_{i+1}.png")
            cv2.imwrite(output_path, rotated)
            generated_images.append(output_path)
            
        return generated_images
    
    def flip_image(self, image_path: str, output_dir: str, prefix: str = "flipped") -> List[str]:
        """
        Flip an image horizontally, vertically, and both, saving each result.

        Args:
            image_path (str): Path to the input image.
            output_dir (str): Directory to save the flipped images.
            prefix (str): Prefix for output filenames.
        Returns:
            List[str]: List of file paths to the generated images.
        """
        image = cv2.imread(image_path)
        if image is None:
            return []
            
        generated_images = []
        
        # Flip horizontally
        flipped_h = cv2.flip(image, 1)
        output_path_h = os.path.join(output_dir, f"{prefix}_horizontal.png")
        cv2.imwrite(output_path_h, flipped_h)
        generated_images.append(output_path_h)
        
        # Flip vertically
        flipped_v = cv2.flip(image, 0)
        output_path_v = os.path.join(output_dir, f"{prefix}_vertical.png")
        cv2.imwrite(output_path_v, flipped_v)
        generated_images.append(output_path_v)
        
        # Flip both directions
        flipped_both = cv2.flip(image, -1)
        output_path_both = os.path.join(output_dir, f"{prefix}_both.png")
        cv2.imwrite(output_path_both, flipped_both)
        generated_images.append(output_path_both)
        
        return generated_images
    
    def scale_image(self, image_path: str, output_dir: str, prefix: str = "scaled") -> List[str]:
        """
        Scale an image by several factors and save the results.

        Args:
            image_path (str): Path to the input image.
            output_dir (str): Directory to save the scaled images.
            prefix (str): Prefix for output filenames.
        Returns:
            List[str]: List of file paths to the generated images.
        """
        image = cv2.imread(image_path)
        if image is None:
            return []
            
        generated_images = []
        scales = [0.8, 0.9, 1.1, 1.2, 1.5]
        
        for i, scale in enumerate(scales):
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(image, (new_w, new_h))
            
            output_path = os.path.join(output_dir, f"{prefix}_{scale}x_{i+1}.png")
            cv2.imwrite(output_path, scaled)
            generated_images.append(output_path)
            
        return generated_images
    
    def adjust_brightness(self, image_path: str, output_dir: str, prefix: str = "brightness") -> List[str]:
        """
        Adjust the brightness of an image by several factors and save the results.

        Args:
            image_path (str): Path to the input image.
            output_dir (str): Directory to save the brightness-adjusted images.
            prefix (str): Prefix for output filenames.
        Returns:
            List[str]: List of file paths to the generated images.
        """
        pil_image = Image.open(image_path)
        generated_images = []
        brightness_values = [0.7, 0.8, 1.2, 1.3, 1.5]
        
        for i, brightness in enumerate(brightness_values):
            enhancer = ImageEnhance.Brightness(pil_image)
            bright_image = enhancer.enhance(brightness)
            
            output_path = os.path.join(output_dir, f"{prefix}_{brightness}_{i+1}.png")
            bright_image.save(output_path)
            generated_images.append(output_path)
            
        return generated_images
    
    def adjust_contrast(self, image_path: str, output_dir: str, prefix: str = "contrast") -> List[str]:
        """
        Adjust the contrast of an image by several factors and save the results.

        Args:
            image_path (str): Path to the input image.
            output_dir (str): Directory to save the contrast-adjusted images.
            prefix (str): Prefix for output filenames.
        Returns:
            List[str]: List of file paths to the generated images.
        """
        pil_image = Image.open(image_path)
        generated_images = []
        contrast_values = [0.7, 0.8, 1.2, 1.3, 1.5]
        
        for i, contrast in enumerate(contrast_values):
            enhancer = ImageEnhance.Contrast(pil_image)
            contrast_image = enhancer.enhance(contrast)
            
            output_path = os.path.join(output_dir, f"{prefix}_{contrast}_{i+1}.png")
            contrast_image.save(output_path)
            generated_images.append(output_path)
            
        return generated_images
    
    def add_noise(self, image_path: str, output_dir: str, prefix: str = "noise") -> List[str]:
        """
        Add Gaussian noise to an image at several levels and save the results.

        Args:
            image_path (str): Path to the input image.
            output_dir (str): Directory to save the noisy images.
            prefix (str): Prefix for output filenames.
        Returns:
            List[str]: List of file paths to the generated images.
        """
        image = cv2.imread(image_path)
        if image is None:
            return []
            
        generated_images = []
        noise_levels = [10, 20, 30, 40, 50]
        
        for i, noise_level in enumerate(noise_levels):
            noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
            noisy_image = cv2.add(image, noise)
            
            output_path = os.path.join(output_dir, f"{prefix}_{noise_level}_{i+1}.png")
            cv2.imwrite(output_path, noisy_image)
            generated_images.append(output_path)
            
        return generated_images
    
    def blur_image(self, image_path: str, output_dir: str, prefix: str = "blur") -> List[str]:
        """
        Apply Gaussian blur to an image with several radii and save the results.

        Args:
            image_path (str): Path to the input image.
            output_dir (str): Directory to save the blurred images.
            prefix (str): Prefix for output filenames.
        Returns:
            List[str]: List of file paths to the generated images.
        """
        pil_image = Image.open(image_path)
        generated_images = []
        blur_radii = [1, 2, 3, 4, 5]
        
        for i, radius in enumerate(blur_radii):
            blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
            
            output_path = os.path.join(output_dir, f"{prefix}_{radius}_{i+1}.png")
            blurred.save(output_path)
            generated_images.append(output_path)
            
        return generated_images
    
    def crop_image(self, image_path: str, output_dir: str, prefix: str = "crop") -> List[str]:
        """
        Crop an image to several ratios and save the results.

        Args:
            image_path (str): Path to the input image.
            output_dir (str): Directory to save the cropped images.
            prefix (str): Prefix for output filenames.
        Returns:
            List[str]: List of file paths to the generated images.
        """
        image = cv2.imread(image_path)
        if image is None:
            return []
            
        generated_images = []
        h, w = image.shape[:2]
        
        # Different crop ratios
        crop_ratios = [0.8, 0.85, 0.9, 0.95]
        
        for i, ratio in enumerate(crop_ratios):
            new_h, new_w = int(h * ratio), int(w * ratio)
            start_y = (h - new_h) // 2
            start_x = (w - new_w) // 2
            
            cropped = image[start_y:start_y+new_h, start_x:start_x+new_w]
            
            output_path = os.path.join(output_dir, f"{prefix}_{ratio}_{i+1}.png")
            cv2.imwrite(output_path, cropped)
            generated_images.append(output_path)
            
        return generated_images
    
    def color_shift(self, image_path: str, output_dir: str, prefix: str = "color_shift") -> List[str]:
        """
        Shift the hue of an image by several values and save the results.

        Args:
            image_path (str): Path to the input image.
            output_dir (str): Directory to save the color-shifted images.
            prefix (str): Prefix for output filenames.
        Returns:
            List[str]: List of file paths to the generated images.
        """
        image = cv2.imread(image_path)
        if image is None:
            return []
            
        generated_images = []
        
        # Convert to HSV for easier color adjustment
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_shifts = [10, 20, 30, -10, -20, -30]
        
        for i, shift in enumerate(hue_shifts):
            hsv_shifted = hsv.copy()
            hsv_shifted[:, :, 0] = (hsv_shifted[:, :, 0] + shift) % 180
            bgr_shifted = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)
            
            output_path = os.path.join(output_dir, f"{prefix}_{shift}_{i+1}.png")
            cv2.imwrite(output_path, bgr_shifted)
            generated_images.append(output_path)
            
        return generated_images
    
    # ========================================================================
    # MAIN IMAGE GENERATION FUNCTION
    # ========================================================================
    def generate_images(self, folder_name: str, transform_functions: List[Callable], 
                       output_folder: str = None, max_images_per_transform: int = None) -> dict:
        """
        Generate augmented images from all images in a subfolder using selected transformation functions.

        Args:
            folder_name (str): Name of the subfolder containing original images.
            transform_functions (List[Callable]): List of transformation functions to apply.
            output_folder (str, optional): Output folder for generated images. Defaults to folder_name + "_generated".
            max_images_per_transform (int, optional): Maximum number of images to generate per transformation.
        Returns:
            dict: Statistics about the image generation process, including counts per transformation.
        """
        # Create output directory
        if output_folder is None:
            output_folder = f"{folder_name}_generated"
        
        output_path = os.path.join(self.base_path, output_folder)
        os.makedirs(output_path, exist_ok=True)
        
        # Get list of original images
        original_images = self.get_images_in_folder(folder_name)
        if not original_images:
            print(f"No images found in folder {folder_name}")
            return {}
        
        stats = {
            'original_images': len(original_images),
            'transform_functions': len(transform_functions),
            'generated_images': 0,
            'details': {}
        }
        
        print(f"Start generating images from folder: {folder_name}")
        print(f"Number of original images: {len(original_images)}")
        print(f"Number of transformations: {len(transform_functions)}")
        
        # Apply transformations to each image
        for img_idx, image_path in enumerate(original_images):
            img_name = os.path.splitext(os.path.basename(image_path))[0]
            print(f"Processing image {img_idx + 1}/{len(original_images)}: {img_name}")
            
            for func_idx, transform_func in enumerate(transform_functions):
                func_name = transform_func.__name__
                print(f"  Applying {func_name}...")
                
                try:
                    generated_paths = transform_func(
                        image_path, 
                        output_path,  # Save directly to the common directory
                        f"{img_name}_{func_name}"
                    )
                    
                    # Limit the number of generated images if needed
                    if max_images_per_transform and len(generated_paths) > max_images_per_transform:
                        generated_paths = generated_paths[:max_images_per_transform]
                    
                    stats['generated_images'] += len(generated_paths)
                    
                    if func_name not in stats['details']:
                        stats['details'][func_name] = 0
                    stats['details'][func_name] += len(generated_paths)
                    
                    print(f"    Generated {len(generated_paths)} images")
                    
                except Exception as e:
                    print(f"    Error applying {func_name}: {str(e)}")
        
        print(f"\nDone! Total generated images: {stats['generated_images']}")
        print(f"Images are saved in: {output_path}")
        
        return stats
    
    def get_all_transform_functions(self) -> List[Callable]:
        """
        Get a list of all available image transformation functions.

        Returns:
            List[Callable]: List of transformation function references.
        """
        return [
            self.rotate_image,
            self.flip_image,
            self.scale_image,
            self.adjust_brightness,
            self.adjust_contrast,
            self.add_noise,
            self.blur_image,
            self.crop_image,
            self.color_shift
        ]

# ========================================================================
# EXAMPLE USAGE (for standalone testing)
# ========================================================================

if __name__ == "__main__":
    # Initialize GenImage
    gen_image = GenImage("recognizeData")
    
    # Show available folders
    print("Available folders:")
    subfolders = gen_image.get_subfolders()
    for i, folder in enumerate(subfolders):
        print(f"{i+1}. {folder}")
    
    # Select folder to generate images (example: garden)
    selected_folder = "garden"  # Change to your desired folder
    
    # Select transformation functions to use
    # selected_transforms = [
    #     gen_image.rotate_image,
    #     gen_image.flip_image,
    #     gen_image.adjust_brightness,
    #     gen_image.scale_image
    # ]
    
    # Or use all available functions
    selected_transforms = gen_image.get_all_transform_functions()
    
    # Generate images
    results = gen_image.generate_images(
        folder_name=selected_folder,
        transform_functions=selected_transforms,
        max_images_per_transform=5  # Limit to 5 images per transformation
    )
    
    # Print statistics
    print("\n" + "="*50)
    print("STATISTICS:")
    print(f"Original images: {results['original_images']}")
    print(f"Number of transformations: {results['transform_functions']}")
    print(f"Total generated images: {results['generated_images']}")
    print("\nDetails by transformation:")
    for func_name, count in results['details'].items():
        print(f"  {func_name}: {count} images")