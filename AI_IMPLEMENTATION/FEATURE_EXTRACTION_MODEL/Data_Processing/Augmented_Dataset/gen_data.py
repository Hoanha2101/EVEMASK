import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import os
import random
import shutil
from tqdm import tqdm
from matplotlib.patches import Rectangle

# Set up matplotlib to display Vietnamese (if needed)
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.style.use('dark_background')

class ImageAugmentation:
    def __init__(self, overlay_folder=None):
        self.overlay_folder = overlay_folder
        self.overlay_images = []
        
        # Get list of overlay images if any
        if overlay_folder and os.path.exists(overlay_folder):
            self.overlay_images = [f for f in os.listdir(overlay_folder) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Found {len(self.overlay_images)} overlay images in {overlay_folder}")
        elif overlay_folder:
            print(f"Folder {overlay_folder} does not exist! Overlay functionality will be skipped.")
    
    def change_color_random(self, image):
        """Method 1: Random color adjustment"""
        results = []
        
        for _ in range(3):
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hue_shift = random.randint(-50, 50)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            
            sat_factor = random.uniform(0.7, 1.3)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
            
            val_factor = random.uniform(0.8, 1.2)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_factor, 0, 255)
            
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            results.append(result)
        
        return results
    
    def rotate_random(self, image):
        """Method 2: Random rotation with padding"""
        results = []
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        for _ in range(3):
            angle = random.uniform(-90, 90)
            rad = np.radians(abs(angle))
            new_w = int(h * np.sin(rad) + w * np.cos(rad))
            new_h = int(h * np.cos(rad) + w * np.sin(rad))
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2
            
            padding_colors = [
                [0, 0, 0], [255, 255, 255], [128, 128, 128], [64, 64, 64],
                [192, 192, 192], [255, 240, 245], [240, 248, 255],
                [255, 248, 220], [248, 248, 255], [245, 245, 220],
            ]
            background_color = random.choice(padding_colors)
            rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=background_color)
            results.append(rotated)
        
        return results
    
    def add_overlay(self, image):
        """Method 3: Add PNG overlay (1/6 of image area)"""
        results = []
        
        if not self.overlay_images:
            return [image] * 3
        
        h, w = image.shape[:2]
        image_area = h * w
        overlay_area = image_area / 6
        overlay_size = int(np.sqrt(overlay_area))
        
        for _ in range(3):
            result = image.copy()
            overlay_name = random.choice(self.overlay_images)
            overlay_path = os.path.join(self.overlay_folder, overlay_name)
            
            try:
                overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
                if overlay is None:
                    continue
                
                if overlay.shape[2] == 4:
                    overlay_bgr = overlay[:, :, :3]
                    alpha_channel = overlay[:, :, 3]
                    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                    overlay_rgb = cv2.resize(overlay_rgb, (overlay_size, overlay_size))
                    alpha_channel = cv2.resize(alpha_channel, (overlay_size, overlay_size))
                    
                    max_x = max(0, w - overlay_size)
                    max_y = max(0, h - overlay_size)
                    
                    if max_x > 0 and max_y > 0:
                        x = random.randint(0, max_x)
                        y = random.randint(0, max_y)
                        roi = result[y:y+overlay_size, x:x+overlay_size]
                        alpha_mask = alpha_channel.astype(float) / 255.0
                        for c in range(3):
                            roi[:, :, c] = (alpha_mask * overlay_rgb[:, :, c] + 
                                            (1 - alpha_mask) * roi[:, :, c])
                        result[y:y+overlay_size, x:x+overlay_size] = roi.astype(np.uint8)
                
                else:
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    overlay = cv2.resize(overlay, (overlay_size, overlay_size))
                    max_x = max(0, w - overlay_size)
                    max_y = max(0, h - overlay_size)
                    
                    if max_x > 0 and max_y > 0:
                        x = random.randint(0, max_x)
                        y = random.randint(0, max_y)
                        alpha = 0.8
                        roi = result[y:y+overlay_size, x:x+overlay_size]
                        blended = cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0)
                        result[y:y+overlay_size, x:x+overlay_size] = blended
                
            except Exception as e:
                print(f"Error overlaying {overlay_name}: {e}")
            
            results.append(result)
        
        return results
    
    def zoom_random(self, image):
        """Method 4: Random zoom in or out"""
        results = []
        h, w = image.shape[:2]
        
        for _ in range(3):
            zoom_factor = random.uniform(0.7, 1.5)
            
            if zoom_factor < 1:
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                resized = cv2.resize(image, (new_w, new_h))
                result = np.zeros_like(image)
                start_y = (h - new_h) // 2
                start_x = (w - new_w) // 2
                result[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            else:
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                resized = cv2.resize(image, (new_w, new_h))
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                result = resized[start_y:start_y+h, start_x:start_x+w]
            
            results.append(result)
        
        return results
    
    def skew_random(self, image):
        """Method 5: Random skew"""
        results = []
        h, w = image.shape[:2]
        
        for _ in range(3):
            skew_factor = random.uniform(0.1, 0.3)
            src_points = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
            skew_type = random.choice(['left', 'right', 'top', 'bottom'])
            
            if skew_type == 'left':
                dst_points = np.float32([
                    [w*skew_factor, 0], [w-1, 0], [w-1, h-1], [0, h-1]
                ])
            elif skew_type == 'right':
                dst_points = np.float32([
                    [0, 0], [w-1-w*skew_factor, 0], [w-1, h-1], [w*skew_factor, h-1]
                ])
            elif skew_type == 'top':
                dst_points = np.float32([
                    [0, h*skew_factor], [w-1, 0], [w-1, h-1-h*skew_factor], [0, h-1]
                ])
            else:
                dst_points = np.float32([
                    [0, 0], [w-1, h*skew_factor], [w-1, h-1], [0, h-1-h*skew_factor]
                ])
            
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            background_color = [0, 0, 0] if random.random() > 0.5 else [255, 255, 255]
            skewed = cv2.warpPerspective(image, M, (w, h), 
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=background_color)
            results.append(skewed)
        
        return results
    
    def process_single_image(self, image_path):
        """Process a single image and return the results"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot read image: {image_path}")
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = {}
        
        try:
            results['color'] = self.change_color_random(image)
            results['rotate'] = self.rotate_random(image)
            results['overlay'] = self.add_overlay(image)
            results['zoom'] = self.zoom_random(image)
            results['skew'] = self.skew_random(image)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
        
        return results, image
    
    def save_image(self, image, save_path):
        """Save image to disk"""
        try:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, image_bgr)
            return True
        except Exception as e:
            print(f"Error saving image {save_path}: {e}")
            return False
    
    def process_batch(self, input_folder, output_folder, copy_original=True):
        """Process a batch of images in folders"""
        print(f"Starting batch processing from {input_folder} → {output_folder}")
        print("="*60)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")
        
        total_images = 0
        total_generated = 0
        total_folders = 0
        
        for folder_name in os.listdir(input_folder):
            folder_path = os.path.join(input_folder, folder_name)
            if not os.path.isdir(folder_path):
                continue
            
            total_folders += 1
            print(f"\n Processing folder: {folder_name}")
            
            output_folder_path = os.path.join(output_folder, folder_name)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            
            image_files = [f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if not image_files:
                print(f"No images found in {folder_name}")
                continue
            
            print(f"Found {len(image_files)} images")
            
            for image_file in tqdm(image_files, desc=f"  Processing {folder_name}"):
                image_path = os.path.join(folder_path, image_file)
                file_name = os.path.splitext(image_file)[0]
                file_ext = os.path.splitext(image_file)[1]
                
                if copy_original:
                    original_save_path = os.path.join(output_folder_path, image_file)
                    try:
                        shutil.copy2(image_path, original_save_path)
                    except Exception as e:
                        print(f"Error copying original image {image_file}: {e}")
                
                result = self.process_single_image(image_path)
                if result is None:
                    print(f"Cannot process {image_file}")
                    continue
                
                augmented_results, _ = result
                total_images += 1
                
                for method_name, method_results in augmented_results.items():
                    for i, aug_image in enumerate(method_results):
                        new_filename = f"{method_name}_{i+1}_{file_name}{file_ext}"
                        save_path = os.path.join(output_folder_path, new_filename)
                        
                        if self.save_image(aug_image, save_path):
                            total_generated += 1
                        else:
                            print(f"Error saving {new_filename}")
        
        print("\n" + "="*60)
        print("BATCH PROCESSING COMPLETE!")
        print("="*60)
        print("Summary:")
        print(f"  • Total folders processed: {total_folders}")
        print(f"  • Total original images processed: {total_images}")
        print(f"  • Total generated images: {total_generated}")
        print(f"  • Success rate: {total_generated/(total_images*15)*100:.1f}%" if total_images > 0 else "  • Success rate: 0%")
        print(f"  • Output folder: {output_folder}")
        
        return total_folders, total_images, total_generated

# Main function to use the class
def main():
    """Main function"""
    input_folder = "data"
    output_folder = "data_augment_new"
    overlay_folder = "cau_thu"
    
    if not os.path.exists(input_folder):
        print(f"Input folder not found: {input_folder}")
        print("Please create a 'data' folder and add subfolders containing images.")
        return
    
    aug = ImageAugmentation(overlay_folder)
    aug.process_batch(input_folder, output_folder, copy_original=True)

if __name__ == "__main__":
    main()
