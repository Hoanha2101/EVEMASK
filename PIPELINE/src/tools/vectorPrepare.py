"""
Vector preparation utilities for the EVEMASK Pipeline system.
Handles image preprocessing, feature extraction, and vector preparation for similarity matching and classification tasks.

Author: EVEMASK Team
Version: 1.0.0
"""

import cv2
import torch
import numpy as np
from glob import glob
import os
from ..models.initNet import net2
from ..tools import set_seed

# ========================================================================
# SEED INITIALIZATION
# ========================================================================
set_seed(42)

# ========================================================================
# VECTOR PREPARATION CLASS
# ========================================================================
class VectorPrepare:
    """
    The VectorPrepare class handles the preparation of feature vectors from images for similarity matching.
    Processes images through a neural network to extract embeddings for comparison.
    """

    def __init__(self, orgFolderPath, enginePlan):
        """
        Initialize the vector preparation system.
        Args:
            orgFolderPath (str): Path to the folder containing image subfolders
            enginePlan: Neural network engine for feature extraction
        """
        self.orgFolderPath = orgFolderPath
        self.enginePlan = enginePlan

    def preprocess_image(self, image_path):
        """
        Preprocess image exactly like the standard embedding creation process.
        Matches the preprocessing used in create_embedding_tensor_deterministic function.
        Args:
            image_path (str): Path to the input image
        Returns:
            torch.Tensor: Preprocessed image tensor ready for inference
        Raises:
            ValueError: If the image cannot be read
        """
        # Read image with fixed flag
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Resize to 224x224 and convert BGR -> RGB
        img_process = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        img_process = cv2.cvtColor(img_process, cv2.COLOR_BGR2RGB)
        img_process = img_process.astype(np.float32)

        # Convert HWC -> CHW -> NCHW and move to CUDA with half precision
        img_tensor = torch.from_numpy(img_process).permute(2, 0, 1).unsqueeze(0).cuda().half()
        
        # Ensure correct shape (batch dimension)
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Make tensor contiguous for optimal memory access
        img_tensor = img_tensor.contiguous()
        
        return img_tensor

    def load_images_from_folder(self, folder_path):
        """
        Load all supported image files from a folder.
        Args:
            folder_path (str): Path to the folder containing images
        Returns:
            list: Sorted list of image file paths
        """
        # Supported image extensions
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
        image_paths = []
        
        # Collect all image files with supported extensions
        for ext in exts:
            image_paths.extend(glob(os.path.join(folder_path, ext)))
        
        # Return sorted list for consistent ordering
        image_paths = sorted(image_paths)
        return image_paths

    def extract_features(self, image_paths):
        """
        Extract feature vectors from a list of images using the neural network engine.
        Args:
            image_paths (list): List of image file paths
        Returns:
            torch.Tensor: Concatenated feature vectors for all images
        """
        features = []
        for path in image_paths:
            # Preprocess each image
            tensor = self.preprocess_image(path)
            # Extract features using the neural network engine
            emb = self.enginePlan.infer(tensor)
            features.append(emb)
        
        # Concatenate all feature vectors
        return torch.cat(features, dim=0)

    def run(self):
        """
        Main execution method that processes all subfolders and extracts features.
        Returns:
            tuple: (feature_matrix, image_names)
                - feature_matrix: numpy array of all extracted features
                - image_names: list of all processed image paths
        """
        all_features = []
        all_names = []
        # Get all subfolders in the organization folder
        subfolders = sorted(os.listdir(self.orgFolderPath))
        
        for folder in subfolders:
            full_folder_path = os.path.join(self.orgFolderPath, folder)
            
            # Skip if not a directory
            if not os.path.isdir(full_folder_path):
                continue

            # Load all images from this subfolder
            image_paths = self.load_images_from_folder(full_folder_path)
            if not image_paths:
                continue
            
            # Extract features from all images in this subfolder
            feature_tensor = self.extract_features(image_paths)
            all_features.append(feature_tensor)
            all_names.extend(image_paths)

        # Concatenate all features and convert to numpy
        return torch.cat(all_features, dim=0).cpu().numpy(), all_names
