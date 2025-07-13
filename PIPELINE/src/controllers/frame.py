"""
Frame Processing Module
Handles individual video frame data and preprocessing for AI inference.

This module provides:
- Frame data encapsulation with metadata
- Preprocessing pipeline for YOLO model input
- Memory management for frame resources
- Image transformation utilities

Key Features:
- Letterboxing for consistent input dimensions
- Color space conversion (BGR to RGB)
- Blob preprocessing for neural network input
- Memory cleanup utilities

Author: EVEMASK Team
"""

import cv2
from ..tools import *

class Frame:
    """
    Frame data container with preprocessing capabilities.
    
    This class encapsulates a single video frame with its metadata
    and provides methods for preprocessing the frame for AI inference.
    
    Attributes:
        frame_id (int): Unique identifier for the frame
        frame_data (numpy.ndarray): Raw frame image data
        processed (bool): Flag indicating if frame has been processed by AI
    """
    
    def __init__(self, frame_id, frame_data):
        """
        Initialize a new frame object.
        
        Args:
            frame_id (int): Unique identifier for the frame
            frame_data (numpy.ndarray): Raw frame image data in BGR format
        """
        self.frame_id = frame_id
        self.frame_data = frame_data
        self.processed = False  # Initially unprocessed

    def get_data(self):
        """
        Get frame data and ID.
        
        Returns:
            tuple: (frame_id, frame_data) containing frame metadata and image
        """
        return self.frame_id, self.frame_data

    def framePreprocessing(self):
        """
        Preprocess frame for YOLO model inference.
        
        This method performs the complete preprocessing pipeline:
        1. Letterboxing to resize image to 640x640 while maintaining aspect ratio
        2. Color space conversion from BGR to RGB
        3. Blob preprocessing for neural network input
        4. Preparation of multiple data formats for different processing stages
        
        Returns:
            tuple: (original_frame, img_tensor, ratio, dwdh, img_no_255) containing:
                - original_frame: Original frame data
                - img_tensor: Preprocessed tensor for model input
                - ratio: Scaling ratio from letterboxing
                - dwdh: Padding offsets from letterboxing
                - img_no_255: Normalized image data (0-1 range)
        """
        # Store original frame for reference
        original_frame = self.frame_data
        
        # Letterboxing: resize image to 640x640 while maintaining aspect ratio
        # This adds padding if necessary to avoid distortion
        input_image, ratio, dwdh = letterbox(original_frame, (640, 640))
        
        # Convert from BGR to RGB color space (OpenCV uses BGR, models expect RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        # Blob preprocessing: normalize and prepare for neural network input
        # This includes normalization and optional segmentation preparation
        input_image, img_no_255, seg_img = blob(input_image, return_seg=True)
        
        # Ensure tensor is contiguous in memory for efficient processing
        img_tensor = np.ascontiguousarray(input_image)
        
        # Package all data for return
        data = (original_frame, img_tensor, ratio, dwdh, img_no_255)
        return data

    def destroy(self):
        """
        Clean up frame data to free memory.
        
        This method should be called when the frame is no longer needed
        to prevent memory leaks in long-running applications.
        """
        self.frame_id = None
        self.frame_data = None
        self.processed = None
    