"""
Unit tests for Frame module.
Tests frame data encapsulation and preprocessing functionality.
"""

import unittest
import sys
import os
import numpy as np
import cv2

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from controllers.frame import Frame


class TestFrame(unittest.TestCase):
    """Test cases for Frame class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create test frame data
        self.test_frame_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.frame_id = 1
        self.frame = Frame(self.frame_id, self.test_frame_data.copy())
    
    def tearDown(self):
        """Clean up after each test method."""
        self.frame = None
    
    def test_initialization(self):
        """Test Frame initialization."""
        self.assertEqual(self.frame.frame_id, self.frame_id)
        np.testing.assert_array_equal(self.frame.frame_data, self.test_frame_data)
        self.assertFalse(self.frame.processed)
    
    def test_get_data(self):
        """Test get_data method returns correct tuple."""
        frame_id, frame_data = self.frame.get_data()
        self.assertEqual(frame_id, self.frame_id)
        np.testing.assert_array_equal(frame_data, self.test_frame_data)
    
    def test_frame_preprocessing_basic(self):
        """Test basic frame preprocessing functionality."""
        # Test that preprocessing doesn't raise exceptions
        try:
            result = self.frame.framePreprocessing()
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 5)
            
            original_frame, img_tensor, ratio, dwdh, img_no_255 = result
            
            # Check types
            self.assertIsInstance(original_frame, np.ndarray)
            self.assertIsInstance(img_tensor, np.ndarray)
            self.assertIsInstance(ratio, float)
            self.assertIsInstance(dwdh, tuple)
            self.assertIsInstance(img_no_255, np.ndarray)
            
            # Check shapes
            self.assertEqual(img_tensor.shape[1:], (640, 640))  # Height, width
            self.assertEqual(img_no_255.shape[1:], (640, 640))
            
        except Exception as e:
            self.fail(f"Frame preprocessing failed with exception: {e}")
    
    def test_frame_preprocessing_letterboxing(self):
        """Test letterboxing functionality in preprocessing."""
        # Create a non-square image
        non_square_data = np.random.randint(0, 255, (300, 800, 3), dtype=np.uint8)
        frame = Frame(1, non_square_data)
        
        result = frame.framePreprocessing()
        original_frame, img_tensor, ratio, dwdh, img_no_255 = result
        
        # Check that output is square (640x640)
        self.assertEqual(img_tensor.shape[1:], (640, 640))
        self.assertEqual(img_no_255.shape[1:], (640, 640))
        
        # Check ratio is reasonable
        self.assertGreater(ratio, 0)
        self.assertLess(ratio, 1)  # Should be scaled down
    
    def test_frame_preprocessing_color_conversion(self):
        """Test color space conversion in preprocessing."""
        # Create BGR image (OpenCV format)
        bgr_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame = Frame(1, bgr_data)
        
        result = frame.framePreprocessing()
        original_frame, img_tensor, ratio, dwdh, img_no_255 = result
        
        # Check that tensor is in RGB format (different from original BGR)
        # This is a basic check - in practice, the conversion should be visible
        self.assertIsInstance(img_tensor, np.ndarray)
        self.assertEqual(img_tensor.shape[0], 3)  # RGB channels
    
    def test_frame_preprocessing_normalization(self):
        """Test normalization in preprocessing."""
        result = self.frame.framePreprocessing()
        original_frame, img_tensor, ratio, dwdh, img_no_255 = result
        
        # Check that img_no_255 is normalized (0-1 range)
        self.assertGreaterEqual(img_no_255.min(), 0)
        self.assertLessEqual(img_no_255.max(), 1)
        
        # Check that img_tensor is also normalized
        self.assertGreaterEqual(img_tensor.min(), 0)
        self.assertLessEqual(img_tensor.max(), 1)
    
    def test_frame_preprocessing_contiguous_memory(self):
        """Test that output tensors are contiguous in memory."""
        result = self.frame.framePreprocessing()
        original_frame, img_tensor, ratio, dwdh, img_no_255 = result
        
        # Check memory contiguity
        self.assertTrue(img_tensor.flags['C_CONTIGUOUS'])
        self.assertTrue(img_no_255.flags['C_CONTIGUOUS'])
    
    def test_frame_preprocessing_different_sizes(self):
        """Test preprocessing with different input sizes."""
        sizes = [(100, 100), (720, 1280), (1080, 1920)]
        
        for height, width in sizes:
            with self.subTest(size=(height, width)):
                test_data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                frame = Frame(1, test_data)
                
                try:
                    result = frame.framePreprocessing()
                    original_frame, img_tensor, ratio, dwdh, img_no_255 = result
                    
                    # All should output 640x640
                    self.assertEqual(img_tensor.shape[1:], (640, 640))
                    self.assertEqual(img_no_255.shape[1:], (640, 640))
                    
                except Exception as e:
                    self.fail(f"Preprocessing failed for size {height}x{width}: {e}")
    
    def test_destroy(self):
        """Test frame destruction and memory cleanup."""
        # Store references
        frame_id = self.frame.frame_id
        frame_data = self.frame.frame_data
        
        # Destroy frame
        self.frame.destroy()
        
        # Check that attributes are set to None
        self.assertIsNone(self.frame.frame_id)
        self.assertIsNone(self.frame.frame_data)
        self.assertIsNone(self.frame.processed)
    
    def test_processed_flag(self):
        """Test processed flag functionality."""
        self.assertFalse(self.frame.processed)
        
        # Set processed flag
        self.frame.processed = True
        self.assertTrue(self.frame.processed)
        
        # Reset processed flag
        self.frame.processed = False
        self.assertFalse(self.frame.processed)
    
    def test_frame_id_assignment(self):
        """Test frame ID assignment and retrieval."""
        new_id = 999
        self.frame.frame_id = new_id
        self.assertEqual(self.frame.frame_id, new_id)
    
    def test_frame_data_assignment(self):
        """Test frame data assignment and retrieval."""
        new_data = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        self.frame.frame_data = new_data
        np.testing.assert_array_equal(self.frame.frame_data, new_data)
    
    def test_preprocessing_with_none_data(self):
        """Test preprocessing behavior with None frame data."""
        frame = Frame(1, None)
        
        with self.assertRaises(Exception):
            frame.framePreprocessing()
    
    def test_preprocessing_with_invalid_data(self):
        """Test preprocessing behavior with invalid frame data."""
        # Test with 2D array (missing channel dimension)
        invalid_data = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        frame = Frame(1, invalid_data)
        
        with self.assertRaises(Exception):
            frame.framePreprocessing()
    
    def test_preprocessing_performance(self):
        """Test preprocessing performance with timing."""
        import time
        
        start_time = time.time()
        result = self.frame.framePreprocessing()
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Preprocessing should complete in reasonable time (< 1 second)
        self.assertLess(processing_time, 1.0)
        
        # Verify result is still valid
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)


if __name__ == '__main__':
    unittest.main() 