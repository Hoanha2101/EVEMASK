"""
Unit tests for utils module.
Tests utility functions for image processing and TensorRT operations.
"""

import unittest
import sys
import os
import numpy as np
import torch
import cv2

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tools import utils


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Set seed for reproducible tests
        utils.set_seed(42)
        
        # Create test data
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=np.float32)
        self.test_scores = np.array([0.8, 0.9], dtype=np.float32)
        self.test_class_ids = np.array([0, 1], dtype=np.int32)
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_set_seed(self):
        """Test seed setting functionality."""
        # Test that seed setting doesn't raise exceptions
        try:
            utils.set_seed(123)
            utils.set_seed(42)
        except Exception as e:
            self.fail(f"set_seed failed with exception: {e}")
    
    def test_bounding_box_initialization(self):
        """Test BoundingBox class initialization."""
        bbox = utils.BoundingBox(0, 0.8, 100, 200, 100, 200, 640, 480)
        
        self.assertEqual(bbox.classID, 0)
        self.assertEqual(bbox.confidence, 0.8)
        self.assertEqual(bbox.x1, 100)
        self.assertEqual(bbox.x2, 200)
        self.assertEqual(bbox.y1, 100)
        self.assertEqual(bbox.y2, 200)
        self.assertEqual(bbox.u1, 100/640)
        self.assertEqual(bbox.u2, 200/640)
        self.assertEqual(bbox.v1, 100/480)
        self.assertEqual(bbox.v2, 200/480)
    
    def test_bounding_box_methods(self):
        """Test BoundingBox class methods."""
        bbox = utils.BoundingBox(0, 0.8, 100, 200, 100, 200, 640, 480)
        
        # Test box method
        box_coords = bbox.box()
        self.assertEqual(box_coords, (100, 100, 200, 200))
        
        # Test width and height
        self.assertEqual(bbox.width(), 100)
        self.assertEqual(bbox.height(), 100)
        
        # Test center methods
        center_abs = bbox.center_absolute()
        self.assertEqual(center_abs, (150.0, 150.0))
        
        center_norm = bbox.center_normalized()
        expected_u = 0.5 * (100/640 + 200/640)
        expected_v = 0.5 * (100/480 + 200/480)
        self.assertEqual(center_norm, (expected_u, expected_v))
        
        # Test size methods
        size_abs = bbox.size_absolute()
        self.assertEqual(size_abs, (100, 100))
        
        size_norm = bbox.size_normalized()
        expected_w = 200/640 - 100/640
        expected_h = 200/480 - 100/480
        self.assertEqual(size_norm, (expected_w, expected_h))
    
    def test_letterbox_basic(self):
        """Test basic letterbox functionality."""
        result = utils.letterbox(self.test_image, (640, 640))
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        
        output_image, ratio, dwdh = result
        
        # Check output image shape
        self.assertEqual(output_image.shape, (640, 640, 3))
        
        # Check ratio is reasonable
        self.assertGreater(ratio, 0)
        self.assertLessEqual(ratio, 1)
        
        # Check dwdh is tuple of two values
        self.assertIsInstance(dwdh, tuple)
        self.assertEqual(len(dwdh), 2)
    
    def test_letterbox_different_sizes(self):
        """Test letterbox with different target sizes."""
        sizes = [(320, 320), (512, 512), (1024, 1024)]
        
        for target_size in sizes:
            with self.subTest(size=target_size):
                result = utils.letterbox(self.test_image, target_size)
                output_image, ratio, dwdh = result
                
                self.assertEqual(output_image.shape, (target_size[1], target_size[0], 3))
                self.assertGreater(ratio, 0)
                self.assertLessEqual(ratio, 1)
    
    def test_letterbox_square_input(self):
        """Test letterbox with square input image."""
        square_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = utils.letterbox(square_image, (640, 640))
        
        output_image, ratio, dwdh = result
        self.assertEqual(output_image.shape, (640, 640, 3))
    
    def test_blob_basic(self):
        """Test basic blob functionality."""
        result = utils.blob(self.test_image)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        output_image, img_no_255 = result
        
        # Check shapes
        self.assertEqual(output_image.shape, (1, 3, 480, 640))
        self.assertEqual(img_no_255.shape, (1, 3, 480, 640))
        
        # Check normalization
        self.assertGreaterEqual(output_image.min(), 0)
        self.assertLessEqual(output_image.max(), 1)
    
    def test_blob_with_segmentation(self):
        """Test blob with segmentation return."""
        result = utils.blob(self.test_image, return_seg=True)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        
        output_image, img_no_255, seg = result
        
        # Check shapes
        self.assertEqual(output_image.shape, (1, 3, 480, 640))
        self.assertEqual(img_no_255.shape, (1, 3, 480, 640))
        self.assertEqual(seg.shape, (480, 640, 3))
        
        # Check segmentation normalization
        self.assertGreaterEqual(seg.min(), 0)
        self.assertLessEqual(seg.max(), 1)
    
    def test_blob_different_image_sizes(self):
        """Test blob with different image sizes."""
        sizes = [(100, 100), (720, 1280), (1080, 1920)]
        
        for height, width in sizes:
            with self.subTest(size=(height, width)):
                test_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                result = utils.blob(test_img)
                
                output_image, img_no_255 = result
                self.assertEqual(output_image.shape, (1, 3, height, width))
                self.assertEqual(img_no_255.shape, (1, 3, height, width))
    
    def test_process_trt_output_basic(self):
        """Test basic TensorRT output processing."""
        # Create mock TensorRT output
        batch_size = 2
        num_classes = 80
        num_boxes = 100
        
        # Create tensor with format [B, C, N]
        tensor_pred = torch.randn(batch_size, 4 + num_classes + 160, num_boxes)
        
        # Set some scores above threshold
        tensor_pred[:, 4:4+num_classes, :] = torch.randn(batch_size, num_classes, num_boxes) * 0.1 + 0.6
        
        result = utils.process_trt_output(tensor_pred, num_classes, conf_threshold=0.5)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 7)
        
        boxes, scores, class_ids, mask_coeffs, batch_idx, start_idx, end_idx = result
        
        # Check that we have some detections
        if len(boxes) > 0:
            self.assertEqual(len(boxes), len(scores))
            self.assertEqual(len(boxes), len(class_ids))
            self.assertEqual(len(boxes), len(mask_coeffs))
    
    def test_process_trt_output_no_detections(self):
        """Test TensorRT output processing with no detections."""
        batch_size = 1
        num_classes = 80
        num_boxes = 100
        
        # Create tensor with all scores below threshold
        tensor_pred = torch.randn(batch_size, 4 + num_classes + 160, num_boxes) * 0.1
        
        result = utils.process_trt_output(tensor_pred, num_classes, conf_threshold=0.5)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 7)
        
        # All should be empty lists
        for item in result:
            self.assertEqual(len(item), 0)
    
    def test_process_trt_output_high_threshold(self):
        """Test TensorRT output processing with high confidence threshold."""
        batch_size = 1
        num_classes = 80
        num_boxes = 100
        
        # Create tensor with mixed scores
        tensor_pred = torch.randn(batch_size, 4 + num_classes + 160, num_boxes)
        tensor_pred[:, 4:4+num_classes, :] = torch.randn(batch_size, num_classes, num_boxes) * 0.1 + 0.6
        
        # Test with very high threshold
        result = utils.process_trt_output(tensor_pred, num_classes, conf_threshold=0.95)
        
        # Should have fewer or no detections
        boxes, scores, class_ids, mask_coeffs, batch_idx, start_idx, end_idx = result
        self.assertLessEqual(len(boxes), 100)  # Should have fewer detections
    
    def test_copy_trt_output_to_torch_tensor(self):
        """Test copying TensorRT output to PyTorch tensor."""
        # Mock TensorRT output info
        shape = (1, 3, 640, 640)
        size = np.prod(shape)
        dtype = np.float32
        
        # Create mock device pointer (just a number for testing)
        device_ptr = 12345
        
        output_info = {
            'device_ptr': device_ptr,
            'shape': shape,
            'size': size,
            'dtype': dtype
        }
        
        # This test might fail if CUDA is not available, so we'll catch the exception
        try:
            result = utils.copy_trt_output_to_torch_tensor(output_info)
            # If successful, check the result
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(result.shape, shape)
            self.assertEqual(result.dtype, torch.float32)
        except Exception as e:
            # If CUDA is not available, this is expected
            self.assertIn("CUDA", str(e) or "cuda", str(e).lower())
    
    def test_draw_bboxes_basic(self):
        """Test basic bounding box drawing."""
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create test bounding boxes
        bboxes = [
            utils.BoundingBox(0, 0.8, 100, 200, 100, 200, 640, 480),
            utils.BoundingBox(1, 0.9, 300, 400, 300, 400, 640, 480)
        ]
        
        # Test drawing
        result = utils.draw_bboxes(test_image, bboxes)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, test_image.shape)
    
    def test_draw_bboxes_with_class_names(self):
        """Test bounding box drawing with class names."""
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        bboxes = [utils.BoundingBox(0, 0.8, 100, 200, 100, 200, 640, 480)]
        class_names = ['person']
        
        result = utils.draw_bboxes(test_image, bboxes, class_names)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, test_image.shape)
    
    def test_draw_bboxes_custom_color(self):
        """Test bounding box drawing with custom color."""
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        bboxes = [utils.BoundingBox(0, 0.8, 100, 200, 100, 200, 640, 480)]
        custom_color = (255, 0, 0)  # Red
        
        result = utils.draw_bboxes(test_image, bboxes, color=custom_color)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, test_image.shape)
    
    def test_draw_bboxes_empty_list(self):
        """Test bounding box drawing with empty list."""
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        bboxes = []
        
        result = utils.draw_bboxes(test_image, bboxes)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, test_image.shape)
        # Should return original image unchanged
        np.testing.assert_array_equal(result, test_image)
    
    def test_censored_options(self):
        """Test censored options functionality."""
        # Create test image tensor
        image_tensor = torch.randn(1, 3, 480, 640)
        
        result = utils.censored_options(image_tensor, downscale_factor=20)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, image_tensor.shape)
    
    def test_resize_mask_to_image(self):
        """Test mask resizing functionality."""
        # Create test mask
        mask = np.random.rand(160, 160)
        target_h, target_w = 480, 640
        
        result = utils.resize_mask_to_image(mask, target_h, target_w)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (target_h, target_w))
    
    def test_apply_blur_to_masked_area(self):
        """Test blur application to masked areas."""
        # Create test image region and mask
        image_region = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.random.rand(100, 100)
        
        result = utils.apply_blur_to_masked_area(image_region, mask)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, image_region.shape)
    
    def test_draw_masks_conditional_blur(self):
        """Test conditional mask drawing with blur."""
        # Create test frame and objects
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detected_objects = [
            utils.BoundingBox(0, 0.8, 100, 200, 100, 200, 640, 480)
        ]
        polygons = [np.array([[100, 100], [200, 100], [200, 200], [100, 200]])]
        class_ids = [0]
        
        result = utils.draw_masks_conditional_blur(frame, detected_objects, polygons, class_ids)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, frame.shape)


if __name__ == '__main__':
    unittest.main() 