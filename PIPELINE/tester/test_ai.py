"""
Unit tests for AI module in the EVEMASK Pipeline system.

This module provides comprehensive tests for the AI processing engine, including initialization, singleton pattern, and inference pipeline (with heavy dependencies mocked).

Author: EVEMASK Team
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.brain import AI

# Minimal valid config for AI initialization (based on cfg/default.yaml)
MINIMAL_CFG = {
                "CLASSES_NO_BLUR": [0],
                "DELAY_TIME": 2,
                "TARGET_FPS": 10,
                "batch_size": 3,
                "INPUT_SOURCE": "test_video.mp4",
                "OUTPUT_TYPE": "udp",
                "OUTPUT_STREAM_URL_RTMP": None,
                "OUTPUT_STREAM_URL_RTSP": None,
                "OUTPUT_STREAM_URL_UDP": "udp://@225.1.9.254:30133?pkt_size=1316",
                "conf_threshold": 0.5,
                "iou_threshold": 0.7,
                "nc": 14,
                "names": {
                    0: "unbet",
                    1: "betrivers",
                    2: "fanduel",
                    3: "betway",
                    4: "caesars",
                    5: "bally",
                    6: "draftkings",
                    7: "pointsbet",
                    8: "bet365",
                    9: "fanatics",
                    10: "betparx",
                    11: "betmgm",
                    12: "gilariver",
                    13: "casino",
                },
                "recognizeData_path": "recognizeData",
                "segment_model": {
                    "all_output_names": [
                        "pred0",
                        "pred1_0_0",
                        "pred1_0_1",
                        "pred1_0_2",
                        "pred1_1",
                        "pred1_2"
                    ],
                    "dynamic_factor": 3,
                    "get_to": "cuda",
                    "input_names": ["input"],
                    "max_batch_size": 3,
                    "path": "weights/trtPlans/yolov8_seg_aug_best_l_trimmed.trt"
                },
                "extract_model": {
                    "input_names": ["input"],
                    "len_emb": 256,
                    "max_batch_size": 32,
                    "output_names": ["output"],
                    "path": "weights/trtPlans/supconloss_bbresnet50_50e.trt"
                }
            }

class TestAI(unittest.TestCase):
    """Test cases for the AI processing engine."""

    def setUp(self):
        """Reset the singleton instance before each test."""
        AI._instance = None

    @patch("src.tools.VectorPrepare")
    @patch("src.controllers.CircleQueue")
    @patch("src.logger.EveMaskLogger")
    def test_initialization(self, mock_logger, mock_circle_queue, mock_vector_prepare):
        """
        Test AI initialization with minimal config and mocked dependencies.
        """
        # Mock the logger singleton
        mock_logger_instance = MagicMock()
        mock_logger.get_instance.return_value = mock_logger_instance
        
        # Mock the circle queue singleton
        mock_queue_instance = MagicMock()
        mock_circle_queue.get_instance.return_value = mock_queue_instance
        
        # Mock the vectorizer's run method to return dummy data
        mock_vector_instance = MagicMock()
        mock_vector_prepare.return_value = mock_vector_instance
        mock_vector_instance.run.return_value = (np.zeros((2, 256)), ["img1", "img2"])
        
        ai = AI(MINIMAL_CFG)
        
        # Verify initialization
        self.assertIsNotNone(ai)
        self.assertEqual(ai.batch_size, MINIMAL_CFG["batch_size"])
        self.assertEqual(ai.number_of_class, MINIMAL_CFG["nc"])
        self.assertEqual(ai.conf_threshold, MINIMAL_CFG["conf_threshold"])
        self.assertEqual(ai.iou_threshold, MINIMAL_CFG["iou_threshold"])
        self.assertEqual(ai.CLASS_NAMES, MINIMAL_CFG["names"])
        self.assertEqual(ai.CLASSES_NO_BLUR, MINIMAL_CFG["CLASSES_NO_BLUR"])

    @patch("src.tools.VectorPrepare")
    @patch("src.controllers.CircleQueue")
    @patch("src.logger.EveMaskLogger")
    def test_singleton_pattern(self, mock_logger, mock_circle_queue, mock_vector_prepare):
        """
        Test that AI class implements the singleton pattern.
        """
        # Mock the logger singleton
        mock_logger_instance = MagicMock()
        mock_logger.get_instance.return_value = mock_logger_instance
        
        # Mock the circle queue singleton
        mock_queue_instance = MagicMock()
        mock_circle_queue.get_instance.return_value = mock_queue_instance
        
        # Mock the vectorizer's run method to return dummy data
        mock_vector_instance = MagicMock()
        mock_vector_prepare.return_value = mock_vector_instance
        mock_vector_instance.run.return_value = (np.zeros((2, 256)), ["img1", "img2"])
        
        ai1 = AI(MINIMAL_CFG)
        ai2 = AI(MINIMAL_CFG)
        
        # Both instances should be the same object
        self.assertIs(ai1, ai2)

    @patch("src.models.net1")
    @patch("src.tools.VectorPrepare")
    @patch("src.controllers.CircleQueue")
    @patch("src.logger.EveMaskLogger")
    def test_inference_mocked(self, mock_logger, mock_circle_queue, mock_vector_prepare, mock_net1):
        """
        Test the inference method with a mocked processed batch.
        """
        # Mock the logger singleton
        mock_logger_instance = MagicMock()
        mock_logger.get_instance.return_value = mock_logger_instance
        
        # Mock the circle queue singleton
        mock_queue_instance = MagicMock()
        mock_circle_queue.get_instance.return_value = mock_queue_instance
        
        # Mock the vectorizer's run method to return dummy data
        mock_vector_instance = MagicMock()
        mock_vector_prepare.return_value = mock_vector_instance
        mock_vector_instance.run.return_value = (np.zeros((2, 256)), ["img1", "img2"])
        
        # Mock net1 inference
        mock_net1.cuda_ctx = MagicMock()
        mock_net1.infer.return_value = [
            np.zeros((2, 84, 8400)),  # Detection output shape
            np.zeros((2, 32, 160, 160))  # Mask prototypes shape
        ]
        
        ai = AI(MINIMAL_CFG)
        
        # Create a dummy processed batch (shape and type as expected by inference)
        processed_batch = [
            (
                np.zeros((640, 640, 3), dtype=np.uint8),  # current_frame
                np.zeros((1, 3, 640, 640), dtype=np.float32),  # batch_tensor
                1.0,  # frame_ratio
                (0, 0),  # frame_dwdh
                np.zeros((1, 3, 640, 640), dtype=np.float32)  # origin_imno255
            )
            for _ in range(2)
        ]
        
        # Mock the frame instances
        frame_instances = []
        for i in range(2):
            frame_mock = MagicMock()
            frame_mock.frame_id = i
            frame_instances.append(frame_mock)
        
        ai._instance_list_ = frame_instances
        
        # Mock the global functions that would be imported
        with patch("src.tools.utils.copy_trt_output_to_torch_tensor") as mock_copy, \
             patch("src.tools.utils.process_trt_output") as mock_process, \
             patch("src.tools.utils.postprocess_torch_cpu") as mock_postprocess, \
             patch("src.tools.utils.draw_bboxes") as mock_draw_bboxes, \
             patch("src.tools.utils.draw_masks_conditional_blur") as mock_draw_masks, \
             patch("src.tools.SimilarityMethod") as mock_similarity:
            
            # Configure mocks
            mock_copy.side_effect = [
                MagicMock(),  # tensor_pred0
                MagicMock()   # tensor_pred1_2
            ]
            
            # Mock process_trt_output to return empty results (no detections)
            mock_process.return_value = (
                MagicMock(),  # boxes_kept
                MagicMock(),  # scores_kept
                MagicMock(),  # class_ids_kept
                MagicMock(),  # mask_coeffs_kept
                MagicMock(),  # batch_indices
                [],  # start_idx (empty)
                []   # end_idx (empty)
            )
            
            mock_postprocess.return_value = ([], [])  # No detected objects, no polygons
            mock_draw_bboxes.return_value = np.zeros((640, 640, 3))
            mock_draw_masks.return_value = np.zeros((640, 640, 3))
            
            try:
                ai.inference(processed_batch)
                # If we get here, the inference ran without exceptions
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"AI inference failed with exception: {e}")

    @patch("src.tools.VectorPrepare")
    @patch("src.controllers.CircleQueue")
    @patch("src.logger.EveMaskLogger")
    def test_get_skip_frame_info(self, mock_logger, mock_circle_queue, mock_vector_prepare):
        """
        Test get_skip_frame_info method returns correct information.
        """
        # Mock the logger singleton
        mock_logger_instance = MagicMock()
        mock_logger.get_instance.return_value = mock_logger_instance
        
        # Mock the circle queue singleton
        mock_queue_instance = MagicMock()
        mock_circle_queue.get_instance.return_value = mock_queue_instance
        
        # Mock the vectorizer's run method to return dummy data
        mock_vector_instance = MagicMock()
        mock_vector_prepare.return_value = mock_vector_instance
        mock_vector_instance.run.return_value = (np.zeros((2, 256)), ["img1", "img2"])
        
        ai = AI(MINIMAL_CFG)
        
        # Set some FPS values
        ai._instream_fps_ = 25
        ai._ai_fps_ = 10
        
        info = ai.get_skip_frame_info()
        
        # Verify the returned information
        self.assertEqual(info['input_fps'], 25)
        self.assertEqual(info['ai_fps'], 10)
        self.assertIn('skip_frames', info)
        self.assertIn('ratio', info)
        self.assertIn('strategy', info)

    @patch("src.tools.VectorPrepare")
    @patch("src.controllers.CircleQueue")
    @patch("src.logger.EveMaskLogger")
    def test_update_input_fps(self, mock_logger, mock_circle_queue, mock_vector_prepare):
        """
        Test update_input_fps method.
        """
        # Mock the logger singleton
        mock_logger_instance = MagicMock()
        mock_logger.get_instance.return_value = mock_logger_instance
        
        # Mock the circle queue singleton
        mock_queue_instance = MagicMock()
        mock_circle_queue.get_instance.return_value = mock_queue_instance
        
        # Mock the vectorizer's run method to return dummy data
        mock_vector_instance = MagicMock()
        mock_vector_prepare.return_value = mock_vector_instance
        mock_vector_instance.run.return_value = (np.zeros((2, 256)), ["img1", "img2"])
        
        ai = AI(MINIMAL_CFG)
        
        # Test updating input FPS
        ai.update_input_fps(30)
        self.assertEqual(ai._instream_fps_, 30)

if __name__ == "__main__":
    unittest.main()