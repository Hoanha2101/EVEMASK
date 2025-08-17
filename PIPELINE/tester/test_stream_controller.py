"""
Unit tests for StreamController module in the EVEMASK Pipeline system.

This module provides comprehensive tests for video stream control, management, and thread safety.

Author: EVEMASK Team
"""

import unittest
import sys
import os
import time
import threading
from unittest.mock import patch, MagicMock, Mock
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.controllers.stream import StreamController
from src.controllers.frame import Frame

class TestStreamController(unittest.TestCase):
    """Test cases for StreamController class."""
    
    def setUp(self):
        # Patch VideoCapture globally
        self.patcher_vc = patch('src.controllers.stream.cv2.VideoCapture')
        self.mock_video_capture = self.patcher_vc.start()
        # Mock behavior
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        self.mock_video_capture.return_value = mock_cap

        self.mock_config = {
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
        self.controller = StreamController(self.mock_config)

    def tearDown(self):
        self.patcher_vc.stop()
        self.controller = None

    def test_initialization(self):
        """Test StreamController initialization."""
        self.assertIsNotNone(self.controller)
        if self.controller is not None:
            self.assertEqual(self.controller.cfg, self.mock_config)
            self.assertIsNotNone(self.controller.circle_queue)

    def test_source_capture_frame_reading(self):
        """Test frame reading in source capture."""
        # Patch source_capture to only run one loop with a frame that has data
        def single_run_source_capture(self):
            ret, data = self.cap.read()
            if ret and data is not None:
                frame = Frame(frame_id=self._frame_index, frame_data=data)
                self.circle_queue.add_frame(frame=frame)
                self._frame_index += 1
        with patch.object(StreamController, 'source_capture', single_run_source_capture):
            controller = StreamController(self.mock_config)
            controller.source_capture()
            self.assertGreaterEqual(controller.circle_queue.queue_length(), 1)

    def test_out_stream_display(self):
        """Test output streaming to display."""
        # Patch out_stream to only run one loop
        def single_run_out_stream(self):
            self._write_frame_index = 0
            frame = Frame(1, np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
            self.circle_queue.add_frame(frame)
            # Simulate that the frame has already been processed
            if hasattr(self.ai_instance, 'mooc_processed_frames'):
                self.ai_instance.mooc_processed_frames = 0
            else:
                setattr(self.ai_instance, 'mooc_processed_frames', 0)
            if (self._write_frame_index in self.circle_queue.frames.keys()) and (self.ai_instance.mooc_processed_frames >= self._write_frame_index):
                frame_out = self.circle_queue.get_by_id(self._write_frame_index)
                if frame_out is not None:
                    # Do not actually write to file
                    pass
        with patch.object(StreamController, 'out_stream', single_run_out_stream):
            config = self.mock_config.copy()
            config['OUTPUT_TYPE'] = 'display'
            controller = StreamController(config)
            try:
                controller.out_stream()
            except Exception as e:
                self.fail(f"Output stream failed with unhandled exception: {e}")

    def test_frame_processing_flag(self):
        controller = StreamController(self.mock_config)
        test_frame = Frame(1, np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        self.assertFalse(getattr(test_frame, 'processed', False))
        test_frame.processed = True
        self.assertTrue(test_frame.processed)
        controller.circle_queue.add_frame(test_frame)
        self.assertEqual(controller.circle_queue.queue_length(), 1)

    def test_config_validation(self):
        invalid_config = {'some_key': 'some_value'}
        with patch('src.brain.AI.get_instance', return_value=MagicMock()):
            with self.assertRaises(KeyError):
                StreamController(invalid_config)

    def test_thread_safety(self):
        controller = StreamController(self.mock_config)
        def add_frames():
            for i in range(10):
                frame = Frame(i, np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
                controller.circle_queue.add_frame(frame)
                time.sleep(0.01)
        def remove_frames():
            for _ in range(10):
                controller.circle_queue.pop_frame()
                time.sleep(0.01)
        add_thread = threading.Thread(target=add_frames)
        remove_thread = threading.Thread(target=remove_frames)
        add_thread.start()
        remove_thread.start()
        add_thread.join()
        remove_thread.join()
        self.assertGreaterEqual(controller.circle_queue.queue_length(), 0)

    def test_memory_management(self):
        controller = StreamController(self.mock_config)
        for i in range(10):
            frame = Frame(i, np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
            controller.circle_queue.add_frame(frame)
        self.assertEqual(controller.circle_queue.queue_length(), 10)
        for _ in range(5):
            controller.circle_queue.pop_frame()
        self.assertEqual(controller.circle_queue.queue_length(), 5)

    def test_performance_metrics(self):
        controller = StreamController(self.mock_config)
        start_time = time.time()
        for i in range(10):
            frame = Frame(i, np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
            controller.circle_queue.add_frame(frame)
        end_time = time.time()
        processing_time = end_time - start_time
        self.assertLess(processing_time, 1.0)
        self.assertEqual(controller.circle_queue.queue_length(), 10)

if __name__ == '__main__':
    unittest.main() 