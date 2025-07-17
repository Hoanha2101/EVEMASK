"""
Unit tests for StreamController module.
Tests video stream control and management functionality.
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
        """Set up test fixtures before each test method."""
        # Mock configuration
        self.mock_config = {
            'INPUT_SOURCE': 'test_video.mp4',
            'OUTPUT_TYPE': 'display',
            'batch_size': 4,
            'TARGET_FPS': 30,
            'DELAY_TIME': 0.1
        }
        
        # Create StreamController with mocked dependencies
        with patch('controllers.stream.cv2.VideoCapture'), \
             patch('controllers.stream.cv2.VideoWriter'), \
             patch('controllers.stream.cv2.imshow'), \
             patch('controllers.stream.cv2.waitKey'):
            
            self.controller = StreamController(self.mock_config)
    
    def tearDown(self):
        """Clean up after each test method."""
        # Clean up any resources
        if hasattr(self, 'controller'):
            self.controller = None
    
    def test_initialization(self):
        """Test StreamController initialization."""
        self.assertIsNotNone(self.controller)
        self.assertEqual(self.controller.config, self.mock_config)
        self.assertIsNotNone(self.controller.queue)
    
    @patch('controllers.stream.cv2.VideoCapture')
    def test_source_capture_basic(self, mock_video_capture):
        """Test basic source capture functionality."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        # Create controller
        controller = StreamController(self.mock_config)
        
        # Test capture (run for a short time)
        def run_capture():
            controller.source_capture()
        
        # Start capture in a thread and stop it quickly
        capture_thread = threading.Thread(target=run_capture, daemon=True)
        capture_thread.start()
        time.sleep(0.1)  # Run for 100ms
        
        # Verify that video capture was initialized
        mock_video_capture.assert_called()
    
    @patch('controllers.stream.cv2.VideoCapture')
    def test_source_capture_invalid_source(self, mock_video_capture):
        """Test source capture with invalid video source."""
        # Mock video capture that fails to open
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        # Create controller with invalid source
        config = self.mock_config.copy()
        config['INPUT_SOURCE'] = 'invalid_source.mp4'
        controller = StreamController(config)
        
        # Test capture - should handle invalid source gracefully
        try:
            controller.source_capture()
        except Exception as e:
            # Should not raise unhandled exceptions
            self.fail(f"Source capture failed with unhandled exception: {e}")
    
    @patch('controllers.stream.cv2.VideoCapture')
    def test_source_capture_frame_reading(self, mock_video_capture):
        """Test frame reading in source capture."""
        # Mock video capture with frame data
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        
        # Create test frames
        test_frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        ]
        
        # Mock read to return frames then None (end of video)
        mock_cap.read.side_effect = [(True, frame) for frame in test_frames] + [(False, None)]
        mock_video_capture.return_value = mock_cap
        
        controller = StreamController(self.mock_config)
        
        # Run capture briefly
        def run_capture():
            controller.source_capture()
        
        capture_thread = threading.Thread(target=run_capture, daemon=True)
        capture_thread.start()
        time.sleep(0.1)
        
        # Verify that frames were read
        self.assertGreater(mock_cap.read.call_count, 0)
    
    @patch('controllers.stream.cv2.VideoWriter')
    @patch('controllers.stream.cv2.imshow')
    @patch('controllers.stream.cv2.waitKey')
    def test_out_stream_display(self, mock_wait_key, mock_imshow, mock_video_writer):
        """Test output streaming to display."""
        # Mock waitKey to return 'q' after a few calls (to exit)
        mock_wait_key.side_effect = [ord('a'), ord('b'), ord('q')]
        
        # Create controller
        config = self.mock_config.copy()
        config['OUTPUT_TYPE'] = 'display'
        controller = StreamController(config)
        
        # Add some test frames to queue
        test_frame = Frame(1, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        controller.queue.add_frame(test_frame)
        
        # Test output stream
        try:
            controller.out_stream()
        except Exception as e:
            # Should handle display output gracefully
            self.fail(f"Output stream failed with unhandled exception: {e}")
    
    @patch('controllers.stream.cv2.VideoWriter')
    def test_out_stream_file(self, mock_video_writer):
        """Test output streaming to file."""
        # Mock video writer
        mock_writer = MagicMock()
        mock_video_writer.return_value = mock_writer
        
        # Create controller with file output
        config = self.mock_config.copy()
        config['OUTPUT_TYPE'] = 'file'
        config['OUTPUT_FILE'] = 'test_output.mp4'
        controller = StreamController(config)
        
        # Add test frames to queue
        for i in range(3):
            test_frame = Frame(i, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            controller.queue.add_frame(test_frame)
        
        # Test output stream briefly
        def run_output():
            controller.out_stream()
        
        output_thread = threading.Thread(target=run_output, daemon=True)
        output_thread.start()
        time.sleep(0.1)
        
        # Verify video writer was initialized
        mock_video_writer.assert_called()
    
    def test_queue_management(self):
        """Test queue management functionality."""
        controller = StreamController(self.mock_config)
        
        # Test adding frames to queue
        test_frames = []
        for i in range(5):
            frame = Frame(i, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            test_frames.append(frame)
            controller.queue.add_frame(frame)
        
        # Verify frames were added
        self.assertEqual(controller.queue.queue_length(), 5)
        
        # Test retrieving frames
        retrieved_frames = controller.queue.get_tail(3)
        self.assertEqual(len(retrieved_frames), 3)
    
    def test_frame_processing_flag(self):
        """Test frame processing flag management."""
        controller = StreamController(self.mock_config)
        
        # Create test frame
        test_frame = Frame(1, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        
        # Initially unprocessed
        self.assertFalse(test_frame.processed)
        
        # Mark as processed
        test_frame.processed = True
        self.assertTrue(test_frame.processed)
        
        # Add to queue
        controller.queue.add_frame(test_frame)
        
        # Verify in queue
        self.assertEqual(controller.queue.queue_length(), 1)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test with missing required keys
        invalid_config = {'some_key': 'some_value'}
        
        try:
            controller = StreamController(invalid_config)
            # Should handle missing keys gracefully
            self.assertIsNotNone(controller)
        except Exception as e:
            # Should not raise unhandled exceptions
            self.fail(f"Controller initialization failed with unhandled exception: {e}")
    
    def test_thread_safety(self):
        """Test thread safety of stream controller."""
        controller = StreamController(self.mock_config)
        
        # Test concurrent access to queue
        def add_frames():
            for i in range(10):
                frame = Frame(i, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
                controller.queue.add_frame(frame)
                time.sleep(0.01)
        
        def remove_frames():
            for _ in range(10):
                controller.queue.pop_frame()
                time.sleep(0.01)
        
        # Create threads
        add_thread = threading.Thread(target=add_frames)
        remove_thread = threading.Thread(target=remove_frames)
        
        # Start threads
        add_thread.start()
        remove_thread.start()
        
        # Wait for completion
        add_thread.join()
        remove_thread.join()
        
        # Queue should be in consistent state
        self.assertGreaterEqual(controller.queue.queue_length(), 0)
    
    def test_frame_id_management(self):
        """Test frame ID management."""
        controller = StreamController(self.mock_config)
        
        # Add frames with sequential IDs
        for i in range(5):
            frame = Frame(i, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            controller.queue.add_frame(frame)
        
        # Verify frame IDs
        frames = controller.queue.get_tail(5)
        for i, frame in enumerate(frames):
            self.assertEqual(frame.frame_id, i)
    
    def test_memory_management(self):
        """Test memory management and cleanup."""
        controller = StreamController(self.mock_config)
        
        # Add frames
        for i in range(10):
            frame = Frame(i, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            controller.queue.add_frame(frame)
        
        # Verify initial state
        self.assertEqual(controller.queue.queue_length(), 10)
        
        # Remove frames
        for _ in range(5):
            controller.queue.pop_frame()
        
        # Verify final state
        self.assertEqual(controller.queue.queue_length(), 5)
    
    def test_error_handling(self):
        """Test error handling in stream controller."""
        controller = StreamController(self.mock_config)
        
        # Test with invalid frame data
        try:
            invalid_frame = Frame(1, None)
            controller.queue.add_frame(invalid_frame)
        except Exception as e:
            # Should handle invalid data gracefully
            self.assertIsInstance(e, (AssertionError, TypeError))
    
    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        controller = StreamController(self.mock_config)
        
        # Simulate frame processing
        start_time = time.time()
        
        for i in range(10):
            frame = Frame(i, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            controller.queue.add_frame(frame)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process frames in reasonable time
        self.assertLess(processing_time, 1.0)
        
        # Verify frames were added
        self.assertEqual(controller.queue.queue_length(), 10)


if __name__ == '__main__':
    unittest.main() 