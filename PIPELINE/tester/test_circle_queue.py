"""
Unit tests for CircleQueue module.
Tests thread-safe circular buffer implementation for frame management.
"""

import unittest
import threading
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.controllers.circle_queue import CircleQueue
from src.controllers.frame import Frame
import numpy as np


class TestCircleQueue(unittest.TestCase):
    """Test cases for CircleQueue class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.queue: CircleQueue = CircleQueue(buffer_size=5)
        self.test_frame_data = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def test_initialization(self):
        """Test CircleQueue initialization."""
        self.assertEqual(self.queue.buffer_size, 5)
        self.assertEqual(self.queue.queue_length(), 0)
        self.assertEqual(self.queue.first_frame_id, 0)
        self.assertEqual(self.queue.last_frame_id, 0)
        self.assertEqual(self.queue.last_seen_id, 0)
    
    def test_add_frame(self):
        """Test adding frames to the queue."""
        frame1 = Frame(1, self.test_frame_data.copy())
        frame2 = Frame(2, self.test_frame_data.copy())
        
        self.queue.add_frame(frame1)
        self.assertEqual(self.queue.queue_length(), 1)
        self.assertEqual(self.queue.last_frame_id, 2)
        
        self.queue.add_frame(frame2)
        self.assertEqual(self.queue.queue_length(), 2)
        self.assertEqual(self.queue.last_frame_id, 3)
    
    def test_add_frame_invalid_type(self):
        """Test adding invalid frame type raises AssertionError."""
        with self.assertRaises(AssertionError):
            self.queue.add_frame("invalid_frame")  # type: ignore
    
    def test_buffer_overflow(self):
        """Test automatic removal of old frames when buffer overflows."""
        # Add frames up to buffer size
        for i in range(6):  # More than buffer_size (5)
            frame = Frame(i, self.test_frame_data.copy())
            self.queue.add_frame(frame)
        # Should maintain buffer_size
        self.assertEqual(self.queue.queue_length(), 5)
        # Oldest frame (ID 0) should be removed
        self.assertEqual(self.queue.first_frame_id, 1)
    
    def test_pop_frame(self):
        """Test removing frames from the queue."""
        frame1 = Frame(1, self.test_frame_data.copy())
        frame2 = Frame(2, self.test_frame_data.copy())
        
        self.queue.add_frame(frame1)
        self.queue.add_frame(frame2)
        
        popped_frame = self.queue.pop_frame()
        self.assertIsNotNone(popped_frame)
        if popped_frame is not None:
            self.assertEqual(popped_frame.frame_id, 1)
        self.assertEqual(self.queue.queue_length(), 1)
        self.assertEqual(self.queue.last_frame_id, 3)
    
    def test_pop_frame_empty_queue(self):
        """Test popping from empty queue returns None."""
        popped_frame = self.queue.pop_frame()
        self.assertIsNone(popped_frame)
    
    def test_get_tail(self):
        """Test retrieving recent frames from the queue."""
        # Add multiple frames
        for i in range(4):
            frame = Frame(i, self.test_frame_data.copy())
            self.queue.add_frame(frame)
        
        # Get last 2 frames
        tail_frames = self.queue.get_tail(2)
        self.assertEqual(len(tail_frames), 2)
        self.assertEqual(tail_frames[0].frame_id, 2)
        self.assertEqual(tail_frames[1].frame_id, 3)
    
    def test_get_tail_more_than_available(self):
        """Test getting more frames than available."""
        frame = Frame(1, self.test_frame_data.copy())
        self.queue.add_frame(frame)
        
        tail_frames = self.queue.get_tail(5)
        self.assertEqual(len(tail_frames), 1)
    
    def test_get_frame_non_processed(self):
        """Test retrieving unprocessed frames."""
        # Add frames with different processed states
        frame1 = Frame(1, self.test_frame_data.copy())
        frame2 = Frame(2, self.test_frame_data.copy())
        frame3 = Frame(3, self.test_frame_data.copy())
        
        frame1.processed = True
        frame2.processed = False
        frame3.processed = False
        
        self.queue.add_frame(frame1)
        self.queue.add_frame(frame2)
        self.queue.add_frame(frame3)
        
        unprocessed = self.queue.get_frame_non_processed(2)
        self.assertEqual(len(unprocessed), 2)
        self.assertFalse(unprocessed[0].processed)
        self.assertFalse(unprocessed[1].processed)
    
    def test_get_frame_non_processed_with_skip(self):
        """Test retrieving unprocessed frames with skipping."""
        # Add multiple unprocessed frames
        for i in range(5):
            frame = Frame(i, self.test_frame_data.copy())
            frame.processed = False
            self.queue.add_frame(frame)
        
        # Get 2 frames with skip=1
        frames = self.queue.get_frame_non_processed(2, n_skip=1)

        self.assertEqual(len(frames), 2)
        # Should skip every other frame
        self.assertEqual(frames[0].frame_id, 1)
        self.assertEqual(frames[1].frame_id, 2)
    
    def test_get_range(self):
        """Test retrieving a range of frames by ID."""
        # Add frames
        for i in range(5):
            frame = Frame(i, self.test_frame_data.copy())
            self.queue.add_frame(frame)
        
        # Get range from ID 1 to 3
        next_id, frames = self.queue.get_range(1, 3)
        self.assertEqual(next_id, 4)
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[0].frame_id, 1)
        self.assertEqual(frames[1].frame_id, 2)
        self.assertEqual(frames[2].frame_id, 3)
    
    def test_get_range_out_of_bounds(self):
        """Test getting range with out-of-bounds IDs."""
        # Add frames
        for i in range(3):
            frame = Frame(i, self.test_frame_data.copy())
            self.queue.add_frame(frame)
        
        # Try to get range starting before available frames
        next_id, frames = self.queue.get_range(-1, 2)
        self.assertEqual(next_id, 2)
        self.assertEqual(len(frames), 2)
    
    def test_get_by_id(self):
        """Test retrieving and removing a specific frame by ID."""
        frame = Frame(1, self.test_frame_data.copy())
        self.queue.add_frame(frame)
        
        retrieved_frame = self.queue.get_by_id(1)
        self.assertIsNotNone(retrieved_frame)
        if retrieved_frame is not None:
            self.assertEqual(retrieved_frame.frame_id, 1)
        self.assertEqual(self.queue.queue_length(), 0)
    
    def test_get_by_id_not_found(self):
        """Test retrieving non-existent frame ID."""
        retrieved_frame = self.queue.get_by_id(1001)
        self.assertIsNone(retrieved_frame)
    
    def test_singleton_pattern(self):
        """Test singleton pattern implementation."""
        instance1 = CircleQueue.get_instance()
        instance2 = CircleQueue.get_instance()
        self.assertIs(instance1, instance2)
    
    def test_thread_safety(self):
        """Test thread safety of queue operations."""
        def add_frames():
            for i in range(10):
                frame = Frame(i, self.test_frame_data.copy())
                self.queue.add_frame(frame)
                time.sleep(0.01)
        
        def remove_frames():
            for _ in range(10):
                self.queue.pop_frame()
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
        
        # Queue should be in a consistent state
        self.assertGreaterEqual(self.queue.queue_length(), 0)
        self.assertLessEqual(self.queue.queue_length(), self.queue.buffer_size)


if __name__ == '__main__':
    unittest.main() 