"""
Circular Queue Module
Thread-safe circular buffer implementation for frame management in real-time video processing.

This module provides:
- Thread-safe frame buffering with automatic overflow management
- Efficient frame retrieval with skipping capabilities
- Memory management for real-time video processing
- Singleton pattern for global access

Key Features:
- Automatic removal of old frames when buffer is full
- Support for frame skipping to optimize processing
- Thread-safe operations using semaphores
- Ordered frame storage with ID-based access

Author: EVEMASK Team
"""

from threading import Semaphore
from typing import List, Optional
from collections import OrderedDict
from .frame import Frame


class CircleQueue:
    """
    Thread-safe circular queue for frame buffering.
    
    This class implements a circular buffer that automatically manages memory
    by removing old frames when the buffer reaches capacity. It provides
    efficient frame storage and retrieval for real-time video processing.
    
    Attributes:
        buffer_size: Maximum number of frames to store
        frames: Ordered dictionary storing frames by ID
        first_frame_id: ID of the oldest frame in buffer
        last_frame_id: ID of the newest frame in buffer
        last_seen_id: Last frame ID that was processed
    """
    
    _global_instance: "CircleQueue" = None
    _tail_lock = Semaphore(1)

    def __init__(self, buffer_size: int = 1000):
        """
        Initialize circular queue with specified buffer size.
        
        Args:
            buffer_size (int): Maximum number of frames to store in buffer
        """
        self.buffer_size = buffer_size
        self.frames: OrderedDict[int, Frame] = OrderedDict()
        self.first_frame_id = 0
        self.last_frame_id = 0
        
        # Track last processed frame for skipping logic
        self.last_seen_id = 0

    def add_frame(self, frame: Frame):
        """
        Add a frame to the circular queue.
        
        This method adds a frame to the buffer and automatically removes
        old frames if the buffer exceeds capacity.
        
        Args:
            frame (Frame): Frame object to add to buffer
            
        Raises:
            AssertionError: If input is not a Frame object
        """
        # Validate input type
        assert isinstance(frame, Frame), "Input must be of type Frame"
        
        # Add frame to buffer
        self.frames[frame.frame_id] = frame
        self.last_frame_id = max(self.last_frame_id, frame.frame_id + 1)

        # Auto-remove old frames if buffer overflows
        # This maintains constant memory usage
        while self.queue_length() >= self.buffer_size:
            self.frames.popitem(last=False)  # Remove oldest frame
            self.first_frame_id += 1

    def pop_frame(self) -> Optional[Frame]:
        """
        Remove and return the oldest frame from the queue.
        
        Returns:
            Frame or None: Oldest frame if available, None if queue is empty
        """
        if self.queue_length() > 0 and self.frames:
            self.first_frame_id += 1
            return self.frames.popitem(last=False)[1]  # Return frame data
        return None

    def queue_length(self) -> int:
        """
        Get current number of frames in the queue.
        
        Returns:
            int: Number of frames currently stored
        """
        return len(self.frames)

    def get_tail(self, count: int) -> List[Frame]:
        """
        Get the most recent frames from the queue.
        
        Args:
            count (int): Number of recent frames to retrieve
            
        Returns:
            List[Frame]: List of the most recent frames
        """
        count = min(count, self.queue_length())
        return list(self.frames.values())[-count:]

    def get_frame_non_processed(self, count: int, n_skip: int = 0) -> List[Frame]:
        """
        Get unprocessed frames with optional skipping.
        
        This method retrieves frames that haven't been processed yet,
        with support for frame skipping to optimize processing performance.
        
        Args:
            count (int): Number of frames to retrieve
            n_skip (int): Number of frames to skip between retrievals
            
        Returns:
            List[Frame]: List of unprocessed frames
        """
        frames_list: List[Frame] = []
        mask_n_skip = 1  # Counter for skip logic
        
        # Iterate through frames in order
        for frame in self.frames.values():
            # Only process frames newer than last seen
            if frame.frame_id > self.last_seen_id:
                # Check if frame is unprocessed and has data
                if not frame.processed and frame.frame_data is not None:
                    # Apply frame skipping logic
                    if mask_n_skip <= n_skip:
                        mask_n_skip += 1
                        continue  # Skip this frame
                    else:
                        # Add frame to result list
                        frames_list.append(frame)
                        if len(frames_list) >= count:
                            # Update last seen ID and stop
                            self.last_seen_id = frame.frame_id
                            break
        return frames_list
        

    def get_range(self, start_id: int, count: int) -> (int, List[Frame]):
        """
        Get a range of frames by ID.
        
        Args:
            start_id (int): Starting frame ID
            count (int): Number of frames to retrieve
            
        Returns:
            tuple: (next_id, list_of_frames) where next_id is the ID after the range
        """
        frames_list: List[Frame] = []
        
        # Adjust start_id if it's before the first available frame
        if start_id < self.first_frame_id:
            start_id = self.first_frame_id
            
        # Adjust count if it exceeds available frames
        if start_id + count > self.last_frame_id:
            count = self.last_frame_id - start_id

        # Collect frames in the specified range
        for i in range(start_id, start_id + count):
            frame = self.frames.get(i)
            if frame:
                frames_list.append(frame)

        return (start_id + count), frames_list

    def get_by_id(self, frame_id: int) -> Optional[Frame]:
        """
        Get and remove a specific frame by ID.
        
        Args:
            frame_id (int): ID of the frame to retrieve
            
        Returns:
            Frame or None: Requested frame if found, None otherwise
        """
        frame = self.frames.get(frame_id)
        if frame:
            del self.frames[frame_id]  # Remove frame from buffer
        return frame

    def buffer_capacity(self) -> int:
        """
        Get the maximum capacity of the buffer.
        
        Returns:
            int: Maximum number of frames the buffer can hold
        """
        return self.buffer_size

    @classmethod
    def get_instance(cls) -> "CircleQueue":
        """
        Get singleton instance of CircleQueue.
        
        This method ensures only one instance of CircleQueue exists
        throughout the application, providing global access to the frame buffer.
        
        Returns:
            CircleQueue: Singleton instance
        """
        if cls._global_instance is None:
            cls._global_instance = CircleQueue()
        return cls._global_instance
