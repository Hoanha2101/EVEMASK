from threading import Lock, RLock
from typing import List, Optional, Any, Iterator
import time


class CircleQueue:
    """
    Thread-safe circular queue implementation with fixed size buffer
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize circular queue
        
        Args:
            max_size: Maximum number of items in the queue
        """
        self.max_size = max_size
        self.buffer = [None] * max_size
        self.head = 0  # Points to the next position to write
        self.tail = 0  # Points to the next position to read
        self.size = 0  # Current number of items
        self.lock = RLock()  # Reentrant lock for thread safety
        
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        with self.lock:
            return self.size == 0
    
    def is_full(self) -> bool:
        """Check if queue is full"""
        with self.lock:
            return self.size == self.max_size
    
    def put(self, item: Any) -> bool:
        """
        Add item to the queue
        
        Args:
            item: Item to add
            
        Returns:
            True if successful, False if queue is full
        """
        with self.lock:
            if self.size >= self.max_size:
                return False
                
            self.buffer[self.head] = item
            self.head = (self.head + 1) % self.max_size
            self.size += 1
            return True
    
    def put_force(self, item: Any) -> Optional[Any]:
        """
        Add item to queue, overwriting oldest item if full
        
        Args:
            item: Item to add
            
        Returns:
            Overwritten item if queue was full, None otherwise
        """
        with self.lock:
            overwritten = None
            
            if self.size >= self.max_size:
                # Queue is full, overwrite oldest item
                overwritten = self.buffer[self.tail]
                self.tail = (self.tail + 1) % self.max_size
                self.size -= 1
            
            self.buffer[self.head] = item
            self.head = (self.head + 1) % self.max_size
            self.size += 1
            return overwritten
    
    def get(self) -> Optional[Any]:
        """
        Remove and return the oldest item from queue
        
        Returns:
            Oldest item or None if queue is empty
        """
        with self.lock:
            if self.size == 0:
                return None
                
            item = self.buffer[self.tail]
            self.buffer[self.tail] = None  # Clear reference
            self.tail = (self.tail + 1) % self.max_size
            self.size -= 1
            return item
    
    def peek(self) -> Optional[Any]:
        """
        Return the oldest item without removing it
        
        Returns:
            Oldest item or None if queue is empty
        """
        with self.lock:
            if self.size == 0:
                return None
            return self.buffer[self.tail]
    
    def peek_newest(self) -> Optional[Any]:
        """
        Return the newest item without removing it
        
        Returns:
            Newest item or None if queue is empty
        """
        with self.lock:
            if self.size == 0:
                return None
            newest_index = (self.head - 1) % self.max_size
            return self.buffer[newest_index]
    
    def get_last_n(self, n: int) -> List[Any]:
        """
        Get the last n items without removing them
        
        Args:
            n: Number of items to get
            
        Returns:
            List of last n items (newest first)
        """
        with self.lock:
            if self.size == 0:
                return []
            
            n = min(n, self.size)
            items = []
            
            for i in range(n):
                index = (self.head - 1 - i) % self.max_size
                items.append(self.buffer[index])
            
            return items
    
    def get_first_n(self, n: int) -> List[Any]:
        """
        Get the first n items without removing them
        
        Args:
            n: Number of items to get
            
        Returns:
            List of first n items (oldest first)
        """
        with self.lock:
            if self.size == 0:
                return []
            
            n = min(n, self.size)
            items = []
            
            for i in range(n):
                index = (self.tail + i) % self.max_size
                items.append(self.buffer[index])
            
            return items
    
    def get_batch(self, batch_size: int) -> List[Any]:
        """
        Remove and return up to batch_size items
        
        Args:
            batch_size: Maximum number of items to get
            
        Returns:
            List of items (oldest first)
        """
        with self.lock:
            items = []
            count = min(batch_size, self.size)
            
            for _ in range(count):
                item = self.get()
                if item is not None:
                    items.append(item)
            
            return items
    
    def clear(self):
        """Clear all items from the queue"""
        with self.lock:
            self.buffer = [None] * self.max_size
            self.head = 0
            self.tail = 0
            self.size = 0
    
    def current_size(self) -> int:
        """Get current number of items in queue"""
        with self.lock:
            return self.size
    
    def remaining_capacity(self) -> int:
        """Get remaining capacity"""
        with self.lock:
            return self.max_size - self.size
    
    def to_list(self) -> List[Any]:
        """Convert queue to list (oldest first)"""
        with self.lock:
            items = []
            for i in range(self.size):
                index = (self.tail + i) % self.max_size
                items.append(self.buffer[index])
            return items
    
    def __len__(self) -> int:
        """Return current size of queue"""
        return self.current_size()
    
    def __bool__(self) -> bool:
        """Return True if queue is not empty"""
        return not self.is_empty()
    
    def __iter__(self) -> Iterator[Any]:
        """Iterate over queue items (oldest first)"""
        with self.lock:
            items = self.to_list()
        return iter(items)
    
    def __str__(self) -> str:
        """String representation"""
        with self.lock:
            return f"CircleQueue(size={self.size}/{self.max_size}, items={self.to_list()})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        with self.lock:
            return f"CircleQueue(max_size={self.max_size}, size={self.size}, head={self.head}, tail={self.tail})"


class TimedCircleQueue(CircleQueue):
    """
    Circle queue with timestamp tracking
    """
    
    def __init__(self, max_size: int = 100, max_age_seconds: Optional[float] = None):
        """
        Initialize timed circular queue
        
        Args:
            max_size: Maximum number of items
            max_age_seconds: Maximum age of items before they expire
        """
        super().__init__(max_size)
        self.max_age_seconds = max_age_seconds
        self.timestamps = [None] * max_size
        
    def put(self, item: Any) -> bool:
        """Add item with timestamp"""
        timestamp = time.time()
        with self.lock:
            if self.size >= self.max_size:
                return False
                
            self.buffer[self.head] = item
            self.timestamps[self.head] = timestamp
            self.head = (self.head + 1) % self.max_size
            self.size += 1
            return True
    
    def put_force(self, item: Any) -> Optional[Any]:
        """Add item with timestamp, overwriting if full"""
        timestamp = time.time()
        with self.lock:
            overwritten = None
            
            if self.size >= self.max_size:
                overwritten = self.buffer[self.tail]
                self.tail = (self.tail + 1) % self.max_size
                self.size -= 1
            
            self.buffer[self.head] = item
            self.timestamps[self.head] = timestamp
            self.head = (self.head + 1) % self.max_size
            self.size += 1
            return overwritten
    
    def get(self) -> Optional[Any]:
        """Get item, removing expired items first"""
        with self.lock:
            self._remove_expired()
            return super().get()
    
    def _remove_expired(self):
        """Remove expired items from the front of queue"""
        if self.max_age_seconds is None:
            return
            
        current_time = time.time()
        while self.size > 0:
            item_time = self.timestamps[self.tail]
            if item_time is None or (current_time - item_time) <= self.max_age_seconds:
                break
                
            # Remove expired item
            self.buffer[self.tail] = None
            self.timestamps[self.tail] = None
            self.tail = (self.tail + 1) % self.max_size
            self.size -= 1
    
    def get_item_age(self, item: Any) -> Optional[float]:
        """Get age of specific item in seconds"""
        current_time = time.time()
        with self.lock:
            for i in range(self.size):
                index = (self.tail + i) % self.max_size
                if self.buffer[index] == item and self.timestamps[index] is not None:
                    return current_time - self.timestamps[index]
        return None
    
    def clear_expired(self) -> int:
        """Clear all expired items and return count"""
        with self.lock:
            initial_size = self.size
            self._remove_expired()
            return initial_size - self.size


# Example usage and test
if __name__ == "__main__":
    # Basic usage
    queue = CircleQueue(max_size=5)
    
    # Add items
    for i in range(3):
        queue.put(f"item_{i}")
    
    print(f"Queue: {queue}")
    print(f"Size: {len(queue)}")
    print(f"Peek: {queue.peek()}")
    print(f"Peek newest: {queue.peek_newest()}")
    
    # Get items
    print(f"Get: {queue.get()}")
    print(f"Get batch: {queue.get_batch(2)}")
    print(f"After getting: {queue}")
    
    # Test force put
    for i in range(10):
        overwritten = queue.put_force(f"new_item_{i}")
        if overwritten:
            print(f"Overwritten: {overwritten}")
    
    print(f"Final queue: {queue}")
    
    # Test timed queue
    print("\n--- Timed Queue Test ---")
    timed_queue = TimedCircleQueue(max_size=5, max_age_seconds=2.0)
    
    for i in range(3):
        timed_queue.put(f"timed_item_{i}")
    
    print(f"Timed queue: {timed_queue}")
    
    # Wait and add more items
    time.sleep(1)
    timed_queue.put("recent_item")
    
    print(f"After 1 second: {timed_queue}")
    
    # Test expiration
    time.sleep(2)
    expired_count = timed_queue.clear_expired()
    print(f"Expired items: {expired_count}")
    print(f"After expiration: {timed_queue}")