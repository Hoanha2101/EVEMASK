from threading import Semaphore
from typing import List, Optional
from collections import OrderedDict
from .frame import Frame


class CircleQueue:
    _global_instance: "CircleQueue" = None
    _tail_lock = Semaphore(1)

    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.frames: OrderedDict[int, Frame] = OrderedDict()
        self.first_frame_id = 0
        self.last_frame_id = 0
        
        self.last_seen_id = 0

    def add_frame(self, frame: Frame):
        assert isinstance(frame, Frame), "Input must be of type Frame"
        self.frames[frame.frame_id] = frame
        self.last_frame_id = max(self.last_frame_id, frame.frame_id + 1)

        # Auto-remove old frames if buffer overflows
        while self.queue_length() >= self.buffer_size:
            self.frames.popitem(last=False)
            self.first_frame_id += 1

    def pop_frame(self) -> Optional[Frame]:
        if self.queue_length() > 0 and self.frames:
            self.first_frame_id += 1
            return self.frames.popitem(last=False)[1]
        return None

    def queue_length(self) -> int:
        return len(self.frames)

    def get_tail(self, count: int) -> List[Frame]:
        count = min(count, self.queue_length())
        return list(self.frames.values())[-count:]

    def get_frame_non_processed(self, count: int, n_skip: int = 0) -> List[Frame]:
        frames_list: List[Frame] = []
        mask_n_skip = 1
        for frame in self.frames.values():
            if frame.frame_id > self.last_seen_id:
                if not frame.processed and frame.frame_data is not None:
                    if mask_n_skip <= n_skip:
                        mask_n_skip += 1
                        continue
                    else:
                        frames_list.append(frame)
                        if len(frames_list) >= count:
                            self.last_seen_id = frame.frame_id
                            break
        return frames_list
        

    def get_range(self, start_id: int, count: int) -> (int, List[Frame]):
        frames_list: List[Frame] = []
        if start_id < self.first_frame_id:
            start_id = self.first_frame_id
        if start_id + count > self.last_frame_id:
            count = self.last_frame_id - start_id

        for i in range(start_id, start_id + count):
            frame = self.frames.get(i)
            if frame:
                frames_list.append(frame)

        return (start_id + count), frames_list

    def get_by_id(self, frame_id: int) -> Optional[Frame]:
        frame = self.frames.get(frame_id)
        del self.frames[frame_id]
        return frame

    def buffer_capacity(self) -> int:
        return self.buffer_size

    @classmethod
    def get_instance(cls) -> "CircleQueue":
        if cls._global_instance is None:
            cls._global_instance = CircleQueue()
        return cls._global_instance
