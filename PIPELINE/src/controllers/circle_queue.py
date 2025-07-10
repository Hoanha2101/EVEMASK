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

    def add_frame(self, frame: Frame):
        assert isinstance(frame, Frame), "Input must be of type Frame"
        self.frames[frame.frame_id] = frame
        self.last_frame_id = max(self.last_frame_id, frame.frame_id + 1)

        # Auto-remove old frames if buffer overflows
        while self.queue_length() > self.buffer_size:
            self.frames.popitem(last=False)
            self.first_frame_id += 1

    def pop_frame(self) -> Optional[Frame]:
        if self.queue_length() > 0 and self.frames:
            self.first_frame_id += 1
            return self.frames.popitem(last=False)[1]
        return None

    def queue_length(self) -> int:
        return self.last_frame_id - self.first_frame_id

    def get_tail(self, count: int) -> List[Frame]:
        count = min(count, self.queue_length())
        return list(self.frames.values())[-count:]

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
        return self.frames.get(frame_id)


    def get_list_by_id_range(self, from_id: int, to_id: int) -> List[Frame]:
        """
        Lấy danh sách các Frame theo khoảng frame_id (bao gồm from_id và to_id - 1).
        """
        frames_list: List[Frame] = []
        for frame_id in range(from_id, to_id):
            frame = self.frames.get(frame_id)
            if frame:
                frames_list.append(frame)
        return frames_list

    def replace_frame_with_data(self, frame_id: int, frame_data, processedbool: bool = True) -> bool:
        """
        Thay thế frame có sẵn trong queue bằng frame_data mới.
        Tự động tạo Frame mới từ frame_id và frame_data.
        """
        if frame_id in self.frames:
            new_frame = Frame(frame_id=frame_id, frame_data=frame_data)
            new_frame.processed = processedbool
            self.frames[frame_id] = new_frame
            return True
        return False

    def remove_by_id(self, frame_id: int) -> Optional[Frame]:
        return self.frames.pop(frame_id, None)

    def buffer_capacity(self) -> int:
        return self.buffer_size

    @classmethod
    def get_instance(cls) -> "CircleQueue":
        if cls._global_instance is None:
            cls._global_instance = CircleQueue()
        return cls._global_instance
