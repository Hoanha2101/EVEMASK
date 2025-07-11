import cv2
from ..tools import *

class Frame:
    def __init__(self, frame_id, frame_data):
        self.frame_id = frame_id
        self.frame_data = frame_data
        self.processed = False

    def get_data(self):
        return self.frame_id ,self.frame_data

    def framePreprocessing(self):
        original_frame = self.frame_data
        input_image, ratio, dwdh = letterbox(original_frame, (640, 640))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image, img_no_255, seg_img = blob(input_image, return_seg=True)
        img_tensor = np.ascontiguousarray(input_image)
        data = (original_frame, img_tensor, ratio, dwdh, img_no_255)
        return data

    def destroy(self):
        """
        Xóa dữ liệu ảnh để giải phóng bộ nhớ.
        """
        self.frame_id = None
        self.frame_data = None
        self.processed = None
    