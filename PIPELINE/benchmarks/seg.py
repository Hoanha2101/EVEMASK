import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.initNet import net1
from src.tools.utils import *


path_images = "DATASET/images/val"

folders = os.listdir(path_images)

def seg_benchmark_trt(folder):
    for file in os.listdir(folder):
        if file.endswith(".jpg", ".png", ".jpeg"):
            frame_data = cv2.imread(os.path.join(folder, file))
            # Store original frame for reference
            original_frame = frame_data
            
            # Letterboxing: resize image to 640x640 while maintaining aspect ratio
            # This adds padding if necessary to avoid distortion
            input_image, ratio, dwdh = letterbox(original_frame, (640, 640))
            
            # Convert from BGR to RGB color space (OpenCV uses BGR, models expect RGB)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            
            # Blob preprocessing: normalize and prepare for neural network input
            # This includes normalization and optional segmentation preparation
            input_image, img_no_255, seg_img = blob(input_image, return_seg=True)
            
            # Ensure tensor is contiguous in memory for efficient processing
            img_tensor = np.ascontiguousarray(input_image)
            
            # 0 - original_frame: (1080, 1920, 3) <class 'numpy.ndarray'> unit 8
            # 1 - img_tensor: (1, 3, 640, 640) <class 'numpy.ndarray'> float16
            # 2 - ratio: 0.3333333333333333
            # 3 - dwdh: (0.0, 140.0)
            # 4 - img_no_255: (1, 3, 640, 640) <class 'numpy.ndarray'> float16
            
            # Package all data for return
            data = (original_frame, img_tensor, ratio, dwdh, img_no_255)
            
            







