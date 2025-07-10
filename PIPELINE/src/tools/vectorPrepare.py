import cv2
import torch
import numpy as np
from glob import glob
import os
from ..models.initNet import net2
from ..tools import set_seed

set_seed(42)

class VectorPrepare:
    def __init__(self, orgFolderPath, enginePlan):
        self.orgFolderPath = orgFolderPath
        self.enginePlan = enginePlan

    def preprocess_image(self, image_path):
        """
        Preprocessing giống hệt như tạo emb1 trong code chuẩn
        (hàm create_embedding_tensor_deterministic)
        """
        # Đọc ảnh với flag cố định
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Resize và chuyển BGR -> RGB
        img_process = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        img_process = cv2.cvtColor(img_process, cv2.COLOR_BGR2RGB)
        img_process = img_process.astype(np.float32)

        # HWC -> CHW -> NCHW và to CUDA
        img_tensor = torch.from_numpy(img_process).permute(2, 0, 1).unsqueeze(0).cuda().half()
        
        # Ensure correct shape
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Make tensor contiguous
        img_tensor = img_tensor.contiguous()
        
        return img_tensor

    def load_images_from_folder(self, folder_path):
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
        image_paths = []
        for ext in exts:
            image_paths.extend(glob(os.path.join(folder_path, ext)))
        
        image_paths = sorted(image_paths)
        return image_paths

    def extract_features(self, image_paths):
        features = []
        for path in image_paths:
            tensor = self.preprocess_image(path)
            emb = self.enginePlan.infer(tensor)
            features.append(emb)
        return torch.cat(features, dim=0)

    def run(self):
        all_features = []
        all_names = []

        subfolders = sorted(os.listdir(self.orgFolderPath))
        for folder in subfolders:
            full_folder_path = os.path.join(self.orgFolderPath, folder)
            if not os.path.isdir(full_folder_path):
                continue

            image_paths = self.load_images_from_folder(full_folder_path)
            if not image_paths:
                continue
            
            feature_tensor = self.extract_features(image_paths)
            all_features.append(feature_tensor)
            all_names.extend(image_paths)

        return torch.cat(all_features, dim=0).cpu().numpy(), all_names
